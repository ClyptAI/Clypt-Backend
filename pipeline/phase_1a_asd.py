#!/usr/bin/env python3
"""
Phase 1A-ASD: Local Active Speaker Timeline (ASD Integration Hook)
===================================================================
Consumes raw ASD predictions (TalkNet/LoCoNet-compatible shape) and maps
them onto Video Intelligence face tracks, producing a confidence-gated
active speaker timeline.

Expected raw input (flexible):
  - outputs/phase_1a_loconet_raw.json

Supported shapes:
  1) {"frames":[{"time_ms":123, "detections":[{"bbox":{...},"speaking_score":0.91}, ...]}, ...]}
  2) [{"time_ms":123, "detections":[...]}, ...]
  3) [{"time_ms":123, "bbox":{...}, "speaking_score":0.91}, ...]

Output:
  - outputs/phase_1a_active_speaker_timeline.json
"""

from __future__ import annotations

import json
import logging
import statistics
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VISUAL_PATH = ROOT / "outputs" / "phase_1a_visual.json"
SPEAKER_MAP_PATH = ROOT / "outputs" / "phase_1a_speaker_map.json"
LOCONET_RAW_PATH = ROOT / "outputs" / "phase_1a_loconet_raw.json"
OUTPUT_PATH = ROOT / "outputs" / "phase_1a_active_speaker_timeline.json"

TIME_TOLERANCE_MS = 180
MIN_IOU = 0.1
ENTER_THRESHOLD = 0.65
EXIT_THRESHOLD = 0.45
SWITCH_MARGIN = 0.08
MIN_SWITCH_GAP_MS = 250
MIN_SEGMENT_MS = 220
WORD_SLACK_MS = 120
BOOTSTRAP_ACTIVE_SCORE = 0.82
BOOTSTRAP_INACTIVE_SCORE = 0.2
BOOTSTRAP_UNKNOWN_SCORE = 0.28

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase_1a_asd")


@dataclass
class FaceFrame:
    time_ms: int
    track_id: int
    bbox: dict


@dataclass
class PredFrame:
    time_ms: int
    bbox: dict
    score: float


def _bbox_area(b: dict) -> float:
    w = max(0.0, float(b["right"]) - float(b["left"]))
    h = max(0.0, float(b["bottom"]) - float(b["top"]))
    return w * h


def _bbox_iou(a: dict, b: dict) -> float:
    left = max(float(a["left"]), float(b["left"]))
    top = max(float(a["top"]), float(b["top"]))
    right = min(float(a["right"]), float(b["right"]))
    bottom = min(float(a["bottom"]), float(b["bottom"]))
    inter_w = max(0.0, right - left)
    inter_h = max(0.0, bottom - top)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _bbox_center_distance(a: dict, b: dict) -> float:
    ax = (float(a["left"]) + float(a["right"])) / 2
    ay = (float(a["top"]) + float(a["bottom"])) / 2
    bx = (float(b["left"]) + float(b["right"])) / 2
    by = (float(b["top"]) + float(b["bottom"])) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _normalize_bbox(raw: dict | None) -> dict | None:
    if not isinstance(raw, dict):
        return None
    if all(k in raw for k in ("left", "top", "right", "bottom")):
        return {
            "left": float(raw["left"]),
            "top": float(raw["top"]),
            "right": float(raw["right"]),
            "bottom": float(raw["bottom"]),
        }
    # Fallback keys.
    if all(k in raw for k in ("x1", "y1", "x2", "y2")):
        return {
            "left": float(raw["x1"]),
            "top": float(raw["y1"]),
            "right": float(raw["x2"]),
            "bottom": float(raw["y2"]),
        }
    return None


def _extract_predictions(raw_obj) -> list[PredFrame]:
    frames = raw_obj.get("frames", raw_obj) if isinstance(raw_obj, dict) else raw_obj
    if not isinstance(frames, list):
        return []

    preds: list[PredFrame] = []
    for item in frames:
        if not isinstance(item, dict):
            continue
        t = item.get("time_ms", item.get("timestamp_ms", item.get("time")))
        if t is None:
            continue
        time_ms = int(float(t))

        if isinstance(item.get("detections"), list):
            for d in item["detections"]:
                if not isinstance(d, dict):
                    continue
                bbox = _normalize_bbox(d.get("bbox", d.get("bounding_box")))
                if not bbox:
                    continue
                score_raw = d.get("speaking_score", d.get("score", d.get("confidence", 0.0)))
                try:
                    score = float(score_raw)
                except (TypeError, ValueError):
                    score = 0.0
                preds.append(PredFrame(time_ms=time_ms, bbox=bbox, score=score))
            continue

        bbox = _normalize_bbox(item.get("bbox", item.get("bounding_box")))
        if not bbox:
            continue
        score_raw = item.get("speaking_score", item.get("score", item.get("confidence", 0.0)))
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = 0.0
        preds.append(PredFrame(time_ms=time_ms, bbox=bbox, score=score))

    preds.sort(key=lambda p: p.time_ms)
    return preds


def _load_face_frames() -> list[FaceFrame]:
    visual = json.loads(VISUAL_PATH.read_text())
    out: list[FaceFrame] = []
    for track_id, face_track in enumerate(visual.get("face_detections", [])):
        for ts in face_track.get("timestamped_objects", []):
            bbox = _normalize_bbox(ts.get("bounding_box") or ts.get("normalized_bounding_box"))
            if not bbox:
                continue
            out.append(
                FaceFrame(
                    time_ms=int(ts.get("time_ms", 0)),
                    track_id=track_id,
                    bbox=bbox,
                )
            )
    out.sort(key=lambda f: f.time_ms)
    return out


def _load_track_to_speaker() -> dict[int, str]:
    if not SPEAKER_MAP_PATH.exists():
        return {}
    m = json.loads(SPEAKER_MAP_PATH.read_text())
    track_to_speaker: dict[int, str] = {}
    for tag, tracks in (m.get("speaker_to_tracks") or {}).items():
        if not isinstance(tracks, list):
            continue
        for tid in tracks:
            try:
                track_to_speaker[int(tid)] = str(tag)
            except (TypeError, ValueError):
                pass
    for tag, tid in (m.get("speaker_to_track") or {}).items():
        try:
            track_to_speaker.setdefault(int(tid), str(tag))
        except (TypeError, ValueError):
            pass
    return track_to_speaker


def _merge_intervals(intervals: list[tuple[int, int]], gap_ms: int = 120) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap_ms:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _build_word_intervals_by_speaker() -> tuple[dict[str, list[tuple[int, int]]], dict[str, list[int]]]:
    audio_path = ROOT / "outputs" / "phase_1a_audio.json"
    if not audio_path.exists():
        return {}, {}

    audio = json.loads(audio_path.read_text())
    words = audio.get("words", [])
    by_tag: dict[str, list[tuple[int, int]]] = {}
    for w in words:
        tag = w.get("speaker_tag")
        if not tag or tag == "unknown":
            continue
        try:
            start = int(w.get("start_time_ms", 0))
            end = int(w.get("end_time_ms", start))
        except (TypeError, ValueError):
            continue
        if end < start:
            end = start
        by_tag.setdefault(str(tag), []).append((start, end))

    merged: dict[str, list[tuple[int, int]]] = {
        tag: _merge_intervals(intervals)
        for tag, intervals in by_tag.items()
    }
    starts: dict[str, list[int]] = {
        tag: [s for s, _ in intervals]
        for tag, intervals in merged.items()
    }
    return merged, starts


def _is_speaker_active_at(
    speaker_tag: str,
    time_ms: int,
    intervals_by_tag: dict[str, list[tuple[int, int]]],
    starts_by_tag: dict[str, list[int]],
) -> bool:
    intervals = intervals_by_tag.get(speaker_tag)
    starts = starts_by_tag.get(speaker_tag)
    if not intervals or not starts:
        return False

    idx = bisect_left(starts, time_ms)
    candidate_idxs = (idx - 1, idx)
    for ci in candidate_idxs:
        if 0 <= ci < len(intervals):
            s, e = intervals[ci]
            if s - WORD_SLACK_MS <= time_ms <= e + WORD_SLACK_MS:
                return True
    return False


def _bootstrap_raw_from_ledgers() -> dict | None:
    """Build a pseudo-ASD raw file from STT diarization + face tracks.

    This is a fallback only. If a true ASD model output is present, that file
    is used directly instead.
    """
    if not VISUAL_PATH.exists():
        return None

    faces = _load_face_frames()
    if not faces:
        return None

    track_to_speaker = _load_track_to_speaker()
    intervals_by_tag, starts_by_tag = _build_word_intervals_by_speaker()

    by_time: dict[int, list[dict]] = {}
    for face in faces:
        speaker_tag = track_to_speaker.get(face.track_id)
        if speaker_tag and _is_speaker_active_at(
            speaker_tag,
            face.time_ms,
            intervals_by_tag,
            starts_by_tag,
        ):
            score = BOOTSTRAP_ACTIVE_SCORE
        elif speaker_tag:
            score = BOOTSTRAP_INACTIVE_SCORE
        else:
            score = BOOTSTRAP_UNKNOWN_SCORE

        by_time.setdefault(face.time_ms, []).append({
            "bbox": face.bbox,
            "speaking_score": round(float(score), 4),
        })

    frames = [
        {"time_ms": t, "detections": dets}
        for t, dets in sorted(by_time.items())
        if dets
    ]
    if not frames:
        return None

    return {
        "source": "bootstrap_stt_face",
        "frames": frames,
    }


def _map_predictions_to_tracks(preds: list[PredFrame], faces: list[FaceFrame]) -> list[dict]:
    mapped: list[dict] = []
    if not preds or not faces:
        return mapped

    start = 0
    for pred in preds:
        while start < len(faces) and faces[start].time_ms < pred.time_ms - TIME_TOLERANCE_MS:
            start += 1
        j = start
        best = None
        best_score = -1e9
        while j < len(faces) and faces[j].time_ms <= pred.time_ms + TIME_TOLERANCE_MS:
            f = faces[j]
            iou = _bbox_iou(pred.bbox, f.bbox)
            if iou < MIN_IOU:
                j += 1
                continue
            dist = _bbox_center_distance(pred.bbox, f.bbox)
            # Weighted match score: prioritize overlap, then center distance.
            score = iou - 0.25 * dist
            if score > best_score:
                best = f
                best_score = score
            j += 1
        if best is None:
            continue
        mapped.append({
            "time_ms": pred.time_ms,
            "track_id": best.track_id,
            "score": max(0.0, min(1.0, pred.score)),
        })
    return mapped


def _build_timeline(mapped: list[dict], track_to_speaker: dict[int, str]) -> list[dict]:
    if not mapped:
        return []

    by_time: dict[int, dict[int, float]] = {}
    for e in mapped:
        t = int(e["time_ms"])
        tid = int(e["track_id"])
        s = float(e["score"])
        by_time.setdefault(t, {})
        by_time[t][tid] = max(by_time[t].get(tid, 0.0), s)

    times = sorted(by_time.keys())
    segments: list[dict] = []
    current_track: int | None = None
    current_start: int | None = None
    current_scores: list[float] = []
    last_switch_ms = -10**9

    for t in times:
        scores = by_time[t]
        top_track, top_score = max(scores.items(), key=lambda kv: kv[1])
        current_score = scores.get(current_track, 0.0) if current_track is not None else 0.0

        if current_track is None:
            if top_score >= ENTER_THRESHOLD:
                current_track = top_track
                current_start = t
                current_scores = [top_score]
            continue

        if top_track == current_track:
            if top_score >= EXIT_THRESHOLD:
                current_scores.append(top_score)
                continue
            # Drop if same track confidence collapses.
            seg_end = t
            if current_start is not None and seg_end - current_start >= MIN_SEGMENT_MS:
                segments.append({
                    "start_ms": current_start,
                    "end_ms": seg_end,
                    "track_id": current_track,
                    "confidence": round(float(statistics.mean(current_scores)), 4),
                    "speaker_tag": track_to_speaker.get(current_track),
                })
            current_track = None
            current_start = None
            current_scores = []
            continue

        can_switch = (
            top_score >= ENTER_THRESHOLD
            and top_score >= current_score + SWITCH_MARGIN
            and t - last_switch_ms >= MIN_SWITCH_GAP_MS
        )
        if can_switch:
            seg_end = t
            if current_start is not None and seg_end - current_start >= MIN_SEGMENT_MS:
                segments.append({
                    "start_ms": current_start,
                    "end_ms": seg_end,
                    "track_id": current_track,
                    "confidence": round(float(statistics.mean(current_scores)), 4),
                    "speaker_tag": track_to_speaker.get(current_track),
                })
            current_track = top_track
            current_start = t
            current_scores = [top_score]
            last_switch_ms = t
        elif current_score >= EXIT_THRESHOLD:
            current_scores.append(current_score)

    if current_track is not None and current_start is not None and current_scores:
        seg_end = times[-1]
        if seg_end - current_start >= MIN_SEGMENT_MS:
            segments.append({
                "start_ms": current_start,
                "end_ms": seg_end,
                "track_id": current_track,
                "confidence": round(float(statistics.mean(current_scores)), 4),
                "speaker_tag": track_to_speaker.get(current_track),
            })

    return segments


def main():
    log.info("=" * 60)
    log.info("PHASE 1A-ASD — Local Active Speaker Timeline")
    log.info("=" * 60)

    if not LOCONET_RAW_PATH.exists():
        fallback_raw = _bootstrap_raw_from_ledgers()
        if fallback_raw is None:
            log.warning(
                f"No ASD raw file at {LOCONET_RAW_PATH} and bootstrap fallback could not be built; "
                "skipping ASD timeline generation."
            )
            return
        LOCONET_RAW_PATH.write_text(json.dumps(fallback_raw, indent=2))
        log.info(
            "No ASD raw file found; generated bootstrap ASD raw from STT+face ledgers "
            f"at {LOCONET_RAW_PATH}."
        )
        log.info(
            "For true model-based ASD, replace this file with TalkNet/LoCoNet (or equivalent) predictions."
        )
    if not VISUAL_PATH.exists():
        log.warning(f"No visual ledger at {VISUAL_PATH}; skipping ASD timeline generation.")
        return

    raw = json.loads(LOCONET_RAW_PATH.read_text())
    preds = _extract_predictions(raw)
    faces = _load_face_frames()
    track_to_speaker = _load_track_to_speaker()

    log.info(f"Predictions parsed: {len(preds)}")
    log.info(f"Face frames loaded: {len(faces)}")

    mapped = _map_predictions_to_tracks(preds, faces)
    log.info(f"Predictions mapped to tracks: {len(mapped)}")

    timeline = _build_timeline(mapped, track_to_speaker)
    OUTPUT_PATH.write_text(json.dumps(timeline, indent=2))

    log.info(f"Timeline segments: {len(timeline)}")
    log.info(f"Saved → {OUTPUT_PATH}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
