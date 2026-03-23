"""Render vertical active-speaker-follow clips from current Phase 1 outputs.

This is a lightweight QA renderer:
- picks strong ~40s windows from speaker_bindings
- follows the active speaker's face center when available
- falls back to the person track center when face detections are missing
- outputs 9:16 clips with original audio preserved
"""

from __future__ import annotations

import bisect
import json
import math
import os
import subprocess
import statistics
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VIDEO_PATH = ROOT / "downloads" / "video.mp4"
VISUAL_PATH = ROOT / "outputs" / "phase_1_visual.json"
AUDIO_PATH = ROOT / "outputs" / "phase_1_audio.json"
OUTPUT_DIR = ROOT / "outputs" / "clips"

NUM_CLIPS = 3
CLIP_DURATION_S = 40
WINDOW_STEP_S = 5
KEYFRAME_STEP_S = 0.5
WINDOW_MIN_GAP_S = 20

OUT_W = 1080
OUT_H = 1920
CAMERA_ZOOM = 1.18
Y_HEAD_BIAS = 0.16
MAX_INTERP_GAP_S = 1.5
SPEAKER_GAP_TOLERANCE_MS = 900
EMA_SMOOTHING = 6
OVERLAY_BOX_COLOR = "0x00FF88"
OVERLAY_SECONDARY_BOX_COLOR = "0xFFB347"
OVERLAY_BOX_THICKNESS = 6
MAX_EXPR_KEYFRAMES = 96
FACE_PLAUSIBILITY_MIN_WIDTH_RATIO = 0.10
FACE_PLAUSIBILITY_MAX_WIDTH_RATIO = 0.78
FACE_PLAUSIBILITY_MIN_HEIGHT_RATIO = 0.10
FACE_PLAUSIBILITY_MAX_HEIGHT_RATIO = 0.80
FACE_PLAUSIBILITY_MAX_CENTER_Y_OFFSET_RATIO = 0.10
FACE_PLAUSIBILITY_MAX_TOP_Y_OFFSET_RATIO = 0.03
FACE_PLAUSIBILITY_MIN_ASPECT_RATIO = 0.45
FACE_PLAUSIBILITY_MAX_ASPECT_RATIO = 1.80
SINGLE_PERSON_MIN_DOMINANCE = 0.64
TWO_SPEAKER_MIN_COMBINED_SHARE = 0.82
TWO_SPEAKER_MIN_SECONDARY_SHARE = 0.26
TWO_SPEAKER_MIN_BOTH_VISIBLE_FRACTION = 0.55
SHARED_TWO_SHOT_MAX_CENTER_GAP_RATIO = 0.42
SHARED_TWO_SHOT_MAX_UNION_WIDTH_RATIO = 0.66
SHARED_TWO_SHOT_MAX_UNION_HEIGHT_RATIO = 0.88
SHARED_CAMERA_ZOOM = 1.10
SPLIT_PANEL_HEIGHT = OUT_H // 2


@dataclass
class Detection:
    frame_idx: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float = 0.0


@dataclass(frozen=True)
class TrackWindowStats:
    track_id: str
    speech_ms: float
    word_count: int
    visible_fraction: float
    median_x: float | None
    median_y: float | None
    median_width: float | None
    median_height: float | None


@dataclass(frozen=True)
class CompositionPlan:
    mode: str
    primary_track_id: str | None
    secondary_track_id: str | None = None


@dataclass(frozen=True)
class MotionProfile:
    camera_zoom: float
    keyframe_step_s: float
    ema_smoothing: int
    y_head_bias: float


@dataclass(frozen=True)
class OverlayPath:
    x_keyframes: list[tuple[float, float]]
    y_keyframes: list[tuple[float, float]]
    box_x_keyframes: list[tuple[float, float]]
    box_y_keyframes: list[tuple[float, float]]
    box_w_keyframes: list[tuple[float, float]]
    box_h_keyframes: list[tuple[float, float]]
    color: str = OVERLAY_BOX_COLOR


def _face_detection_is_plausible(person_det: Detection | None, face_det: Detection | None) -> bool:
    if person_det is None or face_det is None:
        return False

    person_w = max(1.0, float(person_det.width))
    person_h = max(1.0, float(person_det.height))
    face_w = max(1.0, float(face_det.width))
    face_h = max(1.0, float(face_det.height))

    width_ratio = face_w / person_w
    height_ratio = face_h / person_h
    if not (FACE_PLAUSIBILITY_MIN_WIDTH_RATIO <= width_ratio <= FACE_PLAUSIBILITY_MAX_WIDTH_RATIO):
        return False
    if not (FACE_PLAUSIBILITY_MIN_HEIGHT_RATIO <= height_ratio <= FACE_PLAUSIBILITY_MAX_HEIGHT_RATIO):
        return False

    aspect = face_w / max(1.0, face_h)
    if not (FACE_PLAUSIBILITY_MIN_ASPECT_RATIO <= aspect <= FACE_PLAUSIBILITY_MAX_ASPECT_RATIO):
        return False

    person_left = person_det.x_center - (person_det.width / 2.0)
    person_right = person_det.x_center + (person_det.width / 2.0)
    person_top = person_det.y_center - (person_det.height / 2.0)
    person_bottom = person_det.y_center + (person_det.height / 2.0)

    face_left = face_det.x_center - (face_det.width / 2.0)
    face_right = face_det.x_center + (face_det.width / 2.0)
    face_top = face_det.y_center - (face_det.height / 2.0)
    face_bottom = face_det.y_center + (face_det.height / 2.0)

    overlap_w = min(person_right, face_right) - max(person_left, face_left)
    overlap_h = min(person_bottom, face_bottom) - max(person_top, face_top)
    if overlap_w <= 0 or overlap_h <= 0:
        return False

    center_y_offset = (face_det.y_center - person_det.y_center) / person_h
    top_y_offset = (face_top - person_top) / person_h
    if center_y_offset > FACE_PLAUSIBILITY_MAX_CENTER_Y_OFFSET_RATIO:
        return False
    if top_y_offset > FACE_PLAUSIBILITY_MAX_TOP_Y_OFFSET_RATIO:
        return False

    return True


def _track_anchor_candidate(
    track_id: str | None,
    frame_idx: int,
    fps: float,
    person_track_index: dict[str, list[Detection]],
    face_track_index: dict[str, list[Detection]],
) -> Detection | None:
    if not track_id:
        return None
    person_det = interpolate_detection(person_track_index.get(track_id), frame_idx, fps)
    face_det = interpolate_detection(face_track_index.get(track_id), frame_idx, fps)
    if _face_detection_is_plausible(person_det, face_det):
        return face_det
    return person_det


def motion_profile_for_composition(mode: str, *, out_h: int = OUT_H) -> MotionProfile:
    if mode == "two_shared":
        return MotionProfile(
            camera_zoom=SHARED_CAMERA_ZOOM,
            keyframe_step_s=max(KEYFRAME_STEP_S, 0.5),
            ema_smoothing=max(EMA_SMOOTHING, 8),
            y_head_bias=Y_HEAD_BIAS,
        )
    if mode == "two_split":
        split_zoom = max(CAMERA_ZOOM, 1.24 if out_h < OUT_H else 1.20)
        return MotionProfile(
            camera_zoom=split_zoom,
            keyframe_step_s=min(KEYFRAME_STEP_S, 0.3),
            ema_smoothing=min(max(EMA_SMOOTHING, 4), 5),
            y_head_bias=0.14,
        )
    return MotionProfile(
        camera_zoom=max(CAMERA_ZOOM, 1.34),
        keyframe_step_s=min(KEYFRAME_STEP_S, 0.25),
        ema_smoothing=min(EMA_SMOOTHING, 3),
        y_head_bias=0.12,
    )


def crop_dimensions(
    src_w: int,
    src_h: int,
    *,
    out_w: int = OUT_W,
    out_h: int = OUT_H,
    camera_zoom: float = CAMERA_ZOOM,
) -> tuple[int, int]:
    crop_h = int(round(src_h / camera_zoom))
    crop_w = int(round(crop_h * (out_w / out_h)))
    return min(crop_w, src_w), min(crop_h, src_h)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def build_interp_expr(keyframes: list[tuple[float, float]], var_name: str = "t") -> str:
    keyframes = simplify_keyframes(keyframes)
    if not keyframes:
        return "0"
    if len(keyframes) == 1:
        return f"{keyframes[0][1]:.4f}"

    parts = []
    for i in range(len(keyframes) - 1):
        t0, v0 = keyframes[i]
        t1, v1 = keyframes[i + 1]
        dt = max(0.001, t1 - t0)
        slope = (v1 - v0) / dt
        parts.append((t0, t1, v0, slope))

    last = parts[-1]
    expr = f"{last[2]:.6f}+{last[3]:.6f}*({var_name}-{last[0]:.4f})"
    for seg in range(len(parts) - 2, -1, -1):
        t0, t1, v0, slope = parts[seg]
        seg_expr = f"{v0:.6f}+{slope:.6f}*({var_name}-{t0:.4f})"
        expr = f"if(lt({var_name},{t1:.4f}),{seg_expr},{expr})"
    return expr


def simplify_keyframes(
    keyframes: list[tuple[float, float]],
    *,
    tolerance: float = 1.0,
    max_points: int = MAX_EXPR_KEYFRAMES,
) -> list[tuple[float, float]]:
    if len(keyframes) <= max_points:
        return keyframes

    def _rdp(points: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
        if len(points) <= 2:
            return points
        t0, v0 = points[0]
        t1, v1 = points[-1]
        span = max(1e-6, t1 - t0)
        max_dist = -1.0
        split_idx = -1
        for idx in range(1, len(points) - 1):
            t, v = points[idx]
            alpha = (t - t0) / span
            interp = ((1.0 - alpha) * v0) + (alpha * v1)
            dist = abs(v - interp)
            if dist > max_dist:
                max_dist = dist
                split_idx = idx
        if max_dist <= eps or split_idx < 0:
            return [points[0], points[-1]]
        left = _rdp(points[: split_idx + 1], eps)
        right = _rdp(points[split_idx:], eps)
        return left[:-1] + right

    simplified = keyframes
    eps = max(0.25, float(tolerance))
    while len(simplified) > max_points:
        simplified = _rdp(keyframes, eps)
        eps *= 1.5
        if eps > 128:
            break

    if len(simplified) <= max_points:
        return simplified

    stride = max(1, math.ceil((len(keyframes) - 2) / max(1, max_points - 2)))
    reduced = [keyframes[0]]
    reduced.extend(keyframes[1:-1:stride])
    if reduced[-1] != keyframes[-1]:
        reduced.append(keyframes[-1])
    return reduced[: max_points - 1] + [keyframes[-1]]


def ema_smooth(values: list[float], smoothing: int) -> list[float]:
    if not values:
        return values
    alpha = 2.0 / (smoothing + 1.0)
    out = [values[0]]
    for i in range(1, len(values)):
        out.append((alpha * values[i]) + ((1.0 - alpha) * out[-1]))
    return out


def build_track_index(tracks: list[dict]) -> dict[str, list[Detection]]:
    by_track: dict[str, dict[int, list[Detection]]] = {}
    for t in tracks:
        tid = str(t.get("track_id", ""))
        if not tid:
            continue
        frame_idx = int(t["frame_idx"])
        by_track.setdefault(tid, {}).setdefault(frame_idx, []).append(
            Detection(
                frame_idx=frame_idx,
                x_center=float(t["x_center"]),
                y_center=float(t["y_center"]),
                width=float(t["width"]),
                height=float(t["height"]),
                confidence=float(t.get("confidence", 0.0)),
            )
        )
    resolved_by_track: dict[str, list[Detection]] = {}
    for tid, dets_by_frame in by_track.items():
        frame_items = sorted(dets_by_frame.items())
        next_singleton: dict[int, Detection | None] = {}
        upcoming_singleton: Detection | None = None
        for frame_idx, candidates in reversed(frame_items):
            next_singleton[frame_idx] = upcoming_singleton
            if len(candidates) == 1:
                upcoming_singleton = candidates[0]

        resolved: list[Detection] = []
        last_selected: Detection | None = None
        for frame_idx, candidates in frame_items:
            candidates = sorted(
                candidates,
                key=lambda det: (
                    -float(det.confidence),
                    float(det.width * det.height),
                    float(det.x_center),
                    float(det.y_center),
                ),
            )
            if len(candidates) == 1:
                chosen = candidates[0]
            else:
                next_ref = next_singleton.get(frame_idx)
                scored: list[tuple[float, Detection]] = []
                for det in candidates:
                    score = (5.0 * float(det.confidence)) - (0.001 * float(det.width * det.height))
                    if last_selected is not None:
                        avg_w = max(1.0, 0.5 * (float(last_selected.width) + float(det.width)))
                        avg_h = max(1.0, 0.5 * (float(last_selected.height) + float(det.height)))
                        score -= abs(float(det.x_center) - float(last_selected.x_center)) / avg_w
                        score -= 0.5 * (abs(float(det.y_center) - float(last_selected.y_center)) / avg_h)
                    if next_ref is not None:
                        avg_w = max(1.0, 0.5 * (float(next_ref.width) + float(det.width)))
                        avg_h = max(1.0, 0.5 * (float(next_ref.height) + float(det.height)))
                        score -= abs(float(det.x_center) - float(next_ref.x_center)) / avg_w
                        score -= 0.5 * (abs(float(det.y_center) - float(next_ref.y_center)) / avg_h)
                    scored.append((score, det))
                scored.sort(
                    key=lambda item: (
                        -float(item[0]),
                        -float(item[1].confidence),
                        float(item[1].width * item[1].height),
                        float(item[1].x_center),
                        float(item[1].y_center),
                    )
                )
                chosen = scored[0][1]
                if last_selected is not None and len(scored) > 1:
                    top_score = float(scored[0][0])
                    second_score = float(scored[1][0])
                    far_apart = abs(float(scored[0][1].x_center) - float(scored[1][1].x_center)) > (
                        0.7 * max(1.0, 0.5 * (float(scored[0][1].width) + float(scored[1][1].width)))
                    )
                    if far_apart and abs(top_score - second_score) < 0.35:
                        chosen = Detection(
                            frame_idx=frame_idx,
                            x_center=float(last_selected.x_center),
                            y_center=float(last_selected.y_center),
                            width=float(last_selected.width),
                            height=float(last_selected.height),
                            confidence=float(last_selected.confidence),
                        )
            resolved.append(chosen)
            last_selected = chosen

        resolved_by_track[tid] = resolved
    return resolved_by_track


def build_face_index(face_detections: list[dict], src_w: int, src_h: int, fps: float) -> dict[str, list[Detection]]:
    by_track: dict[str, list[Detection]] = {}
    for track_block in face_detections:
        tid = str(track_block.get("track_id", ""))
        if not tid:
            continue
        for obj in track_block.get("timestamped_objects", []) or []:
            bbox = dict(obj.get("bounding_box", {}))
            left = float(bbox.get("left", 0.0))
            top = float(bbox.get("top", 0.0))
            right = float(bbox.get("right", left))
            bottom = float(bbox.get("bottom", top))
            if max(abs(left), abs(top), abs(right), abs(bottom)) <= 1.5:
                x1 = left * src_w
                y1 = top * src_h
                x2 = right * src_w
                y2 = bottom * src_h
            else:
                x1, y1, x2, y2 = left, top, right, bottom
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            time_ms = int(obj.get("time_ms", 0))
            by_track.setdefault(tid, []).append(
                Detection(
                    frame_idx=int(round((time_ms / 1000.0) * fps)),
                    x_center=x1 + (width / 2.0),
                    y_center=y1 + (height / 2.0),
                    width=width,
                    height=height,
                    confidence=float(obj.get("confidence", 0.0)),
                )
            )
    for tid in list(by_track.keys()):
        by_track[tid].sort(key=lambda d: d.frame_idx)
    return by_track


def interpolate_detection(
    dets: list[Detection] | None,
    frame_idx: int,
    fps: float,
) -> Detection | None:
    if not dets:
        return None

    frames = [d.frame_idx for d in dets]
    pos = bisect.bisect_left(frames, frame_idx)
    max_gap = int(round(MAX_INTERP_GAP_S * fps))

    if pos < len(dets) and dets[pos].frame_idx == frame_idx:
        return dets[pos]

    left = dets[pos - 1] if pos > 0 else None
    right = dets[pos] if pos < len(dets) else None

    if left and right:
        gap = right.frame_idx - left.frame_idx
        if gap <= max_gap and gap > 0:
            a = (frame_idx - left.frame_idx) / float(gap)
            return Detection(
                frame_idx=frame_idx,
                x_center=((1.0 - a) * left.x_center) + (a * right.x_center),
                y_center=((1.0 - a) * left.y_center) + (a * right.y_center),
                width=((1.0 - a) * left.width) + (a * right.width),
                height=((1.0 - a) * left.height) + (a * right.height),
            )

    nearest = None
    if left:
        nearest = left
    if right and (nearest is None or abs(right.frame_idx - frame_idx) < abs(nearest.frame_idx - frame_idx)):
        nearest = right
    if nearest and abs(nearest.frame_idx - frame_idx) <= max_gap:
        return nearest
    return None


def active_track_at(bindings: list[dict], target_ms: int) -> str | None:
    for b in bindings:
        if int(b["start_time_ms"]) <= target_ms <= int(b["end_time_ms"]):
            return str(b["track_id"])

    nearest = None
    nearest_gap = None
    for b in bindings:
        start_ms = int(b["start_time_ms"])
        end_ms = int(b["end_time_ms"])
        gap = min(abs(target_ms - start_ms), abs(target_ms - end_ms))
        if nearest_gap is None or gap < nearest_gap:
            nearest_gap = gap
            nearest = b
    if nearest is not None and nearest_gap is not None and nearest_gap <= SPEAKER_GAP_TOLERANCE_MS:
        return str(nearest["track_id"])
    return None


def choose_windows(bindings: list[dict], duration_s: float) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    max_start = max(0, int(math.floor(duration_s)) - CLIP_DURATION_S)
    for start_s in range(0, max_start + 1, WINDOW_STEP_S):
        end_s = start_s + CLIP_DURATION_S
        overlaps = []
        for b in bindings:
            bs = int(b["start_time_ms"]) / 1000.0
            be = int(b["end_time_ms"]) / 1000.0
            if be <= start_s or bs >= end_s:
                continue
            overlaps.append(b)
        if not overlaps:
            continue

        switches = 0
        last_tid = None
        unique_tracks = set()
        total_words = 0
        speech_ms = 0.0
        for b in overlaps:
            tid = str(b["track_id"])
            unique_tracks.add(tid)
            total_words += int(b.get("word_count", 0))
            seg_start = max(start_s * 1000.0, float(b["start_time_ms"]))
            seg_end = min(end_s * 1000.0, float(b["end_time_ms"]))
            speech_ms += max(0.0, seg_end - seg_start)
            if last_tid is not None and tid != last_tid:
                switches += 1
            last_tid = tid

        speech_cov = speech_ms / max(1.0, CLIP_DURATION_S * 1000.0)
        score = (3.0 * switches) + (1.5 * len(unique_tracks)) + (total_words / 40.0) + (2.0 * speech_cov)
        if switches == 0 and len(unique_tracks) < 2:
            score -= 3.0
        candidates.append((score, start_s, end_s))

    candidates.sort(reverse=True)
    picked: list[tuple[int, int, float]] = []
    for score, start_s, end_s in candidates:
        if any(not (end_s <= s or e <= start_s) for s, e, _ in picked):
            continue
        if any(abs(start_s - s) < WINDOW_MIN_GAP_S for s, _, _ in picked):
            continue
        picked.append((start_s, end_s, score))
        if len(picked) >= NUM_CLIPS:
            break

    if not picked:
        return [(0, CLIP_DURATION_S, 0.0)]
    return picked


def _track_anchor_at_frame(
    track_id: str | None,
    frame_idx: int,
    fps: float,
    person_track_index: dict[str, list[Detection]],
    face_track_index: dict[str, list[Detection]],
) -> Detection | None:
    return _track_anchor_candidate(track_id, frame_idx, fps, person_track_index, face_track_index)


def _sample_window_track_stats(
    clip_start_s: int,
    clip_end_s: int,
    fps: float,
    bindings: list[dict],
    person_track_index: dict[str, list[Detection]],
    face_track_index: dict[str, list[Detection]],
) -> dict[str, TrackWindowStats]:
    track_speech_ms: dict[str, float] = {}
    track_word_count: dict[str, int] = {}
    track_samples: dict[str, list[Detection]] = {}
    track_visible_samples: dict[str, int] = {}
    sample_count = 0
    track_ids = sorted({str(b["track_id"]) for b in bindings if str(b.get("track_id", ""))})

    for b in bindings:
        tid = str(b["track_id"])
        seg_start_s = max(float(clip_start_s), float(b["start_time_ms"]) / 1000.0)
        seg_end_s = min(float(clip_end_s), float(b["end_time_ms"]) / 1000.0)
        if seg_end_s <= seg_start_s:
            continue
        track_speech_ms[tid] = track_speech_ms.get(tid, 0.0) + ((seg_end_s - seg_start_s) * 1000.0)
        track_word_count[tid] = track_word_count.get(tid, 0) + int(b.get("word_count", 0))

    t = 0.0
    while t <= (clip_end_s - clip_start_s + 1e-6):
        sample_count += 1
        abs_ms = int(round((clip_start_s + t) * 1000.0))
        frame_idx = int(round((abs_ms / 1000.0) * fps))
        for tid in track_ids:
            det = _track_anchor_at_frame(tid, frame_idx, fps, person_track_index, face_track_index)
            if det is None:
                continue
            track_visible_samples[tid] = track_visible_samples.get(tid, 0) + 1
            track_samples.setdefault(tid, []).append(det)
        t += KEYFRAME_STEP_S

    stats: dict[str, TrackWindowStats] = {}
    for tid in set(track_speech_ms) | set(track_word_count) | set(track_samples) | set(track_visible_samples):
        dets = track_samples.get(tid, [])
        xs = [d.x_center for d in dets]
        ys = [d.y_center for d in dets]
        ws = [d.width for d in dets]
        hs = [d.height for d in dets]
        stats[tid] = TrackWindowStats(
            track_id=tid,
            speech_ms=float(track_speech_ms.get(tid, 0.0)),
            word_count=int(track_word_count.get(tid, 0)),
            visible_fraction=(track_visible_samples.get(tid, 0) / max(1, sample_count)),
            median_x=statistics.median(xs) if xs else None,
            median_y=statistics.median(ys) if ys else None,
            median_width=statistics.median(ws) if ws else None,
            median_height=statistics.median(hs) if hs else None,
        )
    return stats


def choose_window_composition(
    clip_start_s: int,
    clip_end_s: int,
    fps: float,
    src_w: int,
    src_h: int,
    bindings: list[dict],
    person_track_index: dict[str, list[Detection]],
    face_track_index: dict[str, list[Detection]],
) -> CompositionPlan:
    stats = _sample_window_track_stats(
        clip_start_s=clip_start_s,
        clip_end_s=clip_end_s,
        fps=fps,
        bindings=bindings,
        person_track_index=person_track_index,
        face_track_index=face_track_index,
    )
    if not stats:
        return CompositionPlan(mode="single_person", primary_track_id=None, secondary_track_id=None)

    ordered = sorted(
        stats.values(),
        key=lambda s: (s.speech_ms, s.word_count, s.visible_fraction),
        reverse=True,
    )
    primary = ordered[0]
    if len(ordered) == 1:
        return CompositionPlan(mode="single_person", primary_track_id=primary.track_id, secondary_track_id=None)

    secondary = ordered[1]
    total_speech = sum(item.speech_ms for item in ordered)
    if total_speech <= 0:
        return CompositionPlan(mode="single_person", primary_track_id=primary.track_id, secondary_track_id=None)

    primary_share = primary.speech_ms / total_speech
    secondary_share = secondary.speech_ms / total_speech
    top2_share = primary_share + secondary_share

    if primary_share >= SINGLE_PERSON_MIN_DOMINANCE or secondary_share < TWO_SPEAKER_MIN_SECONDARY_SHARE or top2_share < TWO_SPEAKER_MIN_COMBINED_SHARE:
        return CompositionPlan(mode="single_person", primary_track_id=primary.track_id, secondary_track_id=None)

    visible_enough = (
        primary.visible_fraction >= TWO_SPEAKER_MIN_BOTH_VISIBLE_FRACTION
        and secondary.visible_fraction >= TWO_SPEAKER_MIN_BOTH_VISIBLE_FRACTION
    )
    if not visible_enough:
        return CompositionPlan(mode="single_person", primary_track_id=primary.track_id, secondary_track_id=None)

    if primary.median_x is None or secondary.median_x is None:
        return CompositionPlan(mode="two_split", primary_track_id=primary.track_id, secondary_track_id=secondary.track_id)

    center_gap = abs(primary.median_x - secondary.median_x)
    primary_w = primary.median_width or 0.0
    secondary_w = secondary.median_width or 0.0
    primary_h = primary.median_height or 0.0
    secondary_h = secondary.median_height or 0.0
    union_left = min(
        primary.median_x - (primary_w / 2.0),
        secondary.median_x - (secondary_w / 2.0),
    )
    union_right = max(
        primary.median_x + (primary_w / 2.0),
        secondary.median_x + (secondary_w / 2.0),
    )
    union_top = min(
        (primary.median_y or 0.0) - (primary_h / 2.0),
        (secondary.median_y or 0.0) - (secondary_h / 2.0),
    )
    union_bottom = max(
        (primary.median_y or 0.0) + (primary_h / 2.0),
        (secondary.median_y or 0.0) + (secondary_h / 2.0),
    )
    union_w = union_right - union_left
    union_h = union_bottom - union_top
    shared_crop_w, shared_crop_h = crop_dimensions(
        src_w,
        src_h,
        camera_zoom=SHARED_CAMERA_ZOOM,
    )

    if (
        center_gap <= (src_w * SHARED_TWO_SHOT_MAX_CENTER_GAP_RATIO)
        and union_w <= min(src_w * SHARED_TWO_SHOT_MAX_UNION_WIDTH_RATIO, shared_crop_w * 0.92)
        and union_h <= min(src_h * SHARED_TWO_SHOT_MAX_UNION_HEIGHT_RATIO, shared_crop_h * 0.92)
    ):
        return CompositionPlan(mode="two_shared", primary_track_id=primary.track_id, secondary_track_id=secondary.track_id)

    ordered_by_x = sorted(
        [primary, secondary],
        key=lambda item: (item.median_x if item.median_x is not None else 0.0, -item.speech_ms),
    )
    return CompositionPlan(
        mode="two_split",
        primary_track_id=ordered_by_x[0].track_id,
        secondary_track_id=ordered_by_x[1].track_id,
    )


def build_camera_path(
    clip_start_s: int,
    clip_end_s: int,
    fps: float,
    src_w: int,
    src_h: int,
    bindings: list[dict],
    person_track_index: dict[str, list[Detection]],
    face_track_index: dict[str, list[Detection]],
    motion_profile: MotionProfile | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    motion_profile = motion_profile or motion_profile_for_composition("single_person")
    crop_w, crop_h = crop_dimensions(src_w, src_h, camera_zoom=motion_profile.camera_zoom)
    half_w = crop_w / 2.0
    half_h = crop_h / 2.0

    last_cx = src_w / 2.0
    last_cy = src_h / 2.0
    x_keyframes: list[tuple[float, float]] = []
    y_keyframes: list[tuple[float, float]] = []
    segment_samples: list[tuple[float, float, float]] = []
    segment_tid: str | None = None
    cut_epsilon_s = 0.001

    def flush_segment() -> None:
        nonlocal segment_samples, x_keyframes, y_keyframes
        if not segment_samples:
            return
        times = [round(t, 3) for t, _, _ in segment_samples]
        xs = ema_smooth([x for _, x, _ in segment_samples], motion_profile.ema_smoothing)
        ys = ema_smooth([y for _, _, y in segment_samples], motion_profile.ema_smoothing)
        if x_keyframes:
            prev_t, prev_x = x_keyframes[-1]
            _, prev_y = y_keyframes[-1]
            next_t = times[0]
            next_x = xs[0]
            next_y = ys[0]
            if next_t > prev_t and (abs(next_x - prev_x) > 1.0 or abs(next_y - prev_y) > 1.0):
                hold_t = max(prev_t, round(next_t - cut_epsilon_s, 3))
                if hold_t > prev_t:
                    x_keyframes.append((hold_t, prev_x))
                    y_keyframes.append((hold_t, prev_y))
        x_keyframes.extend(zip(times, xs))
        y_keyframes.extend(zip(times, ys))
        segment_samples = []

    t = 0.0
    while t <= (clip_end_s - clip_start_s + 1e-6):
        abs_ms = int(round((clip_start_s + t) * 1000.0))
        frame_idx = int(round((abs_ms / 1000.0) * fps))
        tid = active_track_at(bindings, abs_ms)
        person_det = interpolate_detection(person_track_index.get(tid), frame_idx, fps) if tid else None
        det = _track_anchor_candidate(tid, frame_idx, fps, person_track_index, face_track_index)
        if det is not None and person_det is not None and det is person_det:
            last_cx = person_det.x_center
            last_cy = person_det.y_center - (motion_profile.y_head_bias * person_det.height)
        elif det is not None:
            last_cx = det.x_center
            last_cy = det.y_center
        elif person_det is not None:
            last_cx = person_det.x_center
            last_cy = person_det.y_center - (motion_profile.y_head_bias * person_det.height)

        crop_x = clamp(last_cx, half_w, src_w - half_w) - half_w
        crop_y = clamp(last_cy, half_h, src_h - half_h) - half_h
        if segment_samples and tid != segment_tid:
            flush_segment()
        segment_tid = tid
        segment_samples.append((t, crop_x, crop_y))
        t += motion_profile.keyframe_step_s

    flush_segment()
    return x_keyframes, y_keyframes


def build_single_track_path(
    track_id: str,
    clip_start_s: int,
    clip_end_s: int,
    fps: float,
    src_w: int,
    src_h: int,
    person_track_index: dict[str, list[Detection]],
    face_track_index: dict[str, list[Detection]],
    out_w: int = OUT_W,
    out_h: int = OUT_H,
    camera_zoom: float = CAMERA_ZOOM,
    y_head_bias: float = Y_HEAD_BIAS,
    motion_profile: MotionProfile | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    motion_profile = motion_profile or MotionProfile(
        camera_zoom=camera_zoom,
        keyframe_step_s=KEYFRAME_STEP_S,
        ema_smoothing=EMA_SMOOTHING,
        y_head_bias=y_head_bias,
    )
    camera_zoom = motion_profile.camera_zoom
    y_head_bias = motion_profile.y_head_bias
    crop_w, crop_h = crop_dimensions(src_w, src_h, out_w=out_w, out_h=out_h, camera_zoom=camera_zoom)
    half_w = crop_w / 2.0
    half_h = crop_h / 2.0

    last_cx = src_w / 2.0
    last_cy = src_h / 2.0
    x_keyframes: list[tuple[float, float]] = []
    y_keyframes: list[tuple[float, float]] = []
    samples: list[tuple[float, float, float]] = []

    t = 0.0
    while t <= (clip_end_s - clip_start_s + 1e-6):
        abs_ms = int(round((clip_start_s + t) * 1000.0))
        frame_idx = int(round((abs_ms / 1000.0) * fps))
        person_det = interpolate_detection(person_track_index.get(track_id), frame_idx, fps)
        det = _track_anchor_candidate(track_id, frame_idx, fps, person_track_index, face_track_index)
        if det is not None and person_det is not None and det is person_det:
            last_cx = person_det.x_center
            last_cy = person_det.y_center - (y_head_bias * person_det.height)
        elif det is not None:
            last_cx = det.x_center
            last_cy = det.y_center
        elif person_det is not None:
            last_cx = person_det.x_center
            last_cy = person_det.y_center - (y_head_bias * person_det.height)

        crop_x = clamp(last_cx, half_w, src_w - half_w) - half_w
        crop_y = clamp(last_cy, half_h, src_h - half_h) - half_h
        samples.append((t, crop_x, crop_y))
        t += motion_profile.keyframe_step_s

    if samples:
        times = [round(t, 3) for t, _, _ in samples]
        xs = ema_smooth([x for _, x, _ in samples], motion_profile.ema_smoothing)
        ys = ema_smooth([y for _, _, y in samples], motion_profile.ema_smoothing)
        x_keyframes.extend(zip(times, xs))
        y_keyframes.extend(zip(times, ys))

    return x_keyframes, y_keyframes


def build_shared_two_shot_path(
    primary_track_id: str,
    secondary_track_id: str,
    clip_start_s: int,
    clip_end_s: int,
    fps: float,
    src_w: int,
    src_h: int,
    person_track_index: dict[str, list[Detection]],
    face_track_index: dict[str, list[Detection]],
    out_w: int = OUT_W,
    out_h: int = OUT_H,
    motion_profile: MotionProfile | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    motion_profile = motion_profile or motion_profile_for_composition("two_shared", out_h=out_h)
    crop_w, crop_h = crop_dimensions(
        src_w,
        src_h,
        out_w=out_w,
        out_h=out_h,
        camera_zoom=motion_profile.camera_zoom,
    )
    half_w = crop_w / 2.0
    half_h = crop_h / 2.0

    last_cx = src_w / 2.0
    last_cy = src_h / 2.0
    x_samples: list[float] = []
    y_samples: list[float] = []
    x_keyframes: list[tuple[float, float]] = []
    y_keyframes: list[tuple[float, float]] = []

    t = 0.0
    while t <= (clip_end_s - clip_start_s + 1e-6):
        abs_ms = int(round((clip_start_s + t) * 1000.0))
        frame_idx = int(round((abs_ms / 1000.0) * fps))
        primary_det = _track_anchor_at_frame(primary_track_id, frame_idx, fps, person_track_index, face_track_index)
        secondary_det = _track_anchor_at_frame(secondary_track_id, frame_idx, fps, person_track_index, face_track_index)
        dets = [det for det in (primary_det, secondary_det) if det is not None]
        if dets:
            last_cx = sum(det.x_center for det in dets) / len(dets)
            last_cy = sum(det.y_center for det in dets) / len(dets)
        x_samples.append(clamp(last_cx, half_w, src_w - half_w) - half_w)
        y_samples.append(clamp(last_cy, half_h, src_h - half_h) - half_h)
        t += motion_profile.keyframe_step_s

    if x_samples:
        times = [round(i * motion_profile.keyframe_step_s, 3) for i in range(len(x_samples))]
        x_keyframes.extend(zip(times, ema_smooth(x_samples, motion_profile.ema_smoothing)))
        y_keyframes.extend(zip(times, ema_smooth(y_samples, motion_profile.ema_smoothing)))

    return x_keyframes, y_keyframes


def interpolate_keyframes(keyframes: list[tuple[float, float]], t: float) -> float:
    if not keyframes:
        return 0.0
    if t <= keyframes[0][0]:
        return keyframes[0][1]
    if t >= keyframes[-1][0]:
        return keyframes[-1][1]

    for idx in range(1, len(keyframes)):
        t0, v0 = keyframes[idx - 1]
        t1, v1 = keyframes[idx]
        if t <= t1:
            span = max(0.001, t1 - t0)
            alpha = (t - t0) / span
            return ((1.0 - alpha) * v0) + (alpha * v1)
    return keyframes[-1][1]


def build_overlay_path(
    track_id: str | None,
    clip_start_s: int,
    clip_end_s: int,
    fps: float,
    src_w: int,
    src_h: int,
    x_keyframes: list[tuple[float, float]],
    y_keyframes: list[tuple[float, float]],
    person_track_index: dict[str, list[Detection]],
    face_track_index: dict[str, list[Detection]],
    out_w: int = OUT_W,
    out_h: int = OUT_H,
    camera_zoom: float = CAMERA_ZOOM,
    keyframe_step_s: float = KEYFRAME_STEP_S,
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
]:
    crop_w, crop_h = crop_dimensions(
        src_w,
        src_h,
        out_w=out_w,
        out_h=out_h,
        camera_zoom=camera_zoom,
    )

    box_x_keyframes: list[tuple[float, float]] = []
    box_y_keyframes: list[tuple[float, float]] = []
    box_w_keyframes: list[tuple[float, float]] = []
    box_h_keyframes: list[tuple[float, float]] = []
    last_rel = (0.18 * OUT_W, 0.16 * OUT_H, 0.28 * OUT_W, 0.22 * OUT_H)
    saw_anchor = False

    if not track_id:
        return ([], [], [], [])

    t = 0.0
    while t <= (clip_end_s - clip_start_s + 1e-6):
        abs_ms = int(round((clip_start_s + t) * 1000.0))
        frame_idx = int(round((abs_ms / 1000.0) * fps))
        det = _track_anchor_candidate(track_id, frame_idx, fps, person_track_index, face_track_index)

        crop_x = interpolate_keyframes(x_keyframes, t)
        crop_y = interpolate_keyframes(y_keyframes, t)
        if det is not None:
            saw_anchor = True
            x1 = det.x_center - (det.width / 2.0)
            y1 = det.y_center - (det.height / 2.0)
            rel_x = ((x1 - crop_x) / max(1.0, crop_w)) * out_w
            rel_y = ((y1 - crop_y) / max(1.0, crop_h)) * out_h
            rel_w = (det.width / max(1.0, crop_w)) * out_w
            rel_h = (det.height / max(1.0, crop_h)) * out_h
            rel_w = clamp(rel_w, 40.0, out_w * 0.9)
            rel_h = clamp(rel_h, 40.0, out_h * 0.9)
            rel_x = clamp(rel_x, 0.0, out_w - rel_w)
            rel_y = clamp(rel_y, 0.0, out_h - rel_h)
            last_rel = (rel_x, rel_y, rel_w, rel_h)

        box_x_keyframes.append((round(t, 3), float(last_rel[0])))
        box_y_keyframes.append((round(t, 3), float(last_rel[1])))
        box_w_keyframes.append((round(t, 3), float(last_rel[2])))
        box_h_keyframes.append((round(t, 3), float(last_rel[3])))
        t += keyframe_step_s

    if not saw_anchor:
        return ([], [], [], [])

    return (
        box_x_keyframes,
        box_y_keyframes,
        box_w_keyframes,
        box_h_keyframes,
    )


def build_overlay_box_path(
    track_id: str,
    clip_start_s: int,
    clip_end_s: int,
    fps: float,
    src_w: int,
    src_h: int,
    person_track_index: dict[str, list[Detection]],
    face_track_index: dict[str, list[Detection]],
    out_w: int = OUT_W,
    out_h: int = OUT_H,
    camera_zoom: float = CAMERA_ZOOM,
    keyframe_step_s: float = KEYFRAME_STEP_S,
    color: str = OVERLAY_BOX_COLOR,
) -> OverlayPath | None:
    motion_profile = MotionProfile(
        camera_zoom=camera_zoom,
        keyframe_step_s=keyframe_step_s,
        ema_smoothing=EMA_SMOOTHING,
        y_head_bias=Y_HEAD_BIAS,
    )
    x_keyframes, y_keyframes = build_single_track_path(
        track_id=track_id,
        clip_start_s=clip_start_s,
        clip_end_s=clip_end_s,
        fps=fps,
        src_w=src_w,
        src_h=src_h,
        person_track_index=person_track_index,
        face_track_index=face_track_index,
        out_w=out_w,
        out_h=out_h,
        camera_zoom=camera_zoom,
        motion_profile=motion_profile,
    )
    box_x_keyframes, box_y_keyframes, box_w_keyframes, box_h_keyframes = build_overlay_path(
        track_id=track_id,
        clip_start_s=clip_start_s,
        clip_end_s=clip_end_s,
        fps=fps,
        src_w=src_w,
        src_h=src_h,
        x_keyframes=x_keyframes,
        y_keyframes=y_keyframes,
        person_track_index=person_track_index,
        face_track_index=face_track_index,
        out_w=out_w,
        out_h=out_h,
        camera_zoom=camera_zoom,
        keyframe_step_s=keyframe_step_s,
    )
    if not box_x_keyframes:
        return None
    return OverlayPath(
        x_keyframes=x_keyframes,
        y_keyframes=y_keyframes,
        box_x_keyframes=box_x_keyframes,
        box_y_keyframes=box_y_keyframes,
        box_w_keyframes=box_w_keyframes,
        box_h_keyframes=box_h_keyframes,
        color=color,
    )


def render_clip(
    video_path: Path,
    out_path: Path,
    start_s: int,
    duration_s: int,
    x_keyframes: list[tuple[float, float]],
    y_keyframes: list[tuple[float, float]],
    overlay_paths: list[OverlayPath],
    src_h: int,
    camera_zoom: float = CAMERA_ZOOM,
) -> None:
    crop_h = int(round(src_h / camera_zoom))
    crop_w = int(round(crop_h * (OUT_W / OUT_H)))
    x_expr = build_interp_expr(x_keyframes)
    y_expr = build_interp_expr(y_keyframes)
    filter_chain_parts = [
        f"crop=w={crop_w}:h={crop_h}:x='{x_expr}':y='{y_expr}':exact=1",
        f"scale={OUT_W}:{OUT_H}:flags=lanczos",
    ]
    for idx, overlay in enumerate(overlay_paths):
        filter_chain_parts.append(
            f"drawbox=x='{build_interp_expr(overlay.box_x_keyframes)}':"
            f"y='{build_interp_expr(overlay.box_y_keyframes)}':"
            f"w='{build_interp_expr(overlay.box_w_keyframes)}':"
            f"h='{build_interp_expr(overlay.box_h_keyframes)}':"
            f"color={overlay.color}:t={OVERLAY_BOX_THICKNESS}"
        )
    filter_chain = ",".join(filter_chain_parts)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_s),
        "-t",
        str(duration_s),
        "-i",
        str(video_path),
        "-vf",
        filter_chain,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {out_path.name}:\n{result.stderr[-1200:]}"
        )


def render_split_clip(
    video_path: Path,
    out_path: Path,
    start_s: int,
    duration_s: int,
    upper_x_keyframes: list[tuple[float, float]],
    upper_y_keyframes: list[tuple[float, float]],
    upper_overlay_paths: list[OverlayPath],
    lower_x_keyframes: list[tuple[float, float]],
    lower_y_keyframes: list[tuple[float, float]],
    lower_overlay_paths: list[OverlayPath],
    src_w: int,
    src_h: int,
    camera_zoom: float = CAMERA_ZOOM,
) -> None:
    panel_h = SPLIT_PANEL_HEIGHT
    crop_h = int(round(src_h / camera_zoom))
    crop_w = int(round(crop_h * (OUT_W / panel_h)))
    crop_w = min(crop_w, src_w)
    crop_h = min(crop_h, src_h)
    x_expr_upper = build_interp_expr(upper_x_keyframes)
    y_expr_upper = build_interp_expr(upper_y_keyframes)
    x_expr_lower = build_interp_expr(lower_x_keyframes)
    y_expr_lower = build_interp_expr(lower_y_keyframes)

    upper_draw_filters = [
        (
            f"drawbox=x='{build_interp_expr(overlay.box_x_keyframes)}':"
            f"y='{build_interp_expr(overlay.box_y_keyframes)}':"
            f"w='{build_interp_expr(overlay.box_w_keyframes)}':"
            f"h='{build_interp_expr(overlay.box_h_keyframes)}':"
            f"color={overlay.color}:t={OVERLAY_BOX_THICKNESS}"
        )
        for overlay in upper_overlay_paths
        if overlay.box_x_keyframes
    ]
    lower_draw_filters = [
        (
            f"drawbox=x='{build_interp_expr(overlay.box_x_keyframes)}':"
            f"y='{build_interp_expr(overlay.box_y_keyframes)}':"
            f"w='{build_interp_expr(overlay.box_w_keyframes)}':"
            f"h='{build_interp_expr(overlay.box_h_keyframes)}':"
            f"color={overlay.color}:t={OVERLAY_BOX_THICKNESS}"
        )
        for overlay in lower_overlay_paths
        if overlay.box_x_keyframes
    ]

    filter_complex = (
        f"[0:v]split=2[upper_src][lower_src];"
        f"[upper_src]crop=w={crop_w}:h={crop_h}:x='{x_expr_upper}':y='{y_expr_upper}':exact=1,"
        f"scale={OUT_W}:{panel_h}:flags=lanczos"
        f"{(',' + ','.join(upper_draw_filters)) if upper_draw_filters else ''}[upper];"
        f"[lower_src]crop=w={crop_w}:h={crop_h}:x='{x_expr_lower}':y='{y_expr_lower}':exact=1,"
        f"scale={OUT_W}:{panel_h}:flags=lanczos"
        f"{(',' + ','.join(lower_draw_filters)) if lower_draw_filters else ''}[lower];"
        f"[upper][lower]vstack=inputs=2[vout]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_s),
        "-t",
        str(duration_s),
        "-i",
        str(video_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {out_path.name}:\n{result.stderr[-1200:]}"
        )


def main() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Missing source video: {VIDEO_PATH}")
    if not VISUAL_PATH.exists():
        raise FileNotFoundError(f"Missing visual ledger: {VISUAL_PATH}")
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(f"Missing audio ledger: {AUDIO_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    visual = load_json(VISUAL_PATH)
    audio = load_json(AUDIO_PATH)
    tracks = list(visual.get("tracks", []))
    face_detections = list(visual.get("face_detections", []))
    bindings = list(audio.get("speaker_bindings", []))
    if not tracks or not bindings:
        raise RuntimeError("Need non-empty tracks and speaker_bindings to render clips")

    meta = dict(visual.get("video_metadata", {}))
    fps = float(meta.get("fps", 23.976))
    src_w = int(meta.get("width", 1920))
    src_h = int(meta.get("height", 1080))
    duration_s = float(meta.get("duration_ms", 0)) / 1000.0
    if duration_s <= 0:
        raise RuntimeError("Missing video duration in phase_1_visual.json video_metadata")

    person_track_index = build_track_index(tracks)
    face_track_index = build_face_index(face_detections, src_w=src_w, src_h=src_h, fps=fps)
    windows = choose_windows(bindings, duration_s)

    print(f"Selected {len(windows)} windows from current outputs:")
    print(
        "Camera target mode: "
        f"face-first ({len(face_track_index)} tracks with face detections), "
        f"person-fallback ({len(person_track_index)} person tracks)"
    )
    outputs = []
    for idx, (start_s, end_s, score) in enumerate(windows, start=1):
        clip_name = f"speaker_follow_clip{idx}_{start_s}s_{CLIP_DURATION_S}s.mp4"
        out_path = OUTPUT_DIR / clip_name
        composition = choose_window_composition(
            clip_start_s=start_s,
            clip_end_s=end_s,
            fps=fps,
            src_w=src_w,
            src_h=src_h,
            bindings=bindings,
            person_track_index=person_track_index,
            face_track_index=face_track_index,
        )
        if composition.mode == "two_split" and composition.primary_track_id and composition.secondary_track_id:
            split_profile = motion_profile_for_composition("two_split", out_h=SPLIT_PANEL_HEIGHT)
            upper_x_keyframes, upper_y_keyframes = build_single_track_path(
                track_id=composition.primary_track_id,
                clip_start_s=start_s,
                clip_end_s=end_s,
                fps=fps,
                src_w=src_w,
                src_h=src_h,
                person_track_index=person_track_index,
                face_track_index=face_track_index,
                out_h=SPLIT_PANEL_HEIGHT,
                motion_profile=split_profile,
            )
            lower_x_keyframes, lower_y_keyframes = build_single_track_path(
                track_id=composition.secondary_track_id,
                clip_start_s=start_s,
                clip_end_s=end_s,
                fps=fps,
                src_w=src_w,
                src_h=src_h,
                person_track_index=person_track_index,
                face_track_index=face_track_index,
                out_h=SPLIT_PANEL_HEIGHT,
                motion_profile=split_profile,
            )
            upper_overlay = build_overlay_box_path(
                track_id=composition.primary_track_id,
                clip_start_s=start_s,
                clip_end_s=end_s,
                fps=fps,
                src_w=src_w,
                src_h=src_h,
                person_track_index=person_track_index,
                face_track_index=face_track_index,
                out_h=SPLIT_PANEL_HEIGHT,
                camera_zoom=split_profile.camera_zoom,
                keyframe_step_s=split_profile.keyframe_step_s,
                color=OVERLAY_BOX_COLOR,
            )
            lower_overlay = build_overlay_box_path(
                track_id=composition.secondary_track_id,
                clip_start_s=start_s,
                clip_end_s=end_s,
                fps=fps,
                src_w=src_w,
                src_h=src_h,
                person_track_index=person_track_index,
                face_track_index=face_track_index,
                out_h=SPLIT_PANEL_HEIGHT,
                camera_zoom=split_profile.camera_zoom,
                keyframe_step_s=split_profile.keyframe_step_s,
                color=OVERLAY_SECONDARY_BOX_COLOR,
            )
            print(
                f"  clip{idx}: {start_s}s-{end_s}s score={score:.2f} "
                f"mode={composition.mode} -> {clip_name}"
            )
            render_split_clip(
                video_path=VIDEO_PATH,
                out_path=out_path,
                start_s=start_s,
                duration_s=(end_s - start_s),
                upper_x_keyframes=upper_x_keyframes,
                upper_y_keyframes=upper_y_keyframes,
                upper_overlay_paths=[upper_overlay] if upper_overlay is not None else [],
                lower_x_keyframes=lower_x_keyframes,
                lower_y_keyframes=lower_y_keyframes,
                lower_overlay_paths=[lower_overlay] if lower_overlay is not None else [],
                src_w=src_w,
                src_h=src_h,
                camera_zoom=split_profile.camera_zoom,
            )
        elif composition.mode == "two_shared" and composition.primary_track_id and composition.secondary_track_id:
            shared_profile = motion_profile_for_composition("two_shared")
            x_keyframes, y_keyframes = build_shared_two_shot_path(
                primary_track_id=composition.primary_track_id,
                secondary_track_id=composition.secondary_track_id,
                clip_start_s=start_s,
                clip_end_s=end_s,
                fps=fps,
                src_w=src_w,
                src_h=src_h,
                person_track_index=person_track_index,
                face_track_index=face_track_index,
                motion_profile=shared_profile,
            )
            overlay_paths = [
                overlay
                for overlay in (
                    build_overlay_box_path(
                        track_id=composition.primary_track_id,
                        clip_start_s=start_s,
                        clip_end_s=end_s,
                        fps=fps,
                        src_w=src_w,
                        src_h=src_h,
                        person_track_index=person_track_index,
                        face_track_index=face_track_index,
                        out_h=OUT_H,
                        camera_zoom=shared_profile.camera_zoom,
                        keyframe_step_s=shared_profile.keyframe_step_s,
                        color=OVERLAY_BOX_COLOR,
                    ),
                    build_overlay_box_path(
                        track_id=composition.secondary_track_id,
                        clip_start_s=start_s,
                        clip_end_s=end_s,
                        fps=fps,
                        src_w=src_w,
                        src_h=src_h,
                        person_track_index=person_track_index,
                        face_track_index=face_track_index,
                        out_h=OUT_H,
                        camera_zoom=shared_profile.camera_zoom,
                        keyframe_step_s=shared_profile.keyframe_step_s,
                        color=OVERLAY_SECONDARY_BOX_COLOR,
                    ),
                )
                if overlay is not None
            ]
            print(
                f"  clip{idx}: {start_s}s-{end_s}s score={score:.2f} "
                f"mode={composition.mode} -> {clip_name}"
            )
            render_clip(
                video_path=VIDEO_PATH,
                out_path=out_path,
                start_s=start_s,
                duration_s=(end_s - start_s),
                x_keyframes=x_keyframes,
                y_keyframes=y_keyframes,
                overlay_paths=overlay_paths,
                src_h=src_h,
                camera_zoom=shared_profile.camera_zoom,
            )
        else:
            single_profile = motion_profile_for_composition("single_person")
            x_keyframes, y_keyframes = build_camera_path(
                clip_start_s=start_s,
                clip_end_s=end_s,
                fps=fps,
                src_w=src_w,
                src_h=src_h,
                bindings=bindings,
                person_track_index=person_track_index,
                face_track_index=face_track_index,
                motion_profile=single_profile,
            )
            overlay_path = build_overlay_box_path(
                track_id=composition.primary_track_id,
                clip_start_s=start_s,
                clip_end_s=end_s,
                fps=fps,
                src_w=src_w,
                src_h=src_h,
                person_track_index=person_track_index,
                face_track_index=face_track_index,
                out_h=OUT_H,
                camera_zoom=single_profile.camera_zoom,
                keyframe_step_s=single_profile.keyframe_step_s,
            )
            overlay_paths = [overlay_path] if overlay_path is not None else []
            print(
                f"  clip{idx}: {start_s}s-{end_s}s score={score:.2f} "
                f"mode={composition.mode} -> {clip_name}"
            )
            render_clip(
                video_path=VIDEO_PATH,
                out_path=out_path,
                start_s=start_s,
                duration_s=(end_s - start_s),
                x_keyframes=x_keyframes,
                y_keyframes=y_keyframes,
                overlay_paths=overlay_paths,
                src_h=src_h,
                camera_zoom=single_profile.camera_zoom,
            )
        outputs.append(out_path)

    print("\nRendered clips:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
