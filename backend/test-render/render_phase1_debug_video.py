from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent
DEFAULT_VISUAL = ROOT / "outputs" / "phase_1_visual.json"
DEFAULT_AUDIO = ROOT / "outputs" / "phase_1_audio.json"
DEFAULT_VIDEO = ROOT / "downloads" / "video.mp4"
DEFAULT_OUTPUT = ROOT / "outputs" / "debug" / "phase1_debug_overlay.mp4"


def _use_local_clip_bindings_experiment() -> bool:
    return str(os.getenv("CLYPT_EXPERIMENT_LOCAL_CLIP_BINDINGS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def flatten_segmented_detections(
    segments: list[dict],
    *,
    fps: float,
    frame_width: int,
    frame_height: int,
    kind: str,
) -> dict[int, list[dict]]:
    frame_index: dict[int, list[dict]] = {}
    for segment in segments or []:
        segment_track_id = str(segment.get("track_id", ""))
        for obj in segment.get("timestamped_objects", []):
            bbox = obj.get("bounding_box") or {}
            left = float(bbox.get("left", 0.0))
            top = float(bbox.get("top", 0.0))
            right = float(bbox.get("right", 0.0))
            bottom = float(bbox.get("bottom", 0.0))
            time_ms = int(obj.get("time_ms", 0))
            frame_idx = int(round((time_ms / 1000.0) * fps))
            det = {
                "track_id": str(obj.get("track_id") or segment_track_id),
                "kind": kind,
                "time_ms": time_ms,
                "frame_idx": frame_idx,
                "confidence": float(obj.get("confidence", segment.get("confidence", 0.0)) or 0.0),
                "x1": int(round(left * frame_width)),
                "y1": int(round(top * frame_height)),
                "x2": int(round(right * frame_width)),
                "y2": int(round(bottom * frame_height)),
            }
            frame_index.setdefault(frame_idx, []).append(det)
    return frame_index


def flatten_tracks(
    tracks: list[dict],
    *,
    kind: str,
) -> dict[int, list[dict]]:
    frame_index: dict[int, list[dict]] = {}
    for track in tracks or []:
        frame_idx = int(track.get("frame_idx", 0))
        x1 = int(round(float(track.get("x1", 0.0))))
        y1 = int(round(float(track.get("y1", 0.0))))
        x2 = int(round(float(track.get("x2", 0.0))))
        y2 = int(round(float(track.get("y2", 0.0))))
        det = {
            "track_id": str(track.get("track_id", "")),
            "kind": kind,
            "frame_idx": frame_idx,
            "time_ms": None,
            "confidence": float(track.get("confidence", 0.0) or 0.0),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }
        frame_index.setdefault(frame_idx, []).append(det)
    return frame_index


def select_binding_sets(audio: dict) -> tuple[list[dict], list[dict], str]:
    if _use_local_clip_bindings_experiment():
        raw_local = list(audio.get("speaker_bindings_local", []))
        follow_local = list(audio.get("speaker_follow_bindings_local", []))
        if follow_local:
            return raw_local, follow_local, "local"
        if raw_local:
            return raw_local, (follow_local or raw_local), "local"
    raw_global = list(audio.get("speaker_bindings", []))
    follow_global = list(audio.get("speaker_follow_bindings", []))
    return raw_global, (follow_global or raw_global), "global"


def select_track_frame_index(visual: dict, *, fps: float, frame_width: int, frame_height: int) -> tuple[dict[int, list[dict]], str]:
    if _use_local_clip_bindings_experiment():
        tracks_local = list(visual.get("tracks_local", []))
        if tracks_local:
            return flatten_tracks(tracks_local, kind="person"), "tracks_local"
    tracks = list(visual.get("tracks", []))
    if tracks:
        return flatten_tracks(tracks, kind="person"), "tracks"
    person_detections = list(visual.get("person_detections", []))
    if person_detections:
        return flatten_segmented_detections(
            person_detections,
            fps=fps,
            frame_width=frame_width,
            frame_height=frame_height,
            kind="person",
        ), "person_detections"
    return {}, "none"


def active_binding_at_ms(bindings: list[dict], timestamp_ms: int) -> str | None:
    for binding in bindings or []:
        if int(binding.get("start_time_ms", 0)) <= timestamp_ms < int(binding.get("end_time_ms", 0)):
            return str(binding.get("track_id")) if binding.get("track_id") else None
    return None


def _normalized_id_list(values: list[object] | None) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _select_visible_track_ids_for_span(
    span: dict,
    *,
    prefer_local_track_ids: bool,
    available_track_ids: set[str] | None = None,
) -> list[str]:
    candidate_lists = (
        [
            _normalized_id_list(span.get("visible_local_track_ids")),
            _normalized_id_list(span.get("visible_track_ids")),
        ]
        if prefer_local_track_ids
        else [
            _normalized_id_list(span.get("visible_track_ids")),
            _normalized_id_list(span.get("visible_local_track_ids")),
        ]
    )

    available = {track_id for track_id in (available_track_ids or set()) if track_id}
    for candidate_ids in candidate_lists:
        if not candidate_ids:
            continue
        if not available:
            return candidate_ids
        matched = [track_id for track_id in candidate_ids if track_id in available]
        if matched:
            return matched
    return []


def active_speaker_state_at_ms(
    audio: dict,
    timestamp_ms: int,
    *,
    available_track_ids: set[str] | None = None,
) -> dict:
    prefer_local_track_ids = _use_local_clip_bindings_experiment()
    for span in audio.get("active_speakers_local", []) or []:
        start_ms = int(span.get("start_time_ms", 0) or 0)
        end_ms = int(span.get("end_time_ms", start_ms) or start_ms)
        if not (start_ms <= timestamp_ms < end_ms):
            continue
        visible_track_ids = _select_visible_track_ids_for_span(
            span,
            prefer_local_track_ids=prefer_local_track_ids,
            available_track_ids=available_track_ids,
        )
        offscreen_audio_speaker_ids = _normalized_id_list(span.get("offscreen_audio_speaker_ids"))
        return {
            "visible_track_ids": visible_track_ids,
            "offscreen_audio_speaker_ids": offscreen_audio_speaker_ids,
            "overlap": bool(span.get("overlap")) or len(visible_track_ids) > 1 or bool(offscreen_audio_speaker_ids),
            "decision_source": str(span.get("decision_source") or "unknown"),
        }
    return {
        "visible_track_ids": [],
        "offscreen_audio_speaker_ids": [],
        "overlap": False,
        "decision_source": "none",
    }


def current_word_text_at_ms(words: list[dict], timestamp_ms: int) -> str:
    for word in words or []:
        if int(word.get("start_time_ms", 0)) <= timestamp_ms < int(word.get("end_time_ms", 0)):
            return str(word.get("text") or word.get("word") or "").strip()
    return ""


def hybrid_debug_entry_at_ms(speaker_candidate_debug: list[dict], timestamp_ms: int) -> dict | None:
    for entry in speaker_candidate_debug or []:
        start_ms = int(entry.get("start_time_ms", 0) or 0)
        end_ms = int(entry.get("end_time_ms", start_ms) or start_ms)
        if start_ms <= timestamp_ms < end_ms:
            return dict(entry)
    return None


def role_style_for_track(
    track_id: str,
    *,
    raw_track_id: str | None,
    follow_track_id: str | None,
    active_track_ids: set[str] | None = None,
) -> dict:
    if track_id and raw_track_id == track_id and follow_track_id == track_id:
        return {"color": (0, 200, 255), "thickness": 4, "label_suffix": "RAW+FOLLOW"}
    if track_id and follow_track_id == track_id:
        return {"color": (80, 220, 80), "thickness": 4, "label_suffix": "FOLLOW"}
    if track_id and raw_track_id == track_id:
        return {"color": (0, 235, 255), "thickness": 3, "label_suffix": "RAW"}
    if track_id and track_id in (active_track_ids or set()):
        return {"color": (0, 170, 255), "thickness": 3, "label_suffix": "ACTIVE"}
    return {"color": (200, 200, 200), "thickness": 2, "label_suffix": ""}


def nearest_frame_detections(
    frame_index: dict[int, list[dict]],
    frame_idx: int,
    *,
    max_delta: int = 1,
) -> list[dict]:
    if frame_idx in frame_index:
        return frame_index[frame_idx]
    for delta in range(1, max_delta + 1):
        if (frame_idx - delta) in frame_index:
            return frame_index[frame_idx - delta]
        if (frame_idx + delta) in frame_index:
            return frame_index[frame_idx + delta]
    return []


def _det_geometry_metrics(det: dict, frame_width: int, frame_height: int) -> dict[str, float]:
    norm_w = max(1.0, float(frame_width))
    norm_h = max(1.0, float(frame_height))
    x1 = float(det.get("x1", 0.0)) / norm_w
    x2 = float(det.get("x2", 0.0)) / norm_w
    y1 = float(det.get("y1", 0.0)) / norm_h
    y2 = float(det.get("y2", 0.0)) / norm_h
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return {
        "w": w,
        "h": h,
        "area": w * h,
        "aspect": (w / h) if h > 1e-6 else 0.0,
        "center_y": (y1 + y2) / 2.0,
        "bottom": y2,
    }


def _score_render_target_candidate(det: dict, frame_width: int, frame_height: int) -> float:
    metrics = _det_geometry_metrics(det, frame_width, frame_height)
    score = float(det.get("confidence", 0.0) or 0.0)
    score += 0.6 * metrics["h"]
    if metrics["h"] < 0.30:
        score -= 2.0
    if metrics["bottom"] > 0.96 and metrics["center_y"] > 0.72:
        score -= 1.5
    if metrics["aspect"] > 0.95:
        score -= 1.0
    if metrics["area"] < 0.05:
        score -= 1.0
    return score


def _is_plausible_render_target(det: dict, frame_width: int, frame_height: int) -> bool:
    return _score_render_target_candidate(det, frame_width, frame_height) >= 0.25


def _render_target_selection_key(det: dict, frame_width: int, frame_height: int) -> tuple[float, float, float, float, float]:
    metrics = _det_geometry_metrics(det, frame_width, frame_height)
    return (
        -_score_render_target_candidate(det, frame_width, frame_height),
        -float(det.get("confidence", 0.0) or 0.0),
        -metrics["area"],
        -metrics["h"],
        float(det.get("frame_idx", 0)),
    )


def _render_target_iou(left: dict, right: dict) -> float:
    left_x1 = float(left.get("x1", 0.0))
    left_y1 = float(left.get("y1", 0.0))
    left_x2 = float(left.get("x2", 0.0))
    left_y2 = float(left.get("y2", 0.0))
    right_x1 = float(right.get("x1", 0.0))
    right_y1 = float(right.get("y1", 0.0))
    right_x2 = float(right.get("x2", 0.0))
    right_y2 = float(right.get("y2", 0.0))
    ix1 = max(left_x1, right_x1)
    iy1 = max(left_y1, right_y1)
    ix2 = min(left_x2, right_x2)
    iy2 = min(left_y2, right_y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(
        1e-6,
        (left_x2 - left_x1) * (left_y2 - left_y1)
        + (right_x2 - right_x1) * (right_y2 - right_y1)
        - inter,
    )
    return float(inter / union)


def _is_duplicate_fragment(primary: dict, other: dict) -> bool:
    if str(primary.get("track_id", "")) != str(other.get("track_id", "")):
        return False
    return _render_target_iou(primary, other) >= 0.45


def _group_duplicate_track_detections(
    detections: list[dict],
    frame_width: int,
    frame_height: int,
) -> list[dict]:
    grouped: list[dict] = []
    for det in sorted(detections, key=lambda item: _render_target_selection_key(item, frame_width, frame_height)):
        if any(_is_duplicate_fragment(kept, det) for kept in grouped):
            continue
        grouped.append(det)
    return grouped


def _dominant_larger_track_detection(
    candidates: list[dict],
    frame_width: int,
    frame_height: int,
) -> dict | None:
    if len(candidates) != 2:
        return None
    first, second = candidates
    first_metrics = _det_geometry_metrics(first, frame_width, frame_height)
    second_metrics = _det_geometry_metrics(second, frame_width, frame_height)
    if first_metrics["area"] >= second_metrics["area"]:
        larger, larger_metrics = first, first_metrics
        smaller, smaller_metrics = second, second_metrics
    else:
        larger, larger_metrics = second, second_metrics
        smaller, smaller_metrics = first, first_metrics

    area_ratio = larger_metrics["area"] / max(1e-6, smaller_metrics["area"])
    height_ratio = larger_metrics["h"] / max(1e-6, smaller_metrics["h"])
    if area_ratio < 1.8 or height_ratio < 1.2:
        return None
    if larger_metrics["h"] < 0.55 or larger_metrics["area"] < 0.14:
        return None

    suspicious_smaller = (
        smaller_metrics["h"] < 0.42
        or smaller_metrics["area"] < 0.09
        or (smaller_metrics["bottom"] > 0.84 and smaller_metrics["center_y"] > 0.66)
    )
    if not suspicious_smaller:
        return None
    return larger


def choose_clean_track_detection(
    *,
    target_track_id: str,
    frame_detections: list[dict],
    frame_width: int,
    frame_height: int,
) -> dict | None:
    candidates = [
        det
        for det in frame_detections
        if str(det.get("track_id", "")) == target_track_id and _is_plausible_render_target(det, frame_width, frame_height)
    ]
    if not candidates:
        return None
    grouped = _group_duplicate_track_detections(candidates, frame_width, frame_height)
    dominant = _dominant_larger_track_detection(grouped, frame_width, frame_height)
    if dominant is not None:
        return dominant
    return grouped[0] if grouped else None


def select_render_detections(
    frame_detections: list[dict],
    *,
    raw_track_id: str | None,
    follow_track_id: str | None,
    active_track_ids: set[str] | None,
    frame_width: int,
    frame_height: int,
) -> list[dict]:
    targeted_track_ids = {
        track_id
        for track_id in ([raw_track_id, follow_track_id] + list(active_track_ids or set()))
        if track_id
    }
    selected: list[dict] = []
    chosen_by_track_id: dict[str, dict | None] = {}
    for track_id in targeted_track_ids:
        chosen_by_track_id[track_id] = choose_clean_track_detection(
            target_track_id=track_id,
            frame_detections=frame_detections,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    for det in frame_detections:
        track_id = str(det.get("track_id", ""))
        if track_id not in targeted_track_ids:
            selected.append(det)
            continue
        chosen = chosen_by_track_id.get(track_id)
        if chosen is None:
            selected.append(det)
            continue
        if det is chosen:
            selected.append(det)
    return selected


def build_hud_lines(
    *,
    timestamp_ms: int,
    raw_track_id: str | None,
    follow_track_id: str | None,
    current_word: str,
    binding_source: str,
    track_source: str,
    hybrid_debug: dict | None = None,
    active_visible_track_ids: list[str] | None = None,
    offscreen_audio_speaker_ids: list[str] | None = None,
    overlap_active: bool = False,
) -> list[str]:
    lines = [
        f"time: {timestamp_ms / 1000.0:07.2f}s",
        f"binding source: {binding_source}",
        f"track source: {track_source}",
        f"raw speaker: {raw_track_id or 'unknown'}",
        f"follow speaker: {follow_track_id or 'unknown'}",
        f"word: {current_word or '-'}",
    ]
    if overlap_active or active_visible_track_ids or offscreen_audio_speaker_ids:
        lines.append(f"overlap: {'yes' if overlap_active else 'no'}")
    if active_visible_track_ids:
        lines.append(f"active visible: {','.join(active_visible_track_ids)}")
    if offscreen_audio_speaker_ids:
        lines.append(f"offscreen active audio: {','.join(offscreen_audio_speaker_ids)}")
    if hybrid_debug:
        lines.extend(
            [
                f"audio speaker: {hybrid_debug.get('active_audio_speaker_id') or 'unknown'}",
                f"decision: {hybrid_debug.get('decision_source') or 'unknown'}",
                f"ambiguous: {'yes' if hybrid_debug.get('ambiguous') else 'no'}",
                f"audio local: {hybrid_debug.get('active_audio_local_track_id') or 'unknown'}",
                f"chosen local: {hybrid_debug.get('chosen_local_track_id') or 'unknown'}",
                (
                    f"margin: {float(hybrid_debug.get('top_1_top_2_margin')):.3f}"
                    if hybrid_debug.get("top_1_top_2_margin") not in (None, "")
                    else "margin: unknown"
                ),
            ]
        )
    return lines


def draw_hud(
    frame,
    *,
    timestamp_ms: int,
    raw_track_id: str | None,
    follow_track_id: str | None,
    current_word: str,
    binding_source: str,
    track_source: str,
    hybrid_debug: dict | None = None,
    active_visible_track_ids: list[str] | None = None,
    offscreen_audio_speaker_ids: list[str] | None = None,
    overlap_active: bool = False,
) -> None:
    lines = build_hud_lines(
        timestamp_ms=timestamp_ms,
        raw_track_id=raw_track_id,
        follow_track_id=follow_track_id,
        current_word=current_word,
        binding_source=binding_source,
        track_source=track_source,
        hybrid_debug=hybrid_debug,
        active_visible_track_ids=active_visible_track_ids,
        offscreen_audio_speaker_ids=offscreen_audio_speaker_ids,
        overlap_active=overlap_active,
    )
    x = 24
    y = 42
    for line in lines:
        cv2.putText(frame, line, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        y += 34


def draw_detection_box(
    frame,
    det: dict,
    *,
    raw_track_id: str | None,
    follow_track_id: str | None,
    active_track_ids: set[str] | None = None,
) -> None:
    style = role_style_for_track(
        str(det.get("track_id", "")),
        raw_track_id=raw_track_id,
        follow_track_id=follow_track_id,
        active_track_ids=active_track_ids,
    )
    x1 = int(det["x1"])
    y1 = int(det["y1"])
    x2 = int(det["x2"])
    y2 = int(det["y2"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), style["color"], style["thickness"])
    label = str(det.get("track_id", ""))
    if style["label_suffix"]:
        label = f"{label} {style['label_suffix']}"
    cv2.putText(frame, label, (x1 + 4, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (x1 + 4, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, style["color"], 2, cv2.LINE_AA)


def draw_face_box(frame, det: dict) -> None:
    x1 = int(det["x1"])
    y1 = int(det["y1"])
    x2 = int(det["x2"])
    y2 = int(det["y2"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 255), 1)


def mux_audio(silent_video_path: Path, source_video_path: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(silent_video_path),
        "-i",
        str(source_video_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio mux failed:\n{result.stderr[-1200:]}")


def render_debug_video(
    *,
    video_path: Path,
    visual_path: Path,
    audio_path: Path,
    output_path: Path,
    max_frames: int | None = None,
) -> Path:
    with visual_path.open() as f:
        visual = json.load(f)
    with audio_path.open() as f:
        audio = json.load(f)

    video_meta = visual.get("video_metadata") or {}
    fps = float(video_meta.get("fps", 23.976) or 23.976)
    frame_width = int(video_meta.get("width", 0) or 0)
    frame_height = int(video_meta.get("height", 0) or 0)
    if frame_width <= 0 or frame_height <= 0:
        raise RuntimeError("Missing video dimensions in phase_1_visual metadata")

    person_frame_index, track_source = select_track_frame_index(
        visual,
        fps=fps,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    available_track_ids = {
        str(det.get("track_id", "")).strip()
        for detections in person_frame_index.values()
        for det in detections
        if str(det.get("track_id", "")).strip()
    }
    face_frame_index = flatten_segmented_detections(
        visual.get("face_detections", []),
        fps=fps,
        frame_width=frame_width,
        frame_height=frame_height,
        kind="face",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source video: {video_path}")

    with tempfile.TemporaryDirectory(prefix="clypt-debug-render-") as tmpdir:
        silent_output = Path(tmpdir) / "phase1_debug_silent.mp4"
        writer = cv2.VideoWriter(
            str(silent_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Unable to open output video writer: {silent_output}")

        raw_bindings, follow_bindings, binding_source = select_binding_sets(audio)
        words = list(audio.get("words", []))
        speaker_candidate_debug = list(audio.get("speaker_candidate_debug", []))

        frame_idx = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            timestamp_ms = int(round((frame_idx / fps) * 1000.0))
            raw_track_id = active_binding_at_ms(raw_bindings, timestamp_ms)
            follow_track_id = active_binding_at_ms(follow_bindings, timestamp_ms)
            current_word = current_word_text_at_ms(words, timestamp_ms)
            hybrid_debug = hybrid_debug_entry_at_ms(speaker_candidate_debug, timestamp_ms)
            active_speaker_state = active_speaker_state_at_ms(
                audio,
                timestamp_ms,
                available_track_ids=available_track_ids,
            )
            active_track_ids = set(active_speaker_state["visible_track_ids"])

            person_detections = nearest_frame_detections(person_frame_index, frame_idx, max_delta=1)
            for det in select_render_detections(
                person_detections,
                raw_track_id=raw_track_id,
                follow_track_id=follow_track_id,
                active_track_ids=active_track_ids,
                frame_width=frame_width,
                frame_height=frame_height,
            ):
                draw_detection_box(
                    frame,
                    det,
                    raw_track_id=raw_track_id,
                    follow_track_id=follow_track_id,
                    active_track_ids=active_track_ids,
                )
            for det in nearest_frame_detections(face_frame_index, frame_idx, max_delta=1):
                draw_face_box(frame, det)

            draw_hud(
                frame,
                timestamp_ms=timestamp_ms,
                raw_track_id=raw_track_id,
                follow_track_id=follow_track_id,
                current_word=current_word,
                binding_source=binding_source,
                track_source=track_source,
                hybrid_debug=hybrid_debug,
                active_visible_track_ids=active_speaker_state["visible_track_ids"],
                offscreen_audio_speaker_ids=active_speaker_state["offscreen_audio_speaker_ids"],
                overlap_active=bool(active_speaker_state["overlap"]),
            )
            writer.write(frame)
            frame_idx += 1

        capture.release()
        writer.release()
        mux_audio(silent_output, video_path, output_path)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render full-video Phase 1 debug overlays on the original video.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--visual", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = render_debug_video(
        video_path=args.video,
        visual_path=args.visual,
        audio_path=args.audio,
        output_path=args.output,
        max_frames=args.max_frames,
    )
    print(out)


if __name__ == "__main__":
    main()
