"""Render vertical active-speaker-follow clips from current Phase 1 outputs.

This is a lightweight QA renderer:
- picks strong ~40s windows from speaker_bindings
- follows the active speaker's person/body track center
- outputs 9:16 clips with original audio preserved
"""

from __future__ import annotations

import bisect
import json
import math
import os
import subprocess
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VIDEO_PATH = ROOT / "downloads" / "video.mp4"
VISUAL_PATH = ROOT / "outputs" / "phase_1_visual.json"
AUDIO_PATH = ROOT / "outputs" / "phase_1_audio.json"
OUTPUT_DIR = ROOT / "outputs" / "clips"

NUM_CLIPS = int(os.getenv("CLYPT_RENDER_NUM_CLIPS", "3"))
CLIP_DURATION_S = int(os.getenv("CLYPT_RENDER_CLIP_DURATION_S", "40"))
WINDOW_STEP_S = 5
KEYFRAME_STEP_S = 0.5
WINDOW_MIN_GAP_S = 20
RENDER_DEBUG_MODE = os.getenv("CLYPT_RENDER_DEBUG_MODE", "1").strip() != "0"
RENDER_DEBUG_SHOW_FACES = os.getenv("CLYPT_RENDER_DEBUG_SHOW_FACES", "1").strip() != "0"

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
DEBUG_DEFAULT_BOX_COLOR = "0x4FC3F7"
DEBUG_FOLLOW_BOX_COLOR = "0x00FF88"
DEBUG_RAW_ACTIVE_BOX_COLOR = "0xFFD54F"
DEBUG_FACE_BOX_COLOR = "0xFF6F61"
DEBUG_TIMELINE_RAW_Y = OUT_H - 96
DEBUG_TIMELINE_FOLLOW_Y = OUT_H - 52
DEBUG_TIMELINE_HEIGHT = 24
DEBUG_TIMELINE_LEFT = 120
DEBUG_TIMELINE_RIGHT_PAD = 32
DEBUG_HUD_FONT_SIZE = 28
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
SPLIT_ADAPTIVE_STEP_S = 0.5
SPLIT_MIN_SEGMENT_S = 1.0
SPLIT_MIN_DISTINCT_CENTER_GAP_RATIO = 0.16
SPLIT_MAX_OVERLAP_IOU = 0.28
SINGLE_SPEAKER_MIN_SEGMENT_S = 0.45


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
    label: str | None = None
    box_thickness: int = OVERLAY_BOX_THICKNESS
    text_color: str = "white"


@dataclass(frozen=True)
class AdaptiveSegment:
    mode: str
    start_s: float
    end_s: float
    primary_track_id: str | None
    secondary_track_id: str | None = None


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
    src_w: int,
    src_h: int,
    person_track_index: dict[str, list[Detection]],
    face_track_index: dict[str, list[Detection]],
    frame_detection_index: dict[int, list[dict]] | None = None,
) -> Detection | None:
    if not track_id:
        return None
    if frame_detection_index:
        frame_detections = frame_detection_index.get(frame_idx, [])
        chosen = resolve_follow_box(
            track_id,
            frame_detections,
            src_w,
            src_h,
        )
        if chosen is not None:
            return _render_target_to_detection(chosen, frame_idx, src_w, src_h)
        fallback = fallback_anchor_for_missing_clean_target(
            frame_detections,
            "single_person",
            src_w,
            src_h,
        )
        if fallback is not None:
            return fallback
    person_det = interpolate_detection(person_track_index.get(track_id), frame_idx, fps)
    # Renderer is currently body-led by design. We keep the face index plumbed
    # through for compatibility with older call sites, but camera/overlay
    # anchors intentionally ignore face tracks here.
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


def build_frame_detection_index(tracks: list[dict]) -> dict[int, list[dict]]:
    frame_index: dict[int, list[dict]] = {}
    for track in tracks:
        frame_idx = int(track.get("frame_idx", 0))
        frame_index.setdefault(frame_idx, []).append(track)
    return frame_index


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

def _normalized_id_list(values: list[object] | None) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def overlap_follow_decision_at_ms(overlap_follow_decisions: list[dict] | None, abs_ms: int) -> dict | None:
    for decision in overlap_follow_decisions or []:
        start_ms = int(decision.get("start_time_ms", 0) or 0)
        end_ms = int(decision.get("end_time_ms", start_ms) or start_ms)
        if start_ms <= abs_ms < end_ms:
            return dict(decision)
    return None


def overlap_follow_decisions_for_interval(
    overlap_follow_decisions: list[dict] | None,
    clip_start_s: float,
    clip_end_s: float,
) -> list[dict]:
    clip_start_ms = int(round(float(clip_start_s) * 1000.0))
    clip_end_ms = int(round(float(clip_end_s) * 1000.0))
    selected: list[dict] = []
    for decision in overlap_follow_decisions or []:
        start_ms = int(decision.get("start_time_ms", 0) or 0)
        end_ms = int(decision.get("end_time_ms", start_ms) or start_ms)
        if end_ms <= clip_start_ms or start_ms >= clip_end_ms:
            continue
        selected.append(dict(decision))
    return selected


def _decision_camera_target_track_id(decision: dict, *, prefer_local_track_ids: bool) -> str | None:
    ordered_candidates = (
        [
            str(decision.get("camera_target_local_track_id") or "").strip(),
            str(decision.get("camera_target_track_id") or "").strip(),
        ]
        if prefer_local_track_ids
        else [
            str(decision.get("camera_target_track_id") or "").strip(),
            str(decision.get("camera_target_local_track_id") or "").strip(),
        ]
    )
    for track_id in ordered_candidates:
        if track_id:
            return track_id
    return None


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


def active_speaker_state_for_interval(
    *,
    clip_start_s: float,
    clip_end_s: float,
    active_speakers_local: list[dict] | None,
    prefer_local_track_ids: bool,
    available_track_ids: set[str] | None = None,
) -> dict:
    clip_start_ms = int(round(float(clip_start_s) * 1000.0))
    clip_end_ms = int(round(float(clip_end_s) * 1000.0))
    visible_track_ids: list[str] = []
    offscreen_audio_speaker_ids: list[str] = []
    overlap_active = False

    for span in active_speakers_local or []:
        start_ms = int(span.get("start_time_ms", 0) or 0)
        end_ms = int(span.get("end_time_ms", start_ms) or start_ms)
        if end_ms <= clip_start_ms or start_ms >= clip_end_ms:
            continue
        overlap_active = overlap_active or bool(span.get("overlap"))
        for track_id in _select_visible_track_ids_for_span(
            span,
            prefer_local_track_ids=prefer_local_track_ids,
            available_track_ids=available_track_ids,
        ):
            if track_id not in visible_track_ids:
                visible_track_ids.append(track_id)
        for speaker_id in _normalized_id_list(span.get("offscreen_audio_speaker_ids")):
            if speaker_id not in offscreen_audio_speaker_ids:
                offscreen_audio_speaker_ids.append(speaker_id)

    overlap_active = overlap_active or len(visible_track_ids) > 1 or bool(offscreen_audio_speaker_ids)
    return {
        "visible_track_ids": visible_track_ids,
        "offscreen_audio_speaker_ids": offscreen_audio_speaker_ids,
        "overlap": overlap_active,
    }


def resolve_follow_identity(
    bindings: list[dict],
    abs_ms: int,
    *,
    overlap_follow_decisions: list[dict] | None = None,
    prefer_local_track_ids: bool = False,
    frame_detections: list[dict] | None = None,
) -> str | None:
    def _single_visible_track_id() -> str | None:
        track_ids: list[str] = []
        for det in frame_detections or []:
            track_id = str(det.get("track_id") or "").strip()
            if track_id and track_id not in track_ids:
                track_ids.append(track_id)
        if len(track_ids) == 1:
            return track_ids[0]
        return None

    active_decision = overlap_follow_decision_at_ms(overlap_follow_decisions, abs_ms)
    if active_decision is not None:
        decision_target = None
        if not bool(active_decision.get("stay_wide")):
            decision_target = _decision_camera_target_track_id(
                active_decision,
                prefer_local_track_ids=prefer_local_track_ids,
            )
        if decision_target:
            return decision_target
        singleton_track_id = _single_visible_track_id()
        if singleton_track_id:
            return singleton_track_id
        if bool(active_decision.get("stay_wide")):
            return None

    singleton_track_id = _single_visible_track_id()
    if singleton_track_id:
        return singleton_track_id

    for binding in reversed(bindings):
        if int(binding["start_time_ms"]) == abs_ms:
            return str(binding["track_id"])

    binding = active_track_at(bindings, abs_ms)
    if binding is None:
        return None
    return binding


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
    src_w: int,
    src_h: int,
    person_track_index: dict[str, list[Detection]],
    face_track_index: dict[str, list[Detection]],
    frame_detection_index: dict[int, list[dict]] | None = None,
) -> Detection | None:
    return _track_anchor_candidate(
        track_id,
        frame_idx,
        fps,
        src_w,
        src_h,
        person_track_index,
        face_track_index,
        frame_detection_index,
    )


def _detection_iou(left: Detection, right: Detection) -> float:
    left_x1 = float(left.x_center) - (0.5 * float(left.width))
    left_y1 = float(left.y_center) - (0.5 * float(left.height))
    left_x2 = left_x1 + float(left.width)
    left_y2 = left_y1 + float(left.height)
    right_x1 = float(right.x_center) - (0.5 * float(right.width))
    right_y1 = float(right.y_center) - (0.5 * float(right.height))
    right_x2 = right_x1 + float(right.width)
    right_y2 = right_y1 + float(right.height)
    ix1 = max(left_x1, right_x1)
    iy1 = max(left_y1, right_y1)
    ix2 = min(left_x2, right_x2)
    iy2 = min(left_y2, right_y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(
        1.0,
        (left_x2 - left_x1) * (left_y2 - left_y1)
        + (right_x2 - right_x1) * (right_y2 - right_y1)
        - inter,
    )
    return float(inter / union)


def _bbox_in_unit_space(bbox: list[float], frame_width: int, frame_height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        return x1, y1, x2, y2

    norm_w = max(1.0, float(frame_width))
    norm_h = max(1.0, float(frame_height))
    return x1 / norm_w, y1 / norm_h, x2 / norm_w, y2 / norm_h


def bbox_geometry_metrics(bbox: list[float], frame_width: int, frame_height: int) -> dict[str, float]:
    x1, y1, x2, y2 = _bbox_in_unit_space(bbox, frame_width, frame_height)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    aspect = w / h if h > 1e-6 else 0.0
    return {
        "w": w,
        "h": h,
        "area": area,
        "aspect": aspect,
        "center_y": (y1 + y2) / 2.0,
        "bottom": y2,
    }


def score_render_target_candidate(det: dict, frame_width: int, frame_height: int) -> float:
    metrics = bbox_geometry_metrics(det["bbox"], frame_width, frame_height)
    score = float(det.get("score", 0.0))

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


def is_plausible_render_target(det: dict, frame_width: int, frame_height: int) -> bool:
    return score_render_target_candidate(det, frame_width, frame_height) >= 0.25


def _render_target_selection_key(det: dict, frame_width: int, frame_height: int) -> tuple[float, float, float, float, float]:
    metrics = bbox_geometry_metrics(det["bbox"], frame_width, frame_height)
    return (
        -score_render_target_candidate(det, frame_width, frame_height),
        -float(det.get("score", 0.0)),
        -metrics["area"],
        -metrics["h"],
        float(det.get("frame_idx", 0)),
    )


def _render_target_bbox_iou(left: dict, right: dict, frame_width: int, frame_height: int) -> float:
    left_x1, left_y1, left_x2, left_y2 = _bbox_in_unit_space(left["bbox"], frame_width, frame_height)
    right_x1, right_y1, right_x2, right_y2 = _bbox_in_unit_space(right["bbox"], frame_width, frame_height)
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


def is_duplicate_fragment(primary: dict, other: dict, frame_width: int, frame_height: int) -> bool:
    if str(primary.get("track_id", "")) != str(other.get("track_id", "")):
        return False
    if "bbox" not in primary or "bbox" not in other:
        return False
    return _render_target_bbox_iou(primary, other, frame_width, frame_height) >= 0.45


def _group_duplicate_render_targets(frame_detections: list[dict], frame_width: int, frame_height: int) -> list[dict]:
    grouped: list[dict] = []
    for det in sorted(frame_detections, key=lambda item: _render_target_selection_key(item, frame_width, frame_height)):
        if any(is_duplicate_fragment(kept, det, frame_width, frame_height) for kept in grouped):
            continue
        grouped.append(det)
    return grouped


def _dominant_larger_render_target(
    candidates: list[dict],
    frame_width: int,
    frame_height: int,
) -> dict | None:
    if len(candidates) != 2:
        return None
    first, second = candidates
    first_metrics = bbox_geometry_metrics(first["bbox"], frame_width, frame_height)
    second_metrics = bbox_geometry_metrics(second["bbox"], frame_width, frame_height)

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

    # Small, lower-frame boxes are often hands / cropped fragments. When the
    # larger candidate looks like a full person, prefer it decisively.
    suspicious_smaller = (
        smaller_metrics["h"] < 0.42
        or smaller_metrics["area"] < 0.09
        or (smaller_metrics["bottom"] > 0.84 and smaller_metrics["center_y"] > 0.66)
    )
    if not suspicious_smaller:
        return None

    return larger


def choose_clean_render_target(
    *,
    target_track_id: str,
    frame_detections: list[dict],
    frame_width: int,
    frame_height: int,
) -> dict | None:
    candidates = [
        det
        for det in frame_detections
        if str(det.get("track_id", "")) == target_track_id
        and "bbox" in det
        and is_plausible_render_target(det, frame_width, frame_height)
    ]
    if not candidates:
        return None

    grouped = _group_duplicate_render_targets(candidates, frame_width, frame_height)
    dominant = _dominant_larger_render_target(grouped, frame_width, frame_height)
    if dominant is not None:
        return dominant
    return grouped[0] if grouped else None


def resolve_follow_box(
    track_id: str | None,
    frame_detections: list[dict],
    frame_width: int,
    frame_height: int,
) -> dict | None:
    if not track_id:
        return None
    return choose_clean_render_target(
        target_track_id=track_id,
        frame_detections=frame_detections,
        frame_width=frame_width,
        frame_height=frame_height,
    )


def _detection_bbox_to_pixels(det: dict, frame_width: int, frame_height: int) -> tuple[float, float, float, float] | None:
    bbox = det.get("bbox")
    if not bbox or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = _bbox_in_unit_space(bbox, frame_width, frame_height)
    scale_w = max(1.0, float(frame_width))
    scale_h = max(1.0, float(frame_height))
    return x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h


def estimate_shared_context_bbox(
    frame_detections: list[dict],
    frame_width: int,
    frame_height: int,
) -> Detection | None:
    boxes: list[tuple[float, float, float, float]] = []
    frame_idxs: list[int] = []
    confidences: list[float] = []
    for det in frame_detections:
        box = _detection_bbox_to_pixels(det, frame_width, frame_height)
        if box is None:
            continue
        boxes.append(box)
        frame_idxs.append(int(det.get("frame_idx", 0)))
        confidences.append(float(det.get("score", det.get("confidence", 0.0))))

    if not boxes:
        return None

    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)

    union_w = max(1.0, x2 - x1)
    union_h = max(1.0, y2 - y1)
    pad_x = max(union_w * 0.24, float(frame_width) * 0.08)
    pad_y = max(union_h * 0.20, float(frame_height) * 0.05)
    x1 = clamp(x1 - pad_x, 0.0, float(frame_width))
    x2 = clamp(x2 + pad_x, 0.0, float(frame_width))
    y1 = clamp(y1 - pad_y, 0.0, float(frame_height))
    y2 = clamp(y2 + pad_y, 0.0, float(frame_height))

    if x2 <= x1 or y2 <= y1:
        return None

    return Detection(
        frame_idx=int(round(statistics.median(frame_idxs))) if frame_idxs else 0,
        x_center=(x1 + x2) / 2.0,
        y_center=(y1 + y2) / 2.0,
        width=x2 - x1,
        height=y2 - y1,
        confidence=max(confidences) if confidences else 0.0,
    )


def fallback_anchor_for_missing_clean_target(
    frame_detections: list[dict],
    composition_mode: str,
    frame_width: int,
    frame_height: int,
) -> Detection | None:
    if composition_mode != "single_person":
        return None
    return estimate_shared_context_bbox(frame_detections, frame_width, frame_height)


def score_context_fallback_quality(fallback: Detection | None, frame_width: int, frame_height: int) -> float:
    if fallback is None:
        return 0.0

    frame_w = max(1.0, float(frame_width))
    frame_h = max(1.0, float(frame_height))
    frame_area = frame_w * frame_h
    area_ratio = clamp((float(fallback.width) * float(fallback.height)) / frame_area, 0.0, 1.0)
    coverage_score = clamp(area_ratio * 2.0, 0.0, 1.0)

    fallback_aspect = float(fallback.width) / max(1.0, float(fallback.height))
    frame_aspect = frame_w / frame_h
    aspect_distance = abs(math.log(max(1e-6, fallback_aspect / frame_aspect)))
    aspect_score = clamp(1.0 - (aspect_distance / 1.25), 0.0, 1.0)

    center_x = clamp(float(fallback.x_center) / frame_w, 0.0, 1.0)
    center_y = clamp(float(fallback.y_center) / frame_h, 0.0, 1.0)
    center_distance = math.hypot(center_x - 0.5, center_y - 0.5)
    center_score = clamp(1.0 - (center_distance / 0.75), 0.0, 1.0)

    return round((0.45 * coverage_score) + (0.35 * center_score) + (0.20 * aspect_score), 2)


def _render_target_to_detection(det: dict, frame_idx: int, frame_width: int, frame_height: int) -> Detection:
    x1, y1, x2, y2 = [float(value) for value in det["bbox"][:4]]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1 *= max(1.0, float(frame_width))
        x2 *= max(1.0, float(frame_width))
        y1 *= max(1.0, float(frame_height))
        y2 *= max(1.0, float(frame_height))
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return Detection(
        frame_idx=int(det.get("frame_idx", frame_idx)),
        x_center=x1 + (width / 2.0),
        y_center=y1 + (height / 2.0),
        width=width,
        height=height,
        confidence=float(det.get("score", det.get("confidence", 0.0))),
    )


def _render_target_debug_rejections(
    *,
    chosen: dict | None,
    candidates: list[dict],
    frame_width: int,
    frame_height: int,
) -> list[dict]:
    rejections: list[dict] = []
    for det in candidates:
        if chosen is not None and det is chosen:
            continue
        if not is_plausible_render_target(det, frame_width, frame_height):
            reason = "partial_body"
        elif chosen is not None and is_duplicate_fragment(chosen, det, frame_width, frame_height):
            reason = "duplicate_fragment"
        else:
            reason = "lower_quality_clean_candidate"
        rejections.append(
            {
                "reason": reason,
                "bbox": list(det.get("bbox", [])),
                "score": round(float(det.get("score", det.get("confidence", 0.0))), 3),
            }
        )
    return rejections


def build_render_target_debug_info(
    *,
    track_id: str | None,
    frame_detections: list[dict],
    frame_width: int,
    frame_height: int,
    composition_mode: str,
) -> dict:
    track_id_str = str(track_id or "")
    candidates = [
        det
        for det in frame_detections
        if str(det.get("track_id", "")) == track_id_str and "bbox" in det
    ]
    chosen = choose_clean_render_target(
        target_track_id=track_id_str,
        frame_detections=frame_detections,
        frame_width=frame_width,
        frame_height=frame_height,
    ) if track_id_str else None
    rejections = _render_target_debug_rejections(
        chosen=chosen,
        candidates=candidates,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    if chosen is not None:
        return {
            "target_track_id": track_id_str or None,
            "target_quality": round(score_render_target_candidate(chosen, frame_width, frame_height), 2),
            "target_source": "clean_body_box",
            "fallback_used": False,
            "fallback_quality": None,
            "candidate_count": len(candidates),
            "rejections": rejections,
        }

    fallback = fallback_anchor_for_missing_clean_target(
        frame_detections,
        composition_mode,
        frame_width,
        frame_height,
    )
    if fallback is not None:
        fallback_quality = score_context_fallback_quality(fallback, frame_width, frame_height)
        return {
            "target_track_id": track_id_str or None,
            "target_quality": fallback_quality,
            "target_source": "context_fallback",
            "fallback_used": True,
            "fallback_quality": fallback_quality,
            "candidate_count": len(candidates),
            "rejections": rejections,
        }

    return {
        "target_track_id": track_id_str or None,
        "target_quality": None,
        "target_source": "track_interpolation" if track_id_str else "none",
        "fallback_used": False,
        "fallback_quality": None,
        "candidate_count": len(candidates),
        "rejections": rejections,
    }


def build_render_segment_debug_payload(
    *,
    segment: AdaptiveSegment,
    fps: float,
    src_w: int,
    src_h: int,
    frame_detection_index: dict[int, list[dict]] | None,
    speaker_candidate_debug: list[dict] | None = None,
) -> dict:
    decision_frame_idx = int(round(float(segment.start_s) * float(fps)))
    frame_detections = frame_detection_index.get(decision_frame_idx, []) if frame_detection_index else []
    targets = [
        build_render_target_debug_info(
            track_id=track_id,
            frame_detections=frame_detections,
            frame_width=src_w,
            frame_height=src_h,
            composition_mode=segment.mode,
        )
        for track_id in (segment.primary_track_id, segment.secondary_track_id)
        if track_id
    ]
    payload = dict(segment.__dict__)
    payload["decision_frame_idx"] = decision_frame_idx
    payload["target_debug"] = targets
    payload["hybrid_debug"] = summarize_hybrid_debug_for_interval(
        clip_start_s=float(segment.start_s),
        clip_end_s=float(segment.end_s),
        speaker_candidate_debug=speaker_candidate_debug,
    )
    return payload


def build_render_debug_sidecar_payload(
    *,
    clip_name: str,
    window: dict,
    binding_source: str,
    debug_mode: bool,
    debug_show_faces: bool,
    composition: CompositionPlan,
    segments: list[AdaptiveSegment],
    fps: float,
    src_w: int,
    src_h: int,
    frame_detection_index: dict[int, list[dict]] | None,
    speaker_candidate_debug: list[dict] | None = None,
) -> dict:
    return {
        "clip_name": clip_name,
        "window": window,
        "binding_source": binding_source,
        "debug_mode": debug_mode,
        "debug_show_faces": debug_show_faces,
        "composition": composition.__dict__,
        "segments": [
            build_render_segment_debug_payload(
                segment=segment,
                fps=fps,
                src_w=src_w,
                src_h=src_h,
                frame_detection_index=frame_detection_index,
                speaker_candidate_debug=speaker_candidate_debug,
            )
            for segment in segments
        ],
    }


def _split_frame_is_plausible(
    *,
    primary_det: Detection | None,
    secondary_det: Detection | None,
    src_w: int,
) -> bool:
    if primary_det is None or secondary_det is None:
        return False
    avg_w = max(1.0, 0.5 * (float(primary_det.width) + float(secondary_det.width)))
    center_gap = abs(float(primary_det.x_center) - float(secondary_det.x_center))
    if center_gap < max(src_w * SPLIT_MIN_DISTINCT_CENTER_GAP_RATIO, 0.55 * avg_w):
        return False
    if _detection_iou(primary_det, secondary_det) > SPLIT_MAX_OVERLAP_IOU:
        return False
    return True


def choose_adaptive_split_segments(
    *,
    primary_track_id: str,
    secondary_track_id: str,
    clip_start_s: int,
    clip_end_s: int,
    fps: float,
    src_w: int,
    src_h: int,
    bindings: list[dict],
    person_track_index: dict[str, list[Detection]],
    face_track_index: dict[str, list[Detection]],
    frame_detection_index: dict[int, list[dict]] | None = None,
) -> list[AdaptiveSegment]:
    samples: list[AdaptiveSegment] = []
    t = 0.0
    while t <= (clip_end_s - clip_start_s + 1e-6):
        abs_s = clip_start_s + t
        abs_ms = int(round(abs_s * 1000.0))
        frame_idx = int(round((abs_ms / 1000.0) * fps))
        primary_det = _track_anchor_at_frame(
            primary_track_id,
            frame_idx,
            fps,
            src_w,
            src_h,
            person_track_index,
            face_track_index,
            frame_detection_index,
        )
        secondary_det = _track_anchor_at_frame(
            secondary_track_id,
            frame_idx,
            fps,
            src_w,
            src_h,
            person_track_index,
            face_track_index,
            frame_detection_index,
        )
        if _split_frame_is_plausible(primary_det=primary_det, secondary_det=secondary_det, src_w=src_w):
            sample = AdaptiveSegment(
                mode="two_split",
                start_s=abs_s,
                end_s=min(clip_end_s, abs_s + SPLIT_ADAPTIVE_STEP_S),
                primary_track_id=primary_track_id,
                secondary_track_id=secondary_track_id,
            )
        else:
            active_tid = active_track_at(bindings, abs_ms)
            if active_tid not in {primary_track_id, secondary_track_id}:
                if primary_det is not None and secondary_det is None:
                    active_tid = primary_track_id
                elif secondary_det is not None and primary_det is None:
                    active_tid = secondary_track_id
                else:
                    active_tid = primary_track_id
            sample = AdaptiveSegment(
                mode="single_person",
                start_s=abs_s,
                end_s=min(clip_end_s, abs_s + SPLIT_ADAPTIVE_STEP_S),
                primary_track_id=active_tid,
                secondary_track_id=None,
            )
        samples.append(sample)
        t += SPLIT_ADAPTIVE_STEP_S

    if not samples:
        return [
            AdaptiveSegment(
                mode="single_person",
                start_s=float(clip_start_s),
                end_s=float(clip_end_s),
                primary_track_id=primary_track_id,
                secondary_track_id=None,
            )
        ]

    merged: list[AdaptiveSegment] = []
    for sample in samples:
        if (
            merged
            and merged[-1].mode == sample.mode
            and merged[-1].primary_track_id == sample.primary_track_id
            and merged[-1].secondary_track_id == sample.secondary_track_id
        ):
            prev = merged[-1]
            merged[-1] = AdaptiveSegment(
                mode=prev.mode,
                start_s=prev.start_s,
                end_s=sample.end_s,
                primary_track_id=prev.primary_track_id,
                secondary_track_id=prev.secondary_track_id,
            )
        else:
            merged.append(sample)

    stabilized: list[AdaptiveSegment] = []
    for segment in merged:
        duration = float(segment.end_s - segment.start_s)
        if stabilized and duration < SPLIT_MIN_SEGMENT_S:
            prev = stabilized[-1]
            stabilized[-1] = AdaptiveSegment(
                mode=prev.mode,
                start_s=prev.start_s,
                end_s=segment.end_s,
                primary_track_id=prev.primary_track_id,
                secondary_track_id=prev.secondary_track_id,
            )
        else:
            stabilized.append(segment)

    if stabilized:
        first = stabilized[0]
        stabilized[0] = AdaptiveSegment(
            mode=first.mode,
            start_s=float(clip_start_s),
            end_s=first.end_s,
            primary_track_id=first.primary_track_id,
            secondary_track_id=first.secondary_track_id,
        )
        last = stabilized[-1]
        stabilized[-1] = AdaptiveSegment(
            mode=last.mode,
            start_s=last.start_s,
            end_s=float(clip_end_s),
            primary_track_id=last.primary_track_id,
            secondary_track_id=last.secondary_track_id,
        )
    return stabilized


def choose_active_speaker_segments(
    *,
    clip_start_s: float,
    clip_end_s: float,
    bindings: list[dict],
    fallback_track_id: str | None = None,
) -> list[AdaptiveSegment]:
    clipped: list[AdaptiveSegment] = []
    for binding in bindings:
        track_id = str(binding.get("track_id", "") or "")
        if not track_id:
            continue
        start_s = max(float(clip_start_s), float(binding.get("start_time_ms", 0)) / 1000.0)
        end_s = min(float(clip_end_s), float(binding.get("end_time_ms", 0)) / 1000.0)
        if end_s <= start_s:
            continue
        clipped.append(
            AdaptiveSegment(
                mode="single_person",
                start_s=start_s,
                end_s=end_s,
                primary_track_id=track_id,
                secondary_track_id=None,
            )
        )

    if not clipped:
        return [
            AdaptiveSegment(
                mode="single_person",
                start_s=float(clip_start_s),
                end_s=float(clip_end_s),
                primary_track_id=fallback_track_id,
                secondary_track_id=None,
            )
        ]

    clipped.sort(key=lambda segment: (segment.start_s, segment.end_s, segment.primary_track_id or ""))

    merged: list[AdaptiveSegment] = []
    for segment in clipped:
        if (
            merged
            and merged[-1].primary_track_id == segment.primary_track_id
            and segment.start_s <= (merged[-1].end_s + 1e-3)
        ):
            prev = merged[-1]
            merged[-1] = AdaptiveSegment(
                mode="single_person",
                start_s=prev.start_s,
                end_s=max(prev.end_s, segment.end_s),
                primary_track_id=prev.primary_track_id,
                secondary_track_id=None,
            )
        else:
            merged.append(segment)

    stabilized: list[AdaptiveSegment] = []
    idx = 0
    while idx < len(merged):
        segment = merged[idx]
        duration = float(segment.end_s - segment.start_s)
        prev_segment = stabilized[-1] if stabilized else None
        next_segment = merged[idx + 1] if idx + 1 < len(merged) else None
        if (
            duration < SINGLE_SPEAKER_MIN_SEGMENT_S
            and prev_segment is not None
            and next_segment is not None
            and prev_segment.primary_track_id == next_segment.primary_track_id
        ):
            stabilized[-1] = AdaptiveSegment(
                mode="single_person",
                start_s=prev_segment.start_s,
                end_s=next_segment.end_s,
                primary_track_id=prev_segment.primary_track_id,
                secondary_track_id=None,
            )
            idx += 2
            continue

        if prev_segment is not None and segment.start_s > prev_segment.end_s:
            stabilized[-1] = AdaptiveSegment(
                mode="single_person",
                start_s=prev_segment.start_s,
                end_s=segment.start_s,
                primary_track_id=prev_segment.primary_track_id,
                secondary_track_id=None,
            )
        stabilized.append(segment)
        idx += 1

    if stabilized:
        first = stabilized[0]
        stabilized[0] = AdaptiveSegment(
            mode="single_person",
            start_s=float(clip_start_s),
            end_s=first.end_s,
            primary_track_id=first.primary_track_id or fallback_track_id,
            secondary_track_id=None,
        )
        for idx in range(1, len(stabilized)):
            prev = stabilized[idx - 1]
            cur = stabilized[idx]
            if cur.start_s > prev.end_s:
                stabilized[idx] = AdaptiveSegment(
                    mode="single_person",
                    start_s=prev.end_s,
                    end_s=cur.end_s,
                    primary_track_id=cur.primary_track_id,
                    secondary_track_id=None,
                )
        last = stabilized[-1]
        stabilized[-1] = AdaptiveSegment(
            mode="single_person",
            start_s=last.start_s,
            end_s=float(clip_end_s),
            primary_track_id=last.primary_track_id or fallback_track_id,
            secondary_track_id=None,
        )

    return stabilized


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
            det = _track_anchor_at_frame(tid, frame_idx, fps, 0, 0, person_track_index, face_track_index)
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
    speaker_candidate_debug: list[dict] | None = None,
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
    prefer_wider_hybrid = interval_prefers_wider_hybrid_behavior(
        clip_start_s=float(clip_start_s),
        clip_end_s=float(clip_end_s),
        ordered_stats=ordered,
        speaker_candidate_debug=speaker_candidate_debug,
    )
    primary_dominance_threshold = 0.90 if prefer_wider_hybrid else SINGLE_PERSON_MIN_DOMINANCE
    secondary_share_threshold = 0.12 if prefer_wider_hybrid else TWO_SPEAKER_MIN_SECONDARY_SHARE
    top2_share_threshold = 0.70 if prefer_wider_hybrid else TWO_SPEAKER_MIN_COMBINED_SHARE

    if (
        primary_share >= primary_dominance_threshold
        or secondary_share < secondary_share_threshold
        or top2_share < top2_share_threshold
    ):
        return CompositionPlan(mode="single_person", primary_track_id=primary.track_id, secondary_track_id=None)

    visible_enough = (
        primary.visible_fraction >= (0.40 if prefer_wider_hybrid else TWO_SPEAKER_MIN_BOTH_VISIBLE_FRACTION)
        and secondary.visible_fraction >= (0.40 if prefer_wider_hybrid else TWO_SPEAKER_MIN_BOTH_VISIBLE_FRACTION)
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
    frame_detection_index: dict[int, list[dict]] | None = None,
    motion_profile: MotionProfile | None = None,
    overlap_follow_decisions: list[dict] | None = None,
    prefer_local_track_ids: bool = False,
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
        tid = resolve_follow_identity(
            bindings,
            abs_ms,
            overlap_follow_decisions=overlap_follow_decisions,
            prefer_local_track_ids=prefer_local_track_ids,
            frame_detections=(frame_detection_index or {}).get(frame_idx, []),
        )
        person_det = interpolate_detection(person_track_index.get(tid), frame_idx, fps) if tid else None
        det = _track_anchor_candidate(
            tid,
            frame_idx,
            fps,
            src_w,
            src_h,
            person_track_index,
            face_track_index,
            frame_detection_index,
        )
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
    frame_detection_index: dict[int, list[dict]] | None = None,
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
        det = _track_anchor_candidate(
            track_id,
            frame_idx,
            fps,
            src_w,
            src_h,
            person_track_index,
            face_track_index,
            frame_detection_index,
        )
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
    frame_detection_index: dict[int, list[dict]] | None = None,
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
        primary_det = _track_anchor_at_frame(
            primary_track_id,
            frame_idx,
            fps,
            src_w,
            src_h,
            person_track_index,
            face_track_index,
            frame_detection_index,
        )
        secondary_det = _track_anchor_at_frame(
            secondary_track_id,
            frame_idx,
            fps,
            src_w,
            src_h,
            person_track_index,
            face_track_index,
            frame_detection_index,
        )
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
    frame_detection_index: dict[int, list[dict]] | None = None,
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
        det = _track_anchor_candidate(
            track_id,
            frame_idx,
            fps,
            src_w,
            src_h,
            person_track_index,
            face_track_index,
            frame_detection_index,
        )

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
    frame_detection_index: dict[int, list[dict]] | None = None,
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
        frame_detection_index=frame_detection_index,
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
        frame_detection_index=frame_detection_index,
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


def build_projected_overlay_path(
    *,
    track_id: str,
    clip_start_s: float,
    clip_end_s: float,
    fps: float,
    src_w: int,
    src_h: int,
    x_keyframes: list[tuple[float, float]],
    y_keyframes: list[tuple[float, float]],
    detection_index: dict[str, list[Detection]],
    out_w: int = OUT_W,
    out_h: int = OUT_H,
    camera_zoom: float = CAMERA_ZOOM,
    keyframe_step_s: float = KEYFRAME_STEP_S,
    color: str = OVERLAY_BOX_COLOR,
    label: str | None = None,
    box_thickness: int = OVERLAY_BOX_THICKNESS,
    text_color: str = "white",
) -> OverlayPath | None:
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
    saw_anchor = False
    last_rel = (0.0, 0.0, 0.0, 0.0)

    t = 0.0
    while t <= (clip_end_s - clip_start_s + 1e-6):
        abs_ms = int(round((clip_start_s + t) * 1000.0))
        frame_idx = int(round((abs_ms / 1000.0) * fps))
        det = interpolate_detection(detection_index.get(track_id), frame_idx, fps)
        if det is not None:
            crop_x = interpolate_keyframes(x_keyframes, t)
            crop_y = interpolate_keyframes(y_keyframes, t)
            x1 = det.x_center - (det.width / 2.0)
            y1 = det.y_center - (det.height / 2.0)
            rel_x = ((x1 - crop_x) / max(1.0, crop_w)) * out_w
            rel_y = ((y1 - crop_y) / max(1.0, crop_h)) * out_h
            rel_w = (det.width / max(1.0, crop_w)) * out_w
            rel_h = (det.height / max(1.0, crop_h)) * out_h
            rel_w = clamp(rel_w, 12.0, out_w * 0.95)
            rel_h = clamp(rel_h, 12.0, out_h * 0.95)
            if rel_x + rel_w < 0.0 or rel_y + rel_h < 0.0 or rel_x > out_w or rel_y > out_h:
                t += keyframe_step_s
                continue
            rel_x = clamp(rel_x, 0.0, out_w - rel_w)
            rel_y = clamp(rel_y, 0.0, out_h - rel_h)
            last_rel = (rel_x, rel_y, rel_w, rel_h)
            saw_anchor = True

        if saw_anchor:
            box_x_keyframes.append((round(t, 3), float(last_rel[0])))
            box_y_keyframes.append((round(t, 3), float(last_rel[1])))
            box_w_keyframes.append((round(t, 3), float(last_rel[2])))
            box_h_keyframes.append((round(t, 3), float(last_rel[3])))
        t += keyframe_step_s

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
        label=label,
        box_thickness=box_thickness,
        text_color=text_color,
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
    extra_filters: list[str] | None = None,
) -> None:
    crop_h = int(round(src_h / camera_zoom))
    crop_w = int(round(crop_h * (OUT_W / OUT_H)))
    x_expr = build_interp_expr(x_keyframes)
    y_expr = build_interp_expr(y_keyframes)
    filter_chain_parts = [
        f"crop=w={crop_w}:h={crop_h}:x='{x_expr}':y='{y_expr}':exact=1",
        f"scale={OUT_W}:{OUT_H}:flags=lanczos",
    ]
    filter_chain_parts.extend(build_overlay_filters(overlay_paths))
    if extra_filters:
        filter_chain_parts.extend(extra_filters)
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
    extra_filters: list[str] | None = None,
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

    upper_draw_filters = build_overlay_filters(upper_overlay_paths)
    lower_draw_filters = build_overlay_filters(lower_overlay_paths)

    filter_complex = (
        f"[0:v]split=2[upper_src][lower_src];"
        f"[upper_src]crop=w={crop_w}:h={crop_h}:x='{x_expr_upper}':y='{y_expr_upper}':exact=1,"
        f"scale={OUT_W}:{panel_h}:flags=lanczos"
        f"{(',' + ','.join(upper_draw_filters)) if upper_draw_filters else ''}[upper];"
        f"[lower_src]crop=w={crop_w}:h={crop_h}:x='{x_expr_lower}':y='{y_expr_lower}':exact=1,"
        f"scale={OUT_W}:{panel_h}:flags=lanczos"
        f"{(',' + ','.join(lower_draw_filters)) if lower_draw_filters else ''}[lower];"
        f"[upper][lower]vstack=inputs=2[stacked]"
    )
    if extra_filters:
        filter_complex += f";[stacked]{','.join(extra_filters)}[vout]"
    else:
        filter_complex += ";[stacked]null[vout]"
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


def concat_render_segments(segment_paths: list[Path], out_path: Path) -> None:
    if len(segment_paths) == 1:
        segment_paths[0].replace(out_path)
        return

    with tempfile.TemporaryDirectory(prefix="clypt-render-concat-") as tmpdir:
        concat_path = Path(tmpdir) / "segments.txt"
        concat_path.write_text(
            "".join(f"file '{segment_path.as_posix()}'\n" for segment_path in segment_paths),
            encoding="utf-8",
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c",
            "copy",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg concat failed for {out_path.name}:\n{result.stderr[-1200:]}"
            )


def write_render_debug_sidecar(out_path: Path, payload: dict) -> None:
    sidecar_path = out_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _escape_drawtext_text(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace(",", r"\,")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace("%", r"\%")
    )


def _track_palette_color(track_id: str) -> str:
    palette = [
        "0x4FC3F7",
        "0xCE93D8",
        "0x81C784",
        "0xFF8A65",
        "0x90CAF9",
        "0xF06292",
        "0xAED581",
        "0xFFD54F",
    ]
    if not track_id:
        return DEBUG_DEFAULT_BOX_COLOR
    return palette[sum(ord(ch) for ch in str(track_id)) % len(palette)]


def build_overlay_filters(overlay_paths: list[OverlayPath]) -> list[str]:
    filters: list[str] = []
    for overlay in overlay_paths:
        if not overlay.box_x_keyframes:
            continue
        x_expr = build_interp_expr(overlay.box_x_keyframes)
        y_expr = build_interp_expr(overlay.box_y_keyframes)
        filters.append(
            f"drawbox=x='{x_expr}':"
            f"y='{y_expr}':"
            f"w='{build_interp_expr(overlay.box_w_keyframes)}':"
            f"h='{build_interp_expr(overlay.box_h_keyframes)}':"
            f"color={overlay.color}:t={int(overlay.box_thickness)}"
        )
        if overlay.label:
            filters.append(
                "drawtext="
                f"text='{_escape_drawtext_text(overlay.label)}':"
                "fontcolor="
                f"{overlay.text_color}:"
                f"fontsize={DEBUG_HUD_FONT_SIZE}:"
                "box=1:boxcolor=black@0.55:boxborderw=6:"
                f"x='{x_expr}':"
                f"y='max(0,{y_expr}-34)'"
            )
    return filters


def select_binding_sets(audio: dict) -> tuple[list[dict], list[dict], list[dict], str]:
    use_local = os.getenv("CLYPT_EXPERIMENT_LOCAL_CLIP_BINDINGS", "").strip().lower() in {"1", "true", "yes", "on"}
    if use_local:
        raw_bindings = list(audio.get("speaker_bindings_local", []))
        follow_bindings = list(audio.get("speaker_follow_bindings_local", []))
        if follow_bindings:
            return follow_bindings, raw_bindings, follow_bindings, "speaker_follow_bindings_local"
        if raw_bindings:
            return raw_bindings, raw_bindings, follow_bindings, "speaker_bindings_local"
    raw_bindings = list(audio.get("speaker_bindings", []))
    follow_bindings = list(audio.get("speaker_follow_bindings", []))
    if follow_bindings:
        return follow_bindings, raw_bindings, follow_bindings, "speaker_follow_bindings"
    return raw_bindings, raw_bindings, follow_bindings, "speaker_bindings"


def select_render_bindings(audio: dict) -> list[dict]:
    selected, _, _, _ = select_binding_sets(audio)
    return selected


def select_render_tracks(visual: dict) -> tuple[list[dict], str]:
    use_local = os.getenv("CLYPT_EXPERIMENT_LOCAL_CLIP_BINDINGS", "").strip().lower() in {"1", "true", "yes", "on"}
    if use_local:
        tracks_local = list(visual.get("tracks_local", []))
        if tracks_local:
            return tracks_local, "tracks_local"
    return list(visual.get("tracks", [])), "tracks"


def _bindings_for_interval(bindings: list[dict], clip_start_s: float, clip_end_s: float) -> list[dict]:
    selected: list[dict] = []
    for binding in bindings:
        start_s = float(binding.get("start_time_ms", 0)) / 1000.0
        end_s = float(binding.get("end_time_ms", 0)) / 1000.0
        if end_s <= clip_start_s or start_s >= clip_end_s:
            continue
        selected.append(dict(binding))
    return selected


def _speaker_candidate_debug_for_interval(
    speaker_candidate_debug: list[dict] | None,
    clip_start_s: float,
    clip_end_s: float,
) -> list[dict]:
    selected: list[dict] = []
    clip_start_ms = int(round(float(clip_start_s) * 1000.0))
    clip_end_ms = int(round(float(clip_end_s) * 1000.0))
    for entry in speaker_candidate_debug or []:
        start_ms = int(entry.get("start_time_ms", 0) or 0)
        end_ms = int(entry.get("end_time_ms", start_ms) or start_ms)
        if end_ms <= clip_start_ms or start_ms >= clip_end_ms:
            continue
        selected.append(dict(entry))
    return selected


def summarize_hybrid_debug_for_interval(
    *,
    clip_start_s: float,
    clip_end_s: float,
    speaker_candidate_debug: list[dict] | None,
) -> dict | None:
    interval_entries = _speaker_candidate_debug_for_interval(
        speaker_candidate_debug,
        clip_start_s,
        clip_end_s,
    )
    if not interval_entries:
        return None

    clip_start_ms = int(round(float(clip_start_s) * 1000.0))
    clip_end_ms = int(round(float(clip_end_s) * 1000.0))

    def _selection_key(entry: dict) -> tuple[int, int, int, int]:
        start_ms = int(entry.get("start_time_ms", 0) or 0)
        end_ms = int(entry.get("end_time_ms", start_ms) or start_ms)
        overlap_ms = max(0, min(clip_end_ms, end_ms) - max(clip_start_ms, start_ms))
        decision_source = str(entry.get("decision_source", "") or "")
        return (
            1 if bool(entry.get("ambiguous", False)) else 0,
            1 if decision_source and decision_source != "unknown" else 0,
            overlap_ms,
            -start_ms,
        )

    representative = max(interval_entries, key=_selection_key)
    candidate_track_ids: list[str] = []
    candidate_local_track_ids: list[str] = []
    for candidate in representative.get("candidates", []) or []:
        track_id = str(candidate.get("track_id", "") or "")
        local_track_id = str(candidate.get("local_track_id", "") or "")
        if track_id and track_id not in candidate_track_ids:
            candidate_track_ids.append(track_id)
        if local_track_id and local_track_id not in candidate_local_track_ids:
            candidate_local_track_ids.append(local_track_id)

    decision_sources_seen: list[str] = []
    for entry in interval_entries:
        source = str(entry.get("decision_source", "unknown") or "unknown")
        if source not in decision_sources_seen:
            decision_sources_seen.append(source)

    return {
        "active_audio_speaker_id": representative.get("active_audio_speaker_id"),
        "active_audio_local_track_id": representative.get("active_audio_local_track_id"),
        "chosen_track_id": representative.get("chosen_track_id"),
        "chosen_local_track_id": representative.get("chosen_local_track_id"),
        "decision_source": str(representative.get("decision_source", "unknown") or "unknown"),
        "ambiguous": any(bool(entry.get("ambiguous", False)) for entry in interval_entries),
        "top_1_top_2_margin": representative.get("top_1_top_2_margin"),
        "candidate_track_ids": candidate_track_ids,
        "candidate_local_track_ids": candidate_local_track_ids,
        "decision_sources_seen": decision_sources_seen,
        "entry_count": len(interval_entries),
    }


def interval_prefers_wider_hybrid_behavior(
    *,
    clip_start_s: float,
    clip_end_s: float,
    ordered_stats: list[TrackWindowStats],
    speaker_candidate_debug: list[dict] | None,
) -> bool:
    if len(ordered_stats) < 2:
        return False

    interval_entries = _speaker_candidate_debug_for_interval(
        speaker_candidate_debug,
        clip_start_s,
        clip_end_s,
    )
    if not interval_entries:
        return False

    top_two_ids = {ordered_stats[0].track_id, ordered_stats[1].track_id}
    for entry in interval_entries:
        if not bool(entry.get("ambiguous", False)):
            continue
        entry_track_ids: set[str] = set()
        for key in (
            "chosen_track_id",
            "chosen_local_track_id",
            "active_audio_local_track_id",
        ):
            value = str(entry.get(key, "") or "")
            if value:
                entry_track_ids.add(value)
        for candidate in entry.get("candidates", []) or []:
            for key in ("track_id", "local_track_id"):
                value = str(candidate.get(key, "") or "")
                if value:
                    entry_track_ids.add(value)
        if top_two_ids.issubset(entry_track_ids):
            return True
    return False


def visible_track_ids_for_interval(
    *,
    clip_start_s: float,
    clip_end_s: float,
    fps: float,
    track_index: dict[str, list[Detection]],
) -> list[str]:
    frame_lo = int(math.floor(float(clip_start_s) * fps))
    frame_hi = int(math.ceil(float(clip_end_s) * fps))
    visible: list[str] = []
    for track_id, dets in sorted(track_index.items()):
        if any(frame_lo <= int(det.frame_idx) <= frame_hi for det in dets):
            visible.append(str(track_id))
    return visible


def build_debug_timeline_filters(
    *,
    clip_start_s: float,
    clip_end_s: float,
    raw_bindings: list[dict],
    follow_bindings: list[dict],
) -> list[str]:
    width = max(120, OUT_W - DEBUG_TIMELINE_LEFT - DEBUG_TIMELINE_RIGHT_PAD)
    duration_s = max(0.001, float(clip_end_s) - float(clip_start_s))

    filters = [
        "drawtext=text='raw':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.45:boxborderw=4:"
        f"x=24:y={DEBUG_TIMELINE_RAW_Y - 2}",
        "drawtext=text='follow':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.45:boxborderw=4:"
        f"x=24:y={DEBUG_TIMELINE_FOLLOW_Y - 2}",
    ]

    for binding_set, row_y in (
        (raw_bindings, DEBUG_TIMELINE_RAW_Y),
        (follow_bindings, DEBUG_TIMELINE_FOLLOW_Y),
    ):
        for binding in binding_set:
            start_s = max(float(clip_start_s), float(binding.get("start_time_ms", 0)) / 1000.0)
            end_s = min(float(clip_end_s), float(binding.get("end_time_ms", 0)) / 1000.0)
            if end_s <= start_s:
                continue
            rel_start = (start_s - float(clip_start_s)) / duration_s
            rel_end = (end_s - float(clip_start_s)) / duration_s
            x = DEBUG_TIMELINE_LEFT + int(round(rel_start * width))
            w = max(6, int(round((rel_end - rel_start) * width)))
            color = _track_palette_color(str(binding.get("track_id", "")))
            filters.append(
                f"drawbox=x={x}:y={row_y}:w={w}:h={DEBUG_TIMELINE_HEIGHT}:color={color}:t=fill"
            )
    return filters


def build_debug_hud_filters(
    *,
    mode: str,
    binding_source: str,
    follow_track_ids: list[str],
    raw_track_ids: list[str],
    face_track_ids: list[str],
    hybrid_debug: dict | None = None,
    active_track_ids: list[str] | None = None,
    offscreen_audio_speaker_ids: list[str] | None = None,
    overlap_active: bool = False,
) -> list[str]:
    lines = [
        f"mode={mode}",
        f"binding={binding_source}",
        f"follow={','.join(follow_track_ids) if follow_track_ids else 'none'}",
        f"raw={','.join(raw_track_ids) if raw_track_ids else 'none'}",
        f"faces={','.join(face_track_ids) if face_track_ids else 'none'}",
    ]
    if overlap_active or active_track_ids or offscreen_audio_speaker_ids:
        lines.append(f"overlap={'yes' if overlap_active else 'no'}")
    if active_track_ids:
        lines.append(f"active={','.join(_normalized_id_list(active_track_ids))}")
    if offscreen_audio_speaker_ids:
        lines.append(f"offscreen_audio={','.join(_normalized_id_list(offscreen_audio_speaker_ids))}")
    if hybrid_debug:
        lines.extend(
            [
                f"audio_speaker={hybrid_debug.get('active_audio_speaker_id') or 'unknown'}",
                f"decision={hybrid_debug.get('decision_source') or 'unknown'}",
                f"ambiguous={'yes' if hybrid_debug.get('ambiguous') else 'no'}",
                f"audio_local={hybrid_debug.get('active_audio_local_track_id') or 'unknown'}",
                f"chosen_local={hybrid_debug.get('chosen_local_track_id') or 'unknown'}",
                (
                    "margin="
                    f"{float(hybrid_debug.get('top_1_top_2_margin')):.3f}"
                    if hybrid_debug.get("top_1_top_2_margin") not in (None, "")
                    else "margin=unknown"
                ),
            ]
        )
    filters: list[str] = []
    for idx, line in enumerate(lines):
        filters.append(
            "drawtext="
            f"text='{_escape_drawtext_text(line)}':"
            "fontcolor=white:"
            f"fontsize={DEBUG_HUD_FONT_SIZE}:"
            "box=1:boxcolor=black@0.55:boxborderw=6:"
            f"x=24:y={24 + idx * 38}"
        )
    return filters


def build_debug_overlay_paths(
    *,
    clip_start_s: float,
    clip_end_s: float,
    fps: float,
    src_w: int,
    src_h: int,
    x_keyframes: list[tuple[float, float]],
    y_keyframes: list[tuple[float, float]],
    person_track_index: dict[str, list[Detection]],
    face_track_index: dict[str, list[Detection]],
    follow_track_ids: list[str],
    raw_track_ids: list[str],
    active_track_ids: list[str] | None = None,
    out_w: int = OUT_W,
    out_h: int = OUT_H,
    camera_zoom: float = CAMERA_ZOOM,
    keyframe_step_s: float = KEYFRAME_STEP_S,
    include_faces: bool = False,
) -> list[OverlayPath]:
    overlays: list[OverlayPath] = []
    follow_set = {str(tid) for tid in follow_track_ids if str(tid)}
    raw_set = {str(tid) for tid in raw_track_ids if str(tid)}
    active_set = {str(tid) for tid in (active_track_ids or []) if str(tid)}

    visible_track_ids = visible_track_ids_for_interval(
        clip_start_s=clip_start_s,
        clip_end_s=clip_end_s,
        fps=fps,
        track_index=person_track_index,
    )
    for track_id in visible_track_ids:
        if track_id in follow_set:
            color = DEBUG_FOLLOW_BOX_COLOR
            thickness = 8
        elif track_id in active_set:
            color = OVERLAY_SECONDARY_BOX_COLOR
            thickness = 6
        elif track_id in raw_set:
            color = DEBUG_RAW_ACTIVE_BOX_COLOR
            thickness = 6
        else:
            color = _track_palette_color(track_id)
            thickness = 4
        overlay = build_projected_overlay_path(
            track_id=track_id,
            clip_start_s=clip_start_s,
            clip_end_s=clip_end_s,
            fps=fps,
            src_w=src_w,
            src_h=src_h,
            x_keyframes=x_keyframes,
            y_keyframes=y_keyframes,
            detection_index=person_track_index,
            out_w=out_w,
            out_h=out_h,
            camera_zoom=camera_zoom,
            keyframe_step_s=keyframe_step_s,
            color=color,
            label=track_id,
            box_thickness=thickness,
        )
        if overlay is not None:
            overlays.append(overlay)

    if include_faces:
        visible_face_ids = visible_track_ids_for_interval(
            clip_start_s=clip_start_s,
            clip_end_s=clip_end_s,
            fps=fps,
            track_index=face_track_index,
        )
        for track_id in visible_face_ids:
            overlay = build_projected_overlay_path(
                track_id=track_id,
                clip_start_s=clip_start_s,
                clip_end_s=clip_end_s,
                fps=fps,
                src_w=src_w,
                src_h=src_h,
                x_keyframes=x_keyframes,
                y_keyframes=y_keyframes,
                detection_index=face_track_index,
                out_w=out_w,
                out_h=out_h,
                camera_zoom=camera_zoom,
                keyframe_step_s=keyframe_step_s,
                color=DEBUG_FACE_BOX_COLOR,
                label=None,
                box_thickness=3,
            )
            if overlay is not None:
                overlays.append(overlay)
    return overlays


def build_debug_filters_for_segment(
    *,
    clip_start_s: float,
    clip_end_s: float,
    segment_mode: str,
    binding_source: str,
    raw_bindings: list[dict],
    follow_bindings: list[dict],
    follow_track_ids: list[str],
    face_track_ids: list[str],
    speaker_candidate_debug: list[dict] | None = None,
    active_track_ids: list[str] | None = None,
    offscreen_audio_speaker_ids: list[str] | None = None,
    overlap_active: bool = False,
) -> list[str]:
    raw_track_ids = [str(binding.get("track_id", "")) for binding in raw_bindings if str(binding.get("track_id", ""))]
    hybrid_debug = summarize_hybrid_debug_for_interval(
        clip_start_s=clip_start_s,
        clip_end_s=clip_end_s,
        speaker_candidate_debug=speaker_candidate_debug,
    )
    return build_debug_hud_filters(
        mode=segment_mode,
        binding_source=binding_source,
        follow_track_ids=follow_track_ids,
        raw_track_ids=raw_track_ids,
        face_track_ids=face_track_ids,
        hybrid_debug=hybrid_debug,
        active_track_ids=active_track_ids,
        offscreen_audio_speaker_ids=offscreen_audio_speaker_ids,
        overlap_active=overlap_active,
    ) + build_debug_timeline_filters(
        clip_start_s=clip_start_s,
        clip_end_s=clip_end_s,
        raw_bindings=raw_bindings,
        follow_bindings=follow_bindings,
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
    speaker_candidate_debug = list(audio.get("speaker_candidate_debug", []))
    active_speakers_local = list(audio.get("active_speakers_local", []))
    overlap_follow_decisions = list(audio.get("overlap_follow_decisions", []))
    tracks = list(visual.get("tracks", []))
    bindings, raw_bindings, follow_bindings, binding_source = select_binding_sets(audio)
    tracks, track_source = select_render_tracks(visual)
    if not tracks or not bindings:
        raise RuntimeError("Need non-empty tracks and speaker_bindings to render clips")

    meta = dict(visual.get("video_metadata", {}))
    fps = float(meta.get("fps", 23.976))
    src_w = int(meta.get("width", 1920))
    src_h = int(meta.get("height", 1080))
    duration_s = float(meta.get("duration_ms", 0)) / 1000.0
    if duration_s <= 0:
        raise RuntimeError("Missing video duration in phase_1_visual.json video_metadata")

    frame_detection_index = build_frame_detection_index(tracks)
    person_track_index = build_track_index(tracks)
    available_track_ids = {str(track_id) for track_id in person_track_index.keys() if str(track_id)}
    prefer_local_track_ids = binding_source.endswith("_local")
    face_track_index: dict[str, list[Detection]] = build_face_index(
        list(visual.get("face_detections", [])),
        src_w=src_w,
        src_h=src_h,
        fps=fps,
    )
    windows = choose_windows(bindings, duration_s)

    print(f"Selected {len(windows)} windows from current outputs:")
    print(
        "Camera target mode: "
        f"body-only ({len(person_track_index)} person tracks via {track_source})"
    )
    if RENDER_DEBUG_MODE:
        print(
            "Debug overlays: "
            f"on (binding_source={binding_source}, faces={'on' if RENDER_DEBUG_SHOW_FACES else 'off'})"
        )
    outputs = []
    for idx, (start_s, end_s, score) in enumerate(windows, start=1):
        clip_name = f"speaker_follow_clip{idx}_{start_s}s_{CLIP_DURATION_S}s.mp4"
        out_path = OUTPUT_DIR / clip_name
        clip_raw_bindings = _bindings_for_interval(raw_bindings, float(start_s), float(end_s))
        clip_follow_bindings = _bindings_for_interval(
            follow_bindings if follow_bindings else bindings,
            float(start_s),
            float(end_s),
        )
        composition = choose_window_composition(
            clip_start_s=start_s,
            clip_end_s=end_s,
            fps=fps,
            src_w=src_w,
            src_h=src_h,
            bindings=bindings,
            person_track_index=person_track_index,
            face_track_index=face_track_index,
            speaker_candidate_debug=speaker_candidate_debug,
        )
        if composition.mode == "two_split" and composition.primary_track_id and composition.secondary_track_id:
            split_profile = motion_profile_for_composition("two_split", out_h=SPLIT_PANEL_HEIGHT)
            adaptive_segments = choose_adaptive_split_segments(
                primary_track_id=composition.primary_track_id,
                secondary_track_id=composition.secondary_track_id,
                clip_start_s=start_s,
                clip_end_s=end_s,
                fps=fps,
                src_w=src_w,
                src_h=src_h,
                bindings=bindings,
                person_track_index=person_track_index,
                face_track_index=face_track_index,
                frame_detection_index=frame_detection_index,
            )
            print(
                f"  clip{idx}: {start_s}s-{end_s}s score={score:.2f} "
                f"mode={composition.mode} segments={len(adaptive_segments)} -> {clip_name}"
            )
            with tempfile.TemporaryDirectory(prefix="clypt-render-split-") as tmpdir:
                segment_paths: list[Path] = []
                for segment_idx, segment in enumerate(adaptive_segments, start=1):
                    segment_path = Path(tmpdir) / f"segment_{segment_idx:02d}.mp4"
                    segment_paths.append(segment_path)
                    segment_overlap_state = active_speaker_state_for_interval(
                        clip_start_s=float(segment.start_s),
                        clip_end_s=float(segment.end_s),
                        active_speakers_local=active_speakers_local,
                        prefer_local_track_ids=prefer_local_track_ids,
                        available_track_ids=available_track_ids,
                    )
                    segment_overlap_decisions = overlap_follow_decisions_for_interval(
                        overlap_follow_decisions,
                        float(segment.start_s),
                        float(segment.end_s),
                    )
                    if (
                        segment.mode == "two_split"
                        and segment.primary_track_id
                        and segment.secondary_track_id
                    ):
                        upper_x_keyframes, upper_y_keyframes = build_single_track_path(
                            track_id=segment.primary_track_id,
                            clip_start_s=segment.start_s,
                            clip_end_s=segment.end_s,
                            fps=fps,
                            src_w=src_w,
                            src_h=src_h,
                            person_track_index=person_track_index,
                            face_track_index=face_track_index,
                            frame_detection_index=frame_detection_index,
                            out_h=SPLIT_PANEL_HEIGHT,
                            motion_profile=split_profile,
                        )
                        lower_x_keyframes, lower_y_keyframes = build_single_track_path(
                            track_id=segment.secondary_track_id,
                            clip_start_s=segment.start_s,
                            clip_end_s=segment.end_s,
                            fps=fps,
                            src_w=src_w,
                            src_h=src_h,
                            person_track_index=person_track_index,
                            face_track_index=face_track_index,
                            frame_detection_index=frame_detection_index,
                            out_h=SPLIT_PANEL_HEIGHT,
                            motion_profile=split_profile,
                        )
                        upper_overlay = build_overlay_box_path(
                            track_id=segment.primary_track_id,
                            clip_start_s=segment.start_s,
                            clip_end_s=segment.end_s,
                            fps=fps,
                            src_w=src_w,
                            src_h=src_h,
                            person_track_index=person_track_index,
                            face_track_index=face_track_index,
                            frame_detection_index=frame_detection_index,
                            out_h=SPLIT_PANEL_HEIGHT,
                            camera_zoom=split_profile.camera_zoom,
                            keyframe_step_s=split_profile.keyframe_step_s,
                            color=OVERLAY_BOX_COLOR,
                        )
                        lower_overlay = build_overlay_box_path(
                            track_id=segment.secondary_track_id,
                            clip_start_s=segment.start_s,
                            clip_end_s=segment.end_s,
                            fps=fps,
                            src_w=src_w,
                            src_h=src_h,
                            person_track_index=person_track_index,
                            face_track_index=face_track_index,
                            frame_detection_index=frame_detection_index,
                            out_h=SPLIT_PANEL_HEIGHT,
                            camera_zoom=split_profile.camera_zoom,
                            keyframe_step_s=split_profile.keyframe_step_s,
                            color=OVERLAY_SECONDARY_BOX_COLOR,
                        )
                        upper_overlay_paths = [upper_overlay] if upper_overlay is not None else []
                        lower_overlay_paths = [lower_overlay] if lower_overlay is not None else []
                        extra_filters: list[str] = []
                        if RENDER_DEBUG_MODE:
                            segment_raw_bindings = _bindings_for_interval(
                                clip_raw_bindings,
                                float(segment.start_s),
                                float(segment.end_s),
                            )
                            segment_follow_track_ids = [segment.primary_track_id, segment.secondary_track_id]
                            visible_face_ids = (
                                visible_track_ids_for_interval(
                                    clip_start_s=float(segment.start_s),
                                    clip_end_s=float(segment.end_s),
                                    fps=fps,
                                    track_index=face_track_index,
                                )
                                if RENDER_DEBUG_SHOW_FACES
                                else []
                            )
                            extra_filters = build_debug_filters_for_segment(
                                clip_start_s=float(start_s),
                                clip_end_s=float(end_s),
                                segment_mode="two_split",
                                binding_source=binding_source,
                                raw_bindings=clip_raw_bindings,
                                follow_bindings=clip_follow_bindings,
                                follow_track_ids=[tid for tid in segment_follow_track_ids if tid],
                                face_track_ids=visible_face_ids,
                                speaker_candidate_debug=speaker_candidate_debug,
                                active_track_ids=segment_overlap_state["visible_track_ids"],
                                offscreen_audio_speaker_ids=segment_overlap_state["offscreen_audio_speaker_ids"],
                                overlap_active=bool(segment_overlap_state["overlap"]),
                            )
                            if upper_overlay is not None:
                                upper_overlay_paths = [OverlayPath(**{**upper_overlay.__dict__, "label": segment.primary_track_id})]
                            if lower_overlay is not None:
                                lower_overlay_paths = [OverlayPath(**{**lower_overlay.__dict__, "label": segment.secondary_track_id})]
                        render_split_clip(
                            video_path=VIDEO_PATH,
                            out_path=segment_path,
                            start_s=segment.start_s,
                            duration_s=(segment.end_s - segment.start_s),
                            upper_x_keyframes=upper_x_keyframes,
                            upper_y_keyframes=upper_y_keyframes,
                            upper_overlay_paths=upper_overlay_paths,
                            lower_x_keyframes=lower_x_keyframes,
                            lower_y_keyframes=lower_y_keyframes,
                            lower_overlay_paths=lower_overlay_paths,
                            src_w=src_w,
                            src_h=src_h,
                            camera_zoom=split_profile.camera_zoom,
                            extra_filters=extra_filters,
                        )
                    else:
                        single_profile = motion_profile_for_composition("single_person")
                        if segment_overlap_decisions:
                            x_keyframes, y_keyframes = build_camera_path(
                                clip_start_s=segment.start_s,
                                clip_end_s=segment.end_s,
                                fps=fps,
                                src_w=src_w,
                                src_h=src_h,
                                bindings=clip_follow_bindings if clip_follow_bindings else bindings,
                                person_track_index=person_track_index,
                                face_track_index=face_track_index,
                                frame_detection_index=frame_detection_index,
                                motion_profile=single_profile,
                                overlap_follow_decisions=segment_overlap_decisions,
                                prefer_local_track_ids=prefer_local_track_ids,
                            )
                        else:
                            x_keyframes, y_keyframes = build_single_track_path(
                                track_id=segment.primary_track_id,
                                clip_start_s=segment.start_s,
                                clip_end_s=segment.end_s,
                                fps=fps,
                                src_w=src_w,
                                src_h=src_h,
                                person_track_index=person_track_index,
                                face_track_index=face_track_index,
                                frame_detection_index=frame_detection_index,
                                out_h=OUT_H,
                                motion_profile=single_profile,
                            )
                        overlay_path = build_overlay_box_path(
                            track_id=segment.primary_track_id or composition.primary_track_id,
                            clip_start_s=segment.start_s,
                            clip_end_s=segment.end_s,
                            fps=fps,
                            src_w=src_w,
                            src_h=src_h,
                            person_track_index=person_track_index,
                            face_track_index=face_track_index,
                            frame_detection_index=frame_detection_index,
                            out_h=OUT_H,
                            camera_zoom=single_profile.camera_zoom,
                            keyframe_step_s=single_profile.keyframe_step_s,
                        )
                        overlay_paths = [overlay_path] if overlay_path is not None else []
                        extra_filters: list[str] = []
                        if RENDER_DEBUG_MODE:
                            segment_raw_bindings = _bindings_for_interval(
                                clip_raw_bindings,
                                float(segment.start_s),
                                float(segment.end_s),
                            )
                            overlay_paths = build_debug_overlay_paths(
                                clip_start_s=float(segment.start_s),
                                clip_end_s=float(segment.end_s),
                                fps=fps,
                                src_w=src_w,
                                src_h=src_h,
                                x_keyframes=x_keyframes,
                                y_keyframes=y_keyframes,
                                person_track_index=person_track_index,
                                face_track_index=face_track_index,
                                frame_detection_index=frame_detection_index,
                                follow_track_ids=[segment.primary_track_id] if segment.primary_track_id else [],
                                raw_track_ids=[str(b.get("track_id", "")) for b in segment_raw_bindings],
                                active_track_ids=segment_overlap_state["visible_track_ids"],
                                out_h=OUT_H,
                                camera_zoom=single_profile.camera_zoom,
                                keyframe_step_s=single_profile.keyframe_step_s,
                                include_faces=RENDER_DEBUG_SHOW_FACES,
                            )
                            visible_face_ids = (
                                visible_track_ids_for_interval(
                                    clip_start_s=float(segment.start_s),
                                    clip_end_s=float(segment.end_s),
                                    fps=fps,
                                    track_index=face_track_index,
                                )
                                if RENDER_DEBUG_SHOW_FACES
                                else []
                            )
                            extra_filters = build_debug_filters_for_segment(
                                clip_start_s=float(start_s),
                                clip_end_s=float(end_s),
                                segment_mode="single_person",
                                binding_source=binding_source,
                                raw_bindings=clip_raw_bindings,
                                follow_bindings=clip_follow_bindings,
                                follow_track_ids=[segment.primary_track_id] if segment.primary_track_id else [],
                                face_track_ids=visible_face_ids,
                                speaker_candidate_debug=speaker_candidate_debug,
                                active_track_ids=segment_overlap_state["visible_track_ids"],
                                offscreen_audio_speaker_ids=segment_overlap_state["offscreen_audio_speaker_ids"],
                                overlap_active=bool(segment_overlap_state["overlap"]),
                            )
                        render_clip(
                            video_path=VIDEO_PATH,
                            out_path=segment_path,
                            start_s=segment.start_s,
                            duration_s=(segment.end_s - segment.start_s),
                            x_keyframes=x_keyframes,
                            y_keyframes=y_keyframes,
                            overlay_paths=overlay_paths,
                            src_h=src_h,
                            camera_zoom=single_profile.camera_zoom,
                            extra_filters=extra_filters,
                        )
                concat_render_segments(segment_paths, out_path)
            write_render_debug_sidecar(
                out_path,
                build_render_debug_sidecar_payload(
                    clip_name=clip_name,
                    window={"start_s": start_s, "end_s": end_s, "score": score},
                    binding_source=binding_source,
                    debug_mode=RENDER_DEBUG_MODE,
                    debug_show_faces=RENDER_DEBUG_SHOW_FACES,
                    composition=composition,
                    segments=adaptive_segments,
                    fps=fps,
                    src_w=src_w,
                    src_h=src_h,
                    frame_detection_index=frame_detection_index,
                    speaker_candidate_debug=speaker_candidate_debug,
                ),
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
                frame_detection_index=frame_detection_index,
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
                        frame_detection_index=frame_detection_index,
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
                        frame_detection_index=frame_detection_index,
                        out_h=OUT_H,
                        camera_zoom=shared_profile.camera_zoom,
                        keyframe_step_s=shared_profile.keyframe_step_s,
                        color=OVERLAY_SECONDARY_BOX_COLOR,
                    ),
                )
                if overlay is not None
            ]
            extra_filters: list[str] = []
            if RENDER_DEBUG_MODE:
                clip_overlap_state = active_speaker_state_for_interval(
                    clip_start_s=float(start_s),
                    clip_end_s=float(end_s),
                    active_speakers_local=active_speakers_local,
                    prefer_local_track_ids=prefer_local_track_ids,
                    available_track_ids=available_track_ids,
                )
                overlay_paths = build_debug_overlay_paths(
                    clip_start_s=float(start_s),
                    clip_end_s=float(end_s),
                    fps=fps,
                    src_w=src_w,
                    src_h=src_h,
                    x_keyframes=x_keyframes,
                    y_keyframes=y_keyframes,
                    person_track_index=person_track_index,
                    face_track_index=face_track_index,
                    follow_track_ids=[composition.primary_track_id, composition.secondary_track_id],
                    raw_track_ids=[str(b.get("track_id", "")) for b in clip_raw_bindings],
                    active_track_ids=clip_overlap_state["visible_track_ids"],
                    camera_zoom=shared_profile.camera_zoom,
                    keyframe_step_s=shared_profile.keyframe_step_s,
                    include_faces=RENDER_DEBUG_SHOW_FACES,
                )
                visible_face_ids = (
                    visible_track_ids_for_interval(
                        clip_start_s=float(start_s),
                        clip_end_s=float(end_s),
                        fps=fps,
                        track_index=face_track_index,
                    )
                    if RENDER_DEBUG_SHOW_FACES
                    else []
                )
                extra_filters = build_debug_filters_for_segment(
                    clip_start_s=float(start_s),
                    clip_end_s=float(end_s),
                    segment_mode="two_shared",
                    binding_source=binding_source,
                    raw_bindings=clip_raw_bindings,
                    follow_bindings=clip_follow_bindings,
                    follow_track_ids=[composition.primary_track_id, composition.secondary_track_id],
                    face_track_ids=visible_face_ids,
                    speaker_candidate_debug=speaker_candidate_debug,
                    active_track_ids=clip_overlap_state["visible_track_ids"],
                    offscreen_audio_speaker_ids=clip_overlap_state["offscreen_audio_speaker_ids"],
                    overlap_active=bool(clip_overlap_state["overlap"]),
                )
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
                extra_filters=extra_filters,
            )
            write_render_debug_sidecar(
                out_path,
                build_render_debug_sidecar_payload(
                    clip_name=clip_name,
                    window={"start_s": start_s, "end_s": end_s, "score": score},
                    binding_source=binding_source,
                    debug_mode=RENDER_DEBUG_MODE,
                    debug_show_faces=RENDER_DEBUG_SHOW_FACES,
                    composition=composition,
                    segments=[
                        AdaptiveSegment(
                            mode="two_shared",
                            start_s=float(start_s),
                            end_s=float(end_s),
                            primary_track_id=composition.primary_track_id,
                            secondary_track_id=composition.secondary_track_id,
                        )
                    ],
                    fps=fps,
                    src_w=src_w,
                    src_h=src_h,
                    frame_detection_index=frame_detection_index,
                    speaker_candidate_debug=speaker_candidate_debug,
                ),
            )
        else:
            single_profile = motion_profile_for_composition("single_person")
            print(
                f"  clip{idx}: {start_s}s-{end_s}s score={score:.2f} "
                f"mode={composition.mode} -> {clip_name}"
            )
            single_segments = choose_active_speaker_segments(
                clip_start_s=float(start_s),
                clip_end_s=float(end_s),
                bindings=bindings,
                fallback_track_id=composition.primary_track_id,
            )
            with tempfile.TemporaryDirectory(prefix="clypt-render-single-") as tmpdir:
                segment_paths: list[Path] = []
                for segment_idx, segment in enumerate(single_segments, start=1):
                    segment_path = Path(tmpdir) / f"segment_{segment_idx:02d}.mp4"
                    segment_paths.append(segment_path)
                    segment_overlap_state = active_speaker_state_for_interval(
                        clip_start_s=float(segment.start_s),
                        clip_end_s=float(segment.end_s),
                        active_speakers_local=active_speakers_local,
                        prefer_local_track_ids=prefer_local_track_ids,
                        available_track_ids=available_track_ids,
                    )
                    segment_overlap_decisions = overlap_follow_decisions_for_interval(
                        overlap_follow_decisions,
                        float(segment.start_s),
                        float(segment.end_s),
                    )
                    if segment_overlap_decisions:
                        x_keyframes, y_keyframes = build_camera_path(
                            clip_start_s=segment.start_s,
                            clip_end_s=segment.end_s,
                            fps=fps,
                            src_w=src_w,
                            src_h=src_h,
                            bindings=clip_follow_bindings if clip_follow_bindings else bindings,
                            person_track_index=person_track_index,
                            face_track_index=face_track_index,
                            frame_detection_index=frame_detection_index,
                            motion_profile=single_profile,
                            overlap_follow_decisions=segment_overlap_decisions,
                            prefer_local_track_ids=prefer_local_track_ids,
                        )
                    else:
                        x_keyframes, y_keyframes = build_single_track_path(
                            track_id=segment.primary_track_id,
                            clip_start_s=segment.start_s,
                            clip_end_s=segment.end_s,
                            fps=fps,
                            src_w=src_w,
                            src_h=src_h,
                            person_track_index=person_track_index,
                            face_track_index=face_track_index,
                            frame_detection_index=frame_detection_index,
                            out_h=OUT_H,
                            motion_profile=single_profile,
                        )
                    overlay_path = build_overlay_box_path(
                        track_id=segment.primary_track_id,
                        clip_start_s=segment.start_s,
                        clip_end_s=segment.end_s,
                        fps=fps,
                        src_w=src_w,
                        src_h=src_h,
                        person_track_index=person_track_index,
                        face_track_index=face_track_index,
                        frame_detection_index=frame_detection_index,
                        out_h=OUT_H,
                        camera_zoom=single_profile.camera_zoom,
                        keyframe_step_s=single_profile.keyframe_step_s,
                    )
                    overlay_paths = [overlay_path] if overlay_path is not None else []
                    extra_filters: list[str] = []
                    if RENDER_DEBUG_MODE:
                        segment_raw_bindings = _bindings_for_interval(
                            clip_raw_bindings,
                            float(segment.start_s),
                            float(segment.end_s),
                        )
                        overlay_paths = build_debug_overlay_paths(
                            clip_start_s=float(segment.start_s),
                            clip_end_s=float(segment.end_s),
                            fps=fps,
                            src_w=src_w,
                            src_h=src_h,
                            x_keyframes=x_keyframes,
                            y_keyframes=y_keyframes,
                            person_track_index=person_track_index,
                            face_track_index=face_track_index,
                            frame_detection_index=frame_detection_index,
                            follow_track_ids=[segment.primary_track_id] if segment.primary_track_id else [],
                            raw_track_ids=[str(b.get("track_id", "")) for b in segment_raw_bindings],
                            active_track_ids=segment_overlap_state["visible_track_ids"],
                            camera_zoom=single_profile.camera_zoom,
                            keyframe_step_s=single_profile.keyframe_step_s,
                            include_faces=RENDER_DEBUG_SHOW_FACES,
                        )
                        visible_face_ids = (
                            visible_track_ids_for_interval(
                                clip_start_s=float(segment.start_s),
                                clip_end_s=float(segment.end_s),
                                fps=fps,
                                track_index=face_track_index,
                            )
                            if RENDER_DEBUG_SHOW_FACES
                            else []
                        )
                        extra_filters = build_debug_filters_for_segment(
                            clip_start_s=float(start_s),
                            clip_end_s=float(end_s),
                            segment_mode="single_person",
                            binding_source=binding_source,
                            raw_bindings=clip_raw_bindings,
                            follow_bindings=clip_follow_bindings,
                            follow_track_ids=[segment.primary_track_id] if segment.primary_track_id else [],
                            face_track_ids=visible_face_ids,
                            speaker_candidate_debug=speaker_candidate_debug,
                            active_track_ids=segment_overlap_state["visible_track_ids"],
                            offscreen_audio_speaker_ids=segment_overlap_state["offscreen_audio_speaker_ids"],
                            overlap_active=bool(segment_overlap_state["overlap"]),
                        )
                    render_clip(
                        video_path=VIDEO_PATH,
                        out_path=segment_path,
                        start_s=segment.start_s,
                        duration_s=(segment.end_s - segment.start_s),
                        x_keyframes=x_keyframes,
                        y_keyframes=y_keyframes,
                        overlay_paths=overlay_paths,
                        src_h=src_h,
                        camera_zoom=single_profile.camera_zoom,
                        extra_filters=extra_filters,
                    )
                concat_render_segments(segment_paths, out_path)
            write_render_debug_sidecar(
                out_path,
                build_render_debug_sidecar_payload(
                    clip_name=clip_name,
                    window={"start_s": start_s, "end_s": end_s, "score": score},
                    binding_source=binding_source,
                    debug_mode=RENDER_DEBUG_MODE,
                    debug_show_faces=RENDER_DEBUG_SHOW_FACES,
                    composition=composition,
                    segments=single_segments,
                    fps=fps,
                    src_w=src_w,
                    src_h=src_h,
                    frame_detection_index=frame_detection_index,
                    speaker_candidate_debug=speaker_candidate_debug,
                ),
            )
        outputs.append(out_path)

    print("\nRendered clips:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
