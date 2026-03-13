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


@dataclass
class Detection:
    frame_idx: int
    x_center: float
    y_center: float
    width: float
    height: float


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def build_interp_expr(keyframes: list[tuple[float, float]], var_name: str = "t") -> str:
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


def ema_smooth(values: list[float], smoothing: int) -> list[float]:
    if not values:
        return values
    alpha = 2.0 / (smoothing + 1.0)
    out = [values[0]]
    for i in range(1, len(values)):
        out.append((alpha * values[i]) + ((1.0 - alpha) * out[-1]))
    return out


def build_track_index(tracks: list[dict]) -> dict[str, list[Detection]]:
    by_track: dict[str, list[Detection]] = {}
    for t in tracks:
        tid = str(t.get("track_id", ""))
        if not tid:
            continue
        by_track.setdefault(tid, []).append(
            Detection(
                frame_idx=int(t["frame_idx"]),
                x_center=float(t["x_center"]),
                y_center=float(t["y_center"]),
                width=float(t["width"]),
                height=float(t["height"]),
            )
        )
    for tid in list(by_track.keys()):
        by_track[tid].sort(key=lambda d: d.frame_idx)
    return by_track


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
        if any(abs(start_s - s) < WINDOW_MIN_GAP_S for s, _, _ in picked):
            continue
        picked.append((start_s, end_s, score))
        if len(picked) >= NUM_CLIPS:
            break

    if not picked:
        return [(0, CLIP_DURATION_S, 0.0)]
    return picked


def build_camera_path(
    clip_start_s: int,
    clip_end_s: int,
    fps: float,
    src_w: int,
    src_h: int,
    bindings: list[dict],
    person_track_index: dict[str, list[Detection]],
    face_track_index: dict[str, list[Detection]],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    crop_h = int(round(src_h / CAMERA_ZOOM))
    crop_w = int(round(crop_h * (OUT_W / OUT_H)))
    crop_w = min(crop_w, src_w)
    crop_h = min(crop_h, src_h)
    half_w = crop_w / 2.0
    half_h = crop_h / 2.0

    key_times = []
    xs = []
    ys = []
    last_cx = src_w / 2.0
    last_cy = src_h / 2.0

    t = 0.0
    while t <= (clip_end_s - clip_start_s + 1e-6):
        abs_ms = int(round((clip_start_s + t) * 1000.0))
        frame_idx = int(round((abs_ms / 1000.0) * fps))
        tid = active_track_at(bindings, abs_ms)
        face_det = interpolate_detection(face_track_index.get(tid), frame_idx, fps) if tid else None
        person_det = interpolate_detection(person_track_index.get(tid), frame_idx, fps) if tid else None
        if face_det is not None:
            last_cx = face_det.x_center
            last_cy = face_det.y_center
        elif person_det is not None:
            last_cx = person_det.x_center
            last_cy = person_det.y_center - (Y_HEAD_BIAS * person_det.height)
        key_times.append(round(t, 3))
        xs.append(clamp(last_cx, half_w, src_w - half_w) - half_w)
        ys.append(clamp(last_cy, half_h, src_h - half_h) - half_h)
        t += KEYFRAME_STEP_S

    xs = ema_smooth(xs, EMA_SMOOTHING)
    ys = ema_smooth(ys, EMA_SMOOTHING)
    return list(zip(key_times, xs)), list(zip(key_times, ys))


def render_clip(
    video_path: Path,
    out_path: Path,
    start_s: int,
    duration_s: int,
    x_keyframes: list[tuple[float, float]],
    y_keyframes: list[tuple[float, float]],
    src_h: int,
) -> None:
    crop_h = int(round(src_h / CAMERA_ZOOM))
    crop_w = int(round(crop_h * (OUT_W / OUT_H)))
    x_expr = build_interp_expr(x_keyframes)
    y_expr = build_interp_expr(y_keyframes)
    crop_filter = (
        f"crop=w={crop_w}:h={crop_h}:x='{x_expr}':y='{y_expr}':exact=1,"
        f"scale={OUT_W}:{OUT_H}:flags=lanczos"
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
        "-vf",
        crop_filter,
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
        x_keyframes, y_keyframes = build_camera_path(
            clip_start_s=start_s,
            clip_end_s=end_s,
            fps=fps,
            src_w=src_w,
            src_h=src_h,
            bindings=bindings,
            person_track_index=person_track_index,
            face_track_index=face_track_index,
        )
        print(f"  clip{idx}: {start_s}s-{end_s}s score={score:.2f} -> {clip_name}")
        render_clip(
            video_path=VIDEO_PATH,
            out_path=out_path,
            start_s=start_s,
            duration_s=(end_s - start_s),
            x_keyframes=x_keyframes,
            y_keyframes=y_keyframes,
            src_h=src_h,
        )
        outputs.append(out_path)

    print("\nRendered clips:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
