"""Render 3 vertical (9:16) clips with OpusClip-style speaker tracking.

Motion model:
- Lock onto the active speaker's median position per segment (no jitter).
- On speaker change, ease-in-out pan over ~0.6s to the new speaker.
- Within a segment, gently drift if the speaker moves significantly (deadzone).
"""

import json
import os
import subprocess
import numpy as np
from collections import defaultdict

VIDEO_PATH = "downloads/video.mp4"
VISUAL_PATH = "outputs/phase_1_visual.json"
AUDIO_PATH = "outputs/phase_1_audio.json"
OUTPUT_DIR = "outputs/clips"

SRC_W, SRC_H = 1920, 1080
FPS = 23.976
OUT_W, OUT_H = 1080, 1920  # 9:16

CROP_H = 960
CROP_W = int(CROP_H * (9 / 16))  # 540

# Motion tuning
PAN_DURATION_S = 0.6          # Duration of speaker-change pan (ease in/out)
DEADZONE_PX = 80              # Ignore drift smaller than this
DRIFT_SMOOTHING = 30          # High = very smooth drift within a segment
KF_INTERVAL = 1.0 / FPS       # Per-frame keyframes for butter-smooth output

CLIPS = [
    {"name": "clip_1", "start_s": 15, "dur_s": 20},
    {"name": "clip_2", "start_s": 120, "dur_s": 20},
    {"name": "clip_3", "start_s": 360, "dur_s": 20},
]


def load_data():
    with open(VISUAL_PATH) as f:
        visual = json.load(f)
    with open(AUDIO_PATH) as f:
        audio = json.load(f)
    return visual["tracks"], audio["words"], audio["speaker_bindings"]


def build_track_index(tracks):
    idx = {}
    for t in tracks:
        key = (t["track_id"], int(t["frame_idx"]))
        idx[key] = (float(t["x_center"]), float(t["y_center"]))
    return idx


def get_speaker_segments(words, start_frame, end_frame):
    """Build contiguous speaker segments: [(start_fi, end_fi, person_id), ...]."""
    frame_speaker = {}
    for w in words:
        tid = w.get("speaker_track_id")
        if not tid:
            continue
        sf = int(round((int(w["start_time_ms"]) / 1000.0) * FPS))
        ef = int(round((int(w["end_time_ms"]) / 1000.0) * FPS))
        for fi in range(max(sf, start_frame), min(ef, end_frame) + 1):
            frame_speaker[fi] = tid

    # Merge into contiguous segments
    segments = []
    cur_speaker = None
    cur_start = start_frame
    for fi in range(start_frame, end_frame + 1):
        sp = frame_speaker.get(fi)
        if sp != cur_speaker:
            if cur_speaker is not None:
                segments.append((cur_start, fi - 1, cur_speaker))
            cur_speaker = sp
            cur_start = fi
    if cur_speaker is not None:
        segments.append((cur_start, end_frame, cur_speaker))

    return segments


def get_segment_anchor(track_idx, person_id, start_fi, end_fi):
    """Get the median position of a person within a segment — stable anchor point."""
    xs, ys = [], []
    for fi in range(start_fi, end_fi + 1):
        pos = track_idx.get((person_id, fi))
        if pos:
            xs.append(pos[0])
            ys.append(pos[1])
    if not xs:
        return SRC_W / 2, SRC_H / 2
    return float(np.median(xs)), float(np.median(ys))


def ease_in_out(t):
    """Smooth cubic ease-in-out: 0→1 over t in [0,1]."""
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    if t < 0.5:
        return 4 * t * t * t
    return 1 - (-2 * t + 2) ** 3 / 2


def clamp_crop(cx, cy):
    half_w = CROP_W / 2
    half_h = CROP_H / 2
    x = int(round(max(half_w, min(SRC_W - half_w, cx)) - half_w))
    y = int(round(max(half_h, min(SRC_H - half_h, cy)) - half_h))
    return x, y


def compute_crop_path(track_idx, segments, start_frame, end_frame):
    """Compute per-frame crop center with hold-and-pan motion model."""
    total_frames = end_frame - start_frame + 1
    pan_frames = int(round(PAN_DURATION_S * FPS))

    # Pre-compute anchor for each segment
    anchors = []
    for sf, ef, pid in segments:
        ax, ay = get_segment_anchor(track_idx, pid, sf, ef)
        anchors.append((sf, ef, pid, ax, ay))

    # Build segment lookup: for each frame, which segment index?
    frame_seg = {}
    for i, (sf, ef, pid, ax, ay) in enumerate(anchors):
        for fi in range(sf, ef + 1):
            frame_seg[fi] = i

    # Compute per-frame target with gentle drift within segments
    target_cx = np.full(total_frames, SRC_W / 2)
    target_cy = np.full(total_frames, SRC_H / 2)

    for i, (sf, ef, pid, ax, ay) in enumerate(anchors):
        for fi in range(max(sf, start_frame), min(ef, end_frame) + 1):
            idx = fi - start_frame
            # Start from anchor; allow gentle drift if person moves a lot
            pos = track_idx.get((pid, fi))
            if pos:
                dx = pos[0] - ax
                dy = pos[1] - ay
                # Only drift if outside deadzone
                if abs(dx) > DEADZONE_PX:
                    target_cx[idx] = ax + (dx - np.sign(dx) * DEADZONE_PX) * 0.3
                else:
                    target_cx[idx] = ax
                if abs(dy) > DEADZONE_PX:
                    target_cy[idx] = ay + (dy - np.sign(dy) * DEADZONE_PX) * 0.3
                else:
                    target_cy[idx] = ay
            else:
                target_cx[idx] = ax
                target_cy[idx] = ay

    # Heavy smoothing on the per-frame targets to remove jitter within segments
    alpha = 2.0 / (DRIFT_SMOOTHING + 1)
    for i in range(1, total_frames):
        target_cx[i] = alpha * target_cx[i] + (1 - alpha) * target_cx[i - 1]
        target_cy[i] = alpha * target_cy[i] + (1 - alpha) * target_cy[i - 1]

    # Now apply ease-in-out pans at segment transitions
    crop_cx = target_cx.copy()
    crop_cy = target_cy.copy()

    for seg_i in range(1, len(anchors)):
        prev_sf, prev_ef, _, prev_ax, prev_ay = anchors[seg_i - 1]
        cur_sf, cur_ef, _, cur_ax, cur_ay = anchors[seg_i]

        # Pan region: centered on the transition point
        transition_fi = cur_sf
        pan_start = max(start_frame, transition_fi - pan_frames // 2)
        pan_end = min(end_frame, transition_fi + pan_frames // 2)

        # Get positions just before and after transition
        pre_x = crop_cx[pan_start - start_frame]
        pre_y = crop_cy[pan_start - start_frame]
        post_x = target_cx[min(pan_end - start_frame, total_frames - 1)]
        post_y = target_cy[min(pan_end - start_frame, total_frames - 1)]

        for fi in range(pan_start, pan_end + 1):
            idx = fi - start_frame
            t = (fi - pan_start) / max(1, pan_end - pan_start)
            e = ease_in_out(t)
            crop_cx[idx] = pre_x + (post_x - pre_x) * e
            crop_cy[idx] = pre_y + (post_y - pre_y) * e

    return crop_cx, crop_cy


def render_clip(track_idx, words, clip_cfg):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    name = clip_cfg["name"]
    start_s = clip_cfg["start_s"]
    dur_s = clip_cfg["dur_s"]
    start_frame = int(start_s * FPS)
    end_frame = int((start_s + dur_s) * FPS)
    total_frames = end_frame - start_frame + 1

    segments = get_speaker_segments(words, start_frame, end_frame)
    speakers = set(pid for _, _, pid in segments)
    print(f"  {name}: {start_s}-{start_s+dur_s}s, speakers={speakers}, "
          f"{len(segments)} segments")

    crop_cx, crop_cy = compute_crop_path(track_idx, segments, start_frame, end_frame)

    # Clamp and build keyframes (every frame for maximum smoothness)
    kf_x = []
    kf_y = []
    for i in range(total_frames):
        t = i / FPS
        x, y = clamp_crop(crop_cx[i], crop_cy[i])
        kf_x.append((t, float(x)))
        kf_y.append((t, float(y)))

    # Subsample keyframes to keep ffmpeg expression manageable
    # Every 6 frames (~4 kf/sec) — the ease curves are already baked in
    step = 6
    kf_x_sub = kf_x[::step]
    kf_y_sub = kf_y[::step]
    # Always include the last frame
    if kf_x_sub[-1] != kf_x[-1]:
        kf_x_sub.append(kf_x[-1])
        kf_y_sub.append(kf_y[-1])

    x_expr = _build_interp_expr(kf_x_sub)
    y_expr = _build_interp_expr(kf_y_sub)

    out_path = os.path.join(OUTPUT_DIR, f"{name}.mp4")
    crop_filter = (
        f"crop=w={CROP_W}:h={CROP_H}:x='{x_expr}':y='{y_expr}':exact=1,"
        f"scale={OUT_W}:{OUT_H}:flags=lanczos"
    )

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-t", str(dur_s),
        "-i", VIDEO_PATH,
        "-vf", crop_filter,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        out_path,
    ]

    print(f"  Rendering {out_path}...")
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr[-400:]}")
        raise RuntimeError(f"ffmpeg failed for {name}")

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Done: {out_path} ({size_mb:.1f} MB)\n")
    return out_path


def _build_interp_expr(keyframes):
    """Build ffmpeg expression that linearly interpolates between keyframes."""
    if len(keyframes) <= 1:
        return str(keyframes[0][1])

    parts = []
    for i in range(len(keyframes) - 1):
        t0, v0 = keyframes[i]
        t1, v1 = keyframes[i + 1]
        dt = max(0.001, t1 - t0)
        slope = (v1 - v0) / dt
        parts.append((t0, t1, v0, slope))

    last = parts[-1]
    expr = f"{last[2]:.1f}+{last[3]:.4f}*(t-{last[0]:.4f})"

    for i in range(len(parts) - 2, -1, -1):
        seg = parts[i]
        seg_expr = f"{seg[2]:.1f}+{seg[3]:.4f}*(t-{seg[0]:.4f})"
        expr = f"if(lt(t,{seg[1]:.4f}),{seg_expr},{expr})"

    return expr


def main():
    print("Loading tracking + audio data...")
    tracks, words, bindings = load_data()
    track_idx = build_track_index(tracks)
    print(f"  {len(tracks)} detections, {len(words)} words, {len(bindings)} segments\n")

    outputs = []
    for clip in CLIPS:
        out = render_clip(track_idx, words, clip)
        outputs.append(out)

    print(f"Rendered {len(outputs)} clips:")
    for p in outputs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
