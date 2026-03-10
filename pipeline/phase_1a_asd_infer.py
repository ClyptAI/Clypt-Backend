#!/usr/bin/env python3
"""
Phase 1A-ASD-INFER: True ASD Model Runner (TalkNet Backend)
============================================================
Runs TalkNet ASD inference and converts its outputs into the pipeline's
raw ASD format consumed by phase_1a_asd.py:

  outputs/phase_1a_loconet_raw.json

This script is optional. If it fails or is not configured, phase_1a_asd.py
can still build a bootstrap timeline from STT+face tracks.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_VIDEO_PATH = ROOT / "downloads" / "video.mp4"
OUTPUTS_DIR = ROOT / "outputs"
RAW_OUTPUT_PATH = OUTPUTS_DIR / "phase_1a_loconet_raw.json"

# TalkNet defaults (override via env or CLI flags)
DEFAULT_TALKNET_REPO = ROOT / "third_party" / "TalkNet-ASD"
DEFAULT_TALKNET_PYTHON = "python3"
DEFAULT_TALKNET_VIDEO_NAME = "clypt_input"
DEFAULT_TALKNET_VIDEO_FOLDER = OUTPUTS_DIR / "talknet_demo"
DEFAULT_TALKNET_THREADS = 8
DEFAULT_TALKNET_DEVICE = "auto"
TALKNET_FPS = 25.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase_1a_asd_infer")


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=True,
    )


def _ffprobe_dims(video_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(video_path),
    ]
    result = _run(cmd)
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe returned no video streams for: {video_path}")
    w = int(streams[0]["width"])
    h = int(streams[0]["height"])
    if w <= 0 or h <= 0:
        raise RuntimeError(f"Invalid video dimensions from ffprobe: {w}x{h}")
    return w, h


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _convert_talknet_pickles_to_raw(
    tracks_path: Path,
    scores_path: Path,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    tracks = _load_pickle(tracks_path)
    scores = _load_pickle(scores_path)

    if not isinstance(tracks, list) or not isinstance(scores, list):
        raise RuntimeError("Unexpected TalkNet outputs: tracks/scores are not lists")

    by_time: dict[int, list[dict[str, Any]]] = {}
    total = 0

    for track_idx, track in enumerate(tracks):
        if track_idx >= len(scores):
            continue

        score_seq = list(scores[track_idx])
        track_data = track.get("track", {})
        proc = track.get("proc_track", {})
        frames = list(track_data.get("frame", []))
        xs = list(proc.get("x", []))
        ys = list(proc.get("y", []))
        ss = list(proc.get("s", []))

        n = min(len(frames), len(xs), len(ys), len(ss))
        if n == 0:
            continue

        for i in range(n):
            frame_idx = int(frames[i])
            x = float(xs[i])
            y = float(ys[i])
            s = float(ss[i])
            if s <= 0:
                continue

            left = _clamp((x - s) / frame_width, 0.0, 1.0)
            top = _clamp((y - s) / frame_height, 0.0, 1.0)
            right = _clamp((x + s) / frame_width, 0.0, 1.0)
            bottom = _clamp((y + s) / frame_height, 0.0, 1.0)
            if right <= left or bottom <= top:
                continue

            if i < len(score_seq):
                raw_score = float(score_seq[i])
            else:
                raw_score = float(score_seq[-1]) if score_seq else 0.0
            speaking_score = _sigmoid(raw_score)

            time_ms = int(round((frame_idx / TALKNET_FPS) * 1000))
            by_time.setdefault(time_ms, []).append(
                {
                    "bbox": {
                        "left": round(left, 6),
                        "top": round(top, 6),
                        "right": round(right, 6),
                        "bottom": round(bottom, 6),
                    },
                    "speaking_score": round(speaking_score, 6),
                    "track_id": track_idx,
                    "raw_score": round(raw_score, 6),
                }
            )
            total += 1

    frames = [
        {"time_ms": t, "detections": dets}
        for t, dets in sorted(by_time.items())
        if dets
    ]

    return {
        "source": "talknet_demo",
        "fps": TALKNET_FPS,
        "total_detections": total,
        "frames": frames,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TalkNet ASD and export pipeline raw JSON")
    p.add_argument(
        "--video-path",
        type=Path,
        default=Path(os.getenv("ASD_VIDEO_PATH", str(DOWNLOAD_VIDEO_PATH))),
        help="Input video file (default: downloads/video.mp4)",
    )
    p.add_argument(
        "--repo-path",
        type=Path,
        default=Path(os.getenv("TALKNET_REPO_PATH", str(DEFAULT_TALKNET_REPO))),
        help="Path to TalkNet-ASD repo",
    )
    p.add_argument(
        "--python-bin",
        default=os.getenv("TALKNET_PYTHON_BIN", DEFAULT_TALKNET_PYTHON),
        help="Python executable for TalkNet env",
    )
    p.add_argument(
        "--video-name",
        default=os.getenv("TALKNET_VIDEO_NAME", DEFAULT_TALKNET_VIDEO_NAME),
        help="TalkNet demo video stem (without extension)",
    )
    p.add_argument(
        "--video-folder",
        type=Path,
        default=Path(os.getenv("TALKNET_VIDEO_FOLDER", str(DEFAULT_TALKNET_VIDEO_FOLDER))),
        help="Folder containing <video-name>.mp4 and TalkNet demo outputs",
    )
    p.add_argument(
        "--pretrain-model",
        default=os.getenv("TALKNET_PRETRAIN_MODEL", ""),
        help="Optional path passed to --pretrainModel",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=int(os.getenv("TALKNET_THREADS", str(DEFAULT_TALKNET_THREADS))),
        help="TalkNet --nDataLoaderThread",
    )
    p.add_argument(
        "--device",
        default=os.getenv("TALKNET_DEVICE", DEFAULT_TALKNET_DEVICE),
        help="TalkNet device: auto/cpu/cuda",
    )
    p.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip TalkNet execution if pywork outputs already exist",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    log.info("=" * 60)
    log.info("PHASE 1A-ASD-INFER — TalkNet True Model Runner")
    log.info("=" * 60)

    if not args.video_path.exists():
        raise FileNotFoundError(f"Input video not found: {args.video_path}")
    if not args.repo_path.exists():
        raise FileNotFoundError(
            "TalkNet repo not found. Set TALKNET_REPO_PATH or pass --repo-path. "
            f"Expected path: {args.repo_path}"
        )
    demo_script = args.repo_path / "demoTalkNet.py"
    if not demo_script.exists():
        raise FileNotFoundError(f"TalkNet demo script missing: {demo_script}")

    # Stage video into TalkNet's expected <videoFolder>/<videoName>.<ext>.
    args.video_folder.mkdir(parents=True, exist_ok=True)
    staged_video = args.video_folder / f"{args.video_name}.mp4"
    _safe_link_or_copy(args.video_path, staged_video)
    log.info(f"Staged video → {staged_video}")

    run_root = args.video_folder / args.video_name
    pywork = run_root / "pywork"
    tracks_path = pywork / "tracks.pckl"
    scores_path = pywork / "scores.pckl"

    need_run = True
    if args.reuse_existing and tracks_path.exists() and scores_path.exists():
        need_run = False

    if need_run:
        cmd = [
            args.python_bin,
            str(demo_script),
            "--videoFolder",
            str(args.video_folder),
            "--videoName",
            args.video_name,
            "--nDataLoaderThread",
            str(args.threads),
            "--device",
            str(args.device),
        ]
        if args.pretrain_model:
            cmd.extend(["--pretrainModel", args.pretrain_model])

        log.info("Running TalkNet demo inference...")
        log.info(f"  repo: {args.repo_path}")
        log.info(f"  cmd: {' '.join(cmd)}")
        try:
            result = _run(cmd, cwd=args.repo_path)
            if result.stderr:
                # demoTalkNet prints progress logs to stderr.
                tail = "\n".join(result.stderr.splitlines()[-12:])
                if tail.strip():
                    log.info("TalkNet output tail:\n" + tail)
        except subprocess.CalledProcessError as e:
            stderr_tail = "\n".join((e.stderr or "").splitlines()[-20:])
            hint = ""
            if "cuda" in (e.stderr or "").lower():
                hint = (
                    "\nHint: TalkNet demo defaults to CUDA. On non-NVIDIA machines, "
                    "run TalkNet in a GPU environment or use a CPU-patched fork."
                )
            raise RuntimeError(
                "TalkNet inference failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Exit code: {e.returncode}\n"
                f"stderr tail:\n{stderr_tail}{hint}"
            ) from e

    if not tracks_path.exists() or not scores_path.exists():
        raise FileNotFoundError(
            "TalkNet finished but expected outputs are missing:\n"
            f"  {tracks_path}\n"
            f"  {scores_path}"
        )

    w, h = _ffprobe_dims(staged_video)
    log.info(f"Video dimensions: {w}x{h}")

    raw = _convert_talknet_pickles_to_raw(
        tracks_path=tracks_path,
        scores_path=scores_path,
        frame_width=w,
        frame_height=h,
    )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT_PATH.write_text(json.dumps(raw, indent=2))
    log.info(f"Saved ASD raw JSON → {RAW_OUTPUT_PATH}")
    log.info(f"Frames: {len(raw.get('frames', []))} | detections: {raw.get('total_detections', 0)}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
