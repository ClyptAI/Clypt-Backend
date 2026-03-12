#!/usr/bin/env python3
"""
Phase 1: Modal Deterministic Extraction
===================================
Downloads video + audio locally via yt-dlp, converts audio to 16kHz mono WAV,
then sends both to the Modal GPU worker for extraction (ASR, tracking, speaker
binding). Writes the returned ledgers to outputs/.

Outputs:
  - downloads/video.mp4           (muxed video for downstream use + Remotion)
  - outputs/phase_1_visual.json  (tracking data from Modal worker)
  - outputs/phase_1_audio.json   (word-level transcript from Modal worker)
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path

import modal
import yt_dlp

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
YTDLP_VIDEO_FORMAT = os.getenv(
    "YTDLP_VIDEO_FORMAT",
    (
        "bestvideo[vcodec^=avc1][height<=1080]+bestaudio/"
        "bestvideo[height<=1080]+bestaudio/"
        "best[height<=1080]/best"
    ),
)
YTDLP_MIN_LONG_EDGE = int(os.getenv("YTDLP_MIN_LONG_EDGE", "1080"))
ALLOW_LOW_RES_VIDEO = os.getenv("ALLOW_LOW_RES_VIDEO", "0") == "1"

ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = ROOT / "downloads"
OUTPUT_DIR = ROOT / "outputs"

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase_1")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def probe_video_stream(video_path: str) -> tuple[int, int, str]:
    """Return (width, height, fps_str) for a local video, or (0,0,'?')."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,avg_frame_rate",
                "-of", "json",
                video_path,
            ],
            check=True, capture_output=True, text=True,
        )
        data = json.loads(result.stdout or "{}")
        streams = data.get("streams") or []
        if not streams:
            return 0, 0, "?"
        s = streams[0]
        return (
            int(s.get("width", 0) or 0),
            int(s.get("height", 0) or 0),
            str(s.get("avg_frame_rate", "?")),
        )
    except Exception:
        return 0, 0, "?"


def probe_video_codec(video_path: str) -> str:
    """Return codec_name for v:0 stream (e.g. h264, av1)."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "csv=p=0",
                video_path,
            ],
            check=True, capture_output=True, text=True,
        )
        return (result.stdout or "").strip().lower()
    except Exception:
        return "unknown"


def ensure_h264_local(video_path: str) -> str:
    """Prefer local H.264 so Modal can skip AV1/VP9 re-encode latency."""
    codec = probe_video_codec(video_path)
    log.info(f"Video codec detected: {codec}")

    if codec in {"h264", "avc1"}:
        return video_path
    if codec not in {"av1", "vp9", "hevc", "h265"}:
        return video_path

    h264_path = str(DOWNLOAD_DIR / "video_h264.mp4")
    log.info(f"Re-encoding local video {codec} -> h264 for faster Modal processing…")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-c:a", "copy",
            h264_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log.info(
        f"Local H.264 ready: {h264_path} "
        f"({Path(h264_path).stat().st_size / 1e6:.1f} MB)"
    )
    return h264_path


# ──────────────────────────────────────────────
# Step 1: Download media locally
# ──────────────────────────────────────────────
def download_media(url: str) -> tuple[str, str]:
    """Download video + audio via yt-dlp. Returns (video_path, audio_path)."""
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Clean previous run
    for stale in ("video.mp4", "video_original.mp4", "audio.m4a", "audio.webm",
                   "audio.opus", "audio_16k.wav"):
        p = DOWNLOAD_DIR / stale
        if p.exists():
            p.unlink()
            log.info(f"Removed stale downloads/{stale}")
    for stale in (
        # Current phase naming
        "phase_1_visual.json", "phase_1_audio.json",
        "phase_2a_nodes.json", "phase_2b_narrative_edges.json",
        "phase_3_embeddings.json", "remotion_payloads_array.json",
        "remotion_payload.json",
        # Legacy phase naming (cleanup compatibility)
        "phase_1a_visual.json", "phase_1a_audio.json",
        "phase_1b_nodes.json", "phase_1c_narrative_edges.json",
        "phase_2_embeddings.json",
    ):
        p = OUTPUT_DIR / stale
        if p.exists():
            p.unlink()
            log.info(f"Removed stale outputs/{stale}")

    # ── Video + Audio (muxed) ──
    log.info("Downloading video+audio stream…")
    video_opts = {
        "format": YTDLP_VIDEO_FORMAT,
        "outtmpl": str(DOWNLOAD_DIR / "video.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": False,
    }
    with yt_dlp.YoutubeDL(video_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info)

    video_path = ensure_h264_local(video_path)

    w, h, fps = probe_video_stream(video_path)
    long_edge = max(w, h)
    log.info(
        f"Video saved: {video_path} ({Path(video_path).stat().st_size / 1e6:.1f} MB, "
        f"{w}x{h}, fps={fps})"
    )
    if not ALLOW_LOW_RES_VIDEO and long_edge < YTDLP_MIN_LONG_EDGE:
        raise RuntimeError(
            f"Video resolution too low for 9:16 reframing: {w}x{h}. "
            f"Set ALLOW_LOW_RES_VIDEO=1 to override."
        )

    # ── Convert to 16kHz mono WAV for ASR ──
    audio_path = str(DOWNLOAD_DIR / "audio_16k.wav")
    log.info("Converting audio to 16kHz mono WAV for ASR…")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-ac", "1", "-ar", "16000",
            audio_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log.info(f"Audio saved: {audio_path} ({Path(audio_path).stat().st_size / 1e6:.1f} MB)")

    return video_path, audio_path


# ──────────────────────────────────────────────
# Step 2: Send to Modal GPU worker
# ──────────────────────────────────────────────
async def call_modal_worker(
    video_path: str, audio_path: str, youtube_url: str
) -> dict:
    """Send video + audio bytes to the Modal GPU worker and return results."""
    log.info("Reading files for upload to Modal…")
    video_bytes = Path(video_path).read_bytes()
    audio_bytes = Path(audio_path).read_bytes()
    log.info(
        f"  Video: {len(video_bytes) / 1e6:.1f} MB, "
        f"Audio: {len(audio_bytes) / 1e6:.1f} MB"
    )

    log.info("Calling Modal GPU worker…")
    ClyptWorker = modal.Cls.from_name("clypt-sota-worker", "ClyptWorker")
    worker = ClyptWorker()

    t0 = time.time()
    result = await worker.extract.remote.aio(
        video_bytes=video_bytes,
        audio_wav_bytes=audio_bytes,
        youtube_url=youtube_url,
    )
    elapsed = time.time() - t0
    log.info(f"Modal worker returned in {elapsed:.1f}s")

    return result


# ──────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────
async def main(youtube_url: str | None = None):
    url = youtube_url or "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    log.info("=" * 60)
    log.info("PHASE 1 — Modal Deterministic Extraction")
    log.info("=" * 60)

    # Step 1: Download locally
    log.info("── Step 1: Local Media Acquisition (yt-dlp) ──")
    video_path, audio_path = download_media(url)

    # Step 2: Send to Modal
    log.info("── Step 2: Modal GPU Extraction ──")
    result = await call_modal_worker(video_path, audio_path, url)

    if result.get("status") != "success":
        log.error(f"Modal worker failed: {result.get('message', 'unknown error')}")
        raise RuntimeError(f"Phase 1 extraction failed: {result.get('message')}")

    # Step 3: Write output ledgers
    visual_out = OUTPUT_DIR / "phase_1_visual.json"
    audio_out = OUTPUT_DIR / "phase_1_audio.json"

    phase_1_visual = result.get("phase_1_visual") or result.get("phase_1a_visual")
    phase_1_audio = result.get("phase_1_audio") or result.get("phase_1a_audio")
    if phase_1_visual is None or phase_1_audio is None:
        raise RuntimeError("Modal worker response missing Phase 1 visual/audio payloads")

    with open(visual_out, "w") as f:
        json.dump(phase_1_visual, f, indent=2)
    log.info(f"Visual ledger saved → {visual_out}")

    with open(audio_out, "w") as f:
        json.dump(phase_1_audio, f, indent=2)
    log.info(f"Audio ledger saved → {audio_out}")

    # Summary
    words = phase_1_audio.get("words", [])
    tracks = phase_1_visual.get("tracks", [])
    log.info("=" * 60)
    log.info("PHASE 1 COMPLETE")
    log.info(f"  Words transcribed: {len(words)}")
    log.info(f"  Visual tracks:     {len(tracks)}")
    if words:
        log.info(f"  Audio coverage:    {words[-1]['end_time_ms'] / 1000:.1f}s")
    log.info("=" * 60)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
