#!/usr/bin/env python3
"""
Phase 1 (Fast Transcript): YouTube captions + GCS upload for testing Phases 2-5.
==================================================================================
Bypasses GPU tracking entirely. Downloads the video, fetches YouTube captions
for word-level timestamps (falls back to Whisper if unavailable), uploads the
video to GCS, and writes the same phase_1_audio.json + phase_1_visual.json
interface that Phases 2-5 consume.

Usage:
    python -m backend.pipeline.phase_1_transcript_fast [--video-id ID] [YOUTUBE_URL]

    --video-id ID   Save outputs to backend/videos/<ID>/ so you can maintain a
                    named library and switch between videos without rerunning Phase 1.
                    Omit to use the default backend/outputs/ + backend/downloads/.

Examples:
    python -m backend.pipeline.phase_1_transcript_fast --video-id lex_musk https://youtu.be/...
    python -m backend.pipeline.phase_1_transcript_fast https://youtu.be/...

Optional env vars:
    YTDLP_COOKIES_FILE          path to a Netscape cookies.txt (never locked by browser)
    YTDLP_COOKIES_FROM_BROWSER  browser name e.g. "chrome" (requires browser closed)
    WHISPER_MODEL               fallback model e.g. "large-v3", "medium" (default: large-v3)
    WHISPER_DEVICE              "cpu" or "cuda" (default: cpu)
    WHISPER_COMPUTE_TYPE        "int8", "float16" etc. (default: int8)
    GCS_BUCKET                  override GCS bucket (default: clypt-storage-v2)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import yt_dlp
from google.cloud import storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase_1_fast")

ROOT = Path(__file__).resolve().parent.parent
VIDEOS_LIBRARY_DIR = ROOT / "videos"

# Resolved at runtime by _resolve_dirs() based on --video-id arg
DOWNLOAD_DIR: Path = ROOT / "downloads"
OUTPUT_DIR: Path = ROOT / "outputs"

GCS_BUCKET = os.getenv("GCS_BUCKET", "clypt-storage-v2")
GCS_VIDEO_BLOB = "phase_1/video.mp4"
DEFAULT_VIDEO_GCS_URI = f"gs://{GCS_BUCKET}/{GCS_VIDEO_BLOB}"

YTDLP_VIDEO_FORMAT = (
    "bestvideo[vcodec^=avc1][ext=mp4][height<=1080]+bestaudio[ext=m4a]/"
    "bestvideo[vcodec^=avc1][height<=1080]+bestaudio/"
    "best[vcodec^=avc1][height<=1080]/"
    "best[height<=1080]/best"
)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# Set DO_PHASE1_URL to use Parakeet on the droplet as fallback (e.g. http://107.170.53.44:8000)
DO_PHASE1_URL = os.getenv("DO_PHASE1_URL", "").rstrip("/")


# ──────────────────────────────────────────────
# Step 1: Download
# ──────────────────────────────────────────────
def _make_ydl_opts(out_path: str, *, cookies_file: str | None = None, browser: str | None = None) -> dict:
    opts: dict = {
        "format": YTDLP_VIDEO_FORMAT,
        "outtmpl": out_path,
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
    }
    if cookies_file:
        opts["cookiefile"] = cookies_file
    elif browser:
        opts["cookiesfrombrowser"] = (browser,)
    return opts


def _download_video(url: str) -> str:
    """Download video to downloads/video.mp4. Returns local path.

    Cookie resolution order:
      1. YTDLP_COOKIES_FILE env var — path to a cookies.txt file (never locked)
      2. YTDLP_COOKIES_FROM_BROWSER env var — browser name (requires browser closed)
      3. Retry with no cookies (works for most public videos)
    """
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    out_path = str(DOWNLOAD_DIR / "video.mp4")

    cookies_file = (os.getenv("YTDLP_COOKIES_FILE") or "").strip() or None
    browser = (os.getenv("YTDLP_COOKIES_FROM_BROWSER") or "").strip() or None

    attempts: list[tuple[str, dict]] = []
    if cookies_file:
        attempts.append((f"cookies file ({cookies_file})", _make_ydl_opts(out_path, cookies_file=cookies_file)))
    elif browser:
        attempts.append((f"browser cookies ({browser})", _make_ydl_opts(out_path, browser=browser)))
    attempts.append(("no cookies", _make_ydl_opts(out_path)))

    log.info("Downloading video via yt-dlp → %s", out_path)
    last_error: Exception | None = None
    for label, opts in attempts:
        try:
            log.info("  Trying: %s", label)
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            break
        except Exception as exc:
            log.warning("  Failed (%s): %s", label, exc)
            last_error = exc
            if Path(out_path).exists():
                Path(out_path).unlink(missing_ok=True)
    else:
        raise RuntimeError(f"yt-dlp failed all attempts: {last_error}")

    if not Path(out_path).exists():
        raise RuntimeError(f"yt-dlp did not produce {out_path}")
    log.info("Download complete: %.1f MB", Path(out_path).stat().st_size / 1e6)
    return out_path


# ──────────────────────────────────────────────
# Step 2: Probe
# ──────────────────────────────────────────────
def _probe_video(video_path: str) -> tuple[int, int, float, int]:
    """Return (width, height, fps, duration_ms) via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate",
            "-of", "json", video_path,
        ],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or []

    w, h, fps = 1920, 1080, 25.0
    if streams:
        s = streams[0]
        w = int(s.get("width") or 1920)
        h = int(s.get("height") or 1080)
        fps_str = str(s.get("avg_frame_rate") or "25/1")
        try:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) else 25.0
        except Exception:
            fps = 25.0

    # Probe container duration separately (more reliable)
    r2 = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", video_path],
        check=True, capture_output=True, text=True,
    )
    d2 = json.loads(r2.stdout or "{}")
    duration_s = float((d2.get("format") or {}).get("duration") or 0)
    duration_ms = int(duration_s * 1000)

    log.info("Video probe: %dx%d @ %.2f fps, %.1fs", w, h, fps, duration_s)
    return w, h, fps, duration_ms


# ──────────────────────────────────────────────
# Step 3: Transcript (captions → Whisper fallback)
# ──────────────────────────────────────────────
def _word_entry(text: str, start_ms: int, end_ms: int) -> dict:
    return {
        "word": text,
        "start_time_ms": max(0, start_ms),
        "end_time_ms": max(0, end_ms),
        "speaker_track_id": None,
        "speaker_tag": "SPEAKER_00",
        "speaker_local_track_id": None,
        "speaker_local_tag": None,
    }


def _parse_json3(path: str) -> list[dict]:
    """Parse yt-dlp json3 caption file into word-level dicts.

    json3 structure:
      { "events": [ { "tStartMs": int, "dDurationMs": int,
                      "segs": [ { "utf8": str, "tOffsetMs": int } ] } ] }
    Each seg's absolute start = event tStartMs + seg tOffsetMs.
    End time = next seg start (or event end for last seg in event).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    words: list[dict] = []

    for event in data.get("events", []):
        t_start = int(event.get("tStartMs", 0))
        t_dur = int(event.get("dDurationMs", 0))
        t_end = t_start + t_dur
        segs = event.get("segs") or []

        # Compute absolute start for each seg
        abs_starts = [t_start + int(s.get("tOffsetMs", 0)) for s in segs]

        for i, seg in enumerate(segs):
            text = seg.get("utf8", "").strip()
            if not text or text in ("\n", "\r\n"):
                continue
            seg_start = abs_starts[i]
            # End = next seg start, or event end
            seg_end = abs_starts[i + 1] if i + 1 < len(abs_starts) else t_end
            seg_end = max(seg_start + 1, seg_end)

            # Split multi-word segs (auto-captions sometimes group words)
            tokens = text.split()
            if not tokens:
                continue
            per_word_ms = max(1, (seg_end - seg_start) // len(tokens))
            for j, token in enumerate(tokens):
                w_start = seg_start + j * per_word_ms
                w_end = seg_start + (j + 1) * per_word_ms
                words.append(_word_entry(token, w_start, w_end))

    return words


def _fetch_youtube_captions(url: str) -> list[dict] | None:
    """Download YouTube captions (manual then auto-generated) via yt-dlp.
    Returns word list or None if no captions available."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "en-US", "en-GB"],
            "subtitlesformat": "json3",
            "outtmpl": str(Path(tmpdir) / "caps"),
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception as e:
            log.warning("Caption download error: %s", e)
            return None

        # Find downloaded caption file (manual preferred over auto)
        cap_files = sorted(Path(tmpdir).glob("*.json3"))
        if not cap_files:
            return None

        # Prefer manual (.en.json3) over auto (.en.auto.json3 or .en-orig.json3)
        manual = [f for f in cap_files if ".auto." not in f.name and "-orig" not in f.name]
        chosen = manual[0] if manual else cap_files[0]
        source = "manual" if manual else "auto-generated"
        log.info("Using %s captions: %s", source, chosen.name)

        words = _parse_json3(str(chosen))
        return words if words else None


def _extract_audio_wav(video_path: str) -> str:
    """Extract mono 16 kHz WAV from video for Whisper fallback."""
    audio_path = str(DOWNLOAD_DIR / "audio_whisper.wav")
    log.info("Extracting audio → %s", audio_path)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-f", "wav", audio_path,
        ],
        check=True, capture_output=True,
    )
    return audio_path


def _transcribe_droplet(audio_path: str) -> list[dict]:
    """Send audio to the DO droplet's /asr endpoint and return words."""
    import urllib.request

    url = f"{DO_PHASE1_URL}/asr"
    log.info("Sending audio to droplet ASR: %s", url)

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    boundary = "----ClyptASRBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + audio_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        import json
        result = json.loads(resp.read())

    words = result.get("words", [])
    log.info("Droplet ASR complete: %d words", len(words))
    return words


def _transcribe_whisper(audio_path: str) -> list[dict]:
    """Whisper fallback — only used when YouTube has no captions."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("faster-whisper is required for non-captioned videos: pip install faster-whisper")

    log.info(
        "Loading Whisper model: %s (device=%s, compute_type=%s)",
        WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    )
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

    log.info("Transcribing %s ...", audio_path)
    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        language="en",
        beam_size=5,
        vad_filter=True,
    )
    log.info("Detected language: %s (prob=%.2f)", info.language, info.language_probability)

    words: list[dict] = []
    for segment in segments:
        for w in (segment.words or []):
            text = w.word.strip()
            if not text:
                continue
            words.append(_word_entry(text, int(round(w.start * 1000)), int(round(w.end * 1000))))

    log.info("Whisper transcription complete: %d words", len(words))
    return words


def get_words(url: str, video_path: str) -> list[dict]:
    """Return word list: Droplet Parakeet → Whisper fallback."""
    audio_path = _extract_audio_wav(video_path)

    if DO_PHASE1_URL:
        try:
            log.info("Running Parakeet ASR on droplet (%s)…", DO_PHASE1_URL)
            words = _transcribe_droplet(audio_path)
            if words:
                return words
        except Exception as e:
            log.warning("Droplet ASR failed (%s) — falling back to Whisper", e)

    log.info("Falling back to Whisper (%s)", WHISPER_MODEL)
    return _transcribe_whisper(audio_path)


# ──────────────────────────────────────────────
# Step 4: GCS upload
# ──────────────────────────────────────────────
def _upload_to_gcs(local_path: str, bucket_name: str, blob_name: str) -> str:
    """Upload file to GCS. Returns gs:// URI."""
    log.info("Uploading %s → gs://%s/%s", local_path, bucket_name, blob_name)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path, content_type="video/mp4")
    uri = f"gs://{bucket_name}/{blob_name}"
    log.info("GCS upload complete → %s", uri)
    return uri


# ──────────────────────────────────────────────
# Step 5: Build ledgers
# ──────────────────────────────────────────────
def _build_pseudo_shot_changes(duration_ms: int, window_ms: int = 10_000) -> list[dict]:
    """10-second pseudo-shot windows so Phase 2A chunks correctly."""
    if duration_ms <= 0:
        return [{"start_time_ms": 0, "end_time_ms": window_ms}]
    out = []
    start = 0
    while start < duration_ms:
        end = min(duration_ms, start + window_ms)
        out.append({"start_time_ms": int(start), "end_time_ms": int(end)})
        if end >= duration_ms:
            break
        start = end
    return out


def _build_audio_ledger(words: list[dict], source_url: str, video_gcs_uri: str) -> dict:
    return {
        "uri": video_gcs_uri,
        "source_audio": source_url,
        "video_gcs_uri": video_gcs_uri,
        "words": words,
        "speaker_bindings": [],
        "audio_speaker_turns": [],
        "speaker_bindings_local": [],
        "speaker_follow_bindings_local": [],
        "audio_speaker_local_track_map": [],
        "speaker_candidate_debug": [],
    }


def _build_visual_ledger(
    source_url: str,
    video_gcs_uri: str,
    width: int,
    height: int,
    fps: float,
    duration_ms: int,
) -> dict:
    return {
        "uri": video_gcs_uri,
        "source_video": source_url,
        "video_gcs_uri": video_gcs_uri,
        "schema_version": "2.0.0",
        "task_type": "person_tracking",
        "coordinate_space": "absolute_original_frame_xyxy",
        "geometry_type": "aabb",
        "class_taxonomy": {"0": "person"},
        "tracking_metrics": {},
        "tracks": [],
        "tracks_local": [],
        "face_detections": [],
        "person_detections": [],
        "label_detections": [],
        "object_tracking": [],
        "proxy_face_detections": [],
        "shot_changes": _build_pseudo_shot_changes(duration_ms),
        "video_metadata": {
            "width": width,
            "height": height,
            "fps": fps,
            "duration_ms": duration_ms,
        },
        "runtime_controls": {},
    }


# ──────────────────────────────────────────────
# Library helpers
# ──────────────────────────────────────────────
def _resolve_dirs(video_id: str | None) -> tuple[Path, Path, str, str]:
    """Return (download_dir, output_dir, gcs_blob, gcs_uri) for this run."""
    if video_id:
        video_dir = VIDEOS_LIBRARY_DIR / video_id
        dl = video_dir / "downloads"
        out = video_dir / "outputs"
        blob = f"videos/{video_id}/video.mp4"
    else:
        dl = ROOT / "downloads"
        out = ROOT / "outputs"
        blob = GCS_VIDEO_BLOB

    # VIDEO_GCS_URI env var always wins
    uri = os.getenv("VIDEO_GCS_URI", "").strip() or f"gs://{GCS_BUCKET}/{blob}"
    if os.getenv("VIDEO_GCS_URI", "").strip():
        raw = os.getenv("VIDEO_GCS_URI").strip().removeprefix("gs://")
        _, _, b = raw.partition("/")
        blob = b

    return dl, out, blob, uri


def list_library() -> list[str]:
    if not VIDEOS_LIBRARY_DIR.exists():
        return []
    return sorted(p.name for p in VIDEOS_LIBRARY_DIR.iterdir() if p.is_dir())


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main(url: str | None = None):
    global DOWNLOAD_DIR, OUTPUT_DIR

    # Parse --video-id from argv
    args = sys.argv[1:]
    video_id: str | None = None
    if "--video-id" in args:
        idx = args.index("--video-id")
        if idx + 1 >= len(args):
            raise ValueError("--video-id requires a value")
        video_id = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    if "--list" in args:
        lib = list_library()
        if lib:
            print("Saved videos:")
            for v in lib:
                print(f"  {v}")
        else:
            print("No saved videos yet.")
        return

    url = url or (args[0] if args else None)
    if not url:
        url = input("Enter YouTube URL: ").strip()
    if not url:
        raise ValueError("No URL provided")

    DOWNLOAD_DIR, OUTPUT_DIR, blob_name, video_gcs_uri = _resolve_dirs(video_id)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("PHASE 1 (FAST TRANSCRIPT) — Captions + GCS Upload")
    log.info("=" * 60)
    log.info("Target:   %s", url)
    if video_id:
        log.info("Video ID: %s", video_id)
    log.info("Outputs:  %s", OUTPUT_DIR)

    log.info("── Step 1: Download Video ──")
    video_path = _download_video(url)

    log.info("── Step 2: Probe Video ──")
    width, height, fps, duration_ms = _probe_video(video_path)

    log.info("── Step 3: Transcript ──")
    words = get_words(url, video_path)

    log.info("── Step 4: Upload Video to GCS ──")
    _upload_to_gcs(video_path, GCS_BUCKET, blob_name)

    log.info("── Step 5: Write Phase 1 Ledgers ──")
    audio_ledger = _build_audio_ledger(words, url, video_gcs_uri)
    visual_ledger = _build_visual_ledger(url, video_gcs_uri, width, height, fps, duration_ms)

    audio_out = OUTPUT_DIR / "phase_1_audio.json"
    visual_out = OUTPUT_DIR / "phase_1_visual.json"

    with open(audio_out, "w", encoding="utf-8") as f:
        json.dump(audio_ledger, f, indent=2)
    log.info("Audio ledger → %s", audio_out)

    with open(visual_out, "w", encoding="utf-8") as f:
        json.dump(visual_ledger, f, indent=2)
    log.info("Visual ledger → %s", visual_out)

    log.info("=" * 60)
    log.info("PHASE 1 (FAST TRANSCRIPT) COMPLETE")
    log.info("  Words:      %d", len(words))
    if words:
        log.info(
            "  Coverage:   %.1fs → %.1fs",
            words[0]["start_time_ms"] / 1000,
            words[-1]["end_time_ms"] / 1000,
        )
    log.info("  GCS video:  %s", video_gcs_uri)
    if video_id:
        log.info("  To run phases 2-5:")
        log.info("    START_FROM=2a VIDEO_ID=%s python -m backend.pipeline.run_pipeline", video_id)
    else:
        log.info("  Next: START_FROM=2a python -m backend.pipeline.run_pipeline")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
