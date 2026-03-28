#!/usr/bin/env python3
"""
Phase 1: DigitalOcean async job orchestration
=============================================
Submits the source URL to the DO Phase 1 service, polls until the job completes,
then materializes the returned manifest artifacts into the legacy local files
that downstream phases still consume.

Outputs:
  - downloads/video.mp4           (compatibility bridge for downstream + Remotion)
  - outputs/phase_1_visual.json   (tracking data derived from the manifest)
  - outputs/phase_1_audio.json    (transcript data derived from the manifest)
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import yt_dlp

from backend.pipeline.do_phase1_client import DOPhase1Client
from backend.pipeline.phase1_contract import JobState, Phase1Manifest

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
YTDLP_H264_PREFERRED_FORMAT = os.getenv(
    "YTDLP_H264_PREFERRED_FORMAT",
    (
        "bestvideo[vcodec^=avc1][ext=mp4][height<=1080]+bestaudio[ext=m4a]/"
        "bestvideo[vcodec^=avc1][height<=1080]+bestaudio/"
        "best[vcodec^=avc1][height<=1080]/"
        "best[vcodec^=h264][height<=1080]"
    ),
)
YTDLP_MIN_LONG_EDGE = int(os.getenv("YTDLP_MIN_LONG_EDGE", "1080"))
ALLOW_LOW_RES_VIDEO = os.getenv("ALLOW_LOW_RES_VIDEO", "0") == "1"

ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = ROOT / "downloads"
OUTPUT_DIR = ROOT / "outputs"
PHASE1_NDJSON_PATH = OUTPUT_DIR / "phase_1_visual.ndjson"
PHASE1_RUNTIME_CONTROLS_PATH = OUTPUT_DIR / "phase_1_runtime_controls.json"
DETACHED_STATE_PATH = OUTPUT_DIR / "phase_1_detached_state.json"
DEFAULT_DO_PHASE1_POLL_INTERVAL_SECONDS = 10.0
DEFAULT_DO_PHASE1_TIMEOUT_SECONDS = 60.0 * 30.0
PHASE1_EVAL_PROFILE_NAMES = {
    "eval",
    "evaluation",
    "podcast_eval",
    "podcast_framing_eval",
    "test",
}

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


def parse_fps_value(fps_str: str) -> float:
    """Parse ffprobe avg_frame_rate strings like '30000/1001'."""
    try:
        if "/" in fps_str:
            a, b = fps_str.split("/", 1)
            return float(a) / max(1e-6, float(b))
        return float(fps_str)
    except Exception:
        return 25.0


def probe_duration_seconds(video_path: str) -> float:
    """Return duration in seconds from ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float((result.stdout or "0").strip() or 0.0)
    except Exception:
        return 0.0


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
    """Prefer local H.264 so downstream Remotion/video tooling stays fast."""
    codec = probe_video_codec(video_path)
    log.info(f"Video codec detected: {codec}")

    if codec in {"h264", "avc1"}:
        return video_path
    if codec not in {"av1", "vp9", "hevc", "h265"}:
        return video_path

    h264_path = str(DOWNLOAD_DIR / "video_h264.mp4")
    log.info(f"Re-encoding local video {codec} -> h264 for downstream compatibility…")
    started_at = time.perf_counter()
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-threads", "0",
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
        f"({Path(h264_path).stat().st_size / 1e6:.1f} MB, "
        f"{time.perf_counter() - started_at:.1f}s)"
    )
    return h264_path


def _build_shot_changes(duration_ms: int, window_ms: int = 10000) -> list[dict]:
    """Create deterministic pseudo-shot windows for downstream chunk planning."""
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


def build_phase1_runtime_controls() -> dict:
    """Return the effective phase-1 runtime policy for local contract outputs.

    This is intentionally pipeline-side state only. The worker still controls
    the actual extraction runtime, but downstream consumers and tests can read
    the requested policy from the local manifest bridge and detached state.
    """
    profile_name = (os.getenv("PHASE1_RUNTIME_PROFILE", "production") or "production").strip().lower()
    eval_mode = profile_name in PHASE1_EVAL_PROFILE_NAMES or os.getenv("PHASE1_FORCE_LRASD", "0") == "1"

    requested_speaker_binding_mode = (os.getenv("PHASE1_SPEAKER_BINDING_MODE", "auto") or "auto").strip().lower()
    requested_tracking_mode = (os.getenv("PHASE1_TRACKING_MODE", "direct") or "direct").strip().lower()
    requested_tracker_backend = (os.getenv("PHASE1_TRACKER_BACKEND", "botsort_reid") or "botsort_reid").strip().lower()
    shared_analysis_proxy_enabled = (os.getenv("PHASE1_SHARED_ANALYSIS_PROXY", "1") or "1").strip() != "0"
    heuristic_binding_enabled_env = os.getenv("PHASE1_HEURISTIC_BINDING_ENABLED")

    speaker_binding_mode = "lrasd" if eval_mode else requested_speaker_binding_mode
    tracking_mode = "direct" if eval_mode else requested_tracking_mode
    heuristic_binding_enabled = False if eval_mode else (
        heuristic_binding_enabled_env is None or heuristic_binding_enabled_env.strip() != "0"
    )

    return {
        "profile_name": profile_name,
        "evaluation_mode": eval_mode,
        "speaker_binding_mode": speaker_binding_mode,
        "heuristic_binding_enabled": heuristic_binding_enabled,
        "tracking_mode": tracking_mode,
        "tracker_backend": requested_tracker_backend,
        "shared_analysis_proxy_enabled": shared_analysis_proxy_enabled,
        "framing_policy": "single_person_plus_two_speaker",
        "two_speaker_layout_policy": "shared_two_shot_or_explicit_split",
        "face_detection_provenance": "scrfd_fullframe",
        "notes": (
            "Eval profiles request LR-ASD and disable heuristic binding for inspection. "
            "The worker remains the source of truth for actual extraction runtime."
        ),
    }


def save_phase1_runtime_controls(runtime_controls: dict):
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(PHASE1_RUNTIME_CONTROLS_PATH, "w", encoding="utf-8") as f:
        json.dump(runtime_controls, f, indent=2)
    log.info(f"Phase 1 runtime controls saved → {PHASE1_RUNTIME_CONTROLS_PATH}")


def enrich_visual_ledger_for_downstream(
    phase_1_visual: dict,
    phase_1_audio: dict,
    video_path: str,
    *,
    runtime_controls: dict | None = None,
) -> dict:
    """Add backward-compatible visual fields (shot/person/face/object/label blocks)."""
    tracks = list(phase_1_visual.get("tracks", []))
    w, h, fps_str = probe_video_stream(video_path)
    fps = parse_fps_value(fps_str)
    duration_s = probe_duration_seconds(video_path)
    runtime_controls = runtime_controls or build_phase1_runtime_controls()
    words = phase_1_audio.get("words", [])
    audio_end_ms = int(words[-1]["end_time_ms"]) if words else 0
    duration_ms = max(int(duration_s * 1000), audio_end_ms, 1)

    by_track: dict[str, list[dict]] = {}
    for t in tracks:
        tid = str(t.get("track_id", ""))
        if not tid:
            continue
        by_track.setdefault(tid, []).append(t)
    for tid in list(by_track.keys()):
        by_track[tid].sort(key=lambda d: int(d.get("frame_idx", -1)))

    def _norm_bbox(t: dict) -> dict:
        x1 = float(t.get("x1", 0.0))
        y1 = float(t.get("y1", 0.0))
        x2 = float(t.get("x2", x1 + 1.0))
        y2 = float(t.get("y2", y1 + 1.0))
        if w > 0 and h > 0:
            return {
                "left": max(0.0, min(1.0, x1 / w)),
                "top": max(0.0, min(1.0, y1 / h)),
                "right": max(0.0, min(1.0, x2 / w)),
                "bottom": max(0.0, min(1.0, y2 / h)),
            }
        return {
            "left": max(0.0, x1),
            "top": max(0.0, y1),
            "right": max(0.0, x2),
            "bottom": max(0.0, y2),
        }

    existing_person_detections = list(phase_1_visual.get("person_detections", []))
    existing_face_detections = list(phase_1_visual.get("face_detections", []))
    person_detections = []
    proxy_face_detections = []
    for idx, (tid, dets) in enumerate(sorted(by_track.items())):
        ts_objs = []
        face_ts_objs = []
        for d in dets:
            fi = int(d.get("frame_idx", 0))
            time_ms = int(round((fi / max(1e-6, fps)) * 1000.0))
            bbox = _norm_bbox(d)
            ts_objs.append(
                {
                    "time_ms": time_ms,
                    "bounding_box": bbox,
                    "track_id": tid,
                    "confidence": float(d.get("confidence", 0.0)),
                    "source": "person_track",
                    "provenance": {
                        "kind": "person_track",
                        "derived_from": "tracking_tracks",
                        "track_id": tid,
                        "frame_idx": fi,
                    },
                }
            )
            # Head-biased proxy face box from person bbox.
            bw = max(1e-6, float(bbox["right"] - bbox["left"]))
            bh = max(1e-6, float(bbox["bottom"] - bbox["top"]))
            fx1 = bbox["left"] + 0.18 * bw
            fx2 = bbox["right"] - 0.18 * bw
            fy1 = bbox["top"] + 0.02 * bh
            fy2 = bbox["top"] + 0.48 * bh
            face_ts_objs.append(
                {
                    "time_ms": time_ms,
                    "bounding_box": {
                        "left": max(0.0, min(1.0, fx1)),
                        "top": max(0.0, min(1.0, fy1)),
                        "right": max(0.0, min(1.0, fx2)),
                        "bottom": max(0.0, min(1.0, fy2)),
                    },
                    "track_id": tid,
                    "confidence": float(d.get("confidence", 0.0)),
                    "source": "compatibility_bridge",
                    "provenance": {
                        "kind": "compatibility_bridge",
                        "derived_from": "person_track_bbox",
                        "track_id": tid,
                        "frame_idx": fi,
                        "basis": "head_biased_crop_from_body_box",
                    },
                }
            )
        if not ts_objs:
            continue
        person_detections.append(
            {
                "confidence": float(sum(float(x.get("confidence", 0.0)) for x in ts_objs) / max(1, len(ts_objs))),
                "segment_start_ms": int(ts_objs[0]["time_ms"]),
                "segment_end_ms": int(ts_objs[-1]["time_ms"]),
                "person_track_index": idx,
                "track_id": tid,
                "timestamped_objects": ts_objs,
                "source": "person_track",
                "provenance": {
                    "kind": "person_track",
                    "derived_from": "tracking_tracks",
                    "track_id": tid,
                    "track_count": len(ts_objs),
                },
            }
        )
        proxy_face_detections.append(
            {
                "confidence": float(sum(float(x.get("confidence", 0.0)) for x in face_ts_objs) / max(1, len(face_ts_objs))),
                "segment_start_ms": int(face_ts_objs[0]["time_ms"]),
                "segment_end_ms": int(face_ts_objs[-1]["time_ms"]),
                "face_track_index": idx,
                "track_id": tid,
                "timestamped_objects": face_ts_objs,
                "source": "compatibility_bridge",
                "provenance": {
                    "kind": "compatibility_bridge",
                    "derived_from": "person_track_bbox",
                    "track_id": tid,
                    "track_count": len(face_ts_objs),
                    "basis": "head_biased_crop_from_body_box",
                },
            }
        )

    enriched = dict(phase_1_visual)
    enriched["shot_changes"] = _build_shot_changes(duration_ms=duration_ms)
    enriched["person_detections"] = existing_person_detections or person_detections
    enriched["face_detections"] = existing_face_detections or proxy_face_detections
    enriched["proxy_face_detections"] = proxy_face_detections
    enriched["object_tracking"] = list(enriched.get("object_tracking", []))
    enriched["label_detections"] = list(enriched.get("label_detections", []))
    enriched["runtime_controls"] = runtime_controls
    enriched["video_metadata"] = {
        "width": int(w),
        "height": int(h),
        "fps": float(fps),
        "duration_ms": int(duration_ms),
    }
    return enriched


def write_visual_ndjson(visual_ledger: dict, path: Path):
    """Serialize phase handoff to NDJSON (header + per-detection records)."""
    header = {
        "record_type": "header",
        "schema_version": str(visual_ledger.get("schema_version", "2.0.0")),
        "task_type": str(visual_ledger.get("task_type", "person_tracking")),
        "coordinate_space": str(visual_ledger.get("coordinate_space", "absolute_original_frame_xyxy")),
        "geometry_type": str(visual_ledger.get("geometry_type", "aabb")),
        "class_taxonomy": visual_ledger.get("class_taxonomy", {"0": "person"}),
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for t in visual_ledger.get("tracks", []):
            row = {"record_type": "track", **t}
            f.write(json.dumps(row) + "\n")
    log.info(f"Visual NDJSON saved → {path}")


def validate_phase_handoff(visual_ledger: dict, audio_ledger: dict):
    """Fail fast when phase boundaries violate required contracts."""
    required_visual_keys = (
        "tracks",
        "shot_changes",
        "person_detections",
        "face_detections",
        "object_tracking",
        "label_detections",
    )
    for k in required_visual_keys:
        if k not in visual_ledger:
            raise RuntimeError(f"Phase 1 visual contract missing key: {k}")
        if not isinstance(visual_ledger[k], list):
            raise RuntimeError(f"Phase 1 visual key {k} must be a list")
    if "words" not in audio_ledger or not isinstance(audio_ledger["words"], list):
        raise RuntimeError("Phase 1 audio contract missing words[]")
    for i, t in enumerate(visual_ledger.get("tracks", [])[:200]):
        for key in ("frame_idx", "track_id", "x1", "y1", "x2", "y2", "confidence"):
            if key not in t:
                raise RuntimeError(f"Track[{i}] missing required key: {key}")


def save_detached_state(payload: dict):
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(DETACHED_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log.info(f"Detached Phase 1 state saved → {DETACHED_STATE_PATH}")


def load_detached_state() -> dict | None:
    if not DETACHED_STATE_PATH.exists():
        return None
    try:
        with open(DETACHED_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        log.warning(f"Detached state at {DETACHED_STATE_PATH} is invalid JSON; ignoring it.")
        return None


def clear_detached_state():
    if DETACHED_STATE_PATH.exists():
        DETACHED_STATE_PATH.unlink()
        log.info(f"Cleared detached state → {DETACHED_STATE_PATH}")


def get_phase1_service_base_url() -> str:
    base_url = (
        os.getenv("DO_PHASE1_BASE_URL")
        or os.getenv("PHASE1_SERVICE_URL")
        or os.getenv("DIGITALOCEAN_PHASE1_BASE_URL")
    )
    if not base_url:
        raise RuntimeError(
            "DigitalOcean Phase 1 orchestration requires DO_PHASE1_BASE_URL "
            "(or PHASE1_SERVICE_URL / DIGITALOCEAN_PHASE1_BASE_URL)."
        )
    return base_url


def get_phase1_poll_interval_seconds() -> float:
    return max(
        0.0,
        float(os.getenv("DO_PHASE1_POLL_INTERVAL_SECONDS", str(DEFAULT_DO_PHASE1_POLL_INTERVAL_SECONDS))),
    )


def get_phase1_timeout_seconds() -> float:
    return max(
        0.0,
        float(os.getenv("DO_PHASE1_TIMEOUT_SECONDS", str(DEFAULT_DO_PHASE1_TIMEOUT_SECONDS))),
    )


def build_phase1_client() -> DOPhase1Client:
    return DOPhase1Client(base_url=get_phase1_service_base_url())


def _state_payload(*, job_id: str, source_url: str, status: str, **extra: object) -> dict:
    payload = {
        "provider": "digitalocean",
        "job_id": job_id,
        "source_url": source_url,
        "status": status,
        "updated_at_epoch_s": time.time(),
    }
    payload.update(extra)
    return payload


def _load_resumable_job_id(source_url: str) -> str | None:
    state = load_detached_state()
    if not state:
        return None
    if state.get("provider") != "digitalocean":
        log.info("Ignoring non-DigitalOcean detached state left by an older Phase 1 flow.")
        return None
    if state.get("source_url") != source_url:
        log.info(
            "Ignoring detached state for a different source URL (%s); starting a fresh Phase 1 job.",
            state.get("source_url"),
        )
        return None
    job_id = str(state.get("job_id", "")).strip()
    if not job_id:
        return None
    log.info(
        "Resuming DigitalOcean Phase 1 job %s from detached state (last status=%s).",
        job_id,
        state.get("status", "unknown"),
    )
    return job_id


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
        "phase_1_visual.ndjson",
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
    if _looks_like_direct_media_url(url):
        video_path = _download_direct_media_url(url)
    else:
        video_path = _download_video_with_format_fallback(url)

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


def _looks_like_direct_media_url(url: str) -> bool:
    parsed = urlparse(url)
    path = (parsed.path or "").lower()
    return path.endswith((".mp4", ".mov", ".m4v", ".webm", ".mkv"))


def _download_direct_media_url(url: str) -> str:
    target_path = DOWNLOAD_DIR / "video.mp4"
    with urlopen(url, timeout=120) as response, open(target_path, "wb") as target_file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            target_file.write(chunk)
    return str(target_path)


def _download_video_with_format_fallback(url: str) -> str:
    format_attempts = []
    preferred_format = (os.getenv("YTDLP_H264_PREFERRED_FORMAT", YTDLP_H264_PREFERRED_FORMAT) or "").strip()
    fallback_format = (os.getenv("YTDLP_VIDEO_FORMAT", YTDLP_VIDEO_FORMAT) or "").strip()
    if preferred_format:
        format_attempts.append(("H.264-preferred", preferred_format))
    if fallback_format and fallback_format not in {preferred_format}:
        format_attempts.append(("fallback", fallback_format))

    last_error = None
    for label, format_selector in format_attempts:
        video_opts = {
            "format": format_selector,
            "outtmpl": str(DOWNLOAD_DIR / "video.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "noprogress": False,
        }
        try:
            log.info("yt-dlp attempt %s with format selector: %s", label, format_selector)
            with yt_dlp.YoutubeDL(video_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return ydl.prepare_filename(info)
        except Exception as exc:
            last_error = exc
            log.warning("yt-dlp %s format attempt failed: %s", label, exc)

    raise RuntimeError(f"Failed to download source video with all format selectors: {last_error}")


# ──────────────────────────────────────────────
# Step 2: Submit + poll DigitalOcean job service
# ──────────────────────────────────────────────
async def submit_or_resume_phase1_job(client: DOPhase1Client, source_url: str) -> str:
    resumable_job_id = _load_resumable_job_id(source_url)
    if resumable_job_id:
        return resumable_job_id

    runtime_controls = build_phase1_runtime_controls()
    submission = await client.submit_job(source_url, runtime_controls=runtime_controls)
    job_id = str(submission.job_id)
    log.info("Submitted DigitalOcean Phase 1 job %s (status=%s).", job_id, submission.status)
    save_detached_state(
        _state_payload(
            job_id=job_id,
            source_url=source_url,
            status=str(submission.status),
            submitted_at_epoch_s=time.time(),
            runtime_controls=runtime_controls,
        )
    )
    return job_id


async def wait_for_phase1_manifest(
    client: DOPhase1Client,
    *,
    job_id: str,
    source_url: str,
) -> Phase1Manifest:
    timeout_seconds = get_phase1_timeout_seconds()
    poll_interval_seconds = get_phase1_poll_interval_seconds()
    started_at = time.monotonic()
    runtime_controls = build_phase1_runtime_controls()

    log.info(
        "Polling DigitalOcean Phase 1 job %s every %.1fs (timeout %.1fs).",
        job_id,
        poll_interval_seconds,
        timeout_seconds,
    )

    while True:
        job = await client.get_job(job_id)
        status = str(job.status)
        log.info("DigitalOcean Phase 1 job %s status=%s", job_id, status)

        save_detached_state(
            _state_payload(
                job_id=job_id,
                source_url=source_url,
                status=status,
                manifest_uri=getattr(job, "manifest_uri", None),
                failure=getattr(job, "failure", None),
                runtime_controls=runtime_controls,
            )
        )

        if job.status == JobState.SUCCEEDED or status == JobState.SUCCEEDED.value:
            manifest = await client.get_result(job_id)
            clear_detached_state()
            return manifest

        if job.status == JobState.FAILED or status == JobState.FAILED.value:
            message = getattr(job, "failure", None) or {"error_message": "unknown DigitalOcean Phase 1 failure"}
            save_detached_state(
                _state_payload(
                    job_id=job_id,
                    source_url=source_url,
                    status=status,
                    failure=message,
                    terminal=True,
                    runtime_controls=runtime_controls,
                )
            )
            raise RuntimeError(f"Phase 1 extraction failed for job {job_id}: {message}")

        elapsed = time.monotonic() - started_at
        if elapsed >= timeout_seconds:
            save_detached_state(
                _state_payload(
                    job_id=job_id,
                    source_url=source_url,
                    status=status,
                    timeout_seconds=timeout_seconds,
                    resume_hint="Re-run the pipeline to resume polling this job.",
                    runtime_controls=runtime_controls,
                )
            )
            log.warning(
                "Timeout while waiting for DigitalOcean Phase 1 job %s after %.1fs. "
                "Resume by re-running the pipeline; state saved at %s.",
                job_id,
                elapsed,
                DETACHED_STATE_PATH,
            )
            raise TimeoutError(
                f"DigitalOcean Phase 1 job {job_id} did not complete within {timeout_seconds:.1f}s. "
                f"Resume by re-running the pipeline; state saved at {DETACHED_STATE_PATH}."
            )

        await asyncio.sleep(poll_interval_seconds)


def materialize_phase1_manifest(manifest: Phase1Manifest, *, source_url: str) -> tuple[dict, dict]:
    # Compatibility bridge: later pipeline phases and Remotion still expect a
    # local download alongside the JSON ledgers, even though extraction moved to
    # DigitalOcean jobs.
    log.info("── Step 3: Local Media Acquisition For Downstream Compatibility ──")
    runtime_controls = build_phase1_runtime_controls()
    video_path, _audio_path = download_media(source_url)

    phase_1_visual = enrich_visual_ledger_for_downstream(
        manifest.artifacts.visual_tracking.model_dump(mode="json"),
        manifest.artifacts.transcript.model_dump(mode="json"),
        video_path,
        runtime_controls=runtime_controls,
    )
    phase_1_audio = manifest.artifacts.transcript.model_dump(mode="json")

    phase_1_visual["video_gcs_uri"] = manifest.canonical_video_gcs_uri
    phase_1_audio["video_gcs_uri"] = manifest.canonical_video_gcs_uri

    validate_phase_handoff(phase_1_visual, phase_1_audio)

    visual_out = OUTPUT_DIR / "phase_1_visual.json"
    audio_out = OUTPUT_DIR / "phase_1_audio.json"

    with open(visual_out, "w", encoding="utf-8") as f:
        json.dump(phase_1_visual, f, indent=2)
    log.info(f"Visual ledger saved → {visual_out}")

    write_visual_ndjson(phase_1_visual, PHASE1_NDJSON_PATH)

    with open(audio_out, "w", encoding="utf-8") as f:
        json.dump(phase_1_audio, f, indent=2)
    log.info(f"Audio ledger saved → {audio_out}")

    save_phase1_runtime_controls(runtime_controls)

    return phase_1_visual, phase_1_audio


# ──────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────
async def main(youtube_url: str | None = None):
    url = youtube_url or "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    log.info("=" * 60)
    log.info("PHASE 1 — DigitalOcean Async Extraction")
    log.info("=" * 60)

    log.info("── Step 1: DigitalOcean Job Submission + Polling ──")
    async with build_phase1_client() as client:
        job_id = await submit_or_resume_phase1_job(client, url)
        manifest = await wait_for_phase1_manifest(client, job_id=job_id, source_url=url)

    log.info(
        "DigitalOcean Phase 1 job %s succeeded; consuming manifest artifact URIs %s and %s",
        manifest.job_id,
        manifest.artifacts.visual_tracking.uri,
        manifest.artifacts.transcript.uri,
    )

    phase_1_visual, phase_1_audio = materialize_phase1_manifest(manifest, source_url=url)

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
    return manifest


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
