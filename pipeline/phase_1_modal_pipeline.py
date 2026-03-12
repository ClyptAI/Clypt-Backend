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

import asyncio
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import modal
from modal.functions import FunctionCall
import yt_dlp
from google.cloud import storage

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
GCS_BUCKET = os.getenv("GCS_BUCKET", "clypt-storage-v2")
GCS_VIDEO_OBJECT = os.getenv("GCS_VIDEO_OBJECT", "phase_1/video.mp4")
PHASE1_NDJSON_PATH = OUTPUT_DIR / "phase_1_visual.ndjson"
DETACHED_STATE_PATH = OUTPUT_DIR / "phase_1_detached_state.json"

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


def upload_video_to_gcs(video_path: str) -> str:
    """Upload local video to the canonical GCS URI required by downstream phases."""
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_VIDEO_OBJECT)
    blob.upload_from_filename(video_path)
    uri = f"gs://{GCS_BUCKET}/{GCS_VIDEO_OBJECT}"
    log.info(f"Uploaded source video → {uri}")
    return uri


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


def enrich_visual_ledger_for_downstream(
    phase_1_visual: dict,
    phase_1_audio: dict,
    video_path: str,
) -> dict:
    """Add backward-compatible visual fields (shot/person/face/object/label blocks)."""
    tracks = list(phase_1_visual.get("tracks", []))
    w, h, fps_str = probe_video_stream(video_path)
    fps = parse_fps_value(fps_str)
    duration_s = probe_duration_seconds(video_path)
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

    person_detections = []
    face_detections = []
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
            }
        )
        face_detections.append(
            {
                "confidence": float(sum(float(x.get("confidence", 0.0)) for x in face_ts_objs) / max(1, len(face_ts_objs))),
                "segment_start_ms": int(face_ts_objs[0]["time_ms"]),
                "segment_end_ms": int(face_ts_objs[-1]["time_ms"]),
                "face_track_index": idx,
                "track_id": tid,
                "timestamped_objects": face_ts_objs,
            }
        )

    enriched = dict(phase_1_visual)
    enriched["shot_changes"] = _build_shot_changes(duration_ms=duration_ms)
    enriched["person_detections"] = person_detections
    enriched["face_detections"] = face_detections
    enriched["object_tracking"] = list(enriched.get("object_tracking", []))
    enriched["label_detections"] = list(enriched.get("label_detections", []))
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
    log.info(f"Detached fan-out state saved → {DETACHED_STATE_PATH}")


def load_detached_state() -> dict | None:
    if not DETACHED_STATE_PATH.exists():
        return None
    with open(DETACHED_STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


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


async def call_modal_worker_distributed(
    video_path: str, audio_path: str, youtube_url: str
) -> dict:
    """Client-orchestrated distributed tracking fan-out across Modal workers."""
    log.info("Reading files for upload to Modal…")
    video_bytes = Path(video_path).read_bytes()
    audio_bytes = Path(audio_path).read_bytes()
    log.info(
        f"  Video: {len(video_bytes) / 1e6:.1f} MB, "
        f"Audio: {len(audio_bytes) / 1e6:.1f} MB"
    )

    ClyptWorker = modal.Cls.from_name("clypt-sota-worker", "ClyptWorker")
    worker = ClyptWorker()
    max_gpu_workers = max(1, min(8, int(os.getenv("CLYPT_MAX_GPU_WORKERS", "8"))))
    log.info(f"Distributed fan-out enabled (max GPU workers={max_gpu_workers})")

    # Best-effort autoscaler cap for chunk tracking method.
    try:
        await worker.track_chunk_from_staged.update_autoscaler.aio(max_containers=max_gpu_workers)
    except Exception:
        pass

    t0 = time.time()
    allow_resume = os.getenv("CLYPT_DISTRIBUTED_RESUME", "1") == "1"
    use_detached = os.getenv("CLYPT_DISTRIBUTED_DETACH", "1") == "1"
    state = load_detached_state() if allow_resume else None
    if state and state.get("status") == "submitted":
        log.info(f"Resuming detached fan-out job {state.get('job_id', '')[:8]}…")
        chunks = list(state["chunks"])
        meta = dict(state["meta"])
        job_id = str(state["job_id"])
        staged_video_path = str(state["video_path"])
        asr_handle = FunctionCall.from_id(str(state["asr_call_id"]))
        chunk_mode = str(state.get("chunk_mode", "indexed"))
        if chunk_mode == "indexed":
            chunk_handle = FunctionCall.from_id(str(state["chunk_call_id"]))
            chunk_handles = None
        else:
            chunk_handle = None
            chunk_handles = [FunctionCall.from_id(str(cid)) for cid in state.get("chunk_call_ids", [])]
        tracking_submitted_ts = float(state.get("tracking_submitted_ts", time.time()))
    else:
        stage = await worker.stage_video_for_tracking.remote.aio(video_bytes=video_bytes)
        chunks = list(stage.get("chunks", []))
        meta = dict(stage.get("meta", {}))
        job_id = str(stage.get("job_id", ""))
        staged_video_path = str(stage.get("video_path", ""))
        if not chunks or not job_id or not staged_video_path:
            raise RuntimeError("Distributed staging failed: missing chunks/job metadata")
        log.info(
            f"Staged job {job_id[:8]}… with {len(chunks)} chunks "
            f"({meta.get('total_frames', 0)} frames @ {meta.get('fps', 0):.2f} fps)"
        )

        if use_detached:
            asr_handle = await worker.run_asr_only.spawn.aio(audio_wav_bytes=audio_bytes)
            chunk_obj = await worker.track_chunk_from_staged.spawn_map.aio(
                [job_id] * len(chunks),
                [staged_video_path] * len(chunks),
                [meta] * len(chunks),
                chunks,
            )
            tracking_submitted_ts = time.time()
            if hasattr(chunk_obj, "get") and hasattr(chunk_obj, "object_id"):
                chunk_mode = "indexed"
                chunk_handle = chunk_obj
                chunk_handles = None
                state_payload = {
                    "status": "submitted",
                    "job_id": job_id,
                    "video_path": staged_video_path,
                    "meta": meta,
                    "chunks": chunks,
                    "asr_call_id": str(asr_handle.object_id),
                    "chunk_mode": "indexed",
                    "chunk_call_id": str(chunk_obj.object_id),
                    "tracking_submitted_ts": tracking_submitted_ts,
                }
            elif isinstance(chunk_obj, (list, tuple)):
                chunk_mode = "list"
                chunk_handle = None
                chunk_handles = list(chunk_obj)
                state_payload = {
                    "status": "submitted",
                    "job_id": job_id,
                    "video_path": staged_video_path,
                    "meta": meta,
                    "chunks": chunks,
                    "asr_call_id": str(asr_handle.object_id),
                    "chunk_mode": "list",
                    "chunk_call_ids": [str(c.object_id) for c in chunk_handles],
                    "tracking_submitted_ts": tracking_submitted_ts,
                }
            else:
                raise RuntimeError("spawn_map returned unknown handle type")
            save_detached_state(state_payload)
        else:
            # Non-detached fallback path.
            asr_task = asyncio.create_task(
                worker.run_asr_only.remote.aio(audio_wav_bytes=audio_bytes)
            )

            sem = asyncio.Semaphore(max_gpu_workers)

            async def _run_one(c: dict):
                async with sem:
                    return await worker.track_chunk_from_staged.remote.aio(
                        job_id=job_id,
                        video_path=staged_video_path,
                        meta=meta,
                        chunk=c,
                    )

            tasks = [_run_one(c) for c in chunks]
            chunk_results: list[dict] = []
            done = 0
            for fut in asyncio.as_completed(tasks):
                chunk_results.append(await fut)
                done += 1
                pct = (100.0 * done) / max(1, len(chunks))
                log.info(f"Chunk fan-out progress: {done}/{len(chunks)} ({pct:.1f}%)")
            words = await asr_task
            tracking_wallclock_s = time.time() - t0
            stitched = await worker.stitch_tracking_chunks.remote.aio(
                chunk_results=chunk_results,
                fps=float(meta.get("fps", 25.0)),
            )
            tracks = list(stitched.get("tracks", []))
            tracking_metrics = dict(stitched.get("tracking_metrics", {}))
            tracking_metrics["tracking_wallclock_s"] = float(tracking_wallclock_s)
            total_frames = int(meta.get("total_frames", 0))
            tracking_metrics["throughput_fps"] = float(total_frames / max(1e-6, tracking_wallclock_s))

            result = await worker.finalize_extraction.remote.aio(
                video_bytes=video_bytes,
                audio_wav_bytes=audio_bytes,
                youtube_url=youtube_url,
                words=words,
                tracks=tracks,
                tracking_metrics=tracking_metrics,
            )
            try:
                await worker.cleanup_tracking_job.remote.aio(job_id=job_id)
            except Exception as e:
                log.warning(f"Could not cleanup distributed job {job_id[:8]}… ({type(e).__name__}: {e})")
            elapsed = time.time() - t0
            log.info(f"Distributed Modal workflow returned in {elapsed:.1f}s")
            return result

    # Detached collection path.
    def _collect_chunks_blocking():
        out = []
        if chunk_mode == "indexed":
            assert chunk_handle is not None
            for i in range(len(chunks)):
                out.append(chunk_handle.get(timeout=None, index=i))
                pct = (100.0 * (i + 1)) / max(1, len(chunks))
                log.info(f"Chunk fan-out progress: {i + 1}/{len(chunks)} ({pct:.1f}%)")
        else:
            assert chunk_handles is not None
            for i, h in enumerate(chunk_handles, start=1):
                out.append(h.get(timeout=None))
                pct = (100.0 * i) / max(1, len(chunk_handles))
                log.info(f"Chunk fan-out progress: {i}/{len(chunk_handles)} ({pct:.1f}%)")
        return out

    chunk_results = await asyncio.to_thread(_collect_chunks_blocking)
    words = await asyncio.to_thread(asr_handle.get, None)
    tracking_wallclock_s = float(time.time() - tracking_submitted_ts)

    stitched = await worker.stitch_tracking_chunks.remote.aio(
        chunk_results=chunk_results,
        fps=float(meta.get("fps", 25.0)),
    )
    tracks = list(stitched.get("tracks", []))
    tracking_metrics = dict(stitched.get("tracking_metrics", {}))
    tracking_metrics["tracking_wallclock_s"] = float(tracking_wallclock_s)
    total_frames = int(meta.get("total_frames", 0))
    tracking_metrics["throughput_fps"] = float(total_frames / max(1e-6, tracking_wallclock_s))

    result = await worker.finalize_extraction.remote.aio(
        video_bytes=video_bytes,
        audio_wav_bytes=audio_bytes,
        youtube_url=youtube_url,
        words=words,
        tracks=tracks,
        tracking_metrics=tracking_metrics,
    )

    # Best-effort staged artifact cleanup.
    try:
        await worker.cleanup_tracking_job.remote.aio(job_id=job_id)
    except Exception as e:
        log.warning(f"Could not cleanup distributed job {job_id[:8]}… ({type(e).__name__}: {e})")

    # Mark detached state completed.
    save_detached_state(
        {
            "status": "completed",
            "job_id": job_id,
            "completed_at": time.time(),
            "tracking_metrics": tracking_metrics,
        }
    )

    elapsed = time.time() - t0
    log.info(f"Distributed Modal workflow returned in {elapsed:.1f}s")
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
    distributed = os.getenv("CLYPT_DISTRIBUTED_MODAL_FANOUT", "1") == "1"
    if distributed:
        result = await call_modal_worker_distributed(video_path, audio_path, url)
    else:
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

    # Step 3A: Upload canonical source video to GCS for downstream Gemini phases.
    video_gcs_uri = upload_video_to_gcs(video_path)

    # Step 3B: Enrich visual ledger with backward-compatible fields and metadata.
    phase_1_visual = enrich_visual_ledger_for_downstream(phase_1_visual, phase_1_audio, video_path)
    phase_1_visual["video_gcs_uri"] = video_gcs_uri
    phase_1_audio["video_gcs_uri"] = video_gcs_uri

    # Step 3C: Contract validation before writing.
    validate_phase_handoff(phase_1_visual, phase_1_audio)

    with open(visual_out, "w") as f:
        json.dump(phase_1_visual, f, indent=2)
    log.info(f"Visual ledger saved → {visual_out}")

    write_visual_ndjson(phase_1_visual, PHASE1_NDJSON_PATH)

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
