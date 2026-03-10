#!/usr/bin/env python3
"""
Phase 1A: Deterministic Extraction
===================================
Downloads a YouTube video's separate video/audio streams, uploads them to GCS,
then runs the Visual Engine (Video Intelligence API) and Audio Engine
(Speech-to-Text v2 / Chirp 3) in parallel via asyncio.

Outputs:
  - phase_1a_visual.json  (shot changes, face/person detections, object tracking, label detections)
  - phase_1a_audio.json   (word-level timestamps with speaker diarization)
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import yt_dlp
from google.api_core.client_options import ClientOptions
from google.cloud import storage
from google.cloud import videointelligence_v1 as videointelligence
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
PROJECT_ID = "clypt-preyc"
GCS_BUCKET = "clypt-test-bucket"
YOUTUBE_URL = "https://www.youtube.com/watch?v=KHEZCXfyxjU"
STT_REGION = "us"  # Chirp 3 available in "us" region
STT_MIN_SPEAKERS = int(os.getenv("STT_MIN_SPEAKERS", "2"))
STT_MAX_SPEAKERS = int(os.getenv("STT_MAX_SPEAKERS", "3"))
if STT_MIN_SPEAKERS > STT_MAX_SPEAKERS:
    STT_MIN_SPEAKERS, STT_MAX_SPEAKERS = STT_MAX_SPEAKERS, STT_MIN_SPEAKERS
YTDLP_VIDEO_FORMAT = os.getenv(
    "YTDLP_VIDEO_FORMAT",
    "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
)
YTDLP_MIN_LONG_EDGE = int(os.getenv("YTDLP_MIN_LONG_EDGE", "1080"))
ALLOW_LOW_RES_VIDEO = os.getenv("ALLOW_LOW_RES_VIDEO", "0") == "1"
ENABLE_VIDEO_PERSON_DETECTION = os.getenv("ENABLE_VIDEO_PERSON_DETECTION", "0") == "1"
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
log = logging.getLogger("phase_1a")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def time_offset_to_ms(offset) -> int:
    """Convert a protobuf Duration or Python timedelta to milliseconds."""
    if offset is None:
        return 0
    # Python timedelta (some google-cloud libs return this)
    if hasattr(offset, "total_seconds"):
        return int(offset.total_seconds() * 1000)
    # Protobuf Duration (seconds + nanos)
    seconds = getattr(offset, "seconds", 0) or 0
    nanos = getattr(offset, "nanos", 0) or 0
    return int(seconds * 1000 + nanos / 1_000_000)


def bbox_to_dict(box) -> dict:
    """Convert a NormalizedBoundingBox to a serializable dict."""
    return {
        "left": round(float(box.left), 4),
        "top": round(float(box.top), 4),
        "right": round(float(box.right), 4),
        "bottom": round(float(box.bottom), 4),
    }


def probe_video_stream(video_path: str) -> tuple[int, int, str]:
    """Return (width, height, fps_str) for a local video, or (0,0,'?')."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate",
                "-of",
                "json",
                video_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(result.stdout or "{}")
        streams = data.get("streams") or []
        if not streams:
            return 0, 0, "?"
        s = streams[0]
        return int(s.get("width", 0) or 0), int(s.get("height", 0) or 0), str(s.get("avg_frame_rate", "?"))
    except Exception:
        return 0, 0, "?"


# ══════════════════════════════════════════════
# TASK 1 — Media Acquisition (yt-dlp + GCS)
# ══════════════════════════════════════════════
def download_media(url: str) -> tuple[str, str]:
    """Download a muxed video+audio file and a separate audio-only stream.

    The video file includes an audio track (needed for Gemini 3.1 Pro vocal
    prosody analysis in Phase 1B). The audio-only file is used by the STT v2
    engine.
    """
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Clean previous run's files so yt-dlp doesn't skip and downstream
    # phases don't read stale data from a prior run
    for stale in ("video.mp4", "video_original.mp4", "audio.m4a", "audio.webm", "audio.opus"):
        p = DOWNLOAD_DIR / stale
        if p.exists():
            p.unlink()
            log.info(f"Removed stale downloads/{stale}")
    for stale in ("phase_1a_visual.json", "phase_1a_audio.json",
                   "phase_1b_nodes.json", "phase_1c_narrative_edges.json",
                   "phase_2_embeddings.json", "remotion_payloads_array.json", "remotion_payload.json"):
        p = OUTPUT_DIR / stale
        if p.exists():
            p.unlink()
            log.info(f"Removed stale outputs/{stale}")

    # ── Video + Audio (muxed) ──
    log.info("Downloading video+audio stream…")
    log.info(f"  yt-dlp format: {YTDLP_VIDEO_FORMAT}")
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
    w, h, fps = probe_video_stream(video_path)
    long_edge = max(w, h)
    log.info(
        f"Video saved: {video_path} ({Path(video_path).stat().st_size / 1e6:.1f} MB, "
        f"{w}x{h}, fps={fps})"
    )
    if not ALLOW_LOW_RES_VIDEO and long_edge < YTDLP_MIN_LONG_EDGE:
        raise RuntimeError(
            "Downloaded video resolution is too low for high-quality 9:16 reframing: "
            f"{w}x{h}. Set YTDLP_VIDEO_FORMAT to a higher-quality selector and retry. "
            f"If you intentionally want this quality, set ALLOW_LOW_RES_VIDEO=1 "
            f"(current threshold YTDLP_MIN_LONG_EDGE={YTDLP_MIN_LONG_EDGE})."
        )

    # ── Audio-only ──
    log.info("Downloading audio-only stream…")
    audio_path = str(DOWNLOAD_DIR / "audio.m4a")
    try:
        audio_opts = {
            "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio",
            "outtmpl": str(DOWNLOAD_DIR / "audio.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noprogress": False,
            "postprocessors": [],
        }
        with yt_dlp.YoutubeDL(audio_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_path = ydl.prepare_filename(info)
    except Exception:
        log.warning("No standalone audio stream available — extracting from muxed video via ffmpeg")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn",
             "-acodec", "aac", "-b:a", "192k", audio_path],
            check=True, capture_output=True,
        )
    log.info(f"Audio saved: {audio_path} ({Path(audio_path).stat().st_size / 1e6:.1f} MB)")

    return video_path, audio_path


def upload_to_gcs(video_path: str, audio_path: str) -> tuple[str, str]:
    """Upload video and audio files to GCS. Returns (video_uri, audio_uri)."""
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET)

    video_blob_name = f"phase_1a/{Path(video_path).name}"
    audio_blob_name = f"phase_1a/{Path(audio_path).name}"

    log.info(f"Uploading video → gs://{GCS_BUCKET}/{video_blob_name}")
    bucket.blob(video_blob_name).upload_from_filename(video_path)
    video_uri = f"gs://{GCS_BUCKET}/{video_blob_name}"
    log.info(f"  ✓ {video_uri}")

    log.info(f"Uploading audio → gs://{GCS_BUCKET}/{audio_blob_name}")
    bucket.blob(audio_blob_name).upload_from_filename(audio_path)
    audio_uri = f"gs://{GCS_BUCKET}/{audio_blob_name}"
    log.info(f"  ✓ {audio_uri}")

    return video_uri, audio_uri


# ══════════════════════════════════════════════
# TASK 2 — Visual Engine (Video Intelligence API)
# ══════════════════════════════════════════════
def _poll_video_lro(operation, label: str):
    """Poll a Video Intelligence LRO until done, with progress logging."""
    while not operation.done():
        time.sleep(15)
        try:
            meta = operation.metadata
            if meta and hasattr(meta, "annotation_progress"):
                for p in meta.annotation_progress:
                    if hasattr(p, "progress_percent"):
                        log.info(f"[Visual:{label}] Progress: {p.progress_percent}%")
                        break
                else:
                    log.info(f"[Visual:{label}] Processing…")
            else:
                log.info(f"[Visual:{label}] Processing…")
        except Exception:
            log.info(f"[Visual:{label}] Processing…")
    log.info(f"[Visual:{label}] LRO complete")
    return operation.result()


def _run_visual_engine(video_gcs_uri: str) -> dict:
    """Analyze video via Video Intelligence API.

    Features are split into separate API calls to avoid 'Calculator failure'
    errors that occur when too many features are requested together.
    """
    client = videointelligence.VideoIntelligenceServiceClient()

    # Define each feature call separately for reliability
    feature_calls = [
        {
            "label": "SHOT",
            "features": [videointelligence.Feature.SHOT_CHANGE_DETECTION],
            "video_context": None,
        },
        {
            "label": "FACE",
            "features": [videointelligence.Feature.FACE_DETECTION],
            "video_context": videointelligence.VideoContext(
                face_detection_config=videointelligence.FaceDetectionConfig(
                    include_bounding_boxes=True,
                    include_attributes=True,
                ),
            ),
        },
        {
            "label": "OBJECT",
            "features": [videointelligence.Feature.OBJECT_TRACKING],
            "video_context": None,
        },
        {
            "label": "LABEL",
            "features": [videointelligence.Feature.LABEL_DETECTION],
            "video_context": videointelligence.VideoContext(
                label_detection_config=videointelligence.LabelDetectionConfig(
                    label_detection_mode=videointelligence.LabelDetectionMode.SHOT_AND_FRAME_MODE,
                ),
            ),
        },
    ]
    if ENABLE_VIDEO_PERSON_DETECTION:
        feature_calls.insert(
            2,
            {
                "label": "PERSON",
                "features": [videointelligence.Feature.PERSON_DETECTION],
                "video_context": videointelligence.VideoContext(
                    person_detection_config=videointelligence.PersonDetectionConfig(
                        include_bounding_boxes=True,
                        include_attributes=True,
                        include_pose_landmarks=False,
                    ),
                ),
            },
        )
    else:
        log.info("[Visual] PERSON_DETECTION disabled (ENABLE_VIDEO_PERSON_DETECTION=0)")

    # ── Launch all LROs concurrently ──
    operations = []
    for call in feature_calls:
        request = {
            "features": call["features"],
            "input_uri": video_gcs_uri,
        }
        if call["video_context"]:
            request["video_context"] = call["video_context"]

        log.info(f"[Visual:{call['label']}] Submitting request → {video_gcs_uri}")
        op = client.annotate_video(request=request)
        log.info(f"[Visual:{call['label']}] LRO: {op.operation.name}")
        operations.append((call["label"], op))

    # ── Poll all LROs (sequentially — they run server-side in parallel) ──
    all_annotations = []
    errors = []
    for label, op in operations:
        result = _poll_video_lro(op, label)
        for i, ar in enumerate(result.annotation_results):
            if ar.error and ar.error.message:
                log.error(f"[Visual:{label}] annotation_results[{i}] ERROR: "
                          f"code={ar.error.code} msg={ar.error.message}")
                # Log full error details if available
                if hasattr(ar.error, "details"):
                    for detail in ar.error.details:
                        log.error(f"[Visual:{label}]   detail: {detail}")
                errors.append(f"{label}: {ar.error.message}")
            all_annotations.append(ar)

    if errors:
        log.warning(f"[Visual] {len(errors)} feature(s) had errors: {errors}")

    # ── Merge results across all annotation_results ──
    ledger = {
        "source_video_gcs_uri": video_gcs_uri,
        "shot_changes": [],
        "person_detections": [],
        "face_detections": [],
        "object_tracking": [],
        "label_detections": [],
        "errors": errors,
    }

    for ar in all_annotations:
        # Shot changes
        for shot in ar.shot_annotations:
            ledger["shot_changes"].append({
                "start_time_ms": time_offset_to_ms(shot.start_time_offset),
                "end_time_ms": time_offset_to_ms(shot.end_time_offset),
            })

        # Person detections
        for ann in ar.person_detection_annotations:
            for track in ann.tracks:
                person = {
                    "confidence": round(float(track.confidence), 4),
                    "segment_start_ms": time_offset_to_ms(track.segment.start_time_offset),
                    "segment_end_ms": time_offset_to_ms(track.segment.end_time_offset),
                    "timestamped_objects": [],
                }
                for ts_obj in track.timestamped_objects:
                    obj_data = {
                        "time_ms": time_offset_to_ms(ts_obj.time_offset),
                        "bounding_box": bbox_to_dict(ts_obj.normalized_bounding_box),
                    }
                    if ts_obj.attributes:
                        obj_data["attributes"] = [
                            {
                                "name": a.name,
                                "value": a.value,
                                "confidence": round(float(a.confidence), 4),
                            }
                            for a in ts_obj.attributes
                        ]
                    person["timestamped_objects"].append(obj_data)
                ledger["person_detections"].append(person)

        # Face detections
        for ann in ar.face_detection_annotations:
            for track in ann.tracks:
                face = {
                    "confidence": round(float(track.confidence), 4),
                    "segment_start_ms": time_offset_to_ms(track.segment.start_time_offset),
                    "segment_end_ms": time_offset_to_ms(track.segment.end_time_offset),
                    "timestamped_objects": [],
                }
                for ts_obj in track.timestamped_objects:
                    obj_data = {
                        "time_ms": time_offset_to_ms(ts_obj.time_offset),
                        "bounding_box": bbox_to_dict(ts_obj.normalized_bounding_box),
                    }
                    if ts_obj.attributes:
                        obj_data["attributes"] = [
                            {
                                "name": a.name,
                                "value": a.value,
                                "confidence": round(float(a.confidence), 4),
                            }
                            for a in ts_obj.attributes
                        ]
                    face["timestamped_objects"].append(obj_data)
                ledger["face_detections"].append(face)

        # Object tracking
        for ann in ar.object_annotations:
            obj_entry = {
                "entity": {
                    "entity_id": ann.entity.entity_id or "",
                    "description": ann.entity.description or "",
                },
                "confidence": round(float(ann.confidence), 4),
                "segment_start_ms": time_offset_to_ms(ann.segment.start_time_offset),
                "segment_end_ms": time_offset_to_ms(ann.segment.end_time_offset),
                "frames": [
                    {
                        "time_ms": time_offset_to_ms(frame.time_offset),
                        "bounding_box": bbox_to_dict(frame.normalized_bounding_box),
                    }
                    for frame in ann.frames
                ],
            }
            ledger["object_tracking"].append(obj_entry)

        # Label detections (shot-level, segment-level, frame-level)
        for ann in ar.shot_label_annotations:
            label_entry = {
                "entity": {
                    "entity_id": ann.entity.entity_id or "",
                    "description": ann.entity.description or "",
                },
                "category_entities": [
                    {
                        "entity_id": ce.entity_id or "",
                        "description": ce.description or "",
                    }
                    for ce in ann.category_entities
                ],
                "level": "shot",
                "segments": [
                    {
                        "start_time_ms": time_offset_to_ms(seg.segment.start_time_offset),
                        "end_time_ms": time_offset_to_ms(seg.segment.end_time_offset),
                        "confidence": round(float(seg.confidence), 4),
                    }
                    for seg in ann.segments
                ],
            }
            ledger["label_detections"].append(label_entry)

        for ann in ar.segment_label_annotations:
            label_entry = {
                "entity": {
                    "entity_id": ann.entity.entity_id or "",
                    "description": ann.entity.description or "",
                },
                "category_entities": [
                    {
                        "entity_id": ce.entity_id or "",
                        "description": ce.description or "",
                    }
                    for ce in ann.category_entities
                ],
                "level": "segment",
                "segments": [
                    {
                        "start_time_ms": time_offset_to_ms(seg.segment.start_time_offset),
                        "end_time_ms": time_offset_to_ms(seg.segment.end_time_offset),
                        "confidence": round(float(seg.confidence), 4),
                    }
                    for seg in ann.segments
                ],
            }
            ledger["label_detections"].append(label_entry)

        for ann in ar.frame_label_annotations:
            label_entry = {
                "entity": {
                    "entity_id": ann.entity.entity_id or "",
                    "description": ann.entity.description or "",
                },
                "category_entities": [
                    {
                        "entity_id": ce.entity_id or "",
                        "description": ce.description or "",
                    }
                    for ce in ann.category_entities
                ],
                "level": "frame",
                "frames": [
                    {
                        "time_ms": time_offset_to_ms(frame.time_offset),
                        "confidence": round(float(frame.confidence), 4),
                    }
                    for frame in ann.frames
                ],
            }
            ledger["label_detections"].append(label_entry)

    log.info(f"[Visual] Shot changes: {len(ledger['shot_changes'])}")
    log.info(f"[Visual] Person tracks: {len(ledger['person_detections'])}")
    log.info(f"[Visual] Face tracks: {len(ledger['face_detections'])}")
    log.info(f"[Visual] Tracked objects: {len(ledger['object_tracking'])}")
    log.info(f"[Visual] Label detections: {len(ledger['label_detections'])}")

    return ledger


async def run_visual_engine(video_gcs_uri: str) -> dict:
    """Async wrapper — offloads the blocking Visual Engine to a thread."""
    return await asyncio.to_thread(_run_visual_engine, video_gcs_uri)


# ══════════════════════════════════════════════
# TASK 3 — Audio Engine (STT v2 / Chirp 3)
# ══════════════════════════════════════════════
def _run_audio_engine(audio_gcs_uri: str) -> dict:
    """Analyze audio via Speech-to-Text v2 with Chirp 3. Blocking call."""
    client = SpeechClient(
        client_options=ClientOptions(
            api_endpoint=f"{STT_REGION}-speech.googleapis.com",
        )
    )

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["en-US"],
        model="chirp_3",
        features=cloud_speech.RecognitionFeatures(
            enable_word_time_offsets=True,
            # Note: enable_word_confidence is NOT supported by Chirp 3
            diarization_config=cloud_speech.SpeakerDiarizationConfig(
                min_speaker_count=STT_MIN_SPEAKERS,
                max_speaker_count=STT_MAX_SPEAKERS,
            ),
        ),
    )

    file_metadata = cloud_speech.BatchRecognizeFileMetadata(uri=audio_gcs_uri)

    request = cloud_speech.BatchRecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/{STT_REGION}/recognizers/_",
        config=config,
        files=[file_metadata],
        recognition_output_config=cloud_speech.RecognitionOutputConfig(
            inline_response_config=cloud_speech.InlineOutputConfig(),
        ),
    )

    log.info("[Audio] Submitting STT v2 batch request…")
    log.info(f"  Model: chirp_3 | Region: {STT_REGION}")
    log.info(f"  Input: {audio_gcs_uri}")
    log.info(
        "  Word time offsets: enabled | "
        f"Diarization: enabled ({STT_MIN_SPEAKERS}-{STT_MAX_SPEAKERS} speakers)"
    )

    operation = client.batch_recognize(request=request)
    log.info(f"[Audio] LRO created: {operation.operation.name}")

    # ── Poll for progress ──
    while not operation.done():
        time.sleep(10)
        try:
            meta = operation.metadata
            if meta and hasattr(meta, "progress_percent"):
                log.info(f"[Audio] Progress: {meta.progress_percent}%")
            else:
                log.info("[Audio] Processing…")
        except Exception:
            log.info("[Audio] Processing…")

    log.info("[Audio] LRO complete — parsing results")
    response = operation.result()

    ledger = {
        "source_audio_gcs_uri": audio_gcs_uri,
        "words": [],
        "transcript_segments": [],
    }

    # Navigate the response: results keyed by URI
    file_result = response.results[audio_gcs_uri]

    # The transcript may be under .transcript (Python client) or .inline_result
    transcript_obj = getattr(file_result, "transcript", None)
    if transcript_obj is None:
        transcript_obj = getattr(file_result, "inline_result", None)
    if transcript_obj is None:
        log.error("[Audio] Could not locate transcript in response. Keys available: "
                  f"{[f.name for f in file_result._meta.fields.values()]}")
        return ledger

    for result in transcript_obj.results:
        if not result.alternatives:
            continue
        best = result.alternatives[0]

        # Full segment transcript
        ledger["transcript_segments"].append({
            "transcript": best.transcript,
            "confidence": round(float(best.confidence), 4) if best.confidence else None,
            "language_code": result.language_code or None,
        })

        # Word-level detail
        for w in best.words:
            start_ms = time_offset_to_ms(w.start_offset)
            end_ms = time_offset_to_ms(w.end_offset)
            # v2 uses speaker_label (string); fall back to speaker_tag if present
            speaker = getattr(w, "speaker_label", "") or getattr(w, "speaker_tag", "")
            ledger["words"].append({
                "word": w.word,
                "start_time_ms": start_ms,
                "end_time_ms": end_ms,
                "speaker_tag": str(speaker) if speaker else "unknown",
                "confidence": round(float(w.confidence), 4) if w.confidence else None,
            })

    log.info(f"[Audio] Words: {len(ledger['words'])} | Segments: {len(ledger['transcript_segments'])}")

    return ledger


async def run_audio_engine(audio_gcs_uri: str) -> dict:
    """Async wrapper — offloads the blocking Audio Engine to a thread."""
    return await asyncio.to_thread(_run_audio_engine, audio_gcs_uri)


# ══════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════
async def main(youtube_url: str | None = None):
    url = youtube_url or YOUTUBE_URL
    log.info("=" * 60)
    log.info("PHASE 1A — Deterministic Extraction")
    log.info("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Step 1: Download media ──
    log.info("── Task 1: Media Acquisition (yt-dlp) ──")
    video_path, audio_path = download_media(url)

    # ── Step 2: Upload to GCS ──
    log.info("── Uploading to GCS ──")
    video_gcs_uri, audio_gcs_uri = upload_to_gcs(video_path, audio_path)

    # ── Step 3: Run Visual + Audio engines in parallel ──
    log.info("── Tasks 2 & 3: Visual + Audio Engines (parallel) ──")
    t0 = time.time()

    results = await asyncio.gather(
        run_visual_engine(video_gcs_uri),
        run_audio_engine(audio_gcs_uri),
        return_exceptions=True,
    )

    elapsed = time.time() - t0
    log.info(f"Both engines finished in {elapsed:.1f}s")

    visual_result, audio_result = results

    # ── Step 4: Handle results independently ──
    visual_out = OUTPUT_DIR / "phase_1a_visual.json"
    audio_out = OUTPUT_DIR / "phase_1a_audio.json"
    any_failed = False

    if isinstance(visual_result, BaseException):
        log.error(f"[Visual] ENGINE FAILED: {visual_result}")
        any_failed = True
    else:
        with open(visual_out, "w") as f:
            json.dump(visual_result, f, indent=2)
        log.info(f"Visual ledger saved → {visual_out}")

    if isinstance(audio_result, BaseException):
        log.error(f"[Audio] ENGINE FAILED: {audio_result}")
        any_failed = True
    else:
        with open(audio_out, "w") as f:
            json.dump(audio_result, f, indent=2)
        log.info(f"Audio ledger saved → {audio_out}")

    # ── Summary ──
    log.info("=" * 60)
    if any_failed:
        log.warning("PHASE 1A PARTIAL — one or more engines failed (see errors above)")
    else:
        log.info("PHASE 1A COMPLETE")

    if not isinstance(visual_result, BaseException):
        log.info(f"  Shot changes:      {len(visual_result['shot_changes'])}")
        log.info(f"  Person tracks:     {len(visual_result['person_detections'])}")
        log.info(f"  Face tracks:       {len(visual_result['face_detections'])}")
        log.info(f"  Tracked objects:   {len(visual_result['object_tracking'])}")
        log.info(f"  Label detections:  {len(visual_result['label_detections'])}")
    if not isinstance(audio_result, BaseException):
        log.info(f"  Words transcribed: {len(audio_result['words'])}")
        log.info(f"  Transcript segs:   {len(audio_result['transcript_segments'])}")
    log.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
