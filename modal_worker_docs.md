# modal_worker.py — Documentation

## Overview

`modal_worker.py` is the Clypt Phase 1A serverless GPU microservice. It is deployed to [Modal](https://modal.com) and performs three deterministic multimodal extractions on video+audio content:

| Step | Model | Output |
|------|-------|--------|
| 1 | NVIDIA Parakeet-TDT-1.1B | Word-level ASR with millisecond timestamps |
| 2 | YOLOv11 + BoT-SORT | Dense person bounding boxes with persistent track IDs |
| 3 | TalkNet ASD (+ heuristic fallback) | Speaker-to-word bindings (which person said each word) |

Media (video + audio) is downloaded by the **calling pipeline** on a non-datacenter IP and sent as raw bytes. This avoids YouTube bot detection on Modal's datacenter IPs.

**Deploy:** `modal deploy modal_worker.py`
**Dev mode:** `modal serve modal_worker.py` (hot-reload)

---

## Configuration Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `ASR_MODEL_NAME` | `"nvidia/parakeet-tdt-1.1b"` | NeMo model name for Parakeet |
| `TALKNET_MODEL_PATH` | `/root/.cache/clypt/pretrain_TalkSet.model` | TalkNet checkpoint path in container |
| `TALKNET_REPO_ROOT` | `/root/talknet_asd` | TalkNet source files directory |
| `YOLO_WEIGHTS_PATH` | `yolo11s.pt` | YOLO11s PyTorch weights |
| `YOLO_ENGINE_PATH` | `/root/.cache/clypt/yolo11s.engine` | Optional prebuilt TensorRT engine |

---

## Container Image (`clypt_image`)

Built from `debian:slim` (Python 3.11) with the following layers:

1. **APT packages:** `ffmpeg`, `cmake`, `build-essential`, `libgl1`, `libglib2.0-0`, `libsndfile1`
2. **Core ML packages:** `torch`, `torchvision`, `torchaudio`, `ultralytics`, `insightface`, `onnxruntime-gpu`, `opencv-python-headless`, `decord`, `scikit-learn`, `python_speech_features`, and others
3. **NeMo ASR:** `nemo_toolkit[asr]`
4. **Model weights cached at build time** (see functions below)

### Build-time Download Functions

These run during `modal deploy` and cache weights into the image layer so cold-starts skip downloading.

#### `download_asr_model()`
Downloads the Parakeet-TDT-1.1B NeMo checkpoint via `nemo_asr.models.ASRModel.from_pretrained`.

#### `download_yolo_model()`
Downloads `yolo11s.pt` via the Ultralytics `YOLO` constructor.

#### `download_talknet_model()`
1. Downloads the `pretrain_TalkSet.model` checkpoint from Google Drive via `gdown`.
2. Downloads four TalkNet source files from GitHub (`talkNetModel.py`, `audioEncoder.py`, `visualEncoder.py`, `attentionLayer.py`, `loss.py`) into `TALKNET_REPO_ROOT/model/`.
3. Creates `model/__init__.py` if absent.

#### `download_insightface_model()`
Pre-warms the InsightFace `buffalo_l` model pack using CPU execution so ONNX Runtime downloads the weights. Failures are non-fatal (prints a warning).

---

## `ClyptWorker` Class

Decorated with `@app.cls(image=clypt_image, gpu="H100", timeout=1800)`.
Uses class-based Modal container lifecycle so all three models stay in GPU VRAM between invocations on the same warm container.

### Model Debug Flags

Enabled via the `MODEL_DEBUG_SECRET` environment variables:

| Env var | Effect |
|---------|--------|
| `CLYPT_MODEL_DEBUG=1` | Enable verbose TalkNet tensor logging |
| `CLYPT_MODEL_DEBUG_EVERY=N` | Print debug stats every N TalkNet forward passes (default 20) |

---

### Lifecycle

#### `load_model()` — `@modal.enter()`
Runs once per container cold start. Loads all models into GPU VRAM:

1. **Parakeet-TDT:** loads via NeMo, enables `preserve_alignments` and `compute_timestamps`, computes `self.time_stride` for timestamp conversion.
2. **YOLO11s:** loads PyTorch weights; if a prebuilt TensorRT engine exists at `YOLO_ENGINE_PATH`, loads that instead (no runtime export).
3. **InsightFace buffalo_l:** initialised with `det_thresh=0.15` for profile-face recall; uses CUDA if available, falls back gracefully.
4. **TalkNet:** loads `talkNetModel` + `lossAV` from `TALKNET_REPO_ROOT`, calls `_load_talknet_checkpoint`, moves to GPU. Failure is non-fatal — `self.talknet_model` is set to `None` and the binding step falls back to the heuristic.

---

### Private Helpers

#### `_load_talknet_checkpoint(model, loss_av, ckpt_path)`
Loads a TalkNet `.model` file robustly:
- Handles `module.*`, `model.*`, and flat key prefixes.
- Splits state dict into `model_state` and `loss_state` and loads each separately with `strict=False`.
- Raises `RuntimeError` if there are missing or unexpected keys after loading.

#### `_build_track_indexes(tracks) → (frame_to_dets, track_to_dets)`
Builds two secondary indexes from a flat list of detection dicts:
- `frame_to_dets`: `{frame_idx → [det, ...]}` — all detections in a given frame.
- `track_to_dets`: `{track_id → [det, ...]}` — all detections for a given person, sorted by frame.

#### `_detect_face_in_person_det(frame_rgb, det) → (face_crop_112, anchor)`
Given an RGB frame and a YOLO person detection, runs InsightFace on a broad head-focused ROI (upper 82% / lower 30% of bbox height) and returns:
- A 112×112 RGB face crop (best face by score × area).
- An `anchor` dict with relative offsets (`x_offset`, `y_offset`, `w_ratio`, `h_ratio`) for projecting the face position onto future frames without re-running the detector.

Returns `(None, None)` if InsightFace is not loaded, the ROI is empty, or no face is found.

#### `_interpolate_track_detections(dets, max_gap=5) → {frame_idx: det}`
Fills short detection gaps (≤ `max_gap` frames) via linear interpolation of bounding box coordinates. Gaps larger than `max_gap` are left empty. Returns a dict keyed by frame index.

#### `_tensor_debug_stats(name, tensor) → str`
Returns a compact diagnostic string for a PyTorch tensor: shape, finite element count, min/max/mean/std. Used by the model debug logging path.

#### `_talknet_forward_scores(audio_t, visual_t) → Tensor[B, T]`
Batched TalkNet forward pass returning per-frame speaking probabilities in `[0, 1]`:
1. Encodes audio and visual streams through their respective frontends.
2. Applies cross-attention between the two modalities.
3. Passes the fused representation through the AV backend.
4. Applies softmax over the `lossAV.FC` head and returns the positive (speaking) class probability.

Optionally logs tensor statistics every N calls when model debug is enabled.

---

### ASR

#### `_run_asr(audio_wav_path) → list[dict]`
Runs Parakeet-TDT on a 16 kHz mono WAV file.

**Returns:** list of word dicts:
```python
{
    "word": str,
    "start_time_ms": int,
    "end_time_ms": int,
    "speaker_track_id": None,  # populated later by TalkNet
}
```

Handles both the legacy `timestep` and newer `timestamp` NeMo attribute names. Timestamps are converted from NeMo frame offsets using `self.time_stride`.

---

### Visual Tracking

#### `_ensure_h264(video_path) → str`
Checks the video codec via `ffprobe`. If it is `av1`, `vp9`, `hevc`, or `h265` (which OpenCV often cannot decode), re-encodes to H.264:
- Tries NVENC (`h264_nvenc`) first.
- Falls back to software `libx264` if NVENC is unavailable.
Returns the original path if no re-encoding is needed.

#### `_run_tracking(video_path) → list[dict]`
Runs `YOLO11s.track(tracker="botsort.yaml", classes=[0], stream=True)` for dense person tracking.

Calls `_ensure_h264` first. Streams results frame-by-frame (avoids loading the full video into RAM).

**Returns:** list of detection dicts:
```python
{
    "frame_idx": int,          # 0-based
    "track_id": str,           # e.g. "track_7"
    "x_center": float,         # pixel coords
    "y_center": float,
    "width": float,
    "height": float,
    "confidence": float,
}
```

---

### Tracklet Clustering

#### `_cluster_tracklets(video_path, tracks, track_to_dets=None) → list[dict]`
Merges fragmented BoT-SORT track IDs into stable global person IDs (e.g. `Global_Person_0`).

**Algorithm:**
1. **Frame sampling:** For each track, rank detections by confidence × area and sample up to 6 frames.
2. **Batch frame decode:** Loads all needed frames at once with `decord.VideoReader`.
3. **Embedding extraction:** For each sampled frame, runs InsightFace on a head-focused crop to get a 512-D ArcFace embedding. If InsightFace fails or the face quality is too low (det_score < 0.35, side < 36px, relative area < 3.5%), falls back to a 512-D normalized RGB histogram.
4. **DBSCAN clustering:** Clusters face embeddings with cosine distance (`eps=0.44`). Histogram-fallback tracks are kept separate.
5. **Noise reassignment:** Noise points (label = -1) are assigned to the nearest centroid within cosine distance 0.45.
6. **Tiny cluster merge:** Clusters below a size threshold are merged into the nearest stable cluster.
7. **Cross-cluster deduplication:** Iteratively merges clusters that are embedding-close and rarely co-visible in the same frames (overlap < 8%).
8. **Adaptive merge:** If the cluster count still exceeds the data-driven target, continues merging the least-distant non-overlapping pair (guarded by cos < 0.58 and spatial sig < 3.2).
9. **Histogram attachment:** Histogram-only tracks are assigned to the nearest face cluster by spatial signature (median bbox position + size).
10. **Renumber:** Final IDs are renumbered to contiguous `Global_Person_0, 1, 2, ...`.

---

### Speaker Binding

#### `_run_talknet_binding(...) → list[dict] | None`
Runs full TalkNet Active Speaker Detection to map each word to a visual track.

**Pipeline per track:**
1. Gap-fills detections with `_interpolate_track_detections`.
2. Splits the track's frame sequence into contiguous runs (gap ≤ 4 frames).
3. Slices each run into 120-frame chunks; skips chunks that have no temporal overlap with any word.
4. For each frame in a chunk: extracts a 112×112 face crop via `_detect_face_in_person_det`. If the detector misses, projects the last known face anchor to fill up to 5 frames; beyond 5 consecutive misses, the subchunk is terminated.
5. Computes 13-coefficient MFCC features (window/step scaled by fps) for the audio segment aligned to the visual chunk.
6. Batches chunks of the same temporal length into groups of 8 and runs `_talknet_forward_scores`.
7. Stores per-frame ASD scores in `asd_scores[(track_id, frame_idx)]`.

**Word assignment:**
- Finds the nearest tracked frame to each word's midpoint (within 4 frames).
- Scores each candidate track by `0.995 × talknet_prob + 0.005 × detection_conf`.
- Assigns the word to the winning track if its probability ≥ 0.15 and the margin over the runner-up ≥ `0.18 × (p90 − p50)` of the global score distribution.

**Smoothing:** Applies a ±2-word majority-vote window to suppress single-word speaker flicker.

**Fallback condition:** Returns `None` (triggering heuristic fallback) if:
- TalkNet is not loaded.
- Assignment ratio < 15%.
- Fewer than 2 tracks were scored.

**Returns:** list of speaker segment dicts:
```python
{
    "track_id": str,
    "start_time_ms": int,
    "end_time_ms": int,
    "word_count": int,
}
```

#### `_run_speaker_binding_heuristic(video_path, tracks, words, ...) → list[dict]`
Lightweight fallback when TalkNet is unavailable or low-confidence.

Scores each candidate speaker per word using:
```
score = 0.60 × motion + 0.25 × detection_conf + 0.15 × normalized_area
```
Where `motion` is the normalized bbox displacement between the previous and next frame. If two candidates are within 0.08 of each other, tie-breaks using lip-landmark openness (`face_recognition` library, optional). Applies the same ±2-word smoothing and produces the same output format as TalkNet binding.

#### `_run_speaker_binding(video_path, audio_wav_path, tracks, words, ...) → list[dict]`
Orchestration entrypoint: tries TalkNet first; if it returns `None`, runs the heuristic.

---

### Main Entry Point

#### `extract(video_bytes, audio_wav_bytes, youtube_url) → dict` — `@modal.method()`

**Inputs:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `video_bytes` | `bytes` | Muxed MP4 video file |
| `audio_wav_bytes` | `bytes` | 16 kHz mono WAV file |
| `youtube_url` | `str` | Original URL (metadata only) |

**Execution flow:**
1. Writes bytes to `/tmp/clypt/video.mp4` and `/tmp/clypt/audio_16k.wav`.
2. **Step 1+2 (concurrent):** Runs `_run_asr` and `_run_tracking` in a `ThreadPoolExecutor(max_workers=2)`.
3. **Step 3:** Calls `_cluster_tracklets` to merge fragmented track IDs.
4. **Step 4:** Calls `_run_speaker_binding` to assign speakers to words.
5. Cleans up temp files (including any `_h264.mp4` re-encode).

**Success response:**
```python
{
    "status": "success",
    "phase_1a_visual": {
        "source_video": youtube_url,
        "tracks": list[dict],        # per-frame person detections with global IDs
    },
    "phase_1a_audio": {
        "source_audio": youtube_url,
        "words": list[dict],         # word-level ASR with speaker_track_id populated
        "speaker_bindings": list[dict],  # speaker segments timeline
    },
}
```

**Error response:**
```python
{"status": "error", "message": str}
```

---

## Data Schemas

### Detection dict (in `tracks`)
```python
{
    "frame_idx": int,
    "track_id": str,          # e.g. "Global_Person_0" after clustering
    "x_center": float,        # pixels
    "y_center": float,
    "width": float,
    "height": float,
    "confidence": float,
    "interpolated": bool,     # True if gap-filled, absent otherwise
}
```

### Word dict (in `words`)
```python
{
    "word": str,
    "start_time_ms": int,
    "end_time_ms": int,
    "speaker_track_id": str | None,   # e.g. "Global_Person_1"
    "speaker_tag": str,               # same as speaker_track_id or "unknown"
}
```

### Speaker binding dict (in `speaker_bindings`)
```python
{
    "track_id": str,
    "start_time_ms": int,
    "end_time_ms": int,
    "word_count": int,
}
```
