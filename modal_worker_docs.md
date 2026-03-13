# modal_worker.py Documentation

## Purpose

`modal_worker.py` is the Phase 1 GPU service for Clypt.  
It accepts pre-downloaded media bytes and returns:

- word-level ASR (`phase_1_audio.words`)
- person tracking (`phase_1_visual.tracks`)
- speaker-word bindings (`phase_1_audio.speaker_bindings`)
- tracking quality/runtime metrics (`phase_1_visual.tracking_metrics`)

This worker is designed to run either:

- as one end-to-end call (`extract`)
- or as distributed fan-out primitives (`stage_video_for_tracking`, `track_chunk_from_staged`, `stitch_tracking_chunks`, `finalize_extraction`)

---

## Current Stack

### Models

- ASR: `nvidia/parakeet-tdt-1.1b`
- Detector/tracker: `YOLO26s + BoT-SORT (ReID + GMC)`
- Face ID for clustering: InsightFace (`buffalo_l`, ArcFace embeddings)
- Active speaker detection: LR-ASD

### Runtime

`@app.cls(...)` config:

- `gpu="H100"`
- `timeout=3600`
- `max_containers=8`
- `min_containers=0`
- `scaledown_window=900`
- `enable_memory_snapshot=False`
- shared volume mount: `/vol/clypt-chunks`

---

## Build-Time Caching

During image build, the worker caches:

- Parakeet weights
- YOLO26s weights
- best-effort YOLO ONNX export and TensorRT/OpenVINO artifacts
- LR-ASD checkpoint + source files
- InsightFace model pack

This reduces cold-start time and avoids runtime model downloads.

---

## Public Remote Methods

## 1) End-to-end path

### `extract(video_bytes, audio_wav_bytes, youtube_url) -> dict`

Runs full Phase 1:

1. ASR + tracking concurrently
2. global tracklet clustering
3. speaker binding (LR-ASD, heuristic fallback)
4. contract validation + rollout gates

Response includes both modern and compatibility keys:

- `phase_1_visual`, `phase_1_audio`
- `phase_1a_visual`, `phase_1a_audio` (compat)

---

## 2) Distributed fan-out path

### `stage_video_for_tracking(video_bytes) -> manifest`

- writes video to volume
- ensures decodable codec (H.264 if needed)
- probes metadata
- builds overlapping chunk plan
- returns manifest:
  - `job_id`
  - `video_path`
  - `meta`
  - `chunks`

### `run_asr_only(audio_wav_bytes) -> words`

- ASR-only helper for distributed orchestration

### `track_chunk_from_staged(job_id, video_path, meta, chunk) -> chunk_result`

- tracks one chunk on GPU
- emits per-chunk NDJSON + embeddings

### `stitch_tracking_chunks(chunk_results, fps) -> {tracks, tracking_metrics}`

- merges chunk-local IDs to global IDs on overlap windows

### `finalize_extraction(video_bytes, audio_wav_bytes, youtube_url, words, tracks, tracking_metrics) -> dict`

- runs clustering + speaker binding + final packaging

### `cleanup_tracking_job(job_id)`

- removes staged volume artifacts for that distributed job

---

## Tracking Pipeline (Phase 1 Step 2)

## 1) Codec normalization

`_ensure_h264` checks codec via `ffprobe`.
If source is AV1/VP9/HEVC, it re-encodes to H.264 (`h264_nvenc` preferred, `libx264` fallback).

## 2) Chunk plan

`_build_chunk_plan` uses:

- chunk size: `60s`
- overlap: `2s`

This enables parallel chunk tracking and overlap stitching.

## 3) Per-chunk tracking

`_track_single_chunk`:

- runs YOLO26s track with BoT-SORT config
- uses `vid_stride=2` on long chunks (speed path)
- confidence-triggered dense rerun when sparse pass quality is weak
- optional ROI refinement on propagated detections (`CLYPT_ENABLE_ROI_DETECT=1`)

## 4) Canonical coordinates and metadata

Each detection is canonicalized to:

- absolute original-frame `xyxy`
- label taxonomy (`class_id=0`, `label="person"`)
- normalized `bbox_norm_xywh` (aux field)

Worker explicitly computes and writes letterbox metadata and forward/inverse transforms:

- `_compute_letterbox_meta`
- `_forward_letterbox_xyxy`
- `_inverse_letterbox_xyxy`

Per-chunk NDJSON contains:

- one header record (schema/task/coordinate space/geometry/taxonomy/letterbox)
- one record per detection

## 5) Overlap stitching

`_stitch_chunk_tracks` uses overlap windows + assignment:

- TrackTrack-style local candidate pruning
- cost from IoU + embedding distance
- Hungarian assignment
- track-aware initialization for short unmatched tracks
- dedupe on `(frame_idx, track_id)` keeping higher confidence

Returned tracking metrics include:

- `idf1_proxy`
- `mota_proxy`
- `track_fragmentation_rate`
- `chunk_processed_frames`
- `chunk_elapsed_s`
- `chunk_throughput_fps`
- then wall-clock + throughput appended by caller

---

## Global ID Clustering (Phase 1 Step 3)

`_cluster_tracklets` merges fragmented tracker IDs into `Global_Person_k`:

- multi-frame sampling per tracklet
- ArcFace embeddings for high-quality faces
- histogram fallback for low-quality/no-face cases
- DBSCAN on face embeddings
- conservative tiny-cluster and cross-cluster merges
- histogram tracklet attachment to nearest face cluster
- final contiguous renumbering

This stage is designed to reduce ID fragmentation without hardcoding a fixed speaker/person count.

---

## Speaker Binding (Phase 1 Step 4)

Primary path: `LR-ASD` (`_run_lrasd_binding`)

- builds contiguous frame subchunks
- fault-tolerant face crop path with anchor projection fallback
- computes MFCC audio features aligned to visual chunks
- batches by temporal length
- outputs per-frame speaking probabilities
- assigns words to tracks with confidence + margin gates
- applies local smoothing

Fallback path: `_run_speaker_binding_heuristic`

- motion/confidence/area based scoring
- optional lip-landmark tie-break when available

If LR-ASD confidence/coverage is too low, worker falls back automatically.

---

## Contracts and Rollout Gates

Before final response, worker enforces:

- schema contract validation for tracks
- schema pass-rate computation
- rollout gates from `tracking_metrics`

Configurable gate env vars:

- `CLYPT_ENFORCE_ROLLOUT_GATES`
- `CLYPT_GATE_MIN_IDF1_PROXY`
- `CLYPT_GATE_MIN_MOTA_PROXY`
- `CLYPT_GATE_MAX_FRAGMENTATION`
- `CLYPT_GATE_MIN_THROUGHPUT_FPS`
- `CLYPT_GATE_MAX_WALLCLOCK_S`
- `CLYPT_GATE_MIN_SCHEMA_PASS_RATE`

If enforcement is enabled and gates fail, worker raises.

---

## Output Shape (Final)

## `phase_1_visual`

- `source_video`
- `schema_version` (`2.0.0`)
- `task_type` (`person_tracking`)
- `coordinate_space` (`absolute_original_frame_xyxy`)
- `geometry_type` (`aabb` or `mixed` when OBB records exist)
- `class_taxonomy`
- `tracking_metrics`
- `tracks` (per-frame detections)

## `phase_1_audio`

- `source_audio`
- `words` (word-level timestamps + speaker tags)
- `speaker_bindings` (merged speech segments per track)

---

## Key Environment Variables

General/debug:

- `CLYPT_MODEL_DEBUG`
- `CLYPT_MODEL_DEBUG_EVERY`

Tracking:

- `CLYPT_YOLO_IMGSZ`
- `CLYPT_TRACK_CHUNK_WORKERS`
- `CLYPT_ENABLE_ROI_DETECT`

Rollout gates:

- all `CLYPT_GATE_*` vars listed above

---

## Deploy

```bash
source .venv/bin/activate
.venv/bin/modal deploy modal_worker.py
```

---

## Notes for Teammates

- This worker expects media bytes from local pipeline code (`pipeline/phase_1_modal_pipeline.py`), not direct YouTube access from Modal.
- The recommended production path is distributed fan-out from the client with up to 8 GPU workers.
- Staged chunk artifacts live under `/vol/clypt-chunks/jobs/<job_id>` and should be cleaned with `cleanup_tracking_job`.
