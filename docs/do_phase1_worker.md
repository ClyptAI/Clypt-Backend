# DO Phase 1 Worker

This document describes the active Phase 1 extraction stack implemented in:
- `backend/do_phase1_worker.py`
- `backend/do_phase1_service/extract.py`
- `backend/pipeline/phase_1_do_pipeline.py`

For Phase 1 behavior, those files are the source of truth.

## Purpose

The worker accepts a muxed MP4 video plus a 16kHz mono WAV and returns:
- `phase_1_visual`
- `phase_1_audio`
- compatibility aliases `phase_1a_visual` and `phase_1a_audio`

The active production path runs inside the DigitalOcean Phase 1 service, which calls the worker in-process.

## Current Model Stack

- ASR: `nvidia/parakeet-tdt-1.1b`
- Tracking: `YOLO26s + BoT-SORT`
- Face observations: full-frame `SCRFD` detections on the shared analysis video, then contiguous face tracks
- Identity features: `ArcFace/InsightFace` embeddings on face tracks, short-gap propagation across missed stretches, then signature-only attachment for remaining fragments
- Speaker binding: LR-ASD primary, heuristic fallback, `auto` selection for larger videos

Notes:
- The active path does **not** use TalkNet.
- The active path does **not** use Google Video Intelligence or any `phase_1a_reconcile` stage.
- The active path still exposes optional legacy serverless compatibility shims, but DigitalOcean is the real runtime.

## End-to-End Execution Order

`extract(video_bytes, audio_wav_bytes, youtube_url)` currently runs:

1. Persist incoming media to local temp files.
2. Run Parakeet ASR.
3. Run tracking on the same GPU after ASR completes.
4. Finalize from `words + tracks`:
   - build canonical face observations
   - cluster tracklets into global identities
   - run speaker binding
   - build visual ledgers and metrics
   - package Phase 1 outputs

ASR and tracking are intentionally serialized on the GPU so NeMo CUDA-graph decoding does not conflict with YOLO GPU work.

## Tracking Modes

Supported tracking modes are selected in worker code:
- `direct`
- `chunked`
- `shared_analysis_proxy`
- `auto`

Behavior today:
- `direct` and `shared_analysis_proxy` use the direct full-video tracker path
- `chunked` uses staged chunk tracking plus stitching
- `auto` resolves based on runtime heuristics in the worker

The direct path is the most common path during current podcast evaluation.

## Shared Analysis Proxy

When enabled, the worker prepares one shared analysis video path for:
- tracking
- face observation extraction
- LR-ASD

This avoids duplicating large-video preprocessing for each subsystem.

## Face and Identity Pipeline

The worker builds face observations early and reuses them across later stages.

### Canonical face observations
- detector-derived face observations are attached to person tracks inside ROIs
- provenance is recorded on emitted ledgers
- real face observations are preferred everywhere downstream

### Fallback behavior
- MediaPipe can act as a lightweight fallback when InsightFace coverage is missing
- histogram features are used when high-quality face embeddings are unavailable

### Output ledgers
The worker emits:
- `face_detections`
- `person_detections`
- `tracks`
- `tracking_metrics`

`face_detections` are intended to be real detector-derived tracks.
`proxy_face_detections` are created later by the pipeline bridge only as a compatibility fallback for older consumers.

## Speaker Binding

Supported speaker-binding modes:
- `auto`
- `lrasd`
- `heuristic`
- `shared_analysis_proxy`

Behavior today:
- `lrasd` uses the LR-ASD path and cached assets under `/root/lrasd` and `/root/.cache/clypt/finetuning_TalkSet.model`
- `heuristic` uses a lightweight visual heuristic binder
- `auto` chooses the path based on video size / complexity limits
- `shared_analysis_proxy` resolves through the LR-ASD-oriented path, using the shared proxy pipeline

The active binding path consumes the canonical face stream first instead of rediscovering faces independently.

## Outputs

The worker response includes:
- `phase_1_visual`
- `phase_1_audio`
- `phase_1a_visual`
- `phase_1a_audio`

The DigitalOcean extraction service enriches and validates those artifacts, uploads them to GCS, and persists a contract `v2` manifest.

## DigitalOcean Service Flow

The service in `backend/do_phase1_service/extract.py` does this:
1. Download source media for the submitted URL.
2. Wait for a bounded GPU extraction slot.
3. Run the worker locally in-process.
4. Upload the canonical source video to GCS.
5. Enrich and validate `phase_1_visual` / `phase_1_audio`.
6. Persist `manifest.json`, `phase_1_visual.json`, and `phase_1_audio.json`.

## Runtime Controls

### Local pipeline controls
- `PHASE1_RUNTIME_PROFILE`
- `PHASE1_FORCE_LRASD`
- `PHASE1_SPEAKER_BINDING_MODE`
- `PHASE1_TRACKING_MODE`
- `PHASE1_SHARED_ANALYSIS_PROXY`
- `PHASE1_HEURISTIC_BINDING_ENABLED`

### Worker controls
- `CLYPT_TRACKING_MODE`
- `CLYPT_TRACK_CHUNK_WORKERS`
- `CLYPT_SPEAKER_BINDING_MODE`
- `CLYPT_LRASD_BATCH_SIZE`
- `CLYPT_LRASD_PIPELINE_OVERLAP`
- `CLYPT_LRASD_MAX_INFLIGHT`
- rollout gate thresholds in `backend/do_phase1_worker.py`

### Service controls
- `DO_PHASE1_WORKER_CONCURRENCY`
- `DO_PHASE1_GPU_SLOTS`
- `DO_PHASE1_STATE_ROOT`
- `DO_PHASE1_DB_PATH`
- `DO_PHASE1_OUTPUT_ROOT`
- `DO_PHASE1_LOG_ROOT`

## Current Downstream Compatibility Bridge

After the DO job succeeds, `backend/pipeline/phase_1_do_pipeline.py` still re-downloads the source media locally so later phases and render tooling can keep using local JSON + local video assumptions.

That bridge currently writes:
- `backend/outputs/phase_1_visual.json`
- `backend/outputs/phase_1_audio.json`
- `backend/outputs/phase_1_visual.ndjson`
- `backend/outputs/phase_1_runtime_controls.json`
