# Clypt V3.1 Spec: Phase24 Timeline-Batched Node-Media-Prep, Pipelined Multimodal Embedding, Hybrid Seek/Trim, and Modal L40S

**Status:** Active (implemented)  
**Date:** 2026-04-20  
**Owner:** Phase26 runtime / Modal node-media-prep

## 1. Summary

Phase24 no longer waits for one monolithic node-media-prep result before beginning multimodal embedding. Phase26 now:

1. sorts semantic nodes by timeline,
2. groups nearby nodes into timeline-local batches,
3. submits each batch as its own Modal node-media-prep job,
4. starts Vertex multimodal embedding as soon as each batch completes,
5. reassembles final multimodal embeddings in original node order before the rest of Phase 2 continues.

At the same time, the Modal worker moves from `L4` to `L40S`, and node-media-prep itself shifts from per-node full-source extraction to a hybrid batch flow:

- coarse seek near the batch start,
- decode a shared local batch window once,
- emit exact per-node clips from inside that local window.

## 2. Locked Decisions

1. External Phase26 semantics stay the same: one final multimodal embedding per node keyed by `node_id`.
2. The remote node-media-prep endpoint family stays submit/poll:
   - `POST /tasks/node-media-prep`
   - `GET /tasks/node-media-prep/result/{call_id}`
3. Each remote request now represents one timeline batch, not the full node set.
4. Modal `L40S` is now the active node-media-prep GPU target.
5. Clip outputs remain `video/mp4` objects in GCS.
6. Exact node clip boundaries remain the contract. Batching is only an execution optimization.
7. Phase 6 render/export may reuse submit/poll later, but this spec does not wire render/export into runtime.

## 3. Batch Planning Defaults

- Sort nodes by `start_ms`, then `end_ms`, then `node_id`.
- Start a new batch when the next node starts more than `2000 ms` after the current batch end.
- Cap each batch at `8` nodes.
- Cap each batch at `120000 ms` of raw node span.
- Use `batch_0000`, `batch_0001`, ... identifiers.
- Add `2000 ms` pre/post pad around each batch extraction window, clamped to source bounds.
- Keep at most `3` in-flight node-media-prep batch jobs per Phase24 run.

## 4. Extraction Strategy

The worker-side extraction path now treats one request as one local timeline batch:

1. download the source video once into worker scratch,
2. compute:
   - `batch_start_ms = min(node.start_ms)`
   - `batch_end_ms = max(node.end_ms)`
   - `padded_start_ms = batch_start_ms - 2000 ms`
   - `padded_end_ms = batch_end_ms + 2000 ms`
   - `coarse_start_ms = max(0, padded_start_ms - 10000 ms)`
3. extract one local batch window with a coarse `-ss` before `-i`,
4. trim exact per-node clips from that local batch window with precise local offsets.

The current ffmpeg policy is:

- GPU default: `h264_cuvid` decode + `h264_nvenc` encode
- no `+faststart`
- `-an` by default for node-media-prep clips
- 480p target output preserved for Vertex multimodal embedding

CPU fallback remains an operational escape hatch, but it is not the planned baseline.

## 5. Phase26 Execution Changes

Phase26 now overlaps node-media-prep and multimodal embedding:

- semantic text embeddings still start immediately in parallel,
- each completed media-prep batch triggers immediate multimodal embedding for that batch,
- final media descriptors and multimodal embeddings are reassembled into the original node order,
- duplicate node IDs or missing batch outputs hard-fail the run.

The blocking “prepare all media, then embed all media” path remains only as a compatibility fallback for non-batch-aware preparers.

## 6. Remote Contract and Diagnostics

Request wire shape remains compatible:

- `run_id`
- `video_gcs_uri`
- `object_prefix`
- `max_concurrency`
- `nodes`

Responses may now include optional batch/debug metadata:

- `batch_id`
- `batch_start_ms`
- `batch_end_ms`
- `node_count`
- `ffmpeg_mode`
- `download_ms`
- `extract_ms`
- `upload_ms`
- `total_ms`

Clients must accept these fields when present and still succeed when only `media` is returned.

## 7. Runtime Defaults

Phase26 baseline changes:

- `CLYPT_PHASE24_NODE_MEDIA_PREP_MAX_CONCURRENCY=12`
- `CLYPT_PHASE24_NODE_MEDIA_PREP_TIMEOUT_S=1800` unchanged

Modal deployment changes:

- `gpu="L40S"`
- `min_containers=1`

## 8. Acceptance Criteria

1. Batch planner groups adjacent nodes and splits sparse/heavy windows correctly.
2. Phase26 starts multimodal embedding after the first batch completes, not after all batches complete.
3. Final multimodal embeddings remain aligned one-to-one with original node order.
4. Worker emits batch metadata when available.
5. Focused runtime, provider, Modal, and semantics tests pass.
6. Benchmark target for future replay validation:
   - at least `30%` node-media-prep wall-clock improvement on the heavier Phase24 references,
   - no node-boundary drift over `100 ms`,
   - no missing multimodal embeddings.
