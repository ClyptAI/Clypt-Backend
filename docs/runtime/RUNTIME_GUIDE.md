# RUNTIME GUIDE

**Status:** Active  
**Last updated:** 2026-04-20

This is the runtime source of truth for the current repository state.

## 1) Runtime Topology

The active topology is:

- **Phase1 host (H200 default)**
  - Phase 1 runner/orchestrator
  - local VibeVoice service on `:9100`
  - local visual service on `:9200`
  - co-located VibeVoice vLLM sidecar on `:8000`
  - in-process NFA -> emotion2vec+ -> YAMNet
- **Phase26 host (H200)**
  - Phase26 dispatch API on `:9300`
  - local SQLite queue + local worker
  - SGLang Qwen on `:8001`
- **Modal**
  - node-media-prep service

There is **no local fallback** for the remote boundaries. Config loading fails fast when a required service URL or auth token is missing.

## 2) Host Responsibilities

### 2.1 Phase1 host

- `python -m backend.runtime.run_phase1`
- `python -m backend.runtime.run_phase1_vibevoice_service`
- `python -m backend.runtime.run_phase1_visual_service`
- VibeVoice vLLM sidecar container on `127.0.0.1:8000`
- in-process NFA / emotion2vec+ / YAMNet provider singletons

Key behavior:

- `RemoteVibeVoiceAsrClient` targets the local service at `POST /tasks/vibevoice-asr`
- `RemotePhase1VisualClient` targets the local service at `POST /tasks/visual-extract`
- `RemotePhase26DispatchClient` targets the downstream host at `POST /tasks/phase26-enqueue`
- the VibeVoice service keeps the existing single-pass path through 40 minutes, splits `>40..80` minute jobs into 2 shards, splits `>80..160` minute jobs into 4 shards, and fails fast above 160 minutes
- long-form shard requests still target the same local vLLM sidecar and are stitched back into one global speaker space before the HTTP response returns
- forced alignment now uses duration-bounded global chunks by default instead of a per-turn fallback:
  - `<=20 min`: 1 chunk
  - `>20..40 min`: 2 chunks
  - `>40..80 min`: 4 chunks
  - `>80..160 min`: 8 chunks
  If any chunk-level alignment fails, Phase 1 hard-fails instead of dropping to per-turn alignment.
- Phase 1 audio-post launches immediately after the VibeVoice response returns

Current live non-secret Phase1 env snapshot from `clypt-phase1-h200-rithvik-nyc2` on 2026-04-20:

- dispatch URL: `http://162.243.208.185:9300`
- `VIBEVOICE_VLLM_GPU_MEMORY_UTILIZATION=0.60`
- `VIBEVOICE_VLLM_MAX_NUM_SEQS=4`
- `VIBEVOICE_LONGFORM_SINGLE_PASS_MAX_MINUTES=40`
- `VIBEVOICE_LONGFORM_TWO_SHARD_MAX_MINUTES=80`
- `VIBEVOICE_LONGFORM_FOUR_SHARD_MAX_MINUTES=160`
- `VIBEVOICE_LONGFORM_MAX_SHARDS=4`
- `CLYPT_PHASE1_INPUT_MODE=test_bank`
- visual fast-path settings remain `tensorrt_fp16`, batch `16`, threshold `0.35`, shape `640`, ByteTrack buffer `30`, ByteTrack match `0.7`, GPU decode

### 2.2 Phase26 host

- `python -m backend.runtime.run_phase26_dispatch_service`
- `python -m backend.runtime.run_phase26_worker`
- local queue backend must be `CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite`
- generation backend remains `GENAI_GENERATION_BACKEND=local_openai`

Key behavior:

- `POST /tasks/phase26-enqueue` writes to local SQLite
- the worker still runs current Phase 2-4 business logic
- node-media-prep is called only after node creation

Current live non-secret Phase26 env snapshot from `clypt-phase26-h200-ming-nyc2` on 2026-04-20:

- `GENAI_GENERATION_MODEL=Qwen/Qwen3.6-35B-A3B`
- `CLYPT_LOCAL_LLM_BASE_URL=http://127.0.0.1:8001/v1`
- `VERTEX_EMBEDDING_LOCATION=us-central1`
- `CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite`
- `CLYPT_PHASE24_LOCAL_MAX_INFLIGHT=1`
- `CLYPT_PHASE24_NODE_MEDIA_PREP_URL=https://testifytestprep--clypt-node-media-prep-node-media-prep.modal.run/tasks/node-media-prep`
- `CLYPT_PHASE24_NODE_MEDIA_PREP_TIMEOUT_S=1800`
- `CLYPT_PHASE24_NODE_MEDIA_PREP_MAX_CONCURRENCY=12`
- `CLYPT_PHASE24_NODE_MEDIA_PREP_MAX_INFLIGHT_BATCHES=3`
- `CLYPT_PHASE24_NODE_MEDIA_BATCH_GAP_MS=2000`
- `CLYPT_PHASE24_NODE_MEDIA_BATCH_MAX_NODES=8`
- `CLYPT_PHASE24_NODE_MEDIA_BATCH_MAX_SPAN_MS=120000`
- `CLYPT_PHASE24_NODE_MEDIA_BATCH_PAD_MS=2000`
- `CLYPT_PHASE24_NODE_MEDIA_BATCH_COARSE_SEEK_PAD_MS=10000`

### 2.3 Modal node-media-prep

- app path: `scripts/modal/node_media_prep_app.py`
- web surface: CPU ASGI app for submit/poll
- worker GPU target: `L40S`
- warm GPU pool target: `node_media_prep_job min_containers=1`
- submit/poll contract:
  - `POST /tasks/node-media-prep` -> `202 Accepted` + `call_id`
  - `GET /tasks/node-media-prep/result/{call_id}` -> `202 pending` or `200` final result
- `RemoteNodeMediaPrepClient` hides this async contract from Phase 2 and still returns the same final `media` list shape to the worker
- node-media-prep requests are now timeline-local batches, not the full node set
- batch planning and worker fan-out are env-tunable on the Phase26 side via the `CLYPT_PHASE24_NODE_MEDIA_BATCH_*` knobs and `CLYPT_PHASE24_NODE_MEDIA_PREP_MAX_INFLIGHT_BATCHES`
- clip extraction now downscales to 480p before upload / Vertex multimodal embedding
- Phase26 starts multimodal embedding batch-by-batch as node-media-prep results arrive instead of waiting for all media first

Current live non-secret Modal deployment snapshot on 2026-04-21:

- app name: `clypt-node-media-prep`
- app id: `ap-5cylWYEts4MoJtkNoROVUu`
- secret name present in Modal: `clypt-node-media-prep`
- endpoint: `https://testifytestprep--clypt-node-media-prep-node-media-prep.modal.run/tasks/node-media-prep`
- required secret-backed envs remain `GCS_BUCKET`, `NODE_MEDIA_PREP_AUTH_TOKEN`, and `GOOGLE_APPLICATION_CREDENTIALS_JSON`

## 3) Phase 1 Execution Semantics

```text
Phase 1 visual chain:
  Phase1 local visual service (/tasks/visual-extract)
    -> hot RF-DETR + ByteTrack

Phase 1 audio chain:
  Phase1 local VibeVoice service (/tasks/vibevoice-asr)
    -> probe canonical audio duration
    -> for long-form jobs: split into 2-4 shard WAVs + temporary GCS shard objects
    -> local VibeVoice vLLM sidecar
    -> cross-shard speaker verification + merged turns
    -> response returns to runner
    -> in-process NFA -> emotion2vec+ -> YAMNet

Downstream handoff:
  Phase1 runner
    -> POST /tasks/phase26-enqueue
```

Critical invariant:

- the audio-post chain does not wait for RF-DETR to finish
- Phase 1 remains the owner of ASR, audio-post, and visual extraction
- the queue boundary lives on the Phase26 host, not the Phase1 host
- long-form ASR still preserves a one-call outer contract and one merged `turns` payload

## 4) Visual Fast Path

The intended Phase 1 visual settings are preserved:

- `CLYPT_PHASE1_VISUAL_BACKEND=tensorrt_fp16`
- `CLYPT_PHASE1_VISUAL_BATCH_SIZE=16`
- `CLYPT_PHASE1_VISUAL_THRESHOLD=0.35`
- `CLYPT_PHASE1_VISUAL_SHAPE=640`
- `CLYPT_PHASE1_VISUAL_TRACKER=bytetrack`
- `CLYPT_PHASE1_VISUAL_TRACKER_BUFFER=30`
- `CLYPT_PHASE1_VISUAL_TRACKER_MATCH_THRESH=0.7`
- `CLYPT_PHASE1_VISUAL_DECODE=gpu`

Operationally, the fast path remains:

```text
NVDEC -> scale_cuda -> hwdownload -> CUDA tensor normalize -> TensorRT -> ByteTrack
```

The H100 backup overlay must not change semantic visual behavior.

## 5) Phase26 Worker Contract

- queue rows are stored in local SQLite on the Phase26 host
- `run_id` remains the idempotency key
- default crash mode remains fail-fast:
  - `CLYPT_PHASE24_LOCAL_RECLAIM_EXPIRED_LEASES=0`
  - `CLYPT_PHASE24_LOCAL_FAIL_FAST_ON_STALE_RUNNING=1`
- generation remains local OpenAI-compatible to SGLang
- embeddings remain Vertex-backed
- `VERTEX_EMBEDDING_LOCATION` stays pinned to `us-central1` for `gemini-embedding-2-preview`; the live Clypt project currently receives `404 NOT_FOUND` on `global` even though the locations doc lists global support

## 6) SGLang Settings

Current code-backed Qwen flags remain:

- `--context-length 65536`
- `--kv-cache-dtype fp8_e4m3`
- `--mem-fraction-static 0.78`
- `--speculative-algorithm NEXTN`
- `--speculative-num-steps 3`
- `--speculative-eagle-topk 1`
- `--speculative-num-draft-tokens 4`
- `--mamba-scheduler-strategy extra_buffer`
- `--schedule-policy lpm`
- `--chunked-prefill-size 8192`
- `--grammar-backend xgrammar`
- `--reasoning-parser qwen3`
- systemd env:
  - `HF_HUB_OFFLINE=1`
  - `SGLANG_ENABLE_SPEC_V2=1`

These now belong to the Phase26 host, not the Phase1 host.

## 7) Modal Media-Prep Expectations

- `RemoteNodeMediaPrepClient` continues to own the JSON contract
- Modal web requests must stay short; long-running clip extraction happens in a spawned Modal function call that the Phase26 client polls
- only the spawned worker needs GPU access; the public submit/poll route stays on CPU
- the Modal worker must expose working:
  - `h264_nvenc`
  - `h264_cuvid`
- Modal L40S baseline sets `CLYPT_PHASE24_NODE_MEDIA_PREP_MAX_CONCURRENCY=12`
- `node_media_prep_job min_containers=1` is the intended warm GPU baseline, not a permanent dedicated VM

## 8) Canonical Runtime Files

- Phase1 baseline: [known-good-phase1-h200.env](/Users/rithvik/Clypt-Backend/docs/runtime/known-good-phase1-h200.env)
- Phase1 H100 overlay: [known-good-phase1-h100-backup.env](/Users/rithvik/Clypt-Backend/docs/runtime/known-good-phase1-h100-backup.env)
- Phase26 baseline: [known-good-phase26-h200.env](/Users/rithvik/Clypt-Backend/docs/runtime/known-good-phase26-h200.env)

Legacy env files remain only as migration pointers and should not be treated as canonical baselines.
