# RUNTIME GUIDE

**Status:** Active  
**Last updated:** 2026-04-17

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
- Phase 1 audio-post launches immediately after the VibeVoice response returns

### 2.2 Phase26 host

- `python -m backend.runtime.run_phase26_dispatch_service`
- `python -m backend.runtime.run_phase26_worker`
- local queue backend must be `CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite`
- generation backend remains `GENAI_GENERATION_BACKEND=local_openai`

Key behavior:

- `POST /tasks/phase26-enqueue` writes to local SQLite
- the worker still runs current Phase 2-4 business logic
- node-media-prep is called only after node creation

### 2.3 Modal node-media-prep

- app path: `scripts/modal/node_media_prep_app.py`
- GPU target: `L4`
- warm pool target: `min_containers=1`
- request/response contract matches `RemoteNodeMediaPrepClient`

## 3) Phase 1 Execution Semantics

```text
Phase 1 visual chain:
  Phase1 local visual service (/tasks/visual-extract)
    -> hot RF-DETR + ByteTrack

Phase 1 audio chain:
  Phase1 local VibeVoice service (/tasks/vibevoice-asr)
    -> local VibeVoice vLLM sidecar
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
- Modal must expose working:
  - `h264_nvenc`
  - `h264_cuvid`
- safe concurrency remains capped by `CLYPT_PHASE24_NODE_MEDIA_PREP_MAX_CONCURRENCY`
- `min_containers=1` is the intended warm baseline, not a permanent dedicated VM

## 8) Canonical Runtime Files

- Phase1 baseline: [known-good-phase1-h200.env](/Users/rithvik/Clypt-Backend/docs/runtime/known-good-phase1-h200.env)
- Phase1 H100 overlay: [known-good-phase1-h100-backup.env](/Users/rithvik/Clypt-Backend/docs/runtime/known-good-phase1-h100-backup.env)
- Phase26 baseline: [known-good-phase26-h200.env](/Users/rithvik/Clypt-Backend/docs/runtime/known-good-phase26-h200.env)

Legacy env files remain only as migration pointers and should not be treated as canonical baselines.
