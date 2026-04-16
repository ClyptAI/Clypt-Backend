# Clypt V3.1 Spec: Phase 1-4 L4 Offload and Concurrency Revamp

**Status:** Active  
**Date:** 2026-04-16  
**Owner:** Backend runtime / inference  
**Scope:** Move VibeVoice ASR and Phase 2 node-media GPU encode/decode onto a persistent Cloud Run L4 service, remove the global concurrency knob, and restructure Phase 2-3 scheduling so semantic work overlaps instead of serializing behind media prep.

---

## 1. Locked Decisions

1. VibeVoice ASR is no longer treated as an H200-resident service target for the intended production topology.
2. Cloud Run L4 will host both:
   - persistent VibeVoice vLLM ASR
   - Phase 2 node-media prep with GPU ffmpeg decode/encode
3. The L4 service is treated as a single serialized GPU worker:
   - warm instance
   - request concurrency `1`
   - no assumption that ASR and node-media prep run simultaneously
4. The H200 is reserved for:
   - RF-DETR / visual extraction
   - SGLang Qwen Phase 2-4 generation
5. There is **no runtime fallback** to the legacy topology once this revamp lands:
   - no H200-hosted VibeVoice fallback
   - no “try Cloud Run L4 first, then fall back to local vLLM” behavior
   - no fallback from the combined L4 ASR/media-prep service to the old per-host path
6. `CLYPT_GEMINI_MAX_CONCURRENT` is removed entirely.
7. Every concurrency control must be phase- or subphase-specific.
8. Phase 2 concurrency is split:
   - `CLYPT_PHASE2_MAX_CONCURRENT=8`
   - `CLYPT_PHASE2_BOUNDARY_MAX_CONCURRENT=10`
9. Phase 3 and Phase 4 retain explicit env pins:
   - `CLYPT_PHASE3_LOCAL_MAX_CONCURRENT=8`
   - `CLYPT_PHASE3_LONG_RANGE_MAX_CONCURRENT=8`
   - `CLYPT_PHASE4_SUBGRAPH_MAX_CONCURRENT=10`
10. Semantic text embeddings start immediately after Phase 2 node creation.
11. Phase 3 local-edge batches start immediately after Phase 2 node creation.
12. Node media prep runs in parallel with those lanes.
13. Multimodal embeddings start as soon as media URIs arrive.
14. Phase 3 long-range is gated only on embedding completion.
15. Phase 3 local-edge and long-range stages should overlap where safe, with separate caps, rather than being serialized by default.

---

## 2. Why This Change

The current topology spends scarce H200 capacity on work that does not need to share the same GPU:

- VibeVoice ASR residency competes with Qwen VRAM headroom.
- Phase 2 media prep still sits on the critical path even when it is offloaded.
- Phase 2 and Phase 3 contain unnecessary serialization barriers.
- one global concurrency knob (`CLYPT_GEMINI_MAX_CONCURRENT`) couples multiple subsystems and makes tuning brittle.

The desired outcome is:

1. Move ASR and GPU ffmpeg clip work off the H200.
2. Give SGLang Qwen materially more memory headroom on the H200.
3. Increase safe Phase 2 concurrency.
4. Remove artificial waits between node creation, embeddings, local graph work, and long-range graph work.

---

## 3. Current vs Target Runtime Topology

### 3.1 Current

- H200 host runs:
  - RF-DETR visual pipeline
  - VibeVoice vLLM service
  - SGLang Qwen service
  - Phase 2-4 local worker
- Cloud Run L4 is used only for optional Phase 2 node-media prep offload.
- `CLYPT_GEMINI_MAX_CONCURRENT` acts as a shared fallback/default for multiple Phase 2-4 LLM lanes.

### 3.2 Target

#### Cloud Run L4

- one persistent GPU service in `us-east4`
- one L4 GPU
- non-zonal redundancy
- warm instance (`min-instances=1`)
- request concurrency forced to `1`

Service responsibilities:

- `POST /tasks/asr` for VibeVoice ASR
- `POST /tasks/node-media-prep` for ordered node clip extraction/upload
- `GET /healthz`

#### H200 host

- RF-DETR / TensorRT visual path
- SGLang Qwen only
- Phase 2-4 local worker orchestration
- no VibeVoice residency required for the intended target topology

### 3.3 No-Fallback Policy

Once this revamp is implemented, the intended production path is exclusive:

- ASR must use the combined Cloud Run L4 service
- node-media prep must use the combined Cloud Run L4 service
- the H200 host must not retain a runtime fallback path for VibeVoice
- if the L4 service is unavailable, the run fails explicitly and immediately

This is intentional. The purpose of the redesign is to free H200 memory and simplify tuning around one production topology, not to preserve dual-path operational complexity.

### 3.4 Topology Diagram

```text
Phase 1 video
    |
    +--> H200: RF-DETR / visual extraction
    |
    +--> L4 Cloud Run: VibeVoice ASR
              |
              +--> turns back to H200 worker
                        |
                        +--> H200: forced alignment -> emotion2vec -> YAMNet
                        |
                        +--> H200: Phase 2 merge/boundary on SGLang Qwen
                                   |
                                   +--> raw semantic nodes
                                           |
                                           +--> Lane A: semantic text embeddings
                                           |
                                           +--> Lane B: Phase 3 local-edge batches
                                           |
                                           +--> Lane C: L4 Cloud Run node-media prep
                                                      |
                                                      +--> multimodal embeddings
                                                                 |
                                                                 +--> Phase 3 long-range
```

---

## 4. Cloud Run L4 Combined Service Design

### 4.1 Service Shape

Recommended deployment profile:

- region: `us-east4`
- GPU: `1 x nvidia-l4`
- zonal mode: `--no-gpu-zonal-redundancy`
- `min-instances=1`
- `concurrency=1`
- CPU: `8`
- memory: `32Gi`

Rationale:

- the project currently has effective L4 quota in `us-east4`
- the service is intended to serialize GPU-intensive requests rather than multiplex them
- one warm instance preserves ASR startup performance
- concurrency `1` ensures ASR and ffmpeg media prep do not overlap on the same L4 unless the design is revisited later with measurements

### 4.2 Supported Routes

The combined service exposes:

- `POST /tasks/asr`
- `POST /tasks/node-media-prep`
- `GET /healthz`

### 4.3 Internal Runtime Policy

1. VibeVoice vLLM starts once at container boot and remains warm.
2. Media-prep requests reuse the existing `prepare_node_media_embeddings()` GPU ffmpeg path.
3. The service does not attempt intra-instance GPU multiplexing across ASR and media prep.
4. Ordered request/response guarantees for node-media prep remain unchanged.
5. Failure remains fail-fast:
   - no partial response ordering
   - no silent fallback to unordered descriptors

### 4.4 Why This Combined Service Is Acceptable

- VibeVoice ASR and node-media prep are not expected to overlap in the desired runtime flow.
- GPU ffmpeg decode/encode uses far less VRAM than Qwen or VibeVoice weights.
- the real concern is GPU engine contention, not raw VRAM exhaustion
- request concurrency `1` turns the L4 into a predictable serialized worker instead of a contested multi-tenant GPU

---

## 5. Phase 2 Scheduling Revamp

### 5.1 Current Critical Path

Today `run_phase_2()` effectively behaves like:

```text
merge+boundary
  -> media prep
  -> semantic + multimodal embeddings
  -> persist
  -> return nodes to Phase 3
```

This leaves overlap on the table.

### 5.2 Target Critical Path

After raw nodes are created:

```text
merge+boundary
  -> [semantic text embeddings] ----\
  -> [Phase 3 local-edge batches] ---+--> wait only where needed
  -> [media prep on L4] -> [multimodal embeddings] --/
```

### 5.3 Required Semantic Split

Phase 2 must be refactored to expose a meaningful boundary between:

1. **raw nodes**
   - available immediately after merge/boundary reconciliation
2. **embedded nodes**
   - available after semantic and multimodal embedding lanes complete

That boundary is what enables parallel downstream work.

### 5.4 Phase 2 LLM Concurrency

Remove the old shared fallback and make Phase 2 explicit:

- `CLYPT_PHASE2_MAX_CONCURRENT=8`
- `CLYPT_PHASE2_BOUNDARY_MAX_CONCURRENT=10`

Reason for split:

- merge prompts are larger and heavier
- boundary seam prompts are smaller and can tolerate a higher lane count
- forcing both through one cap is unnecessary and leaves throughput unused

### 5.5 Phase 2 Substeps To Persist

The new split should persist at least:

- `merge_total`
- `merge_batch`
- `boundary_total`
- `boundary_seam`
- `semantic_embedding`
- `media_prep_total`
- `media_prep_item`
- `multimodal_embedding`
- `persist_nodes`

---

## 6. Phase 3 Scheduling Revamp

### 6.1 Constraint That Must Be Preserved

Phase 3 long-range cannot start from raw nodes alone because shortlist generation uses:

- `semantic_embedding`
- `multimodal_embedding`

Therefore:

- **local-edge** can start immediately after raw nodes exist
- **long-range** must wait until embeddings complete

### 6.2 New Phase 3 Shape

```text
raw nodes ready
    |
    +--> local-edge batches start immediately
    |
    +--> semantic text embeddings start immediately
    |
    +--> media prep -> multimodal embeddings
              |
              +--> long-range adjudication starts once embeddings are ready
                           |
                           +--> reconcile local + long-range edges
```

### 6.3 Local vs Long-Range Concurrency

Keep separate caps:

- `CLYPT_PHASE3_LOCAL_MAX_CONCURRENT=8`
- `CLYPT_PHASE3_LONG_RANGE_MAX_CONCURRENT=8`

These remain independently tunable if the shared SGLang server shows queue pressure.

---

## 7. Configuration Surface Changes

### 7.1 Remove

- `CLYPT_GEMINI_MAX_CONCURRENT`

This env must not remain as a hidden default or fallback.

### 7.2 Add / Keep Explicit Phase-Scoped Knobs

#### Phase 2

- `CLYPT_PHASE2_MAX_CONCURRENT=8`
- `CLYPT_PHASE2_BOUNDARY_MAX_CONCURRENT=10`
- `CLYPT_PHASE2_MERGE_MAX_OUTPUT_TOKENS`
- `CLYPT_PHASE2_BOUNDARY_MAX_OUTPUT_TOKENS`

#### Phase 3

- `CLYPT_PHASE3_LOCAL_MAX_CONCURRENT=8`
- `CLYPT_PHASE3_LONG_RANGE_MAX_CONCURRENT=8`
- `CLYPT_PHASE3_LOCAL_MAX_OUTPUT_TOKENS`
- `CLYPT_PHASE3_LONG_RANGE_MAX_OUTPUT_TOKENS`

#### Phase 4

- `CLYPT_PHASE4_SUBGRAPH_MAX_CONCURRENT=10`
- `CLYPT_PHASE4_META_MAX_OUTPUT_TOKENS`
- `CLYPT_PHASE4_SUBGRAPH_MAX_OUTPUT_TOKENS`
- `CLYPT_PHASE4_POOL_MAX_OUTPUT_TOKENS`

#### Cloud Run media / ASR

- `CLYPT_PHASE24_MEDIA_PREP_BACKEND=cloud_run_l4`
- `CLYPT_PHASE24_MEDIA_PREP_SERVICE_URL=...`
- `CLYPT_PHASE24_MEDIA_PREP_AUTH_MODE=id_token`
- `CLYPT_PHASE24_MEDIA_PREP_AUDIENCE=...`
- `CLYPT_PHASE24_MEDIA_PREP_TIMEOUT_S=600`
- `CLYPT_PHASE24_NODE_MEDIA_CONCURRENCY=<start conservative, tune upward>`
- `CLYPT_PHASE24_FFMPEG_DEVICE=gpu`
- `VIBEVOICE_REPETITION_PENALTY=1.03`

### 7.3 Compatibility Rule

If `CLYPT_GEMINI_MAX_CONCURRENT` is still present after this change:

- fail fast at startup
- surface a clear migration error message

Do not silently honor it.

If a legacy local-ASR or legacy local media-prep topology is configured after this change:

- fail fast at startup
- surface a clear migration error message

Do not silently fall back to the pre-revamp path.

---

## 8. H200 Qwen / SGLang Tuning

### 8.1 Why Retune

Moving VibeVoice off the H200 frees VRAM headroom for Qwen. That headroom should be spent on:

- larger static memory allocation
- more stable structured-output concurrency
- reduced queueing at Phase 2 merge/boundary

### 8.2 Keep

- `--schedule-policy lpm`
- `--chunked-prefill-size 8192`
- `--context-length 131072`

### 8.3 Retune

After VibeVoice removal from the H200:

- increase `SG_MEM_FRACTION_STATIC` from `0.55` to a higher measured target
- suggested canary progression:
  - `0.72`
  - `0.78`
  - higher only with evidence

### 8.4 New Benchmark Matrix

At minimum, measure:

1. Phase 2 merge throughput at `4`, `6`, `8`
2. Phase 2 boundary throughput at `6`, `8`, `10`
3. Phase 2 + Phase 3 local overlap wall clock
4. Phase 3 long-range overlap once embeddings complete
5. Phase 4 subgraph review at `10` on the post-L4 topology

Track:

- queue depth
- p50 / p95 / max latency
- malformed structured-output rate
- `finish_reason=length` rate
- timeout rate

---

## 9. Implementation Streams

### 9.1 Stream A - L4 Combined Service

Add or update:

- combined Cloud Run service app entrypoint
- ASR route
- node-media-prep route
- health route
- deployment automation for persistent L4 service

### 9.2 Stream B - Provider / Runtime Wiring

Update:

- VibeVoice client path to target remote ASR route when configured
- node-media prep client path to target the same Cloud Run service
- worker/runtime bootstrapping so Phase 2-4 uses the combined service endpoint correctly

### 9.3 Stream C - Phase 2 Refactor

Update:

- live runner orchestration to expose raw-node completion early
- semantic embedding launch point
- media-prep launch point
- multimodal embedding launch point
- substep logging / Spanner persistence

### 9.4 Stream D - Phase 3 Refactor

Update:

- local-edge launch timing
- long-range launch timing
- final reconciliation timing
- substep logging / Spanner persistence

### 9.5 Stream E - Config and Docs Cleanup

Update:

- `.env.example`
- `docs/runtime/known-good.env`
- deployment/runbook docs
- specs index if needed
- any host scripts still referencing the global concurrency env

---

## 10. Execution Checklist

This section turns the implementation streams into an execution order with expected file targets and validation commands.

### Stage 0 - Guardrails and Baseline

#### Goals

- confirm the active spec and runtime docs are aligned
- capture blast radius before editing runtime symbols
- establish a baseline test pass for the touched areas

#### Expected files to inspect or touch

- `docs/specs/2026-04-16_phase14_l4_offload_concurrency_revamp_spec.md`
- `docs/runtime/RUNTIME_GUIDE.md`
- `docs/deployment/P1_DEPLOY.md`
- `backend/pipeline/config.py`
- `backend/runtime/phase14_live.py`
- `backend/providers/openai_local.py`
- `backend/providers/vibevoice_vllm.py`

#### Validation commands

```bash
python -m pytest tests/backend/pipeline/test_subgraph_review_schema_compat.py -q
python -m pytest tests/backend/runtime -q
```

### Stage 1 - Remove Global Concurrency Knob

#### Goals

- remove `CLYPT_GEMINI_MAX_CONCURRENT`
- add explicit Phase 2 envs and preserve explicit Phase 3 and Phase 4 envs
- fail fast if the removed env is still supplied

#### Expected files to touch

- `backend/pipeline/config.py`
- `.env.example`
- `docs/runtime/known-good.env`
- `docs/runtime/RUNTIME_GUIDE.md`
- `docs/deployment/P1_DEPLOY.md`
- tests under:
  - `tests/backend/providers/`
  - `tests/backend/runtime/`
  - `tests/backend/pipeline/`

#### Validation commands

```bash
python -m pytest tests/backend/providers/test_provider_config_and_clients.py -q
python -m pytest tests/backend/runtime/test_phase24_worker_app.py -q
python -m pytest tests/backend/runtime/test_phase24_local_worker.py -q
```

### Stage 2 - Phase 2 Concurrency Split

#### Goals

- wire `CLYPT_PHASE2_MAX_CONCURRENT=8`
- wire `CLYPT_PHASE2_BOUNDARY_MAX_CONCURRENT=10`
- keep output-token controls explicit per subphase

#### Expected files to touch

- `backend/pipeline/config.py`
- `backend/pipeline/semantics/merge_and_classify.py`
- `backend/pipeline/semantics/boundary_reconciliation.py`
- `backend/pipeline/semantics/runtime.py`
- tests under `tests/backend/pipeline/`

#### Validation commands

```bash
python -m pytest tests/backend/pipeline -q
```

### Stage 3 - Phase 2 Raw-Node / Embedded-Node Split

#### Goals

- make raw nodes available as soon as merge plus boundary completes
- launch semantic text embeddings immediately
- launch media prep independently
- launch multimodal embeddings as soon as media descriptors arrive

#### Expected files to touch

- `backend/runtime/phase14_live.py`
- `backend/pipeline/semantics/runtime.py`
- `backend/pipeline/semantics/merge_and_classify.py`
- `backend/pipeline/semantics/boundary_reconciliation.py`
- `backend/pipeline/semantics/turn_neighborhoods.py`
- `backend/repository/spanner_phase14_repository.py`
- `backend/repository/models.py`
- tests under:
  - `tests/backend/pipeline/`
  - `tests/backend/runtime/`

#### Validation commands

```bash
python -m pytest tests/backend/runtime/test_phase24_local_worker.py -q
python -m pytest tests/backend/runtime/test_phase24_worker_app.py -q
python -m pytest tests/backend/pipeline/test_semantic_edges_phase3.py -q
```

### Stage 4 - Start Phase 3 Local-Edge Early

#### Goals

- start Phase 3 local-edge batches as soon as raw nodes exist
- avoid waiting for media prep or multimodal embeddings

#### Expected files to touch

- `backend/runtime/phase14_live.py`
- `backend/pipeline/graph/local_semantic_edges.py`
- `backend/pipeline/graph/runtime.py`
- tests under:
  - `tests/backend/pipeline/`
  - `tests/backend/runtime/`

#### Validation commands

```bash
python -m pytest tests/backend/pipeline/test_semantic_edges_phase3.py -q
python -m pytest tests/backend/runtime/test_phase24_local_worker.py -q
```

### Stage 5 - Gate Long-Range Only On Embeddings

#### Goals

- allow Phase 3 local-edge and embedding/media lanes to overlap
- start long-range only after semantic plus multimodal embeddings complete
- reconcile local and long-range outputs after both lanes complete

#### Expected files to touch

- `backend/runtime/phase14_live.py`
- `backend/pipeline/graph/long_range_edges.py`
- `backend/pipeline/graph/runtime.py`
- `backend/pipeline/graph/local_semantic_edges.py`
- tests under:
  - `tests/backend/pipeline/`
  - `tests/backend/runtime/`

#### Validation commands

```bash
python -m pytest tests/backend/pipeline/test_semantic_edges_phase3.py -q
python -m pytest tests/backend/runtime/test_phase24_worker_app.py -q
```

### Stage 6 - Combined L4 ASR + Media-Prep Topology

#### Goals

- support a single persistent Cloud Run L4 service for ASR and node-media prep
- keep request concurrency `1`
- preserve fail-fast behavior and ordered media descriptor contract

#### Expected files to touch

- `backend/providers/vibevoice_vllm.py`
- `backend/runtime/phase14_live.py`
- `backend/runtime/phase24_worker_app.py`
- `scripts/deploy_phase24_media_prep_service.sh`
- `scripts/do_phase1/bootstrap_gpu_droplet.sh`
- `scripts/do_phase1/deploy_sglang_qwen_service.sh`
- `scripts/do_phase1/systemd/clypt-vllm-vibevoice.service`
- `docs/runtime/RUNTIME_GUIDE.md`
- `docs/deployment/P1_DEPLOY.md`

#### Validation commands

```bash
python -m pytest tests/backend/providers/test_storage_and_phase1_runtime.py -q
python -m pytest tests/backend/providers/test_vertex_and_pipeline_runtimes.py -q
python -m pytest tests/backend/runtime -q
```

### Stage 7 - Qwen H200 Retuning

#### Goals

- increase `SG_MEM_FRACTION_STATIC` after VibeVoice moves off the H200
- keep the current scheduler/prefill/context settings unless measurement disproves them
- benchmark Phase 2 and Phase 3 overlap under the new topology

#### Expected files to touch

- `scripts/do_phase1/deploy_sglang_qwen_service.sh`
- `scripts/do_phase1/systemd/clypt-sglang-qwen.service`
- `docs/runtime/known-good.env`
- `docs/runtime/RUNTIME_GUIDE.md`
- `docs/deployment/P1_DEPLOY.md`

#### Validation commands

```bash
python scripts/bench_phase24_llm_concurrency.py --scenario phase3_local --concurrency-values 1,2,4,6,8 --rounds 1 --timeout-s 180
python scripts/bench_phase24_llm_concurrency.py --scenario phase4_subgraph --concurrency-values 1,2,4,6,8,10 --rounds 1 --timeout-s 180
```

### Stage 8 - End-to-End Verification

#### Goals

- verify no regression in schema compatibility
- verify phase ordering and overlap behavior
- verify docs and env templates match implemented behavior

#### Expected files to inspect or touch

- `docs/ERROR_LOG.md`
- `docs/runtime/RUN_REFERENCE.md`
- `docs/runtime/RUNTIME_GUIDE.md`
- `docs/deployment/P1_DEPLOY.md`

#### Validation commands

```bash
python -m pytest tests/backend/pipeline -q
python -m pytest tests/backend/runtime -q
python -m pytest tests/backend/providers -q
```

---

## 11. Verification Plan

### 11.1 Functional

1. VibeVoice ASR works through the Cloud Run L4 service.
2. Node media prep works through the same L4 service.
3. Phase 2 raw nodes are available before embeddings complete.
4. Semantic embeddings begin before media prep completes.
5. Phase 3 local-edge batches begin before embeddings complete.
6. Phase 3 long-range waits until embeddings are ready.
7. Reconciled graph output matches expected contract and existing validators.

### 11.2 Performance

For the same video set, compare before vs after:

- Phase 1 wall time
- Phase 2 wall time
- Phase 3 wall time
- Phase 4 wall time
- total Phase 1-4 wall time
- H200 Qwen latency / queue depth
- Cloud Run L4 request latency for ASR and media prep

### 11.3 Resilience

1. If Cloud Run L4 ASR is unavailable, Phase 1 fails with explicit error.
2. If Cloud Run L4 media prep is unavailable, Phase 2 fails with explicit error.
3. No response-order violations are tolerated for node media prep.
4. Old global concurrency env is rejected at startup.

---

## 12. Risks and Mitigations

### 11.1 Single L4 Service Bottleneck

**Risk:** ASR and node-media prep serialize behind one Cloud Run GPU worker.  
**Mitigation:** this is an intentional first-cut tradeoff; request concurrency `1` and `min-instances=1` prioritize predictability over speculative multiplexing.

### 11.2 Phase 2 / Phase 3 Correctness Drift

**Risk:** introducing overlap breaks hidden ordering assumptions.  
**Mitigation:** keep raw-node vs embedded-node boundaries explicit and preserve all existing validators.

### 11.3 SGLang Overload at Higher Phase 2 Concurrency

**Risk:** merge/boundary concurrency increases malformed or truncated JSON.  
**Mitigation:** stepwise canary, separate merge vs boundary caps, explicit diagnostics on malformed structured outputs and capped finishes.

### 11.4 Documentation Drift

**Risk:** env/runbook docs continue to advertise the removed global concurrency knob.  
**Mitigation:** treat docs/env cleanup as part of the same change, not a follow-up.

---

## 13. Acceptance Criteria

This revamp is complete only when all are true:

1. VibeVoice ASR is served from the persistent Cloud Run L4 topology in the intended production path.
2. Phase 2 node-media GPU ffmpeg work is served from the same Cloud Run L4 topology in the intended production path.
3. `CLYPT_GEMINI_MAX_CONCURRENT` is fully removed from runtime config surface.
4. Phase 2 uses explicit merge and boundary concurrency knobs.
5. Semantic embeddings and Phase 3 local-edge batches begin immediately after raw node creation.
6. Multimodal embeddings begin immediately after media descriptors arrive.
7. Phase 3 long-range is gated only on embeddings, not on the old full Phase 2 tail.
8. Phase 3 local and long-range overlap where safe, with separate caps.
9. H200 SGLang Qwen is re-tuned with higher static memory utilization and validated under the new topology.
10. End-to-end Phase 1-4 wall time improves on the target validation set without introducing unacceptable structured-output regressions.

---

## 14. Out of Scope

1. Replacing VibeVoice vLLM with a native persistent worker in this same revamp.
2. Replacing Vertex embeddings.
3. Changing RF-DETR architecture beyond existing optimizations.
4. Changing Phase 4 scoring semantics or candidate logic.
5. Multi-instance or autoscaled GPU multiplexing for ASR/media prep.

