# Clypt V3.1 Spec: Single-H200 Local Phase 2-4 with Qwen3.5-122B-A10B-GPTQ-Int4

**Status:** Draft v1  
**Date:** 2026-04-14  
**Owner:** Backend runtime / inference  
**Scope:** Replace Gemini generation calls in Phase 2-4 and signal LLM paths with self-hosted Qwen on a single H200 while preserving existing typed JSON contracts, validators, Vertex embeddings, GCS storage, and Spanner graph persistence.

---

## 1. Locked Decisions

1. Deployment target is **single H200 (141 GB VRAM)**.
2. Generation model is **`Qwen/Qwen3.5-122B-A10B-GPTQ-Int4`** served locally.
3. **Typed JSON contract and response validators remain unchanged** in pipeline call sites.
4. **Local task queue + local worker** replaces remote Phase 2-4 queue dispatch path.
5. **GCS remains system of record** for media/artifact storage.
6. **Vertex remains embedding backend** for both text-only semantic and multimodal embeddings.
7. **Spanner remains graph DB** for run graph persistence and traversal.
8. Data that was only persisted for cross-service decoupling (Cloud Tasks/Cloud Run boundary) should move to local-GPU lifecycle storage unless required for audit/replay.

---

## 2. Objectives and Non-Goals

### 2.1 Objectives

1. Remove Gemini generation dependency from Phase 2-4 and signals.
2. Keep existing output schemas and validation strictness.
3. Increase throughput by co-locating generation and Phase 2-4 worker with GPU-aware scheduling.
4. Reduce external API spend while maintaining quality and deterministic failure behavior.
5. Preserve current runtime behavior parity: no reduction in existing batching geometry or per-call generation ceilings solely to fit a smaller context window.

### 2.2 Non-Goals

1. Replacing Vertex embeddings.
2. Replacing Spanner with local graph store.
3. Introducing multi-node distributed inference in v1.
4. Rewriting prompt logic or changing Phase scoring semantics in v1.

---

## 3. Current vs Target Runtime Topology

### 3.1 Current (Baseline)

- Phase 1 runs on DO GPU host (VibeVoice + RF-DETR pipeline).
- Phase 2-4 run through remote worker entrypoint (`Cloud Tasks -> phase24_worker_app` path).
- Generation calls route through `VertexGeminiClient`.
- Embeddings route through Vertex.

### 3.2 Target (Single-H200 Full Hosting)

- Same host runs:
  - Phase 1 media pipeline (VibeVoice/RF-DETR/audio chain),
  - local vLLM OpenAI-compatible server for Qwen3.5 GPTQ Int4,
  - local Phase 2-4 queue dispatcher,
  - local Phase 2-4 worker(s) with admission control.
- Vertex embeddings and Spanner/GCS clients remain remote service dependencies.
- Queue ingress and worker execution are local-only by default (no Cloud Tasks hop in normal mode).

---

## 4. Inference Serving Design (Qwen on vLLM)

## 4.1 Serving Interface

1. Deploy vLLM in OpenAI-compatible mode (`/v1/chat/completions`).
2. Use an internal provider client implementing current LLM interface (`generate_json(...)` contract).
3. Keep all pipeline validators as-is (Pydantic/JSON schema checks unchanged).

## 4.2 Qwen-Specific Request Policy

1. Pass `chat_template_kwargs.enable_thinking=false` by default for current "minimal/low thinking" profile parity.
2. Keep per-call temperature/top_p/top_k pinned to deterministic defaults unless explicitly overridden.
3. Continue fail-fast behavior on malformed JSON or schema mismatch.
4. Preserve existing per-call output ceilings used in current pipeline (notably `max_output_tokens=32768` at current call sites).

## 4.3 vLLM Runtime Knobs to Expose

Expose these as environment-driven, versioned tuning parameters:

- `--gpu-memory-utilization`
- `--max-model-len`
- `--max-num-seqs`
- `--max-num-batched-tokens`
- `--enable-prefix-caching`
- `--kv-cache-memory-bytes` (optional override)
- `--kv-cache-dtype` (auto/fp8 when supported + validated)
- `--speculative-config` (MTP; optional gate)

Notes:
- vLLM V1 chunked prefill behavior should be assumed/tuned via batched token limits.
- Preemption signals must be observed in metrics and used to back off concurrency.

## 4.4 Initial Tuning Profiles (Starting Points)

These are start points for calibration, not final fixed values.
The initial default baseline for first implementation/testing is the Conservative profile below.

Context budget policy for parity:
- Because current runtime defaults use high per-call output ceilings (`32768`) and large structured prompt payloads, `max_model_len` must not be set to 16K for parity mode.
- Baseline parity target in v1 is `max_model_len >= 131072` with preferred target `262144` when VRAM allows.

1. **Conservative**
   - `gpu_memory_utilization=0.86`
   - `max_num_seqs=16`
   - `max_num_batched_tokens=16384`
   - `max_model_len=131072`
2. **Balanced**
   - `gpu_memory_utilization=0.90`
   - `max_num_seqs=12`
   - `max_num_batched_tokens=12288`
   - `max_model_len=131072`
3. **Throughput-biased**
   - `gpu_memory_utilization=0.92`
   - `max_num_seqs=16`
   - `max_num_batched_tokens=16384`
   - `max_model_len=262144`

Speculative decoding (MTP) is feature-flagged and benchmark-gated before default enablement.

---

## 5. Local Queue + Worker Architecture

## 5.1 Queue Backend

Adopt a **durable local queue** for Phase 2-4 dispatch with lease semantics:

1. SQLite WAL-backed queue file on local NVMe (`/opt/clypt-queue/phase24.db`) for v1.
2. Job schema includes:
   - `job_id`, `run_id`, `status`, `attempt_count`, `available_at`, `locked_at`, `worker_id`, `payload_json`, `last_error`.
3. Lease timeout and heartbeat allow recovery from process crash.
4. Idempotent key is `run_id` (plus query version when needed).

Rationale: durable local queue without introducing Redis/Kafka dependency in v1.

## 5.2 Worker Model

1. One dispatcher process receives handoff and enqueues local jobs.
2. Local worker pool pops leased jobs and executes `V31LivePhase14Runner`.
3. Worker concurrency is bounded by:
   - available VRAM headroom,
   - active Phase 1 GPU load,
   - live preemption/OOM telemetry.

## 5.3 Failure/Retry Policy

1. Retry only transient classes (LLM 5xx, network timeout, Vertex transient errors).
2. Schema violations and deterministic prompt/contract failures remain terminal.
3. On repeated preemption/OOM, treat as terminal fail-fast: stop dequeue/processing for the affected run, emit structured error logs/metrics, and mark run failed (no automatic profile downshift within that run).

---

## 6. Data Locality Rules

1. Keep temporary Phase 2-4 intermediate artifacts local by default:
   - prompt expansion scratch,
   - candidate intermediate ranking payloads,
   - subgraph transient debug snapshots.
2. Persist to Spanner only:
   - graph primitives and links used by traversal/querying,
   - canonical candidate outputs,
   - attribution records required for UI/debug contracts.
3. Persist to GCS only:
   - source and durable run artifacts already part of runtime contract,
   - replay-critical outputs.

---

## 7. Code-Level Implementation Plan (Spec-Level)

## 7.1 Provider Layer

1. Add `backend/providers/openai_local.py`:
   - `LocalOpenAIQwenClient` implementing current generation interface.
   - strict JSON extraction + validator pass-through.
   - retry/backoff semantics parallel to current Gemini client behavior.
2. Extend `backend/providers/config.py`:
   - add generation backend type `local_openai`,
   - add local server URL/model/timeouts/retry settings,
   - keep existing Vertex embedding settings untouched.
3. Wire backend selection in provider factory paths without touching pipeline contracts.

## 7.2 Runtime Wiring

1. Update `backend/runtime/phase24_worker_app.py` bootstrapping:
   - choose generation client by env backend selector,
   - preserve `VertexEmbeddingClient` usage.
2. Add local queue runtime module:
   - `backend/runtime/phase24_local_queue.py` (enqueue/dequeue/lease API),
   - `backend/runtime/phase24_local_dispatcher.py` (handoff adapter),
   - `backend/runtime/phase24_local_worker.py` (poll + execute loop).
3. Remove remote queue/generation fallback wiring for this branch; local queue + local Qwen path is the only supported execution path.

## 7.3 Runner and Pipeline

1. No schema contract changes in:
   - Phase 2 merge/boundary outputs,
   - Phase 3 edge outputs,
   - Phase 4 meta/subgraph/pool outputs,
   - signal callpoint outputs.
2. Keep all validators in place and fail-fast on mismatch.
3. Add queue-aware structured logs with run/job/attempt correlation IDs.

## 7.4 Config Surface Additions

Add new environment variables (names final after implementation review):

- `CLYPT_GEN_BACKEND=local_openai`
- `CLYPT_LOCAL_LLM_BASE_URL=http://127.0.0.1:8001/v1`
- `CLYPT_LOCAL_LLM_MODEL=Qwen/Qwen3.5-122B-A10B-GPTQ-Int4`
- `CLYPT_LOCAL_LLM_TIMEOUT_S`
- `CLYPT_LOCAL_LLM_MAX_RETRIES`
- `CLYPT_LOCAL_LLM_ENABLE_THINKING=0|1`
- `CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite`
- `CLYPT_PHASE24_LOCAL_QUEUE_PATH`
- `CLYPT_PHASE24_LOCAL_MAX_INFLIGHT`
- `CLYPT_PHASE24_LOCAL_MAX_REQUESTS_PER_WORKER`
- `CLYPT_PHASE24_LOCAL_POLL_INTERVAL_MS`
- `CLYPT_PHASE24_LOCAL_LEASE_TIMEOUT_S`
- `CLYPT_VLLM_PROFILE=conservative|balanced|throughput`
- `CLYPT_VLLM_MAX_NUM_SEQS`
- `CLYPT_VLLM_MAX_NUM_BATCHED_TOKENS`
- `CLYPT_VLLM_GPU_MEMORY_UTILIZATION`
- `CLYPT_VLLM_MAX_MODEL_LEN`
- `CLYPT_VLLM_SPECULATIVE_MODE=off|mtp`
- `CLYPT_VLLM_SPECULATIVE_NUM_TOKENS`

Operational note:
- If the API wrapper runs under gunicorn/uvicorn-worker mode, expose worker recycle controls (`--max-requests` and `--max-requests-jitter`) to bound long-run memory fragmentation risk.
- Preserve existing Phase 2-4 generation ceilings by default (do not downshift `CLYPT_PHASE2_MERGE_MAX_OUTPUT_TOKENS`, `CLYPT_PHASE2_BOUNDARY_MAX_OUTPUT_TOKENS`, or callsite-level 32768 caps in Phase 3/4/signals during initial parity rollout).

---

## 8. GPU Scheduling and Throughput Control

## 8.1 Admission Controller

Introduce a local admission controller that decides Phase 2-4 dequeue eligibility based on:

1. current Phase 1 load (ASR/visual active),
2. vLLM queue depth and decode backlog,
3. recent preemption count and OOM events,
4. p95 latency SLO breaches.

## 8.2 Fail-Fast Rules

1. If preemption rises above threshold: fail the active run(s) with a typed runtime error and emit structured diagnostics; do not auto-adjust `max_num_seqs` or inflight limits in-run.
2. If p95 latency breaches threshold (while GPU memory remains stable): fail the active run(s) with a typed runtime error and emit structured diagnostics; do not auto-adjust `max_num_batched_tokens` in-run.
3. If throughput is below target without preemption/OOM: keep run behavior deterministic; tuning changes are manual/operator-driven between runs, not automatic in-run profile shifts.

---

## 9. Validation and Benchmark Plan

## 9.1 Functional Parity

1. Golden-set replay across representative videos:
   - compare schema validity, not exact text equality.
2. Confirm all Phase 2-4 and signal validators pass unchanged.
3. Confirm Spanner writes/traversal outputs remain query-compatible.

## 9.2 Performance Benchmarks

1. Measure:
   - TTFT, ITL, tokens/sec, requests/sec,
   - end-to-end run latency (Phase 1-4),
   - queue wait time and worker utilization.
2. Run three load envelopes:
   - single-run latency,
   - moderate concurrent runs,
   - stress + recovery (spike then steady).
3. Track vLLM preemption and KV-cache pressure metrics during each run.

## 9.3 Reliability Tests

1. Kill worker process mid-run; ensure lease recovery.
2. Restart vLLM; ensure retry/backoff + terminal behavior are correct.
3. Corrupt one LLM response; ensure validator catches and fails job deterministically.

---

## 10. Rollout Strategy

1. **Stage A:** local queue + local Qwen path in test mode; validate schema/quality/perf on sampled replays.
2. **Stage B:** canary (10-20% runs) on single H200 with conservative profile.
3. **Stage C:** full cutover with balanced profile and automatic backpressure enabled.
4. **Stage D:** optional MTP/speculative enablement only after separate benchmark signoff.

Rollback for this branch does not switch to Gemini/Cloud Tasks. If severe issues occur, stop processing and fail runs until the local path is fixed.

---

## 11. Acceptance Criteria

1. Phase 2-4 and signal generation executes entirely via local Qwen server.
2. All existing typed JSON validators are unchanged and passing.
3. Vertex embeddings, Spanner persistence/traversal, and GCS flows remain operational.
4. Local queue/worker handles retries, leases, and crash recovery.
5. No uncontrolled GPU throttling/preemption at target concurrency envelope.
6. End-to-end quality is acceptable against current baseline on agreed evaluation set.

---

## 12. Risks and Mitigations

1. **Single-GPU contention (VibeVoice/RF-DETR/Qwen)**  
   Mitigation: admission controller + profile tiers + hard concurrency caps.
2. **Schema drift from different model behavior**  
   Mitigation: keep strict validators and add repair loop only if validator-safe.
3. **Queue durability edge cases**  
   Mitigation: WAL + leases + crash-recovery integration tests.
4. **Tail latency spikes under mixed prompt lengths**  
   Mitigation: tune `max_num_batched_tokens` and partial prefill-related settings.

---

## 13. External References

1. Qwen model card: https://huggingface.co/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4  
2. vLLM OpenAI server: https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html  
3. vLLM optimization guide: https://docs.vllm.ai/en/stable/configuration/optimization/  
4. NVIDIA DGX Spark thread (optimization context): https://forums.developer.nvidia.com/t/qwen3-5-122b-a10b-on-single-spark-up-to-51-tok-s-v2-1-patches-quick-start-benchmark/365639
