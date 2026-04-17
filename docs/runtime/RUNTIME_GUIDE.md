# RUNTIME GUIDE

**Status:** Active  
**Last updated:** 2026-04-17

This is the runtime source of truth for current backend code behavior (Phases 1-4).

For the full environment variable catalog, see `docs/runtime/ENV_REFERENCE.md`.

## 1) Runtime Topology

Phase 1 is conceptually split into two sub-chains:

- **Phase 1 audio chain:** VibeVoice vLLM ASR → NeMo forced aligner →
  emotion2vec+ → YAMNet (CPU).
- **Phase 1 visual chain:** RF-DETR + ByteTrack (TensorRT FP16 fast path).

Both chains run in parallel, and the audio chain's completion callback
drives the Phase 2-4 enqueue without waiting for the visual chain.

### 1.1 Current single-host runtime

- **Phase 1 runtime:** local process (`python -m backend.runtime.run_phase1`)
  backed by `Phase1JobRunner`. Audio + visual sub-chains run in the same
  process via a `ThreadPoolExecutor`.
- **Phase 1 runtime env:** `/opt/clypt-phase1/venvs/phase1` on the DO host.
- **ASR backend:** local VibeVoice vLLM on the same host
  (`CLYPT_PHASE1_ASR_BACKEND=vllm`, `VIBEVOICE_VLLM_BASE_URL`). No remote
  ASR offload path exists today.
- **Qwen serving env:** dedicated SGLang env at
  `/opt/clypt-phase1/venvs/sglang`.
- **Qwen serving tuning:** `deploy_sglang_qwen_service.sh` sources
  `/etc/clypt-phase1/v3_1_phase1.env` and can drive SGLang launch with
  `SG_SCHEDULE_POLICY`, `SG_CHUNKED_PREFILL_SIZE`, `SG_MEM_FRACTION_STATIC`,
  `SG_CONTEXT_LENGTH`, and `SG_EXTRA_ARGS`.
- **Phase 2-4 dispatch path:** local SQLite queue + local worker loop.
  - Queue backend must be `CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite`.
  - Phase 1 enqueues to local queue through `Phase24LocalDispatcherClient`.
- **Phase 2-4 worker service:**
  `python -m backend.runtime.run_phase24_local_worker` from the Phase 1
  env, with systemd dependency on `clypt-sglang-qwen.service`.
- **Phase 2-4 node-media prep:** runs in-process on the worker host
  (ffmpeg + GCS upload) via `node_media_preparer=None` fallback. There is
  no remote media-prep service in the current code.
- **Generation backend for Phase 2-4:** local OpenAI-compatible endpoint
  (`GENAI_GENERATION_BACKEND=local_openai` enforced by
  `build_default_phase24_worker_service()`).
- **Embedding backend:** Vertex (`VERTEX_EMBEDDING_BACKEND=vertex` default).
- **Storage and graph persistence:** GCS + Spanner remain active
  dependencies.

### 1.2 Target topology (RTX 6000 Ada refactor, in progress)

Two single-GPU DigitalOcean droplets:

- **Phase 1 audio host — RTX 6000 Ada (48 GB):** VibeVoice vLLM ASR, NFA,
  emotion2vec+, YAMNet (CPU), and NVENC/NVDEC ffmpeg for Phase 2 node-clip
  extraction. VibeVoice runs at native dtype (no bf16 audio-encoder
  patch) thanks to the extra VRAM headroom.
- **Phase 1 visual + Phase 2-4 host — H200:** RF-DETR + ByteTrack
  (TensorRT FP16), SGLang Qwen3.6-35B-A3B on `:8001`, and the Phase 2-4
  local queue + worker.

Why the split:

- H200 NVENC is **not usable** for our ffmpeg clip-extraction path
  (`h264_nvenc` returns `unsupported device (2)`). Node-media prep must
  run on a GPU with working NVENC/NVDEC. The RTX 6000 Ada provides one.
- RTX 6000 Ada's 48 GB VRAM lets VibeVoice run native dtype with room to
  spare for NFA/emotion2vec+, without the L4-era bf16 patch.
- Removing VibeVoice + ffmpeg from the H200 frees SM time and VRAM for
  RF-DETR and SGLang.

Refactor status and migration plan:
[`docs/deployment/REFACTOR_RTX6000ADA.md`](../deployment/REFACTOR_RTX6000ADA.md).

## 2) Phase 1 Execution Semantics

Phase 1 executes visual and audio sub-chains concurrently. Under the
target topology, the two chains run on different hosts; under the current
single-host code path they share a process via `ThreadPoolExecutor`.

```text
Phase 1 visual chain (H200):
  RF-DETR + ByteTrack --------------------------------\
                                                       +--> join
Phase 1 audio chain (RTX 6000 Ada target / H200 today):
  VibeVoice vLLM ASR ---------------------------------/
     |
     +--> NeMo forced aligner -> emotion2vec+ -> YAMNet (CPU)
```

Critical invariant in code: the audio chain launches immediately after
ASR completion and does not wait for visual completion. Phase 2-4
enqueue fires from the audio-chain completion callback, not from the
visual join.

## 3) Phase 1 Input Contract

- `CLYPT_PHASE1_INPUT_MODE` must be `test_bank` (other values raise).
- In strict mode (`CLYPT_PHASE1_TEST_BANK_STRICT=1`), `CLYPT_PHASE1_TEST_BANK_PATH` is required.
- Test-bank mapping supports `video_gcs_uri` and `audio_gcs_uri` hydration/signing paths.
- Known-good DO H200 runtime uses `CLYPT_PHASE1_VISUAL_BACKEND=tensorrt_fp16`.
- That path requires both host `trtexec` (`libnvinfer-bin`) and Python TensorRT in the Phase 1 env; `deploy_vllm_service.sh` now provisions both automatically.
- The current TensorRT fast path decodes directly to detector resolution on GPU (`scale_cuda`) before `hwdownload`, then performs tensor conversion, resize/shape reconciliation, and Imagenet normalization on CUDA with `torch`.
- The runtime preserves original source frame dimensions alongside the resized decode output so TensorRT postprocess can still rescale detection boxes back to source-video coordinates correctly.

### 3.1 Phase 1 visual fast path

For `CLYPT_PHASE1_VISUAL_BACKEND=tensorrt_fp16`, the intended high-throughput path is:

```text
NVDEC -> scale_cuda to detector resolution -> hwdownload RGB -> CUDA tensor/normalize -> TensorRT -> ByteTrack
```

Operationally, this means:

- the host should not be doing full-resolution OpenCV resize work in the hot loop
- CUDA preprocess is part of the intended runtime behavior, not an optional micro-optimization
- box geometry must still be interpreted against the original source dimensions, not the resized decode surface

On the 2026-04-15 H200 reference replay for `source_video.mp4` (Billy Carson test-bank asset), this path sustained about `240.1 fps` over `35705` frames with `batch_size=16`. The immediately previous host-side resize/normalize path on the same workload ran at about `51.5 fps`.

## 4) Phase 2-4 Local Queue / Worker Contract

### 4.1 Queue model

- SQLite WAL queue at `CLYPT_PHASE24_LOCAL_QUEUE_PATH` (default `backend/outputs/phase24_local_queue.sqlite`).
- `run_id` is unique; enqueue is idempotent by `run_id`.
- `claim_next()` supports optional expired-lease reclaim.

### 4.2 Default fail-fast lease behavior

Current defaults loaded from config:

- `CLYPT_PHASE24_LOCAL_RECLAIM_EXPIRED_LEASES=0`
- `CLYPT_PHASE24_LOCAL_FAIL_FAST_ON_STALE_RUNNING=1`

Meaning:

- Worker does **not** auto-requeue stale `running` leases by default.
- If stale running lease is detected, worker raises fail-fast stop and requires manual operator intervention.

### 4.3 Crash classification

`classify_phase24_exception()` maps key failures:

- `connection refused`, `xgrammar`, `compile_json_schema`, `EngineCore` patterns => `FAIL_FAST`
- transient HTTP (`429/500/502/503/504`) => retryable transient
- schema/type/validation/json errors => non-transient

### 4.4 Phase24 worker backend gate

`build_default_phase24_worker_service()` hard-enforces:

- only `GENAI_GENERATION_BACKEND=local_openai`
- local model must be set (`CLYPT_LOCAL_LLM_MODEL` or `GENAI_FLASH_MODEL` fallback)
- node-media prep runs in-process on the worker host (there is no remote backend knob today)

## 5) Local OpenAI Qwen Call Behavior

`LocalOpenAIQwenClient.generate_json()` behavior:

- always sends OpenAI-compatible `/chat/completions`
- always uses `response_format.type=json_schema` with `strict=true`
- enforces `additionalProperties=false` recursively on object schemas
- forces non-thinking payload (top-level `chat_template_kwargs.enable_thinking=false`; sent flat, not under `extra_body`, because the client posts JSON directly via stdlib and does not rely on OpenAI-SDK `extra_body` unwrapping)
- forwards configured `max_output_tokens` as a real `max_tokens` request cap on structured-output calls
- warns when the serving backend returns `finish_reason=length`, because that is the strongest signal that a structured JSON reply was token-capped
- validates parsed response against required/type/enum subset

## 6) Phase 2-4 Concurrency Guidance

All Phase 2-4 stage concurrency envs are **max in-flight LLM request caps**,
not throughput targets. Smaller videos will naturally stay well below these
caps; only large videos will scale up to them. Defaults were validated against
Qwen3.6-35B-A3B on H200 (2026-04-16 bench, see
`docs/specs/2026-04-16_qwen36_swap_and_sglang_tuning_spec.md`).

Current code-backed defaults:

- `CLYPT_GEMINI_MAX_CONCURRENT` has been removed; startup fails if it is still set
- `CLYPT_PHASE2_MAX_CONCURRENT` has been renamed to `CLYPT_PHASE2_MERGE_MAX_CONCURRENT`; the old name now raises at config load
- use explicit per-stage envs:
  - `CLYPT_PHASE2_MERGE_MAX_CONCURRENT=16`
  - `CLYPT_PHASE2_BOUNDARY_MAX_CONCURRENT=16`
  - `CLYPT_SIGNAL_MAX_CONCURRENT=8`
  - `CLYPT_PHASE3_LOCAL_MAX_CONCURRENT=24`
  - `CLYPT_PHASE3_LONG_RANGE_MAX_CONCURRENT=24`
  - `CLYPT_PHASE4_SUBGRAPH_MAX_CONCURRENT=16`
- keep Phase 3 long-range shortlist controls explicit:
  - `CLYPT_PHASE3_LONG_RANGE_TOP_K=2`
  - `CLYPT_PHASE3_LONG_RANGE_PAIRS_PER_SHARD=24`
- full env inventory lives in `docs/runtime/ENV_REFERENCE.md`

Reference measurements captured on `2026-04-15` against the live droplet service:

- `phase4_subgraph` prompt (`~4.4k chars`): concurrency `1` took about `79.6s`; concurrency `2` took about `88.6s` wall-clock for `2` successful requests, improving aggregate throughput from `0.0126 req/s` to `0.0226 req/s`
- `phase3_local` prompt (`~10.5k chars`): concurrency `1` took about `71.6s`; concurrency `2` took about `103.6s` wall-clock for `2` successful requests, improving aggregate throughput from `0.0140 req/s` to `0.0193 req/s`
- a broader Phase 4 subgraph sweep over `1,2,4,6,8` was not operationally useful because the higher-concurrency lanes stalled long enough to require aborting the run

Repeat the benchmark with:

```bash
python scripts/bench_phase24_llm_concurrency.py \
  --scenario phase4_subgraph \
  --concurrency-values 1,2 \
  --rounds 1 \
  --timeout-s 120
```

## 7) Canonical Local Runtime Commands

### 7.1 Load environment

```bash
cd /opt/clypt-phase1/repo
source /opt/clypt-phase1/venvs/phase1/bin/activate
set -a; source /etc/clypt-phase1/v3_1_phase1.env; set +a
```

### 7.2 Verify core services

```bash
systemctl is-active clypt-vllm-vibevoice clypt-sglang-qwen clypt-v31-phase1-api clypt-v31-phase1-worker clypt-v31-phase24-local-worker
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8001/v1/models | python3 -m json.tool
trtexec --help >/dev/null
/opt/clypt-phase1/venvs/phase1/bin/python -c "import tensorrt as trt; print(trt.__version__)"
/opt/clypt-phase1/venvs/sglang/bin/python -c "import sglang, torch; print(sglang.__version__, torch.__version__)"
systemctl cat clypt-sglang-qwen | rg "ExecStart|schedule-policy|chunked-prefill-size|mem-fraction-static|context-length"
```

### 7.3 Run Phase 1 + queue-mode Phase 2-4

```bash
python -m backend.runtime.run_phase1 \
  --job-id "run_$(date +%Y%m%d_%H%M%S)" \
  --source-path /opt/clypt-phase1/videos/<video>.mp4 \
  --run-phase14
```

### 7.4 Inspect local queue state

```bash
python - <<'PY'
import sqlite3
db = "backend/outputs/phase24_local_queue.sqlite"
conn = sqlite3.connect(db)
for row in conn.execute("SELECT run_id,status,attempt_count,locked_at,last_error FROM phase24_jobs ORDER BY updated_at DESC LIMIT 20"):
    print(row)
PY
```

## 8) Required Environment Variables (Code-Enforced)

Always required:

- `GOOGLE_CLOUD_PROJECT`
- `GCS_BUCKET`
- `VIBEVOICE_BACKEND=vllm`
- `GENAI_GENERATION_BACKEND=local_openai` for the local Phase 2-4 worker
- `CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite` for the local queue runtime

Required when `CLYPT_PHASE1_ASR_BACKEND=vllm` (the only supported mode):

- `VIBEVOICE_VLLM_BASE_URL`
- `VIBEVOICE_VLLM_MODEL=vibevoice`

Required for local OpenAI generation:

- `CLYPT_LOCAL_LLM_BASE_URL`
- `CLYPT_LOCAL_LLM_MODEL` or a compatible fallback such as `GENAI_FLASH_MODEL`

For the full env catalog, see `docs/runtime/ENV_REFERENCE.md`.

## 9) Known High-Signal Failure Modes

1. **GPU runtime/container startup failure:** missing host NVIDIA libs (`libnvidia-ml.so.1`) causes vLLM container startup loop failure.
2. **Qwen endpoint crash / refusal:** local worker marks failed and fail-fast stops on `connection refused` signatures.
3. **Stale queue lease:** with default fail-fast settings, worker exits until operator manually resolves stale running row.
4. **Structured output mismatch:** schema/type violations are terminal non-transient failures.
5. **Shared-env drift:** if SGLang and Phase 1 share a venv, SGLang can overwrite Phase 1 runtime packages; the current deploy path avoids this by using separate envs.
6. **Host NVENC mismatch:** node-media prep uses the worker host's ffmpeg; on hosts without NVENC (e.g., H200), `h264_nvenc` will fail with `unsupported device (2)`. This is a primary driver for the RTX 6000 Ada split (see `docs/deployment/REFACTOR_RTX6000ADA.md`). Until the refactor lands, either configure ffmpeg to use an available encoder (e.g., `libx264`) or run node-media prep on a host with NVENC.
7. **Visual throughput unexpectedly low on TensorRT path:** if a visual-only H200 replay drops back toward `~50 fps` instead of `~240 fps`, the most likely causes are stale code on-host, loss of `scale_cuda` decode-to-resolution behavior, or a fallback to host-side preprocessing before TensorRT.

For incident history and recovery notes, see `docs/ERROR_LOG.md`.
