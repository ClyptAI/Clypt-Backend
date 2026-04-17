# RUNTIME GUIDE

**Status:** Active  
**Last updated:** 2026-04-17

This is the runtime source of truth for current backend code behavior (Phases 1-4).

For the full environment variable catalog, see `docs/runtime/ENV_REFERENCE.md`.

## 1) Runtime Topology

Phase 1 runs on two single-GPU DigitalOcean droplets. There is **no local
fallback** — the H200 always calls the RTX host over HTTP for VibeVoice
ASR and node-media prep; config loading fails fast if the remote host
URL/token are missing.

Phase 1 is conceptually split into two sub-chains:

- **Phase 1 audio chain:**
  - ASR leg (RTX 6000 Ada, sole tenant): VibeVoice vLLM ASR, dispatched
    via one `POST /tasks/vibevoice-asr` call per run. Returns
    `{turns, stage_events}`.
  - Post-ASR leg (H200, in-process): NeMo forced aligner →
    emotion2vec+ → YAMNet (CPU). Runs on the H200 immediately when the
    ASR HTTP call returns.
- **Phase 1 visual chain (H200):** RF-DETR + ByteTrack (TensorRT FP16 fast
  path), in-process.

Both chains run in parallel. The audio chain's completion callback drives
the Phase 2-4 enqueue without waiting for visual join.

### 1.1 Host responsibilities

| Host | Runs |
| --- | --- |
| **H200** | Phase 1 orchestrator (`run_phase1`, Phase 1 API/worker), RF-DETR + ByteTrack, in-process NFA + emotion2vec+ + YAMNet (CPU), Phase 2-4 local SQLite queue + worker, SGLang Qwen3.6-35B-A3B on `:8001`, Vertex embedding calls, Spanner/GCS I/O. |
| **RTX 6000 Ada (sole tenant)** | VibeVoice vLLM on `:8000`, ffmpeg NVENC/NVDEC node-clip extraction, FastAPI service (`backend.runtime.phase1_audio_service.app:create_app`) exposing `/tasks/vibevoice-asr` and `/tasks/node-media-prep`. |

### 1.2 H200 runtime

- **Phase 1 runtime:** local process (`python -m backend.runtime.run_phase1`)
  backed by `Phase1JobRunner`. Visual chain runs in-process; ASR is
  dispatched to the RTX host via `RemoteVibeVoiceAsrClient` (legacy
  alias: `RemoteAudioChainClient`), and NFA / emotion2vec+ / YAMNet run
  in-process on the H200 after the ASR response returns.
- **Phase 1 runtime env:** `/opt/clypt-phase1/venvs/phase1` on the H200.
- **ASR backend:** exclusively remote. `backend/providers/config.py`
  requires `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL` and
  `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN`; there is no
  in-process VibeVoice provider on this host. The legacy
  `CLYPT_PHASE1_AUDIO_HOST_URL` / `_TOKEN` names remain accepted for one
  release as deprecated aliases.
- **Qwen serving env:** dedicated SGLang env at
  `/opt/clypt-phase1/venvs/sglang`, driven by
  `SG_SCHEDULE_POLICY`, `SG_CHUNKED_PREFILL_SIZE`, `SG_MEM_FRACTION_STATIC`,
  `SG_CONTEXT_LENGTH`, `SG_EXTRA_ARGS`. Current effective flags:
  `--context-length 65536` (reduced from 131072),
  `--kv-cache-dtype fp8_e4m3`, `--mem-fraction-static 0.78`,
  `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`,
  `--mamba-scheduler-strategy extra_buffer`, `--schedule-policy lpm`,
  `--chunked-prefill-size 8192`, `--grammar-backend xgrammar`,
  `--reasoning-parser qwen3`, plus environment `HF_HUB_OFFLINE=1` and
  `SGLANG_ENABLE_SPEC_V2=1` (both exported from the systemd unit; the Mamba
  hybrid refuses to start with MTP + radix cache unless SPEC_V2 is on).
  `HF_HUB_OFFLINE=1` means SGLang never phones home at startup, so the
  Qwen snapshot must already be resident in `HF_HOME=/opt/clypt-phase1/hf-cache`
  before the unit is started; `bootstrap_h200.sh` pre-downloads it, and
  the model-revision refresh procedure lives in `docs/deployment/P1_DEPLOY.md` §3.2.
- **Phase 2-4 dispatch path:** local SQLite queue + local worker loop.
  `CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite`; Phase 1 enqueues through
  `Phase24LocalDispatcherClient`.
- **Phase 2-4 worker service:**
  `python -m backend.runtime.run_phase24_local_worker` from the Phase 1
  env, with systemd dependency on `clypt-sglang-qwen.service`.
- **Phase 2-4 node-media prep:** delegated to the RTX host via
  `RemoteNodeMediaPrepClient`. `CLYPT_PHASE24_NODE_MEDIA_PREP_URL` +
  `CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN` are required; there is no local
  ffmpeg fallback on the H200.
- **Generation backend for Phase 2-4:** local OpenAI-compatible endpoint
  (`GENAI_GENERATION_BACKEND=local_openai` enforced by
  `build_default_phase24_worker_service()`).
- **Embedding backend:** Vertex (`VERTEX_EMBEDDING_BACKEND=vertex` default).
- **Storage and graph persistence:** GCS + Spanner remain active
  dependencies on the H200.

### 1.3 RTX 6000 Ada runtime (sole tenant)

- **Service:** `uvicorn backend.runtime.phase1_audio_service.app:create_app`
  behind `clypt-audio-host.service`. Bearer-auth bound to
  `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN` (legacy alias:
  `CLYPT_PHASE1_AUDIO_HOST_TOKEN`).
- **vLLM:** `clypt-vllm-vibevoice.service` runs the VibeVoice container at
  `127.0.0.1:8000`. Current sole-tenant vLLM flags:
  - `--gpu-memory-utilization 0.77` (leaves ~8 GiB free for NVDEC contexts)
  - `--max-num-seqs 2`
  - `--dtype bfloat16`
  - CUDA graph capture enabled (`enforce_eager=False`, the default)
  - No speculative decoding (VibeVoice is Whisper encoder-decoder, not
    decoder-only; MTP heads do not apply)
  - The earlier co-tenancy workarounds (`--max-num-seqs 1`,
    `--max-model-len 32768`, `--enforce-eager`,
    `--gpu-memory-utilization 0.60`,
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) are gone.
- **Docker image:** VibeVoice is baked into the image (767 MB layer:
  ffmpeg, libsndfile1, vibevoice[vllm] deps). Model weights are on the
  host mount at `/root/.cache/huggingface`. Cold restart ~45 s.
- **Providers held hot in-process:** VibeVoice vLLM client, GCS storage
  client. NFA, emotion2vec+, and YAMNet are **not** instantiated on this
  host anymore. Singletons are constructed lazily on first request via
  `backend/runtime/phase1_audio_service/deps.py`.
- **Node-media prep NVDEC:** clip extraction uses `-c:v h264_cuvid`
  (explicit NVDEC decode) + `h264_nvenc` (NVENC encode). Max concurrency
  is 8 (`CLYPT_PHASE24_NODE_MEDIA_PREP_MAX_CONCURRENCY=8`); 16 OOMs the
  NVENC input buffers at the current VRAM footprint. `-ss` seek is
  output-side (after `-i`) — intentional; input seek was evaluated but
  rejected.
- **Concurrency:** one global `asyncio.Lock` serializes ASR on the GPU;
  node-media prep runs under a bounded semaphore (cap=8) and can overlap
  with ASR. Concurrency ceilings on the caller side are controlled by
  `CLYPT_PHASE24_NODE_MEDIA_PREP_MAX_CONCURRENCY`.

Why the split: H200 NVENC is unusable (`h264_nvenc` returns
`unsupported device (2)`), and NFA global alignment OOM'd reliably when
co-tenant with VibeVoice on the 48 GiB RTX card. Keeping the RTX as the
VibeVoice + ffmpeg sole-tenant host and moving NFA / emotion2vec+ /
YAMNet back to the H200 gets us a working NVENC path **and** a stable
NFA run against 141 GiB of H200 memory.

## 2) Phase 1 Execution Semantics

Phase 1 executes visual and audio sub-chains concurrently across the two
hosts. The H200 runs the visual chain and the audio post-processing
chain in-process, and awaits the ASR HTTP round-trip to the RTX host.

```text
Phase 1 visual chain (H200, in-process):
  RF-DETR + ByteTrack --------------------------------\
                                                       +--> join
Phase 1 audio chain:
  RTX 6000 Ada (POST /tasks/vibevoice-asr):
    VibeVoice vLLM ASR -------------------------------/
       |
       v   (HTTP response)
  H200 (in-process):
    NeMo forced aligner -> emotion2vec+ -> YAMNet (CPU)
```

Critical invariant in code: the audio post-processing chain launches
immediately after ASR completion and does not wait for visual
completion. Phase 2-4 enqueue fires from the audio-chain completion
callback once the H200-side NFA / emotion2vec+ / YAMNet steps finish —
not from the visual join.

## 3) Phase 1 Input Contract

- `CLYPT_PHASE1_INPUT_MODE` must be `test_bank` (other values raise).
- In strict mode (`CLYPT_PHASE1_TEST_BANK_STRICT=1`), `CLYPT_PHASE1_TEST_BANK_PATH` is required.
- Test-bank mapping supports `video_gcs_uri` and `audio_gcs_uri` hydration/signing paths.
- Known-good DO H200 runtime uses `CLYPT_PHASE1_VISUAL_BACKEND=tensorrt_fp16`.
- That path requires both host `trtexec` (`libnvinfer-bin`) and Python TensorRT in the Phase 1 env; `deploy_visual_service.sh` now provisions both automatically.
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
- node-media prep is always delegated to the remote RTX host via
  `RemoteNodeMediaPrepClient`; config load fails if
  `CLYPT_PHASE24_NODE_MEDIA_PREP_URL` / `CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN`
  are unset

## 5) Local OpenAI Qwen Call Behavior

`LocalOpenAIQwenClient.generate_json()` behavior:

- always sends OpenAI-compatible `/chat/completions`
- always uses `response_format.type=json_schema` with `strict=true`
- enforces `additionalProperties=false` recursively on object schemas
- forces non-thinking payload (top-level `chat_template_kwargs.enable_thinking=false`; sent flat, not under `extra_body`, because the client posts JSON directly via stdlib and does not rely on OpenAI-SDK `extra_body` unwrapping)
- forwards configured `max_output_tokens` as a real `max_tokens` request cap on structured-output calls
- warns when the serving backend returns `finish_reason=length`, because that is the strongest signal that a structured JSON reply was token-capped
- validates parsed response against required/type/enum subset
- every pipeline call site (`backend/pipeline/{candidates,graph,signals,semantics}/`) additionally wraps the returned dict with a stage-local Pydantic `StrictModel(extra="forbid")` declared in `<stage>/responses.py`; any prompt/response schema drift surfaces as a `ValidationError` at the call boundary rather than as a silent downstream dict-access bug. See `docs/ARCHITECTURE.md` §4.4.

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
- per-stage output token caps (DO-speedup-and-OSS-swap baseline, 2026-04-17):
  - `CLYPT_PHASE2_MERGE_MAX_OUTPUT_TOKENS=8192` (down from 32768)
  - `CLYPT_PHASE2_BOUNDARY_MAX_OUTPUT_TOKENS=8192` (down from 32768)
  - `CLYPT_PHASE4_POOL_MAX_OUTPUT_TOKENS=4096` (up from 2048)
  - `CLYPT_PHASE4_META_MAX_OUTPUT_TOKENS=4096` (up from 2048)
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

On the H200:

```bash
systemctl is-active clypt-sglang-qwen clypt-v31-phase1-api clypt-v31-phase1-worker clypt-v31-phase24-local-worker
test -f /etc/clypt/clypt-phase1-runtime.env && cat /etc/clypt/clypt-phase1-runtime.env
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8001/v1/models | python3 -m json.tool
trtexec --help >/dev/null
/opt/clypt-phase1/venvs/phase1/bin/python -c "import tensorrt as trt; print(trt.__version__)"
/opt/clypt-phase1/venvs/sglang/bin/python -c "import sglang, torch; print(sglang.__version__, torch.__version__)"
/opt/clypt-phase1/venvs/phase1/bin/python -c "import nemo, funasr, tensorflow, tensorflow_hub, librosa, resampy; print('audio post ok')"
systemctl cat clypt-sglang-qwen | rg "ExecStart|schedule-policy|chunked-prefill-size|mem-fraction-static|context-length"
curl -fsS -H "Authorization: Bearer ${CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN}" \
  "${CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL%/}/health"
```

On the RTX 6000 Ada:

```bash
systemctl is-active clypt-vllm-vibevoice clypt-audio-host
curl -fsS http://127.0.0.1:8000/health
curl -fsS "http://127.0.0.1:${CLYPT_PHASE1_AUDIO_HOST_PORT}/health"
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

Always required on the H200 (config load fails fast otherwise):

- `GOOGLE_CLOUD_PROJECT`
- `GCS_BUCKET`
- `GENAI_GENERATION_BACKEND=local_openai` for the local Phase 2-4 worker
- `CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite` for the local queue runtime
- `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL` (remote RTX VibeVoice ASR base URL)
- `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN` (bearer for the RTX service)
- `CLYPT_PHASE24_NODE_MEDIA_PREP_URL`
- `CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN`

> **Compat note:** `CLYPT_PHASE1_AUDIO_HOST_URL` /
> `CLYPT_PHASE1_AUDIO_HOST_TOKEN` are still accepted as deprecated
> aliases for one release but should not be used in new deployments.

VibeVoice environment variables (`VIBEVOICE_BACKEND`,
`VIBEVOICE_VLLM_BASE_URL`, `VIBEVOICE_VLLM_MODEL`) must **not** be set on
the H200. They belong to the RTX VibeVoice ASR host only.

Required on the RTX 6000 Ada host:

- `CLYPT_PHASE1_AUDIO_HOST_BIND`, `CLYPT_PHASE1_AUDIO_HOST_PORT`
- `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN` (must match the H200 value)
- `VIBEVOICE_BACKEND=vllm`, `VIBEVOICE_VLLM_BASE_URL`, `VIBEVOICE_VLLM_MODEL=vibevoice`
- `GOOGLE_CLOUD_PROJECT`, `GCS_BUCKET`, `GOOGLE_APPLICATION_CREDENTIALS`

Required for local OpenAI generation (H200):

- `CLYPT_LOCAL_LLM_BASE_URL`
- `CLYPT_LOCAL_LLM_MODEL` or a compatible fallback such as `GENAI_FLASH_MODEL`

For the full env catalog, see `docs/runtime/ENV_REFERENCE.md`.

## 9) Known High-Signal Failure Modes

1. **GPU runtime/container startup failure:** missing host NVIDIA libs (`libnvidia-ml.so.1`) causes vLLM container startup loop failure.
2. **Qwen endpoint crash / refusal:** local worker marks failed and fail-fast stops on `connection refused` signatures.
3. **Stale queue lease:** with default fail-fast settings, worker exits until operator manually resolves stale running row.
4. **Structured output mismatch:** schema/type violations are terminal non-transient failures.
5. **Shared-env drift:** if SGLang and Phase 1 share a venv, SGLang can overwrite Phase 1 runtime packages; the current deploy path avoids this by using separate envs.
6. **Remote VibeVoice ASR unavailable:** Phase 1 fails with a `RemoteVibeVoiceAsrError` (legacy alias: `RemoteAudioChainError`) if the RTX host is down, mis-tokened, or unreachable. There is no local fallback — fix the host or the bearer token and retry. Same applies to `RemoteNodeMediaPrepError` at the Phase 2 boundary.
7. **H200 audio-post import failure:** if `nemo`, `funasr`, `tensorflow`, `tensorflow_hub`, `librosa`, or `resampy` fail to import on the H200, the in-process audio post-processing chain won't start — reinstall `requirements-do-phase1-visual.txt` (these moved back onto the H200 from the RTX host).
8. **Visual throughput unexpectedly low on TensorRT path:** if a visual-only H200 replay drops back toward `~50 fps` instead of `~240 fps`, the most likely causes are stale code on-host, loss of `scale_cuda` decode-to-resolution behavior, or a fallback to host-side preprocessing before TensorRT.

For incident history and recovery notes, see `docs/ERROR_LOG.md`.
