# ENV REFERENCE

**Status:** Active  
**Last updated:** 2026-04-17

This is the code-backed environment variable reference for the current repository state.

Use this file for the full env surface. Use `.env.example` for a starter file, and use `docs/runtime/known-good.env` for the most recent working DO H200 baseline.

## 1) Required Core Inputs

### 1.1 H200 host (orchestrator + visual + audio post + worker)

Validated by `load_provider_settings()` at startup — config load fails
fast if any are missing:

- `GOOGLE_CLOUD_PROJECT`
- `GCS_BUCKET`
- `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL`
- `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN`
- `CLYPT_PHASE24_NODE_MEDIA_PREP_URL`
- `CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN`
- `GENAI_GENERATION_BACKEND=local_openai`
- `CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite`
- `CLYPT_LOCAL_LLM_BASE_URL`
- `CLYPT_LOCAL_LLM_MODEL` (or a compatible fallback such as `GENAI_FLASH_MODEL`)

> **Compat note:** `CLYPT_PHASE1_AUDIO_HOST_URL` /
> `CLYPT_PHASE1_AUDIO_HOST_TOKEN` still work for one release as
> deprecated aliases of the `_VIBEVOICE_ASR_SERVICE_*` names. Prefer
> the new names in new deployments.

`VIBEVOICE_BACKEND` / `VIBEVOICE_VLLM_*` must **not** be set on the H200.

### 1.2 RTX 6000 Ada host (VibeVoice ASR + node-media prep)

Required on the RTX host (loaded by the FastAPI service and by its
embedded VibeVoice client):

- `CLYPT_PHASE1_AUDIO_HOST_BIND`
- `CLYPT_PHASE1_AUDIO_HOST_PORT`
- `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN`
  (legacy alias: `CLYPT_PHASE1_AUDIO_HOST_TOKEN`, accepted for one release)
- `VIBEVOICE_BACKEND=vllm`
- `VIBEVOICE_VLLM_BASE_URL`
- `VIBEVOICE_VLLM_MODEL=vibevoice`
- `GOOGLE_CLOUD_PROJECT`
- `GCS_BUCKET`
- `GOOGLE_APPLICATION_CREDENTIALS`

NeMo / FunASR / TensorFlow / librosa / resampy are no longer required
here — they moved back to the H200 with the NFA / emotion2vec+ / YAMNet
post-processing chain.

## 2) Recommended Working Profile

The profile below describes the deployed two-host reality. There is no
local-fallback audio or node-media-prep mode; `run_phase1_sidecars` and
`build_default_phase24_worker_service` both require the remote clients.

### 2.1 H200 profile

```bash
CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL=http://<rtx-host>:9100
CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN=<shared-bearer>
CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_TIMEOUT_S=7200
CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_HEALTHCHECK_PATH=/health

CLYPT_PHASE24_NODE_MEDIA_PREP_URL=http://<rtx-host>:9100
CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN=<shared-bearer>
CLYPT_PHASE24_NODE_MEDIA_PREP_TIMEOUT_S=600
CLYPT_PHASE24_NODE_MEDIA_PREP_MAX_CONCURRENCY=8

GENAI_GENERATION_BACKEND=local_openai
CLYPT_LOCAL_LLM_BASE_URL=http://127.0.0.1:8001/v1
CLYPT_LOCAL_LLM_MODEL=Qwen/Qwen3.6-35B-A3B
VERTEX_EMBEDDING_BACKEND=vertex

CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite
```

### 2.2 RTX 6000 Ada profile

```bash
CLYPT_PHASE1_AUDIO_HOST_BIND=0.0.0.0
CLYPT_PHASE1_AUDIO_HOST_PORT=9100
CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN=<shared-bearer>
CLYPT_PHASE1_AUDIO_HOST_SCRATCH_ROOT=/opt/clypt-audio-host/scratch

VIBEVOICE_BACKEND=vllm
VIBEVOICE_VLLM_BASE_URL=http://127.0.0.1:8000
VIBEVOICE_VLLM_MODEL=vibevoice
```

Fail-fast guardrails in current code:

- `CLYPT_PHASE1_ASR_BACKEND` (if set on the H200) only accepts `vllm`; any other value raises
- `CLYPT_GEMINI_MAX_CONCURRENT` has been removed and now raises at startup
- Omitting any of the four remote-host vars in §1.1 raises at startup
  (legacy `CLYPT_PHASE1_AUDIO_HOST_URL` / `_TOKEN` satisfy the
  VibeVoice ASR service requirement for one release as deprecated
  aliases)

## 3) Provider and Runtime Env Surface

### 3.1 Phase 1 runtime

| Env | Default | Notes |
|---|---|---|
| `CLYPT_V31_OUTPUT_ROOT` | `backend/outputs/v3_1` | Root output dir for Phase 1-4 artifacts. |
| `CLYPT_PHASE1_WORK_ROOT` | `backend/outputs/v3_1_phase1_work` | Phase 1 workdir root. |
| `CLYPT_PHASE1_YAMNET_DEVICE` | `cpu` | `gpu` makes YAMNet run on GPU. |
| `CLYPT_PHASE1_INPUT_MODE` | `test_bank` | Current code only supports `test_bank`. |
| `CLYPT_PHASE1_TEST_BANK_PATH` | unset | Required when strict mode is enabled. |
| `CLYPT_PHASE1_TEST_BANK_STRICT` | `1` | Strict mapping enforcement. |

### 3.2 Phase 1 ASR routing

| Env | Default | Notes |
|---|---|---|
| `CLYPT_PHASE1_ASR_BACKEND` | `vllm` | Only `vllm` is supported today. |

### 3.3 Remote VibeVoice ASR service (H200 → RTX 6000 Ada)

`RemoteVibeVoiceAsrClient` (legacy class name: `RemoteAudioChainClient`)
settings loaded on the H200. `URL` and `AUTH_TOKEN` are required;
startup fails fast if either is missing.

| Env | Default | Notes |
|---|---|---|
| `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL` | required | Base URL of the RTX 6000 Ada FastAPI service. |
| `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN` | required | Shared bearer for the RTX service. |
| `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_TIMEOUT_S` | `7200` | Request timeout for `/tasks/vibevoice-asr`. |
| `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_HEALTHCHECK_PATH` | `/health` | Used by deploy probe and diagnostics. |

> **Compat note:** `CLYPT_PHASE1_AUDIO_HOST_URL`,
> `CLYPT_PHASE1_AUDIO_HOST_TOKEN`, `CLYPT_PHASE1_AUDIO_HOST_TIMEOUT_S`,
> and `CLYPT_PHASE1_AUDIO_HOST_HEALTHCHECK_PATH` are accepted for one
> release as deprecated aliases and log a warning at startup.

### 3.4 VibeVoice local vLLM settings (RTX 6000 Ada only)

These live on the RTX host, not the H200. The FastAPI service
loads them when building its in-process VibeVoice vLLM client. The
systemd unit launches vLLM with current sole-tenant flags:
`--gpu-memory-utilization 0.77` (leaves ~8 GiB for NVDEC contexts),
`--max-num-seqs 2`, `--dtype bfloat16`, CUDA graph capture enabled
(`enforce_eager=False`). No speculative decoding (VibeVoice is Whisper
encoder-decoder). The earlier co-tenancy workarounds
(`--max-num-seqs 1`, `--max-model-len 32768`, `--enforce-eager`,
`--gpu-memory-utilization 0.60`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) have been removed.

| Env | Default | Notes |
|---|---|---|
| `VIBEVOICE_BACKEND` | `vllm` | Must be `vllm`. |
| `VIBEVOICE_VLLM_BASE_URL` | required | Local vLLM URL on the RTX host. |
| `VIBEVOICE_VLLM_MODEL` | `vibevoice` | Must be `vibevoice`. |
| `VIBEVOICE_VLLM_TIMEOUT_S` | `7200` | ASR timeout. |
| `VIBEVOICE_VLLM_HEALTHCHECK_PATH` | `/health` | Local service health route. |
| `VIBEVOICE_VLLM_MAX_RETRIES` | `1` | ASR retry budget. |
| `VIBEVOICE_VLLM_AUDIO_MODE` | `url` | Current default path. |
| `VIBEVOICE_HOTWORDS_CONTEXT` | built-in pronoun/connective list | Comma-separated context string. |
| `VIBEVOICE_MAX_NEW_TOKENS` | `32768` | Passed through to ASR generation config. |
| `VIBEVOICE_DO_SAMPLE` | `0` | Boolean. |
| `VIBEVOICE_TEMPERATURE` | `0` | Sampling temperature. |
| `VIBEVOICE_TOP_P` | `1.0` | Sampling top-p. |
| `VIBEVOICE_REPETITION_PENALTY` | `1.03` | Current default. |
| `VIBEVOICE_NUM_BEAMS` | `1` | Beam count. |

### 3.5 Local OpenAI generation settings

| Env | Default | Notes |
|---|---|---|
| `GENAI_GENERATION_BACKEND` | `local_openai` | Local Phase 2-4 worker only accepts `local_openai`. |
| `CLYPT_LOCAL_LLM_BASE_URL` | `http://127.0.0.1:8001/v1` | Local OpenAI-compatible Qwen endpoint. |
| `CLYPT_LOCAL_LLM_MODEL` | unset | Should be set explicitly in real deployments. |
| `CLYPT_LOCAL_LLM_TIMEOUT_S` | `600` | Request timeout. |
| `CLYPT_LOCAL_LLM_MAX_RETRIES` | `6` | Retry budget. |
| `CLYPT_LOCAL_LLM_INITIAL_BACKOFF_S` | `1.0` | Retry tuning. |
| `CLYPT_LOCAL_LLM_MAX_BACKOFF_S` | `30.0` | Retry tuning. |
| `CLYPT_LOCAL_LLM_BACKOFF_MULTIPLIER` | `2.0` | Retry tuning. |
| `CLYPT_LOCAL_LLM_JITTER_RATIO` | `0.2` | Retry tuning. |
| `CLYPT_LOCAL_LLM_TEMPERATURE` | `0.0` | Generation tuning; strict-JSON profile (see `2026-04-16_qwen36_swap_and_sglang_tuning_spec.md`). |
| `CLYPT_LOCAL_LLM_TOP_P` | `1.0` | Generation tuning; strict-JSON profile. |
| `CLYPT_LOCAL_LLM_TOP_K` | `40` | Generation tuning; strict-JSON profile. |
| `CLYPT_LOCAL_LLM_MIN_P` | `0.0` | Generation tuning. |
| `CLYPT_LOCAL_LLM_PRESENCE_PENALTY` | `0.0` | Generation tuning; must be `0.0` for strict-schema JSON so repeated keys aren't penalized. |
| `CLYPT_LOCAL_LLM_REPETITION_PENALTY` | `1.0` | Generation tuning. |

The local OpenAI-compatible Qwen path always sends `chat_template_kwargs.enable_thinking=false`.
`CLYPT_LOCAL_LLM_ENABLE_THINKING` has been removed and now fails fast if set.

### 3.6 Vertex / Gemini / storage / persistence

| Env | Default | Notes |
|---|---|---|
| `VERTEX_EMBEDDING_BACKEND` | `vertex` | `vertex` or `developer`. |
| `GOOGLE_CLOUD_LOCATION` | `global` | Fallback generation location. |
| `GENAI_GENERATION_LOCATION` | `global` | Generation location. |
| `VERTEX_EMBEDDING_LOCATION` | `us-central1` | Embedding location. |
| `GENAI_GENERATION_MODEL` | `Qwen/Qwen3.6-35B-A3B` | Provider default. |
| `GENAI_FLASH_MODEL` | `Qwen/Qwen3.6-35B-A3B` | Default flash model. |
| `VERTEX_EMBEDDING_MODEL` | `gemini-embedding-2-preview` | Embedding model. |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | unset | Developer API path only. |
| `GENAI_GENERATION_API_MAX_RETRIES` | `6` | Developer API retry tuning. |
| `GENAI_GENERATION_API_INITIAL_BACKOFF_S` | `1.0` | Developer API retry tuning. |
| `GENAI_GENERATION_API_MAX_BACKOFF_S` | `30.0` | Developer API retry tuning. |
| `GENAI_GENERATION_API_BACKOFF_MULTIPLIER` | `2.0` | Developer API retry tuning. |
| `GENAI_GENERATION_API_JITTER_RATIO` | `0.2` | Developer API retry tuning. |
| `VERTEX_EMBEDDING_API_MAX_RETRIES` | `6` | Vertex embedding retry tuning. |
| `VERTEX_EMBEDDING_API_INITIAL_BACKOFF_S` | `1.0` | Vertex embedding retry tuning. |
| `VERTEX_EMBEDDING_API_MAX_BACKOFF_S` | `30.0` | Vertex embedding retry tuning. |
| `VERTEX_EMBEDDING_API_BACKOFF_MULTIPLIER` | `2.0` | Vertex embedding retry tuning. |
| `VERTEX_EMBEDDING_API_JITTER_RATIO` | `0.2` | Vertex embedding retry tuning. |
| `CLYPT_SPANNER_PROJECT` | `GOOGLE_CLOUD_PROJECT` | Spanner project. |
| `CLYPT_SPANNER_INSTANCE` | `clypt-phase14` | Spanner instance. |
| `CLYPT_SPANNER_DATABASE` | `clypt_phase14` | Spanner database. |
| `CLYPT_SPANNER_DDL_OPERATION_TIMEOUT_S` | `600` | Schema bootstrap timeout. |

## 4) Phase 2-4 Local Queue and Worker Env Surface

### 4.1 Queue and worker runtime

| Env | Default | Notes |
|---|---|---|
| `CLYPT_PHASE24_QUEUE_BACKEND` | `local_sqlite` | Required by local Phase 1 and local worker. |
| `CLYPT_PHASE24_LOCAL_QUEUE_PATH` | `backend/outputs/phase24_local_queue.sqlite` | SQLite queue file. |
| `CLYPT_PHASE24_LOCAL_POLL_INTERVAL_MS` | `500` | Worker poll interval. |
| `CLYPT_PHASE24_LOCAL_LEASE_TIMEOUT_S` | `1800` | Lease timeout. |
| `CLYPT_PHASE24_LOCAL_MAX_INFLIGHT` | `1` | Max inflight queue items. |
| `CLYPT_PHASE24_LOCAL_MAX_REQUESTS_PER_WORKER` | `0` | `0` means unlimited loop. |
| `CLYPT_PHASE24_LOCAL_RECLAIM_EXPIRED_LEASES` | `0` | Default is fail-fast, not reclaim. |
| `CLYPT_PHASE24_LOCAL_FAIL_FAST_ON_STALE_RUNNING` | `1` | Default is fail-fast. |
| `CLYPT_PHASE24_LOCAL_WORKER_ID` | `phase24-local-worker` | Used by `run_phase24_local_worker`. |
| `CLYPT_PHASE24_WORKER_SERVICE_NAME` | `clypt-phase24-worker` | Worker identity. |
| `CLYPT_PHASE24_ENVIRONMENT` | `dev` | Metadata only. |
| `CLYPT_PHASE24_QUERY_VERSION` | `v1` | Query version attached to runs. |
| `CLYPT_PHASE24_CONCURRENCY` | `1` | Worker service config field. |
| `CLYPT_DEBUG_SNAPSHOTS` | `0` | Worker debug snapshots. |
| `CLYPT_PHASE24_MAX_ATTEMPTS` | `3` | Max attempts before terminal failure. |
| `CLYPT_PHASE24_FAILFAST_PREEMPTION_THRESHOLD` | `0` | Admission guard. |
| `CLYPT_PHASE24_FAILFAST_P95_LATENCY_MS` | `0` | Admission guard. |
| `CLYPT_PHASE24_ADMISSION_METRICS_PATH` | unset | Optional JSON metrics file. |
| `CLYPT_PHASE24_BLOCK_ON_PHASE1_ACTIVE` | `0` | Admission guard. |

> Removed (2026-04-16 SGLang cutover): `CLYPT_PHASE24_MAX_VLLM_QUEUE_DEPTH` and
> `CLYPT_PHASE24_MAX_VLLM_DECODE_BACKLOG`. Loading any of these now raises
> `ValueError` at config load. There is no metrics producer under SGLang, so the
> admission guards were dead code.

### 4.2 Node-media prep (remote RTX 6000 Ada)

Node-media prep is always delegated to the RTX host via
`RemoteNodeMediaPrepClient`. There is no in-process ffmpeg path on the
H200 worker anymore; startup fails fast if `URL` or `TOKEN` is missing.

| Env | Default | Notes |
|---|---|---|
| `CLYPT_PHASE24_NODE_MEDIA_PREP_URL` | required | Base URL of the RTX service. Typically the same as `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL`. |
| `CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN` | required | Shared bearer for the RTX service. |
| `CLYPT_PHASE24_NODE_MEDIA_PREP_TIMEOUT_S` | `600` | Per-request timeout. |
| `CLYPT_PHASE24_NODE_MEDIA_PREP_MAX_CONCURRENCY` | `8` | Caller-side concurrency ceiling for `/tasks/node-media-prep`. |

The RTX-side ffmpeg hardware-device selection is still controlled by
`CLYPT_PHASE24_FFMPEG_DEVICE` (`auto`/`gpu`/`cpu`), but that variable
now belongs to the **RTX host** environment, not the H200.

> Removed: `CLYPT_PHASE24_NODE_MEDIA_MAX_CONCURRENT` (replaced by the
> remote client's `_MAX_CONCURRENCY`). Setting the old name raises at
> config load.

## 5) Phase 2-4 Pipeline Tuning Env Surface

Current code enforces explicit per-stage concurrency. There is no global
concurrency fallback anymore. All `*_MAX_CONCURRENT` values are **max in-flight
LLM request caps** per stage (validated 2026-04-16 against Qwen3.6-35B-A3B on
H200). Smaller videos stay well below these caps; only large videos scale up
to them.

### 5.1 Phase 2-4 orchestration

| Env | Default |
|---|---|
| `CLYPT_PHASE2_TARGET_BATCH_COUNT` | `5` |
| `CLYPT_PHASE2_MAX_TURNS_PER_BATCH` | `25` |
| `CLYPT_PHASE2_MERGE_MAX_CONCURRENT` | `16` |
| `CLYPT_PHASE2_BOUNDARY_MAX_CONCURRENT` | `16` |
| `CLYPT_PHASE2_MERGE_MAX_OUTPUT_TOKENS` | `8192` |
| `CLYPT_PHASE2_BOUNDARY_MAX_OUTPUT_TOKENS` | `8192` |
| `CLYPT_PHASE3_TARGET_BATCH_COUNT` | `3` |
| `CLYPT_PHASE3_MAX_NODES_PER_BATCH` | `24` |
| `CLYPT_PHASE3_LOCAL_MAX_OUTPUT_TOKENS` | `4096` |
| `CLYPT_PHASE3_LONG_RANGE_MAX_OUTPUT_TOKENS` | `4096` |
| `CLYPT_PHASE3_LONG_RANGE_TOP_K` | `2` |
| `CLYPT_PHASE3_LONG_RANGE_PAIRS_PER_SHARD` | `24` |
| `CLYPT_PHASE3_LONG_RANGE_MAX_CONCURRENT` | `24` |
| `CLYPT_PHASE3_LOCAL_MAX_CONCURRENT` | `24` |
| `CLYPT_PHASE4_META_MAX_OUTPUT_TOKENS` | `4096` |
| `CLYPT_PHASE4_SUBGRAPH_MAX_OUTPUT_TOKENS` | `4096` |
| `CLYPT_PHASE4_POOL_MAX_OUTPUT_TOKENS` | `4096` |
| `CLYPT_PHASE4_SUBGRAPH_MAX_CONCURRENT` | `16` |
| `CLYPT_PHASE4_SUBGRAPH_OVERLAP_DEDUPE_THRESHOLD` | `0.70` |

> Renamed (2026-04-16): `CLYPT_PHASE2_MAX_CONCURRENT` is now
> `CLYPT_PHASE2_MERGE_MAX_CONCURRENT` (to mirror `*_BOUNDARY_MAX_CONCURRENT`).
> The old name now raises `ValueError` at config load.

> Retuned (2026-04-17 DO-speedup-and-OSS-swap):
> - `CLYPT_PHASE2_MERGE_MAX_OUTPUT_TOKENS` / `CLYPT_PHASE2_BOUNDARY_MAX_OUTPUT_TOKENS`: `32768` → `8192`.
>   The 32768 cap was ~4x typical response size and negated MTP gains.
> - `CLYPT_PHASE4_POOL_MAX_OUTPUT_TOKENS` / `CLYPT_PHASE4_META_MAX_OUTPUT_TOKENS`: `2048` → `4096`.
>   Phase 4 pool/meta on 20+ minute videos routinely exceeded 2048 tokens
>   and was hitting `finish_reason=length` (see `docs/ERROR_LOG.md`).

### 5.2 Phase 4 budget knobs

These are currently code defaults only, not env-driven:

- `max_total_prompts=12`
- `max_subgraphs_per_run=24`
- `max_final_review_calls=1`

## 6) Signal Augmentation Env Surface

### 6.1 Signal enablement and fetch

| Env | Default |
|---|---|
| `CLYPT_SIGNAL_MODE` | `augment` |
| `CLYPT_SIGNAL_FAIL_FAST` | `1` |
| `CLYPT_SIGNAL_LLM_FAIL_FAST` | `1` |
| `CLYPT_ENABLE_COMMENT_SIGNALS` | `0` |
| `CLYPT_ENABLE_TREND_SIGNALS` | `0` |
| `CLYPT_YOUTUBE_DATA_API_KEY` / `YOUTUBE_API_KEY` | unset |
| `CLYPT_YOUTUBE_DATA_API_BASE_URL` | `https://www.googleapis.com/youtube/v3` |
| `CLYPT_COMMENT_MAX_PAGES` | `5` |
| `CLYPT_COMMENT_ORDER` | `relevance` |
| `CLYPT_COMMENT_TOP_THREADS_MIN` | `15` |
| `CLYPT_COMMENT_TOP_THREADS_MAX` | `40` |
| `CLYPT_COMMENT_MAX_REPLIES_PER_THREAD` | `0` |
| `CLYPT_COMMENT_CLUSTER_SIM_THRESHOLD` | `0.82` |
| `CLYPT_TREND_MAX_ITEMS` | `40` |
| `CLYPT_TREND_RELEVANCE_THRESHOLD` | `0.6` |
| `CLYPT_SIGNAL_MAX_HOPS` | `2` |
| `CLYPT_SIGNAL_TIME_WINDOW_MS` | `30000` |
| `CLYPT_SIGNAL_MAX_CONCURRENT` | `8` |

### 6.2 Signal scoring weights

| Env | Default |
|---|---|
| `CLYPT_SIGNAL_EPSILON` | `1e-6` |
| `CLYPT_SIGNAL_CLUSTER_CAP` | `0.12` |
| `CLYPT_SIGNAL_TOTAL_CAP` | `0.20` |
| `CLYPT_SIGNAL_AGREEMENT_CAP` | `0.10` |
| `CLYPT_SIGNAL_MEANINGFUL_MIN_CLUSTER_CONTRIB` | `0.04` |
| `CLYPT_SIGNAL_MEANINGFUL_MIN_SOURCE_COVERAGE` | `0.15` |
| `CLYPT_SIGNAL_AGREEMENT_BONUS_TIER1` | `0.04` |
| `CLYPT_SIGNAL_AGREEMENT_BONUS_TIER2` | `0.07` |
| `CLYPT_SIGNAL_ENGAGEMENT_TOP_LIKE_WEIGHT` | `0.65` |
| `CLYPT_SIGNAL_ENGAGEMENT_TOP_REPLY_WEIGHT` | `0.35` |
| `CLYPT_SIGNAL_ENGAGEMENT_REPLY_LIKE_WEIGHT` | `0.85` |
| `CLYPT_SIGNAL_ENGAGEMENT_REPLY_PARENT_WEIGHT` | `0.15` |
| `CLYPT_SIGNAL_CLUSTER_MEAN_WEIGHT` | `0.45` |
| `CLYPT_SIGNAL_CLUSTER_MAX_WEIGHT` | `0.25` |
| `CLYPT_SIGNAL_CLUSTER_FREQ_WEIGHT` | `0.30` |
| `CLYPT_SIGNAL_CLUSTER_FREQ_REF` | `30` |
| `CLYPT_SIGNAL_HOP_DECAY_1` | `0.75` |
| `CLYPT_SIGNAL_HOP_DECAY_2` | `0.55` |
| `CLYPT_SIGNAL_COVERAGE_WEIGHT` | `0.30` |
| `CLYPT_SIGNAL_DIRECT_RATIO_WEIGHT` | `0.15` |

### 6.3 Signal LLM call routing

| Env | Default |
|---|---|
| `CLYPT_SIGNAL_LLM_MODEL_1` | `Qwen/Qwen3.6-35B-A3B` |
| `CLYPT_SIGNAL_LLM_MODEL_2` | `Qwen/Qwen3.6-35B-A3B` |
| `CLYPT_SIGNAL_LLM_MODEL_3` | `Qwen/Qwen3.6-35B-A3B` |
| `CLYPT_SIGNAL_LLM_MODEL_5` | `Qwen/Qwen3.6-35B-A3B` |
| `CLYPT_SIGNAL_LLM_MODEL_9` | `Qwen/Qwen3.6-35B-A3B` |
| `CLYPT_SIGNAL_LLM_MODEL_10` | `Qwen/Qwen3.6-35B-A3B` |
| `CLYPT_SIGNAL_LLM_MODEL_11` | `Qwen/Qwen3.6-35B-A3B` |

## 7) SGLang Runtime Tuning Surface (Qwen3.6)

SGLang startup flags for the Qwen3.6 service are driven by `SG_*` knobs in
`scripts/do_phase1_visual/deploy_sglang_qwen_service.sh` and the
`clypt-sglang-qwen.service` systemd unit on the H200, not via
Python-level envs.

Current effective SGLang flags (DO-speedup-and-OSS-swap baseline):
- `--context-length 65536` (reduced from 131072 to reclaim KV-cache headroom)
- `--kv-cache-dtype fp8_e4m3`
- `--mem-fraction-static 0.78`
- `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`
- `--mamba-scheduler-strategy extra_buffer` (required by Qwen3.6 hybrid Mamba/Attention when MTP + radix cache are both on)
- `--schedule-policy lpm`
- `--chunked-prefill-size 8192`
- `--grammar-backend xgrammar`
- `--reasoning-parser qwen3`
- `HF_HUB_OFFLINE=1` (set in the systemd unit to prevent DNS failures at startup; requires the `Qwen/Qwen3.6-35B-A3B` snapshot to be pre-cached in `HF_HOME=/opt/clypt-phase1/hf-cache` — `bootstrap_h200.sh` handles the initial pull; revision refreshes follow the temporary-unset procedure in `docs/deployment/P1_DEPLOY.md` §3.2)
- `SGLANG_ENABLE_SPEC_V2=1` (environment variable, set in the systemd unit; required by SGLang 0.5.10 to run speculative decoding + radix cache on the Mamba/Attention hybrid)
- Effective runtime limits: `max_total_num_tokens=1,739,188`, `max_running_requests=48`

> Removed (2026-04-16 Qwen3.6 + SGLang cutover): `CLYPT_VLLM_PROFILE`,
> `CLYPT_VLLM_MAX_NUM_SEQS`, `CLYPT_VLLM_MAX_NUM_BATCHED_TOKENS`,
> `CLYPT_VLLM_GPU_MEMORY_UTILIZATION`, `CLYPT_VLLM_MAX_MODEL_LEN`,
> `CLYPT_VLLM_LANGUAGE_MODEL_ONLY`, `CLYPT_VLLM_SPECULATIVE_MODE`,
> `CLYPT_VLLM_SPECULATIVE_NUM_TOKENS`. They were loaded into
> `VLLMRuntimeSettings` but never consumed. Setting any of them now raises
> `ValueError` at `load_provider_settings()`.

## 8) Deploy-Script-Only Env Surface

These are not read by `load_provider_settings()` or `get_v31_config()`. They are used by deploy/bootstrap scripts.

### 8.1 SGLang deploy knobs (H200)

Used by `scripts/do_phase1_visual/deploy_sglang_qwen_service.sh`:

- `SG_PACKAGE_SPEC`
- `SG_BASE_URL`
- `SG_PORT`
- `SG_MODEL`
- `SG_GRAMMAR_BACKEND`
- `SG_SCHEDULE_POLICY`
- `SG_CHUNKED_PREFILL_SIZE`
- `SG_MEM_FRACTION_STATIC`
- `SG_CONTEXT_LENGTH`
- `SG_EXTRA_ARGS`
- `SG_READY_TIMEOUT_S`
- `SG_SYSTEMD_UNIT`

### 8.2 VibeVoice deploy knobs (RTX 6000 Ada)

Used by `scripts/do_phase1_audio/deploy_vllm_service.sh`:

- `VIBEVOICE_REPO_DIR`
- `VIBEVOICE_REPO_URL`
- `VIBEVOICE_REPO_REF`

The systemd unit also passes through some VibeVoice runtime envs that are not validated by the Python config layer, such as:

- `VIBEVOICE_OUTPUT_MODE`
- `VIBEVOICE_WORD_TURN_GAP_MS`
- `VIBEVOICE_WORD_TIME_TOKEN_MODE`
- `VIBEVOICE_WORD_CHUNK_SECONDS`
- `VIBEVOICE_WORD_STREAMING_SEGMENT_DURATION_S`

## 9) Removed Env Surface

- `CLYPT_GEMINI_MAX_CONCURRENT`

If this variable is still present, current startup should fail fast and tell you to use explicit per-stage concurrency envs instead.
