# ENV REFERENCE

**Status:** Active  
**Last updated:** 2026-04-17

This is the code-backed environment variable reference for the current repository state.

Use this file for the full env surface. Use `.env.example` for a starter file, and use `docs/runtime/known-good.env` for the most recent working DO H200 baseline.

## 1) Required Core Inputs

These are validated by `load_provider_settings()` for normal runtime startup:

- `GOOGLE_CLOUD_PROJECT`
- `GCS_BUCKET`
- `VIBEVOICE_BACKEND=vllm`

Even when Phase 1 ASR is offloaded to the remote L4 combined service (currently
hosted on GCE; backend enum is still `cloud_run_l4`), the current provider layer
still requires `VIBEVOICE_BACKEND=vllm`.

## 2) Recommended Working Profiles

### 2.1 DO H200 host with local VibeVoice

Use this when both Phase 1 ASR and Phase 2-4 run from the host:

```bash
CLYPT_PHASE1_ASR_BACKEND=vllm
VIBEVOICE_BACKEND=vllm
VIBEVOICE_VLLM_BASE_URL=http://127.0.0.1:8000
VIBEVOICE_VLLM_MODEL=vibevoice

GENAI_GENERATION_BACKEND=local_openai
CLYPT_LOCAL_LLM_BASE_URL=http://127.0.0.1:8001/v1
CLYPT_LOCAL_LLM_MODEL=Qwen/Qwen3.6-35B-A3B
VERTEX_EMBEDDING_BACKEND=vertex

CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite
CLYPT_PHASE24_MEDIA_PREP_BACKEND=local
```

### 2.2 DO H200 host with combined GCE L4 offload

Use this when Phase 1 ASR and node-media prep are both offloaded to the same
GCE L4 VM (firewall-gated to the droplet's egress IP; backend enum is still
`cloud_run_l4` for backward compatibility):

```bash
CLYPT_PHASE1_ASR_BACKEND=cloud_run_l4
CLYPT_PHASE1_ASR_SERVICE_URL=http://<GCE_VM_EXTERNAL_IP>:8080
CLYPT_PHASE1_ASR_AUTH_MODE=none
CLYPT_PHASE1_ASR_TIMEOUT_S=7200

VIBEVOICE_BACKEND=vllm
VIBEVOICE_VLLM_MODEL=vibevoice
# IMPORTANT: leave VIBEVOICE_VLLM_BASE_URL unset on the caller in this mode.

GENAI_GENERATION_BACKEND=local_openai
CLYPT_LOCAL_LLM_BASE_URL=http://127.0.0.1:8001/v1
CLYPT_LOCAL_LLM_MODEL=Qwen/Qwen3.6-35B-A3B
VERTEX_EMBEDDING_BACKEND=vertex

CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite
CLYPT_PHASE24_MEDIA_PREP_BACKEND=cloud_run_l4
CLYPT_PHASE24_MEDIA_PREP_SERVICE_URL=http://<GCE_VM_EXTERNAL_IP>:8080
CLYPT_PHASE24_MEDIA_PREP_AUTH_MODE=none
```

> Legacy Cloud Run L4 profile (historical, not recommended — OOMs without the
> bf16 audio-encoder patch applied inside the Cloud Run container):
> `CLYPT_PHASE1_ASR_SERVICE_URL=https://<clypt-phase1-l4-combined>`,
> `CLYPT_PHASE1_ASR_AUTH_MODE=id_token`,
> `CLYPT_PHASE1_ASR_AUDIENCE=https://<clypt-phase1-l4-combined>`, and the same
> pattern for `CLYPT_PHASE24_MEDIA_PREP_*`.

Fail-fast guardrails in current code:

- if `CLYPT_PHASE1_ASR_BACKEND=cloud_run_l4`, then `CLYPT_PHASE1_ASR_SERVICE_URL` is required
- if `CLYPT_PHASE1_ASR_BACKEND=cloud_run_l4`, then `VIBEVOICE_VLLM_BASE_URL` must be unset on the caller
- `CLYPT_GEMINI_MAX_CONCURRENT` has been removed and now raises at startup

## 3) Provider and Runtime Env Surface

### 3.1 Phase 1 runtime

| Env | Default | Notes |
|---|---|---|
| `CLYPT_V31_OUTPUT_ROOT` | `backend/outputs/v3_1` | Root output dir for Phase 1-4 artifacts. |
| `CLYPT_PHASE1_WORK_ROOT` | `backend/outputs/v3_1_phase1_work` | Phase 1 workdir root. |
| `CLYPT_PHASE1_KEEP_WORKDIR` | `0` | Keep temp workdirs when `1`. |
| `CLYPT_PHASE1_YAMNET_DEVICE` | `cpu` | `gpu` makes YAMNet run on GPU. |
| `CLYPT_PHASE1_INPUT_MODE` | `test_bank` | Current code only supports `test_bank`. |
| `CLYPT_PHASE1_TEST_BANK_PATH` | unset | Required when strict mode is enabled. |
| `CLYPT_PHASE1_TEST_BANK_STRICT` | `1` | Strict mapping enforcement. |

### 3.2 Phase 1 ASR routing

| Env | Default | Notes |
|---|---|---|
| `CLYPT_PHASE1_ASR_BACKEND` | `vllm` | `vllm` or `cloud_run_l4` (enum name is historical; currently targets a GCE L4 VM). |
| `CLYPT_PHASE1_ASR_SERVICE_URL` | unset | Required for `cloud_run_l4`. On GCE path use `http://<VM_IP>:8080`; on legacy Cloud Run use `https://<service>`. |
| `CLYPT_PHASE1_ASR_AUTH_MODE` | `id_token` | `id_token` for Cloud Run audience-scoped IAM; `none` for the GCE firewall-gated deployment. |
| `CLYPT_PHASE1_ASR_AUDIENCE` | unset | Optional (Cloud Run path only), defaults to service URL in callers. Unused when `AUTH_MODE=none`. |
| `CLYPT_PHASE1_ASR_TIMEOUT_S` | `7200` | Remote ASR request timeout. |

### 3.3 VibeVoice local vLLM settings

| Env | Default | Notes |
|---|---|---|
| `VIBEVOICE_BACKEND` | `vllm` | Must stay `vllm` on mainline. |
| `VIBEVOICE_VLLM_BASE_URL` | required in local-ASR mode | Must be unset in `cloud_run_l4` caller mode. |
| `VIBEVOICE_VLLM_MODEL` | `vibevoice` | Must stay `vibevoice`. |
| `VIBEVOICE_VLLM_TIMEOUT_S` | `7200` | Local ASR timeout. |
| `VIBEVOICE_VLLM_HEALTHCHECK_PATH` | `/health` | Local service health route. |
| `VIBEVOICE_VLLM_MAX_RETRIES` | `1` | Local ASR retry budget. |
| `VIBEVOICE_VLLM_AUDIO_MODE` | `url` | Current default path. |
| `VIBEVOICE_HOTWORDS_CONTEXT` | built-in pronoun/connective list | Comma-separated context string. |
| `VIBEVOICE_MAX_NEW_TOKENS` | `32768` | Passed through to ASR generation config. |
| `VIBEVOICE_DO_SAMPLE` | `0` | Boolean. |
| `VIBEVOICE_TEMPERATURE` | `0` | Sampling temperature. |
| `VIBEVOICE_TOP_P` | `1.0` | Sampling top-p. |
| `VIBEVOICE_REPETITION_PENALTY` | `1.03` | Current default. |
| `VIBEVOICE_NUM_BEAMS` | `1` | Beam count. |

### 3.4 Local OpenAI generation settings

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

### 3.5 Vertex / Gemini / storage / persistence

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

### 3.6 Cloud Tasks settings

These are still supported by config, but the current local runtime path uses the SQLite queue instead.

| Env | Default |
|---|---|
| `CLYPT_PHASE24_PROJECT` | `GOOGLE_CLOUD_PROJECT` |
| `CLYPT_PHASE24_TASKS_LOCATION` | `us-central1` |
| `CLYPT_PHASE24_TASKS_QUEUE` | `clypt-phase24` |
| `CLYPT_PHASE24_WORKER_URL` | unset |
| `CLYPT_PHASE24_WORKER_SERVICE_ACCOUNT_EMAIL` | unset |

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

### 4.2 Node-media prep

| Env | Default | Notes |
|---|---|---|
| `CLYPT_PHASE24_MEDIA_PREP_BACKEND` | `local` | `local` or `cloud_run_l4`. |
| `CLYPT_PHASE24_MEDIA_PREP_SERVICE_URL` | unset | Required for remote media prep. |
| `CLYPT_PHASE24_MEDIA_PREP_AUTH_MODE` | `id_token` | `id_token` or `none`. |
| `CLYPT_PHASE24_MEDIA_PREP_AUDIENCE` | unset | Optional, defaults to service URL in callers. |
| `CLYPT_PHASE24_MEDIA_PREP_TIMEOUT_S` | `600` | Remote request timeout. |
| `CLYPT_PHASE24_FFMPEG_DEVICE` | `auto` | `auto`, `gpu`, or `cpu`. |
| `CLYPT_PHASE24_NODE_MEDIA_MAX_CONCURRENT` | `8` | Clip extraction worker pool size. Renamed from `CLYPT_PHASE24_NODE_MEDIA_CONCURRENCY`; the old name now raises at config load. |

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
| `CLYPT_PHASE2_MERGE_MAX_OUTPUT_TOKENS` | `32768` |
| `CLYPT_PHASE2_BOUNDARY_MAX_OUTPUT_TOKENS` | `32768` |
| `CLYPT_PHASE3_TARGET_BATCH_COUNT` | `3` |
| `CLYPT_PHASE3_MAX_NODES_PER_BATCH` | `24` |
| `CLYPT_PHASE3_LOCAL_MAX_OUTPUT_TOKENS` | `4096` |
| `CLYPT_PHASE3_LONG_RANGE_MAX_OUTPUT_TOKENS` | `4096` |
| `CLYPT_PHASE3_LONG_RANGE_TOP_K` | `2` |
| `CLYPT_PHASE3_LONG_RANGE_PAIRS_PER_SHARD` | `24` |
| `CLYPT_PHASE3_LONG_RANGE_MAX_CONCURRENT` | `24` |
| `CLYPT_PHASE3_LOCAL_MAX_CONCURRENT` | `24` |
| `CLYPT_PHASE4_META_MAX_OUTPUT_TOKENS` | `2048` |
| `CLYPT_PHASE4_SUBGRAPH_MAX_OUTPUT_TOKENS` | `4096` |
| `CLYPT_PHASE4_POOL_MAX_OUTPUT_TOKENS` | `2048` |
| `CLYPT_PHASE4_SUBGRAPH_MAX_CONCURRENT` | `16` |

> Renamed (2026-04-16): `CLYPT_PHASE2_MAX_CONCURRENT` is now
> `CLYPT_PHASE2_MERGE_MAX_CONCURRENT` (to mirror `*_BOUNDARY_MAX_CONCURRENT`).
> The old name now raises `ValueError` at config load.
| `CLYPT_PHASE4_SUBGRAPH_OVERLAP_DEDUPE_THRESHOLD` | `0.70` |

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
`scripts/do_phase1/deploy_sglang_qwen_service.sh` and the
`clypt-sglang-qwen.service` systemd unit, not via Python-level envs.

> Removed (2026-04-16 Qwen3.6 + SGLang cutover): `CLYPT_VLLM_PROFILE`,
> `CLYPT_VLLM_MAX_NUM_SEQS`, `CLYPT_VLLM_MAX_NUM_BATCHED_TOKENS`,
> `CLYPT_VLLM_GPU_MEMORY_UTILIZATION`, `CLYPT_VLLM_MAX_MODEL_LEN`,
> `CLYPT_VLLM_LANGUAGE_MODEL_ONLY`, `CLYPT_VLLM_SPECULATIVE_MODE`,
> `CLYPT_VLLM_SPECULATIVE_NUM_TOKENS`. They were loaded into
> `VLLMRuntimeSettings` but never consumed. Setting any of them now raises
> `ValueError` at `load_provider_settings()`.

## 8) Deploy-Script-Only Env Surface

These are not read by `load_provider_settings()` or `get_v31_config()`. They are used by deploy/bootstrap scripts.

### 8.1 SGLang deploy knobs

Used by `scripts/do_phase1/deploy_sglang_qwen_service.sh`:

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

### 8.2 VibeVoice deploy knobs

Used by `scripts/do_phase1/deploy_vllm_service.sh`:

- `VIBEVOICE_REPO_DIR`
- `VIBEVOICE_REPO_URL`
- `VIBEVOICE_REPO_REF`

The systemd unit also passes through some VibeVoice runtime envs that are not validated by the Python config layer, such as:

- `VIBEVOICE_OUTPUT_MODE`
- `VIBEVOICE_WORD_TURN_GAP_MS`
- `VIBEVOICE_WORD_TIME_TOKEN_MODE`
- `VIBEVOICE_WORD_CHUNK_SECONDS`
- `VIBEVOICE_WORD_STREAMING_SEGMENT_DURATION_S`

### 8.3 GCE L4 combined service deploy knobs

Used by `scripts/deploy_l4_gce.sh`:

- `PROJECT` (default `clypt-v3`)
- `ZONE` (default `us-central1-a`; script probes additional zones if capacity is exhausted)
- `REGION_AR` (default `us-east4`) — Artifact Registry region
- `VM_NAME` (default `clypt-phase1-l4-gce`)
- `MACHINE_TYPE` (default `g2-standard-8`)
- `GPU_TYPE` (default `nvidia-l4`)
- `BOOT_DISK_SIZE` (default `200GB`)
- `BOOT_DISK_TYPE` (default `pd-balanced`)
- `IMAGE_FAMILY` (default `common-cu129-ubuntu-2204-nvidia-580`)
- `IMAGE_PROJECT` (default `deeplearning-platform-release`)
- `FIREWALL_RULE` (default `clypt-l4-combined-ingress`)
- `NETWORK_TAG` (default `clypt-l4-combined`)
- `DROPLET_IP` — DO droplet egress IP, required for the firewall allowlist
- `SERVICE_ACCOUNT` (default: empty => Compute Engine default SA)
- `TAG` (default: `gce-bf16-<timestamp>`)
- `CONTAINER_NAME` (default `clypt-l4-combined`)
- `HOST_HF_CACHE` (default `/var/clypt/hf-cache`) — persistent HF cache on the VM host disk, bind-mounted to `/root/.cache/huggingface`
- `CONTAINER_PORT` (default `8080`)

### 8.4 L4 combined container runtime knobs

Read by `backend/runtime/l4_combined_bootstrap.py` inside the container on boot
to configure the bundled VibeVoice vLLM process. All are L4-tuned defaults; do
not revert to old H200 settings on a 24 GB L4.

| Env | Default | Notes |
|---|---|---|
| `CLYPT_L4_VIBEVOICE_REPO_DIR` | `/app/vllm_plugin` | VibeVoice source tree baked into the image. |
| `CLYPT_L4_VIBEVOICE_MAX_NUM_SEQS` | `4` | vLLM `--max-num-seqs`. |
| `CLYPT_L4_VIBEVOICE_MAX_MODEL_LEN` | `16384` | vLLM `--max-model-len`. |
| `CLYPT_L4_VIBEVOICE_GPU_MEMORY_UTILIZATION` | `0.90` | vLLM `--gpu-memory-utilization`. |
| `CLYPT_L4_VIBEVOICE_STARTUP_TIMEOUT_S` | `1500` | Max wait for `/health` during container boot. |
| `VIBEVOICE_FFMPEG_MAX_CONCURRENCY` | `16` | Parallel ffmpeg workers for node-media prep. |

## 9) Removed Env Surface

- `CLYPT_GEMINI_MAX_CONCURRENT`

If this variable is still present, current startup should fail fast and tell you to use explicit per-stage concurrency envs instead.
