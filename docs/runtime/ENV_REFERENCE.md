# ENV REFERENCE

**Status:** Active  
**Last updated:** 2026-04-19

This is the code-backed env catalog for the current Phase1 + Phase26 + Modal topology.

Canonical baselines:

- [known-good-phase1-h200.env](/Users/rithvik/Clypt-Backend/docs/runtime/known-good-phase1-h200.env)
- [known-good-phase1-h100-backup.env](/Users/rithvik/Clypt-Backend/docs/runtime/known-good-phase1-h100-backup.env)
- [known-good-phase26-h200.env](/Users/rithvik/Clypt-Backend/docs/runtime/known-good-phase26-h200.env)

## 1) Required Core Inputs

### 1.1 Phase1 host

Required by `load_phase1_host_settings()`:

- `GOOGLE_CLOUD_PROJECT`
- `GCS_BUCKET`
- `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL`
- `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN`
- `CLYPT_PHASE1_VISUAL_SERVICE_URL`
- `CLYPT_PHASE1_VISUAL_SERVICE_AUTH_TOKEN`
- `CLYPT_PHASE24_DISPATCH_URL`
- `CLYPT_PHASE24_DISPATCH_AUTH_TOKEN`

Operational notes:

- Phase 1 is allowed to keep `VIBEVOICE_*` env because it owns the local VibeVoice service and sidecar.
- `CLYPT_PHASE1_INPUT_MODE=test_bank` is still required.

### 1.2 Phase26 host

Required by `load_phase26_host_settings()`:

- `GOOGLE_CLOUD_PROJECT`
- `GCS_BUCKET`
- `GENAI_GENERATION_BACKEND=local_openai`
- `CLYPT_LOCAL_LLM_BASE_URL`
- `CLYPT_LOCAL_LLM_MODEL` or compatible fallback
- `CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite`
- `CLYPT_PHASE24_NODE_MEDIA_PREP_URL`
- `CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN`

### 1.3 Modal node-media-prep

Runtime env depends on the Modal secret/app config. The current deployed app is
`clypt-node-media-prep`, backed by the Modal secret `clypt-node-media-prep`,
and the service expects at minimum:

- `GCS_BUCKET`
- `NODE_MEDIA_PREP_AUTH_TOKEN`
- `GOOGLE_APPLICATION_CREDENTIALS_JSON`

The active HTTP contract is async:

- `POST /tasks/node-media-prep` returns `202 Accepted` with `call_id`
- `GET /tasks/node-media-prep/result/{call_id}` returns `202` while pending, `200` on completion

## 2) Recommended Working Profiles

### 2.1 Phase1 H200 default

```bash
CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL=http://127.0.0.1:9100
CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN=<shared-bearer>
CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_TIMEOUT_S=7200
CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_HEALTHCHECK_PATH=/health

CLYPT_PHASE1_VISUAL_SERVICE_URL=http://127.0.0.1:9200
CLYPT_PHASE1_VISUAL_SERVICE_AUTH_TOKEN=<shared-bearer>
CLYPT_PHASE1_VISUAL_SERVICE_TIMEOUT_S=3600
CLYPT_PHASE1_VISUAL_SERVICE_HEALTHCHECK_PATH=/health

CLYPT_PHASE24_DISPATCH_URL=http://192.241.241.118:9300
CLYPT_PHASE24_DISPATCH_AUTH_TOKEN=<shared-bearer>
CLYPT_PHASE24_DISPATCH_TIMEOUT_S=30

VIBEVOICE_BACKEND=vllm
VIBEVOICE_VLLM_BASE_URL=http://127.0.0.1:8000
VIBEVOICE_VLLM_MODEL=vibevoice
VIBEVOICE_VLLM_GPU_MEMORY_UTILIZATION=0.65
VIBEVOICE_VLLM_MAX_NUM_SEQS=3
VIBEVOICE_LONGFORM_ENABLED=1
VIBEVOICE_LONGFORM_SINGLE_PASS_MAX_MINUTES=60
VIBEVOICE_LONGFORM_TWO_SHARD_MAX_MINUTES=90
VIBEVOICE_LONGFORM_THREE_SHARD_MAX_MINUTES=180
VIBEVOICE_LONGFORM_MAX_SHARDS=3
VIBEVOICE_LONGFORM_SPEAKER_MATCH_THRESHOLD=0.85
VIBEVOICE_LONGFORM_REP_CLIP_MIN_SECONDS=15
VIBEVOICE_LONGFORM_REP_CLIP_MAX_SECONDS=30
VIBEVOICE_LONGFORM_VERIFIER_BACKEND=ecapa_tdnn
VIBEVOICE_LONGFORM_VERIFIER_DEVICE=cpu
VIBEVOICE_LONGFORM_VERIFIER_MODEL_ID=speechbrain/spkrec-ecapa-voxceleb
```

### 2.2 Phase1 H100 overlay

Only memory-sensitive overrides belong here:

```bash
VIBEVOICE_VLLM_GPU_MEMORY_UTILIZATION=0.74
VIBEVOICE_VLLM_MAX_NUM_SEQS=2
```

Do not change visual thresholds, tracker behavior, or semantic runtime defaults in the overlay.

### 2.3 Phase26 H200 default

```bash
GENAI_GENERATION_BACKEND=local_openai
CLYPT_LOCAL_LLM_BASE_URL=http://127.0.0.1:8001/v1
CLYPT_LOCAL_LLM_MODEL=Qwen/Qwen3.6-35B-A3B

CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite
CLYPT_PHASE24_NODE_MEDIA_PREP_URL=https://rithuuu--clypt-node-media-prep-node-media-prep.modal.run/tasks/node-media-prep
CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN=<shared-bearer>
```

`CLYPT_PHASE24_NODE_MEDIA_PREP_URL` accepts either the Modal base URL or the full task endpoint URL. The current known-good records use the full endpoint URL.
`RemoteNodeMediaPrepClient` handles the follow-up result polling internally, so Phase26 still receives a final ordered `media` list.

## 3) Phase1 Service Settings

### 3.1 VibeVoice service routing

| Env | Default | Notes |
| --- | --- | --- |
| `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL` | required | Base URL for the local Phase 1 VibeVoice service. |
| `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN` | required | Bearer token used by the runner client. |
| `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_TIMEOUT_S` | `7200` | Timeout for `/tasks/vibevoice-asr`. |
| `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_HEALTHCHECK_PATH` | `/health` | Health route. |

### 3.2 Visual service routing

| Env | Default | Notes |
| --- | --- | --- |
| `CLYPT_PHASE1_VISUAL_SERVICE_URL` | required | Base URL for the local Phase 1 visual service. |
| `CLYPT_PHASE1_VISUAL_SERVICE_AUTH_TOKEN` | required | Bearer token used by the runner client. |
| `CLYPT_PHASE1_VISUAL_SERVICE_TIMEOUT_S` | `3600` | Timeout for `/tasks/visual-extract`. |
| `CLYPT_PHASE1_VISUAL_SERVICE_HEALTHCHECK_PATH` | `/health` | Health route. |

### 3.3 Visual runtime knobs

| Env | Default | Notes |
| --- | --- | --- |
| `CLYPT_PHASE1_VISUAL_BACKEND` | `tensorrt_fp16` | Required fast path. |
| `CLYPT_PHASE1_VISUAL_BATCH_SIZE` | `16` | Preserve on H200 default. |
| `CLYPT_PHASE1_VISUAL_THRESHOLD` | `0.35` | Preserve. |
| `CLYPT_PHASE1_VISUAL_SHAPE` | `640` | Preserve. |
| `CLYPT_PHASE1_VISUAL_TRACKER` | `bytetrack` | Preserve. |
| `CLYPT_PHASE1_VISUAL_TRACKER_BUFFER` | `30` | Preserve. |
| `CLYPT_PHASE1_VISUAL_TRACKER_MATCH_THRESH` | `0.7` | Preserve. |
| `CLYPT_PHASE1_VISUAL_DECODE` | `gpu` | Preserve. |

### 3.4 Phase1 dispatch routing

| Env | Default | Notes |
| --- | --- | --- |
| `CLYPT_PHASE24_DISPATCH_URL` | required | Base URL for the Phase26 dispatch API. |
| `CLYPT_PHASE24_DISPATCH_AUTH_TOKEN` | required | Bearer token used by the Phase 1 client. |
| `CLYPT_PHASE24_DISPATCH_TIMEOUT_S` | `30` | Timeout for `/tasks/phase26-enqueue`. |

### 3.5 Local VibeVoice sidecar

| Env | Default | Notes |
| --- | --- | --- |
| `VIBEVOICE_BACKEND` | `vllm` | Must be `vllm`. |
| `VIBEVOICE_VLLM_BASE_URL` | required | Local vLLM sidecar URL. |
| `VIBEVOICE_VLLM_MODEL` | `vibevoice` | Must remain `vibevoice`. |
| `VIBEVOICE_VLLM_TIMEOUT_S` | `7200` | ASR timeout. |
| `VIBEVOICE_VLLM_HEALTHCHECK_PATH` | `/health` | Local health route. |
| `VIBEVOICE_VLLM_MAX_RETRIES` | `1` | Retry budget. |
| `VIBEVOICE_VLLM_GPU_MEMORY_UTILIZATION` | deployment-specific | Host bootstrap/deploy knob. |
| `VIBEVOICE_VLLM_MAX_NUM_SEQS` | deployment-specific | Host bootstrap/deploy knob. |
| `VIBEVOICE_VLLM_DTYPE` | `bfloat16` | Host bootstrap/deploy knob. |

### 3.6 Long-form VibeVoice controls

| Env | Default | Notes |
| --- | --- | --- |
| `VIBEVOICE_LONGFORM_ENABLED` | `1` | Enables 2-3 shard orchestration for long-form ASR inside the Phase1 VibeVoice service. |
| `VIBEVOICE_LONGFORM_SINGLE_PASS_MAX_MINUTES` | `60` | Inputs at or below this duration stay on the existing one-request path. |
| `VIBEVOICE_LONGFORM_TWO_SHARD_MAX_MINUTES` | `90` | Inputs above the single-pass cap and at or below this threshold split into exactly 2 shards. |
| `VIBEVOICE_LONGFORM_THREE_SHARD_MAX_MINUTES` | `180` | Inputs above the two-shard threshold and at or below this threshold split into exactly 3 shards. |
| `VIBEVOICE_LONGFORM_MAX_SHARDS` | `3` | Hard upper bound on shard fan-out. Values outside `1..3` are rejected at config load. |
| `VIBEVOICE_LONGFORM_SPEAKER_MATCH_THRESHOLD` | `0.85` | Cosine-similarity threshold for cross-shard speaker stitching. |
| `VIBEVOICE_LONGFORM_REP_CLIP_MIN_SECONDS` | `15` | Minimum representative clip length extracted per shard-local speaker. |
| `VIBEVOICE_LONGFORM_REP_CLIP_MAX_SECONDS` | `30` | Maximum representative clip length extracted per shard-local speaker. |
| `VIBEVOICE_LONGFORM_VERIFIER_BACKEND` | `ecapa_tdnn` | Current supported verifier backend. |
| `VIBEVOICE_LONGFORM_VERIFIER_DEVICE` | `cpu` | Device for the speaker verifier; keep on CPU unless explicitly retuning. |
| `VIBEVOICE_LONGFORM_VERIFIER_MODEL_ID` | `speechbrain/spkrec-ecapa-voxceleb` | Default ECAPA-TDNN model used for speaker verification. |
| `VIBEVOICE_LONGFORM_VERIFIER_CACHE_DIR` | unset | Optional cache directory for the verifier model. |

Operational notes:

- Inputs longer than `VIBEVOICE_LONGFORM_THREE_SHARD_MAX_MINUTES` fail fast.
- Long-form sharding reuses the existing `/tasks/vibevoice-asr` contract and returns one merged `turns` list.
- Shard audio is uploaded to temporary GCS objects under a run-scoped prefix so the URL-based VibeVoice path can stay unchanged.

## 4) Phase26 Settings

### 4.1 Dispatch service

| Env | Default | Notes |
| --- | --- | --- |
| `CLYPT_PHASE24_DISPATCH_AUTH_TOKEN` | required | Bearer token expected by `/tasks/phase26-enqueue`. |

### 4.2 Queue and worker

| Env | Default | Notes |
| --- | --- | --- |
| `CLYPT_PHASE24_QUEUE_BACKEND` | `local_sqlite` | Required. |
| `CLYPT_PHASE24_LOCAL_QUEUE_PATH` | `backend/outputs/phase24_local_queue.sqlite` | Queue file. |
| `CLYPT_PHASE24_LOCAL_POLL_INTERVAL_MS` | `500` | Worker poll interval. |
| `CLYPT_PHASE24_LOCAL_LEASE_TIMEOUT_S` | `1800` | Lease timeout. |
| `CLYPT_PHASE24_LOCAL_MAX_INFLIGHT` | `1` | Max inflight queue items. |
| `CLYPT_PHASE24_LOCAL_MAX_REQUESTS_PER_WORKER` | `0` | `0` means loop forever. |
| `CLYPT_PHASE24_LOCAL_RECLAIM_EXPIRED_LEASES` | `0` | Fail-fast default. |
| `CLYPT_PHASE24_LOCAL_FAIL_FAST_ON_STALE_RUNNING` | `1` | Fail-fast default. |
| `CLYPT_PHASE24_LOCAL_WORKER_ID` | `phase26-worker-1` | Worker identity. |
| `CLYPT_PHASE24_WORKER_SERVICE_NAME` | `clypt-phase26-worker` | Metadata. |

### 4.3 Node-media-prep

| Env | Default | Notes |
| --- | --- | --- |
| `CLYPT_PHASE24_NODE_MEDIA_PREP_URL` | required | Modal submit endpoint base or full task URL. |
| `CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN` | required | Shared bearer token. |
| `CLYPT_PHASE24_NODE_MEDIA_PREP_TIMEOUT_S` | `1800` | Total submit+poll wait budget on the Phase26 side. |
| `CLYPT_PHASE24_NODE_MEDIA_PREP_MAX_CONCURRENCY` | `16` | Modal L4 concurrency cap. |

### 4.4 Local OpenAI generation

| Env | Default | Notes |
| --- | --- | --- |
| `GENAI_GENERATION_BACKEND` | `local_openai` | Required. |
| `CLYPT_LOCAL_LLM_BASE_URL` | `http://127.0.0.1:8001/v1` | SGLang endpoint. |
| `CLYPT_LOCAL_LLM_MODEL` | unset | Set explicitly in deploy env. |
| `CLYPT_LOCAL_LLM_TIMEOUT_S` | `600` | Request timeout. |
| `CLYPT_LOCAL_LLM_MAX_RETRIES` | `6` | Retry budget. |

### 4.5 SGLang knobs

| Env | Default | Notes |
| --- | --- | --- |
| `SG_PACKAGE_SPEC` | `sglang[all]` | SGLang install target. |
| `SG_SCHEDULE_POLICY` | `lpm` | Launch arg source. |
| `SG_CHUNKED_PREFILL_SIZE` | `8192` | Launch arg source. |
| `SG_MEM_FRACTION_STATIC` | `0.78` | Launch arg source. |
| `SG_CONTEXT_LENGTH` | `65536` | Launch arg source. |
| `SG_EXTRA_ARGS` | unset | Optional extra flags. |

## 5) Shared Provider Settings

| Env | Default | Notes |
| --- | --- | --- |
| `GOOGLE_CLOUD_PROJECT` | required | Required on both hosts. |
| `GOOGLE_CLOUD_LOCATION` | `global` | Shared default. |
| `VERTEX_EMBEDDING_LOCATION` | `us-central1` | Active Phase26 baseline. `global` currently returns `404 NOT_FOUND` for `gemini-embedding-2-preview` in the live Clypt project despite the locations doc listing global support. |
| `GCS_BUCKET` | required | Required on both hosts and Modal. |
| `GOOGLE_APPLICATION_CREDENTIALS` | deployment-specific | Service account path or equivalent secret injection. |
| `VERTEX_EMBEDDING_BACKEND` | `vertex` | Used on Phase26 host. |
| `VERTEX_EMBEDDING_MODEL` | `gemini-embedding-2-preview` | Current default. |
| `CLYPT_SPANNER_INSTANCE` | `clypt-spanner-v3` | Persistence target. |
| `CLYPT_SPANNER_DATABASE` | `clypt-graph-db-v3` | Persistence target. |

## 6) Host-Level Naming

The active host-level names are `phase1` and `phase26`.
