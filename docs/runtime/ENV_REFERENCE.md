# ENV REFERENCE

**Status:** Active  
**Last updated:** 2026-05-02

This is the code-backed env catalog for the current AMD-refactor topology:

- Phase1 vCPU orchestrator colocated on the Phase26 MI300X host with ElevenLabs Scribe v2 and Modal RF-DETR.
- Phase26 MI300X GPU service with local SGLang/Qwen.
- Modal L40S visual worker plus one shared Modal L40S media worker.

Canonical baselines:

- [known-good-phase26-mi300x.env](/Users/rithvik/Clypt-Backend/docs/runtime/known-good-phase26-mi300x.env)

Historical H200/H100 and Phase1 MI300X/VibeVoice env files are deleted on AMD-refactor. Historical mentions in `ERROR_LOG.md` are retained only for debugging context.

## 1) Phase1 Orchestrator

Phase1 runs on the MI300X host's vCPUs and uses the same canonical env file as Phase26. Required by `load_phase1_host_settings()`:

- `GOOGLE_CLOUD_PROJECT`
- `GCS_BUCKET` or `CLYPT_GCS_BUCKET`
- `GOOGLE_APPLICATION_CREDENTIALS` pointing at a signing-capable service-account JSON key
- `ELEVENLABS_API_KEY`
- `CLYPT_PHASE1_AUDIO_BACKEND=elevenlabs_scribe_v2`
- `CLYPT_PHASE1_INPUT_MODE=test_bank`
- `CLYPT_PHASE1_TEST_BANK_PATH`
- `CLYPT_PHASE1_VISUAL_SERVICE_URL`
- `CLYPT_PHASE1_VISUAL_SERVICE_AUTH_TOKEN`
- `CLYPT_PHASE24_DISPATCH_URL`
- `CLYPT_PHASE24_DISPATCH_AUTH_TOKEN`
- `CLYPT_YOUTUBE_DATA_API_KEY` or `YOUTUBE_API_KEY` for public YouTube metadata ingress

Scribe defaults:

| Env | Default | Notes |
| --- | --- | --- |
| `CLYPT_PHASE1_SCRIBE_MODEL_ID` | `scribe_v2` | Active ElevenLabs STT model. |
| `CLYPT_PHASE1_SCRIBE_LANGUAGE_CODE` | `en` | Keep English-only for now. |
| `CLYPT_PHASE1_SCRIBE_DIARIZE` | `1` | Required for speaker turns. |
| `CLYPT_PHASE1_SCRIBE_TAG_AUDIO_EVENTS` | `1` | Emits coarse audio events such as laughter when available. |
| `CLYPT_PHASE1_SCRIBE_TIMESTAMPS_GRANULARITY` | `word` | Required for Clypt timeline reconstruction. |
| `CLYPT_PHASE1_SCRIBE_URL_FIELD` | `source_url` | Signed HTTPS GCS URL field sent to ElevenLabs. |
| `CLYPT_PHASE1_SCRIBE_SIGNED_URL_EXPIRY_HOURS` | `24` | Signed audio URL lifetime. |
| `CLYPT_PHASE1_SCRIBE_TIMEOUT_S` | `7200` | Long-form synchronous API timeout. |
| `CLYPT_PHASE1_SCRIBE_MAX_RETRIES` | `2` | Retry budget for transient Scribe failures. |
| `CLYPT_PHASE1_SCRIBE_TURN_GAP_MS` | `1200` | Word gap used to split Scribe words into Clypt turns. |
| `CLYPT_PHASE1_SCRIBE_NUM_SPEAKERS` | unset | Optional frontend-provided override; omit by default. |
| `CLYPT_PHASE1_SCRIBE_KEYTERMS` | unset | Optional frontend-provided comma-separated hints; omit by default. |

Modal visual routing:

| Env | Default | Notes |
| --- | --- | --- |
| `CLYPT_PHASE1_VISUAL_BACKEND` | `modal_rfdetr` | Remote Modal RF-DETR path. |
| `CLYPT_PHASE1_VISUAL_SERVICE_URL` | required | Modal app base URL or `/tasks/visual-extract` URL. |
| `CLYPT_PHASE1_VISUAL_SERVICE_AUTH_TOKEN` | required | Bearer token for submit/poll. |
| `CLYPT_PHASE1_VISUAL_SERVICE_TIMEOUT_S` | `7200` | Poll timeout for RF-DETR completion. |
| `CLYPT_PHASE1_VISUAL_MODEL` | `nano` | Preserve current RF-DETR model. |
| `CLYPT_PHASE1_VISUAL_BATCH_SIZE` | `16` | Preserve current fast path. |
| `CLYPT_PHASE1_VISUAL_THRESHOLD` | `0.35` | Preserve current tuning. |
| `CLYPT_PHASE1_VISUAL_SHAPE` | `640` | Preserve current input shape. |

Phase1 handoff invariant:

- Phase1 uploads canonical audio/video to GCS.
- Phase1 calls Scribe synchronously using a signed HTTPS GCS audio URL.
- Phase1 submits Modal RF-DETR immediately and carries the returned `visual_future`.
- Phase1 enqueues Phase26 as soon as Scribe audio is adapted, without waiting for RF-DETR.
- Phase26 must hard-join the visual future before Phase5/frontend grounding or Phase6 visual use.

## 2) Phase26 MI300X

Required by `load_phase26_host_settings()`:

- `GOOGLE_CLOUD_PROJECT`
- `GCS_BUCKET` or `CLYPT_GCS_BUCKET`
- `GENAI_GENERATION_BACKEND=local_openai`
- `CLYPT_LOCAL_LLM_BASE_URL=http://127.0.0.1:8001/v1`
- `CLYPT_LOCAL_LLM_MODEL=Qwen/Qwen3.6-35B-A3B`
- `CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite`
- `CLYPT_PHASE24_NODE_MEDIA_PREP_URL`
- `CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN`

Current SGLang/Qwen baseline:

```bash
SG_DOCKER_IMAGE=lmsysorg/sglang:v0.5.10-rocm720-mi30x
SG_MODEL=Qwen/Qwen3.6-35B-A3B
SG_LAUNCH_PROFILE=final
SG_ACCEPTANCE_PROFILES=minimal strict_json fp8_kv scheduler_cache speculative
SG_SCHEDULE_POLICY=lpm
SG_CHUNKED_PREFILL_SIZE=8192
SG_MEM_FRACTION_STATIC=0.78
SG_CONTEXT_LENGTH=65536
```

Phase26 local queue defaults:

```bash
CLYPT_PHASE24_LOCAL_MAX_INFLIGHT=1
CLYPT_PHASE24_LOCAL_RECLAIM_EXPIRED_LEASES=0
CLYPT_PHASE24_LOCAL_FAIL_FAST_ON_STALE_RUNNING=1
```

## 3) Modal L40S Workers

### 3.1 Visual RF-DETR Worker

Runtime env:

- `GCS_BUCKET`
- `VISUAL_EXTRACT_AUTH_TOKEN` or `CLYPT_PHASE1_VISUAL_SERVICE_AUTH_TOKEN`
- `GOOGLE_APPLICATION_CREDENTIALS_JSON`

Fast path:

```bash
CLYPT_MODAL_VISUAL_MODEL=nano
CLYPT_MODAL_VISUAL_BATCH_SIZE=16
CLYPT_MODAL_VISUAL_THRESHOLD=0.35
CLYPT_MODAL_VISUAL_SHAPE=640
CLYPT_MODAL_VISUAL_BACKEND=tensorrt
```

The Modal visual worker must fail hard if CUDA ffmpeg hwaccel, `scale_cuda`, TensorRT Python runtime, `trtexec`, or CUDA PyTorch are unavailable.
The deployed Modal image is pinned to Python `3.12` because the CUDA PyTorch
and TensorRT wheels used by the visual fast path are not available for every
new default Python runtime.

### 3.2 Shared Media Worker

Runtime env:

- `GCS_BUCKET`
- `NODE_MEDIA_PREP_AUTH_TOKEN` or `CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN`
- `PHASE6_RENDER_AUTH_TOKEN` or `CLYPT_PHASE24_PHASE6_RENDER_TOKEN`
- `GOOGLE_APPLICATION_CREDENTIALS_JSON`

Phase26 endpoint env:

```bash
CLYPT_PHASE24_NODE_MEDIA_PREP_URL=https://<modal-media-app>/tasks/node-media-prep
CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN=<shared-bearer>
CLYPT_PHASE24_PHASE6_RENDER_URL=https://<modal-media-app>/tasks/render-video
CLYPT_PHASE24_PHASE6_RENDER_TOKEN=<shared-bearer>
```

The media worker uses one persistent L40S pool for node-media-prep and render/export with `min_containers=1`, `max_containers=1`. Keep batching inside each job/structure; do not deploy separate warm GPU pools for node media and render in cost-strict mode.
The media Modal image is also pinned to Python `3.12` for deploy parity and to
avoid unvalidated dependency wheel drift.

## 4) Superseded Env Families

These are not active in the Scribe/Modal topology:

- `VIBEVOICE_*`
- `CLYPT_PHASE1_VIBEVOICE_*`
- `CLYPT_PHASE1_NFA_DEVICE`
- `CLYPT_PHASE1_EMOTION2VEC_DEVICE`
- `CLYPT_PHASE1_YAMNET_DEVICE`
- `CLYPT_PHASE1_VLLM_SLEEP_*`
- Phase1-local ROCm/VAAPI RF-DETR env

If a file is edited in place instead of replaced, its filename must semantically match the new topology. For example, a Phase1 Scribe/Modal env must not keep a `phase1-mi300x` name.
