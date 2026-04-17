# H200 Phase 1 Visual + Phase 2-4 Deployment

**Status:** Active
**Last updated:** 2026-04-17

This runbook covers the **H200 host** only — the visual chain (RF-DETR +
ByteTrack), the Phase 1 orchestrator, the Phase 2-4 local queue + worker, and
the SGLang Qwen service.

The Phase 1 **audio chain** (VibeVoice vLLM ASR, NFA, emotion2vec+, YAMNet)
and **node-media prep** (ffmpeg NVENC clip extraction) live on a separate
**RTX 6000 Ada** droplet. See
[`docs/deployment/P1_AUDIO_HOST_DEPLOY.md`](P1_AUDIO_HOST_DEPLOY.md).

The H200 calls the RTX audio host over HTTP. There is **no local fallback**:
if the RTX host is unreachable or misconfigured, Phase 1 and Phase 2 jobs on
the H200 fail fast.

## 0) Topology Summary

| Responsibility | Host |
| --- | --- |
| RF-DETR + ByteTrack (TensorRT FP16) | H200 |
| Phase 1 orchestrator (`run_phase1`, Phase 1 API/worker) | H200 |
| Phase 2-4 local SQLite queue + worker | H200 |
| SGLang Qwen3.6-35B-A3B on `:8001` | H200 |
| VibeVoice vLLM ASR on `:8000` | RTX 6000 Ada |
| NFA / emotion2vec+ / YAMNet | RTX 6000 Ada |
| ffmpeg NVENC node-clip extraction | RTX 6000 Ada |

## 1) Provisioning Requirements

### 1.1 Image

Use a DigitalOcean **NVIDIA AI/ML** base image (Ubuntu 22.04 + NVIDIA
userland). If host libs are missing (for example `libnvidia-ml.so.1`) GPU
services will fail to start.

### 1.2 Required host paths

- repo: `/opt/clypt-phase1/repo`
- env file: `/etc/clypt-phase1/v3_1_phase1.env`
- service account key: `/opt/clypt-phase1/sa-key.json`
- Phase 1 + worker venv: `/opt/clypt-phase1/venvs/phase1`
- SGLang venv: `/opt/clypt-phase1/venvs/sglang`

## 2) Sync and Bootstrap

```bash
rsync -az --delete \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='outputs' \
  --exclude='*.egg-info' \
  -e "ssh -i ~/.ssh/clypt_do_ed25519 -o IdentitiesOnly=yes" \
  /Users/rithvik/Clypt-Backend/ \
  root@<H200_IP>:/opt/clypt-phase1/repo/
```

```bash
ssh -i ~/.ssh/clypt_do_ed25519 root@<H200_IP>
cd /opt/clypt-phase1/repo
bash scripts/do_phase1_visual/bootstrap_h200.sh
```

## 3) Deploy Runtime Services

### 3.1 Deploy visual + Phase 2-4 worker

```bash
cd /opt/clypt-phase1/repo
bash scripts/do_phase1_visual/deploy_visual_service.sh
```

This script:

- installs `requirements-do-phase1-visual.txt` into the Phase 1 venv (no
  VibeVoice / NFA / emotion2vec+ / YAMNet deps — those live on the RTX host)
- installs TensorRT host/runtime deps when `CLYPT_PHASE1_VISUAL_BACKEND=tensorrt_fp16`
- enforces that `VIBEVOICE_BACKEND` / `VIBEVOICE_VLLM_BASE_URL` / `VIBEVOICE_VLLM_MODEL`
  are **not** set in the H200 env file (those are a deployment bug on this host)
- probes `${CLYPT_PHASE1_AUDIO_HOST_URL}/health` with the shared bearer token
- installs and starts `clypt-v31-phase1-api.service`,
  `clypt-v31-phase1-worker.service`, and `clypt-v31-phase24-local-worker.service`

### 3.2 Deploy SGLang Qwen

```bash
bash scripts/do_phase1_visual/deploy_sglang_qwen_service.sh
```

Installs/starts `clypt-sglang-qwen.service`, isolates SGLang in its own venv,
and validates `/health` + `/v1/models`. Honors `SG_SCHEDULE_POLICY`,
`SG_CHUNKED_PREFILL_SIZE`, `SG_MEM_FRACTION_STATIC`, `SG_CONTEXT_LENGTH`, and
`SG_EXTRA_ARGS` from the env file.

## 4) Minimum Environment Contract

Start from [`docs/runtime/known-good.env`](../runtime/known-good.env) and
edit in your project/bucket/token values. See
[`docs/runtime/ENV_REFERENCE.md`](../runtime/ENV_REFERENCE.md) for the full
catalog.

### 4.1 Remote audio host (required, no fallback)

```bash
CLYPT_PHASE1_AUDIO_HOST_URL=http://<rtx6000ada-private-ip>:9100
CLYPT_PHASE1_AUDIO_HOST_TOKEN=<shared-bearer-token>
CLYPT_PHASE1_AUDIO_HOST_TIMEOUT_S=7200
CLYPT_PHASE1_AUDIO_HOST_HEALTHCHECK_PATH=/health
```

### 4.2 Remote node-media prep (required, no fallback)

Points at the same RTX droplet; `POST /tasks/node-media-prep`.

```bash
CLYPT_PHASE24_NODE_MEDIA_PREP_URL=http://<rtx6000ada-private-ip>:9100
CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN=<shared-bearer-token>
CLYPT_PHASE24_NODE_MEDIA_PREP_TIMEOUT_S=1800
CLYPT_PHASE24_NODE_MEDIA_PREP_MAX_CONCURRENCY=8
```

### 4.3 Other shared core values

```bash
GOOGLE_CLOUD_PROJECT=<project>
GCS_BUCKET=<bucket>
GOOGLE_APPLICATION_CREDENTIALS=/opt/clypt-phase1/sa-key.json

GENAI_GENERATION_BACKEND=local_openai
CLYPT_LOCAL_LLM_BASE_URL=http://127.0.0.1:8001/v1
CLYPT_LOCAL_LLM_MODEL=Qwen/Qwen3.6-35B-A3B

CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite
CLYPT_PHASE24_LOCAL_RECLAIM_EXPIRED_LEASES=0
CLYPT_PHASE24_LOCAL_FAIL_FAST_ON_STALE_RUNNING=1

CLYPT_PHASE1_VISUAL_BACKEND=tensorrt_fp16
CLYPT_PHASE1_VISUAL_TRT_ENGINE_DIR=backend/outputs/tensorrt_engines

CLYPT_PHASE1_INPUT_MODE=test_bank
CLYPT_PHASE1_TEST_BANK_PATH=/opt/clypt-phase1/repo/config/test_bank/phase1_canonical_assets.json
CLYPT_PHASE1_TEST_BANK_STRICT=1
```

## 5) Service Verification

```bash
systemctl is-active \
  clypt-sglang-qwen \
  clypt-v31-phase1-api \
  clypt-v31-phase1-worker \
  clypt-v31-phase24-local-worker

# SGLang
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8001/v1/models | python3 -m json.tool

# Remote audio host from this H200
curl -fsS \
  -H "Authorization: Bearer ${CLYPT_PHASE1_AUDIO_HOST_TOKEN}" \
  "${CLYPT_PHASE1_AUDIO_HOST_URL%/}/health"

# Local TensorRT toolchain
trtexec --help >/dev/null
/opt/clypt-phase1/venvs/phase1/bin/python -c "import tensorrt as trt; print(trt.__version__)"
/opt/clypt-phase1/venvs/sglang/bin/python -c "import sglang, torch; print(sglang.__version__, torch.__version__)"
```

Note: there is **no** `clypt-vllm-vibevoice.service` on this H200 anymore —
that unit lives on the RTX 6000 Ada audio host.

## 6) Runtime Execution

```bash
cd /opt/clypt-phase1/repo
source /opt/clypt-phase1/venvs/phase1/bin/activate
set -a; source /etc/clypt-phase1/v3_1_phase1.env; set +a

python -m backend.runtime.run_phase1 \
  --job-id "run_$(date +%Y%m%d_%H%M%S)" \
  --source-path /opt/clypt-phase1/videos/<video>.mp4 \
  --run-phase14
```

## 7) Operational Notes

- `VIBEVOICE_BACKEND`, `VIBEVOICE_VLLM_BASE_URL`, `VIBEVOICE_VLLM_MODEL`
  must NOT be set on the H200. They belong on the RTX audio host.
- `CLYPT_PHASE1_AUDIO_HOST_URL` + `CLYPT_PHASE1_AUDIO_HOST_TOKEN` and the
  matching `CLYPT_PHASE24_NODE_MEDIA_PREP_*` values are required. Config load
  fails fast if any are missing.
- `GOOGLE_CLOUD_PROJECT` and `GCS_BUCKET` are still required even for
  visual-only smoke tests.
- Phase 2-4 local worker defaults to fail-fast on stale leases and known
  structured-output/backend crash signatures.
- If a run crashes and the queue row remains `running`, resolve it manually
  before restart (expected with current fail-fast policy).
- The Phase 2-4 local worker unit depends on `clypt-sglang-qwen.service`
  and runs from the Phase 1 venv, not the SGLang env.
- `CLYPT_GEMINI_MAX_CONCURRENT` has been removed; do not leave it in the live
  env because startup should fail fast when it is present.
- On the 2026-04-15 Billy Carson reference replay, the optimized TensorRT
  path on H200 sustained about `240.1 fps` (`35705` frames in
  `148678.5 ms`). If a fresh host replay is near `~50 fps`, re-sync the
  repo and verify the runtime includes `scale_cuda` in the decode filter
  chain and CUDA-side preprocess in
  `backend/phase1_runtime/tensorrt_detector.py`.
