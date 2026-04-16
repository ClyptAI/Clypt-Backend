# V3.1 DigitalOcean Phase 1 Deployment

**Status:** Active  
**Last updated:** 2026-04-16

This runbook is aligned to current code and scripts for single-host deployment:

- Phase 1 ASR selectable via `CLYPT_PHASE1_ASR_BACKEND`:
  - local VibeVoice vLLM (`:8000`)
  - or remote Cloud Run L4 combined service
- Qwen generation on local OpenAI-compatible service (`:8001`) via SGLang helper
- Phase 2-4 local SQLite queue worker
- Optional Cloud Run L4 combined service for Phase 1 ASR and node clip extraction/upload
- Separate Python envs on-host:
  - Phase 1 + Phase 2-4 worker: `/opt/clypt-phase1/venvs/phase1`
  - SGLang Qwen service: `/opt/clypt-phase1/venvs/sglang`

For the exhaustive env inventory behind these profiles, see `docs/runtime/ENV_REFERENCE.md`.

## 1) Provisioning Requirements

### 1.1 Use GPU-ready image

Use a DigitalOcean GPU base image with NVIDIA userland preinstalled.  
If host libs are missing (for example `libnvidia-ml.so.1`), Docker GPU services will fail.

### 1.2 Required host paths

- repo: `/opt/clypt-phase1/repo`
- env file: `/etc/clypt-phase1/v3_1_phase1.env`
- service account key: `/opt/clypt-phase1/sa-key.json`
- Phase 1 venv: `/opt/clypt-phase1/venvs/phase1`
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
  root@<DROPLET_IP>:/opt/clypt-phase1/repo/
```

Then:

```bash
ssh -i ~/.ssh/clypt_do_ed25519 root@<DROPLET_IP>
cd /opt/clypt-phase1/repo
bash scripts/do_phase1/bootstrap_gpu_droplet.sh
```

## 3) Deploy Runtime Services

### 3.1 Deploy VibeVoice vLLM

```bash
cd /opt/clypt-phase1/repo
bash scripts/do_phase1/deploy_vllm_service.sh
```

### 3.2 Deploy Qwen SGLang

```bash
cd /opt/clypt-phase1/repo
bash scripts/do_phase1/deploy_sglang_qwen_service.sh
```

`deploy_sglang_qwen_service.sh` installs/starts `clypt-sglang-qwen.service`, disables old Qwen vLLM unit, validates `/health`, and checks `/v1/models`.
It also installs `ninja-build`, keeps SGLang isolated from the Phase 1 runtime env, sources `/etc/clypt-phase1/v3_1_phase1.env`, and now honors cache-aware launch knobs such as `SG_SCHEDULE_POLICY`, `SG_CHUNKED_PREFILL_SIZE`, `SG_MEM_FRACTION_STATIC`, `SG_CONTEXT_LENGTH`, and `SG_EXTRA_ARGS`.

### 3.3 Deploy-time host fixes now handled by scripts

- `deploy_vllm_service.sh`
  - creates/updates the dedicated Phase 1 venv
  - installs TensorRT host/runtime dependencies automatically when `CLYPT_PHASE1_VISUAL_BACKEND=tensorrt_fp16`
  - installs the tracked `clypt-v31-phase24-local-worker.service` unit
  - expects the checked-in TensorRT fast path that decodes to detector resolution on GPU and preprocesses on CUDA before inference
- `deploy_sglang_qwen_service.sh`
  - creates/updates the dedicated SGLang venv
  - installs `ninja-build` so SGLang JIT kernels can compile on fresh droplets

### 3.4 Deploy Cloud Run L4 combined ASR + media-prep service

Use this when the intended topology is:

- Phase 1 ASR via Cloud Run L4 (`POST /tasks/asr`)
- Phase 2 node-media prep via the same Cloud Run L4 service (`POST /tasks/node-media-prep`)

```bash
cd /Users/rithvik/Clypt-Backend
PROJECT=clypt-v3 \
REGION=us-east4 \
SERVICE=clypt-phase1-l4-combined \
bash scripts/deploy_l4_combined_service.sh
```

Recommended defaults encoded in the script:

- `--gpu 1`
- `--gpu-type nvidia-l4`
- `--cpu 8`
- `--memory 32Gi`
- `--concurrency 1`
- `--no-allow-unauthenticated`

Current image behavior:

- builds from the checked-in Cloud Run Dockerfile
- clones the VibeVoice repo during image build
- starts the local VibeVoice vLLM server inside the container at boot
- waits for local `/health` before serving FastAPI traffic

## 4) Minimum Environment Contract

For the full environment catalog, see `docs/runtime/ENV_REFERENCE.md`.

Populate `/etc/clypt-phase1/v3_1_phase1.env` with one of the following working profiles plus the shared core values.

### 4.1 Shared core values

```bash
GOOGLE_CLOUD_PROJECT=<project>
GCS_BUCKET=<bucket>
GOOGLE_APPLICATION_CREDENTIALS=/opt/clypt-phase1/sa-key.json

VIBEVOICE_BACKEND=vllm
GENAI_GENERATION_BACKEND=local_openai
CLYPT_LOCAL_LLM_BASE_URL=http://127.0.0.1:8001/v1
CLYPT_LOCAL_LLM_MODEL=Qwen/Qwen3.6-35B-A3B
CLYPT_PHASE2_MAX_CONCURRENT=8
CLYPT_PHASE2_BOUNDARY_MAX_CONCURRENT=10
CLYPT_SIGNAL_MAX_CONCURRENT=8
CLYPT_PHASE3_LOCAL_MAX_CONCURRENT=8
CLYPT_PHASE3_LONG_RANGE_MAX_CONCURRENT=8
CLYPT_PHASE4_SUBGRAPH_MAX_CONCURRENT=10
SG_SCHEDULE_POLICY=lpm
SG_CHUNKED_PREFILL_SIZE=8192
SG_MEM_FRACTION_STATIC=0.55
SG_CONTEXT_LENGTH=131072
# Optional; pin to an exact release or git SHA when you want fully reproducible installs.
SG_PACKAGE_SPEC=sglang[all]

VERTEX_EMBEDDING_BACKEND=vertex
CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite
CLYPT_PHASE3_LONG_RANGE_TOP_K=2
CLYPT_PHASE3_LONG_RANGE_PAIRS_PER_SHARD=24

CLYPT_PHASE24_LOCAL_RECLAIM_EXPIRED_LEASES=0
CLYPT_PHASE24_LOCAL_FAIL_FAST_ON_STALE_RUNNING=1

CLYPT_PHASE1_VISUAL_BACKEND=tensorrt_fp16
CLYPT_PHASE1_VISUAL_TRT_ENGINE_DIR=backend/outputs/tensorrt_engines

CLYPT_PHASE1_INPUT_MODE=test_bank
CLYPT_PHASE1_TEST_BANK_PATH=/opt/clypt-phase1/repo/config/test_bank/phase1_canonical_assets.json
CLYPT_PHASE1_TEST_BANK_STRICT=1
```

### 4.2 Local VibeVoice profile

```bash
CLYPT_PHASE1_ASR_BACKEND=vllm
VIBEVOICE_VLLM_BASE_URL=http://127.0.0.1:8000
VIBEVOICE_VLLM_MODEL=vibevoice

CLYPT_PHASE24_MEDIA_PREP_BACKEND=local
```

### 4.3 Combined Cloud Run L4 offload profile

```bash
CLYPT_PHASE1_ASR_BACKEND=cloud_run_l4
CLYPT_PHASE1_ASR_SERVICE_URL=https://<l4-combined-service>
CLYPT_PHASE1_ASR_AUTH_MODE=id_token
# Optional; defaults to SERVICE_URL
CLYPT_PHASE1_ASR_AUDIENCE=https://<l4-combined-service>
CLYPT_PHASE1_ASR_TIMEOUT_S=7200

# IMPORTANT: do not set VIBEVOICE_VLLM_BASE_URL on the caller in cloud_run_l4 mode.
VIBEVOICE_VLLM_MODEL=vibevoice

CLYPT_PHASE24_MEDIA_PREP_BACKEND=cloud_run_l4
CLYPT_PHASE24_MEDIA_PREP_SERVICE_URL=https://<l4-combined-service>
CLYPT_PHASE24_MEDIA_PREP_AUTH_MODE=id_token
# Optional; defaults to SERVICE_URL
CLYPT_PHASE24_MEDIA_PREP_AUDIENCE=https://<l4-combined-service>
CLYPT_PHASE24_MEDIA_PREP_TIMEOUT_S=600
```

## 5) Service Verification

```bash
systemctl is-active \
  clypt-vllm-vibevoice \
  clypt-sglang-qwen \
  clypt-v31-phase1-api \
  clypt-v31-phase1-worker \
  clypt-v31-phase24-local-worker

curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models | python3 -m json.tool
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8001/v1/models | python3 -m json.tool

trtexec --help >/dev/null
/opt/clypt-phase1/venvs/phase1/bin/python -c "import tensorrt as trt; print(trt.__version__)"
/opt/clypt-phase1/venvs/sglang/bin/python -c "import sglang, torch; print(sglang.__version__, torch.__version__)"
systemctl cat clypt-sglang-qwen | rg "ExecStart|schedule-policy|chunked-prefill-size|mem-fraction-static|context-length"
curl -fsS https://<l4-combined-service>/healthz
```

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

- VibeVoice model ID must stay `vibevoice`.
- `GOOGLE_CLOUD_PROJECT` and `GCS_BUCKET` are required even for ASR-only checks.
- Phase 2-4 local worker now defaults to fail-fast on stale leases and known structured-output/backend crash signatures.
- If a run crashes and queue row remains `running`, resolve manually before restart (expected with current fail-fast policy).
- The checked-in Phase 2-4 local worker unit now depends on `clypt-sglang-qwen.service` and runs from the Phase 1 venv, not the SGLang env.
- H200 does not expose NVENC; local Phase 2 node-media clip extraction must stay `local` only on NVENC-capable GPUs or be offloaded to the Cloud Run L4 combined service.
- The H200 host should invoke the private Cloud Run L4 combined service with a signing-capable service account so it can mint an ID token for the endpoint.
- The intended TensorRT visual path on H200 is now GPU resize + GPU preprocess, not full-resolution host decode followed by OpenCV resize.
- `CLYPT_GEMINI_MAX_CONCURRENT` has been removed; do not leave it in the live env because startup should fail fast when it is present.
- Current target env surface is explicit per-stage concurrency: `CLYPT_PHASE2_MAX_CONCURRENT`, `CLYPT_PHASE2_BOUNDARY_MAX_CONCURRENT`, `CLYPT_SIGNAL_MAX_CONCURRENT`, `CLYPT_PHASE3_LOCAL_MAX_CONCURRENT`, `CLYPT_PHASE3_LONG_RANGE_MAX_CONCURRENT`, and `CLYPT_PHASE4_SUBGRAPH_MAX_CONCURRENT`.
- If `CLYPT_PHASE1_ASR_BACKEND=cloud_run_l4`, the caller must set `CLYPT_PHASE1_ASR_SERVICE_URL` and must not set `VIBEVOICE_VLLM_BASE_URL`.
- On the 2026-04-15 Billy Carson reference replay, the optimized TensorRT path sustained about `240.1 fps` (`35705` frames in `148678.5 ms`) versus roughly `51.5 fps` before the decode/preprocess move to GPU.
- If a fresh host replay is still near `~50 fps`, re-sync the repo to the droplet and verify the runtime includes `scale_cuda` in the decode filter chain and CUDA-side preprocess in `backend/phase1_runtime/tensorrt_detector.py`.
