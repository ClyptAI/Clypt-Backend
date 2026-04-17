# V3.1 DigitalOcean Phase 1 Deployment

**Status:** Active  
**Last updated:** 2026-04-17

This runbook is aligned to current code and scripts for single-host deployment:

- Phase 1 ASR selectable via `CLYPT_PHASE1_ASR_BACKEND`:
  - local VibeVoice vLLM (`:8000`)
  - or remote GCE L4 combined service (backend enum is still `cloud_run_l4`; deploy target is a GCE `g2-standard-8` L4 VM)
- Qwen generation on local OpenAI-compatible service (`:8001`) via SGLang helper
- Phase 2-4 local SQLite queue worker
- Optional GCE L4 combined service for Phase 1 ASR and node clip extraction/upload
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

### 3.4 Deploy GCE L4 combined ASR + media-prep service

Use this when the intended topology is:

- Phase 1 ASR via GCE L4 VM (`POST /tasks/asr`)
- Phase 2 node-media prep via the same GCE L4 VM (`POST /tasks/node-media-prep`)

> The Cloud Run L4 path (`scripts/deploy_l4_combined_service.sh`) is retained for
> history only. It OOMs on a 24 GB L4 during vLLM `profile_run` because
> VibeVoice's audio encoder upcasts from `bfloat16` to `float32` and bloats the
> model from ~10 GB to ~18 GB. The GCE path uses the same Dockerfile with a
> `sed` patch that pins the audio encoder to `bfloat16`.

#### 3.4.1 One-time GPU quota (`GPUS_ALL_REGIONS`)

Fresh GCP projects get `GPUS_ALL_REGIONS=0`, which blocks all non-SPOT L4 VM
creation. Bump it to at least `1` before first deploy:

```bash
gcloud components install alpha --quiet
gcloud alpha quotas preferences create \
  --service=compute.googleapis.com \
  --quota-id=GPUS-ALL-REGIONS-per-project \
  --preferred-value=1 \
  --email=<your-gcp-login-email>
```

The request is usually auto-approved within seconds. Verify with:

```bash
gcloud compute project-info describe --project=clypt-v3 \
  --format="value(quotas[?metric=='GPUS_ALL_REGIONS'].limit)"
```

#### 3.4.2 Build image + provision VM

```bash
cd /Users/rithvik/Clypt-Backend
PROJECT=clypt-v3 \
ZONE=us-central1-a \
REGION_AR=us-east4 \
VM_NAME=clypt-phase1-l4-gce \
DROPLET_IP=<your-DO-droplet-egress-ip> \
bash scripts/deploy_l4_gce.sh
```

Defaults encoded in the script:

- Machine type: `g2-standard-8`
- GPU: `nvidia-l4 x1`
- Boot disk: `200 GB pd-balanced`
- Image family: `common-cu129-ubuntu-2204-nvidia-580` (project `deeplearning-platform-release`)
- Firewall rule: `clypt-l4-combined-ingress` on tcp:8080, restricted to `DROPLET_IP/32`
- Network tag: `clypt-l4-combined`
- Container port: `8080`
- Host HF cache: `/var/clypt/hf-cache` bind-mounted into `/root/.cache/huggingface`
- Auth: none (firewall-gated; droplet calls `http://<VM_IP>:8080` directly)
- Image tag: `gce-bf16-<timestamp>` in `${REGION_AR}-docker.pkg.dev/${PROJECT}/cloud-run-source-deploy/clypt-phase1-l4-combined`

Current image behavior:

- builds `docker/phase24-media-prep/Dockerfile` via Google Cloud Build
- installs `hf_transfer` and sets `HF_HUB_ENABLE_HF_TRANSFER=1` for ~3-5x faster downloads
- bakes `microsoft/VibeVoice-ASR` into the image via `snapshot_download`
- **sed-patches `vllm_plugin/model.py`** to force `_audio_encoder_dtype = torch.bfloat16` (grep-guarded so the build fails if upstream reshapes that line)
- clones the VibeVoice repo during image build and installs it editable at runtime
- starts VibeVoice vLLM inside the container at boot (via `backend.runtime.l4_combined_bootstrap`)
- waits for local `/health` before serving FastAPI traffic

vLLM tuning for L4 (24 GB VRAM, single-concurrency L4 workload), wired into
`backend/runtime/l4_combined_bootstrap.py` as defaults:

- `--max-num-seqs 4`
- `--max-model-len 16384`
- `--gpu-memory-utilization 0.90`
- startup health-wait: `1500 s`
- `VIBEVOICE_FFMPEG_MAX_CONCURRENCY=16` for node-media prep

> First cold start on a fresh VM re-downloads the model (~3 min with
> `hf_transfer`) into `/var/clypt/hf-cache` because the bind mount overlays the
> baked-in image layer. Subsequent restarts on the same VM are cache hits.

#### 3.4.3 Known startup-script TTY gotcha

If the GCE startup script fails during NVIDIA container toolkit install with
`gpg: cannot open '/dev/tty'`, SSH into the VM and re-run the install with
`gpg --batch --yes --dearmor`, or re-run `scripts/deploy_l4_gce.sh` after
confirming the fix is present in the script.

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
# Max in-flight LLM request caps per stage (Qwen3.6 on H200, 2026-04-16 bench).
# Smaller videos won't saturate these; only large videos scale up.
CLYPT_PHASE2_MERGE_MAX_CONCURRENT=16
CLYPT_PHASE2_BOUNDARY_MAX_CONCURRENT=16
CLYPT_SIGNAL_MAX_CONCURRENT=8
CLYPT_PHASE3_LOCAL_MAX_CONCURRENT=24
CLYPT_PHASE3_LONG_RANGE_MAX_CONCURRENT=24
CLYPT_PHASE4_SUBGRAPH_MAX_CONCURRENT=16
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

### 4.3 Combined GCE L4 offload profile

The backend enum is still `cloud_run_l4`. Point the URL at the GCE VM's
external IP on port `8080` (firewall-gated to the droplet) and disable ID
tokens (`auth_mode=none`), since GCE does not issue the same audience-scoped
tokens Cloud Run did.

```bash
CLYPT_PHASE1_ASR_BACKEND=cloud_run_l4
CLYPT_PHASE1_ASR_SERVICE_URL=http://<GCE_VM_EXTERNAL_IP>:8080
CLYPT_PHASE1_ASR_AUTH_MODE=none
CLYPT_PHASE1_ASR_TIMEOUT_S=7200

# IMPORTANT: do not set VIBEVOICE_VLLM_BASE_URL on the caller in cloud_run_l4 mode.
VIBEVOICE_VLLM_MODEL=vibevoice

CLYPT_PHASE24_MEDIA_PREP_BACKEND=cloud_run_l4
CLYPT_PHASE24_MEDIA_PREP_SERVICE_URL=http://<GCE_VM_EXTERNAL_IP>:8080
CLYPT_PHASE24_MEDIA_PREP_AUTH_MODE=none
CLYPT_PHASE24_MEDIA_PREP_TIMEOUT_S=600
```

If you revert to the legacy Cloud Run L4 target (not recommended — known OOM
without the bf16 patch applied to the serving container), use
`https://<cloud-run-service>` URLs and `auth_mode=id_token`; keep
`CLYPT_PHASE1_ASR_AUDIENCE` / `CLYPT_PHASE24_MEDIA_PREP_AUDIENCE` equal to the
service URL.

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

# GCE L4 combined service (from the droplet; firewall-gated to the droplet IP)
curl -fsS http://<GCE_VM_EXTERNAL_IP>:8080/healthz

# VRAM sanity check (SSH to VM)
gcloud compute ssh clypt-phase1-l4-gce --zone us-central1-a --project clypt-v3 \
  --command "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"
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
- H200 does not expose NVENC; local Phase 2 node-media clip extraction must stay `local` only on NVENC-capable GPUs or be offloaded to the GCE L4 combined service (L4 has NVENC).
- On the GCE L4 path, auth is firewall-based (`AUTH_MODE=none` on the droplet; firewall scoped to droplet egress IP). No ID token is required. If you revert to a Cloud Run L4 deploy, the droplet must use a signing-capable service account and `AUTH_MODE=id_token`.
- The L4 combined container MUST ship with the bf16 audio-encoder sed patch in `docker/phase24-media-prep/Dockerfile`. Without it, vLLM `profile_run` OOMs on a 24 GB L4.
- GCE L4 deploys require `GPUS_ALL_REGIONS >= 1` project quota; see 3.4.1.
- The intended TensorRT visual path on H200 is now GPU resize + GPU preprocess, not full-resolution host decode followed by OpenCV resize.
- `CLYPT_GEMINI_MAX_CONCURRENT` has been removed; do not leave it in the live env because startup should fail fast when it is present.
- Current target env surface is explicit per-stage concurrency: `CLYPT_PHASE2_MERGE_MAX_CONCURRENT`, `CLYPT_PHASE2_BOUNDARY_MAX_CONCURRENT`, `CLYPT_SIGNAL_MAX_CONCURRENT`, `CLYPT_PHASE3_LOCAL_MAX_CONCURRENT`, `CLYPT_PHASE3_LONG_RANGE_MAX_CONCURRENT`, and `CLYPT_PHASE4_SUBGRAPH_MAX_CONCURRENT`. All of these are max in-flight LLM request caps per stage, not targets.
- If `CLYPT_PHASE1_ASR_BACKEND=cloud_run_l4`, the caller must set `CLYPT_PHASE1_ASR_SERVICE_URL` and must not set `VIBEVOICE_VLLM_BASE_URL`.
- On the 2026-04-15 Billy Carson reference replay, the optimized TensorRT path sustained about `240.1 fps` (`35705` frames in `148678.5 ms`) versus roughly `51.5 fps` before the decode/preprocess move to GPU.
- If a fresh host replay is still near `~50 fps`, re-sync the repo to the droplet and verify the runtime includes `scale_cuda` in the decode filter chain and CUDA-side preprocess in `backend/phase1_runtime/tensorrt_detector.py`.
