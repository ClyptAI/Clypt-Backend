# V3.1 DigitalOcean Phase 1 Deployment

**Status:** Active vLLM deployment reference
**Droplet:** `clypt-v3-phase1-worker` — NVIDIA H200 141GB, Atlanta (`atl1`)
**SSH alias:** `clypt-do` (configured in `~/.ssh/config` — update the IP there when the droplet is recreated)

This document is the authoritative operational reference for deploying and running Phase 1 with the vLLM VibeVoice sidecar.

For copy/paste end-to-end repro commands (Phase 1 + queue-mode Phase 2-4 + verification), use:

- [Canonical Repro Checklist](../runtime/RUNTIME_GUIDE.md#4-canonical-repro-checklist) in `docs/runtime/RUNTIME_GUIDE.md`

## Current Working Setup

- **ASR:** VibeVoice via persistent Docker-managed vLLM service (`clypt-vllm-vibevoice.service`)
- **Visual:** RF-DETR Small + ByteTrack (`pytorch_cuda_fp16` default)
- **Execution:** visual extraction + ASR run **concurrently**; NFA / emotion2vec+ / YAMNet start immediately after ASR (not waiting for RF-DETR), running concurrent with visual — serial with each other
- **Phase 2-4 queue worker default profile:** `us-east4` L4 GPU-accelerated (CPU encoder path is fallback/degraded)
- **Model served as:** `vibevoice` (NOT `microsoft/VibeVoice-ASR` — see model ID gotcha below)
- **vLLM version:** `v0.14.1` (pinned — this is the tested/compatible version for the VibeVoice plugin)

### Run snapshots (vLLM backend, includes historical baselines)


| Video                | Duration | Turns | Wall time            | ASR RTF | Mode                                                                                                       |
| -------------------- | -------- | ----- | -------------------- | ------- | ---------------------------------------------------------------------------------------------------------- |
| mrbeastflagrant.mp4  | 392.9s   | 102   | 30.8s                | 0.07x   | ASR-only                                                                                                   |
| joeroganflagrant.mp4 | 788.7s   | 200   | 64.3s                | 0.07x   | ASR-only                                                                                                   |
| mrbeastflagrant.mp4  | 392.9s   | 104   | 179.2s (3.0 min)     | 0.07x   | Full Phase 1 (audio chain sequential)                                                                      |
| joeroganflagrant.mp4 | 788.7s   | 201   | 342.4s (5.7 min)     | 0.07x   | Full Phase 1 (audio chain sequential)                                                                      |
| joeroganflagrant.mp4 | 788.7s   | 201   | 299.6s (5.0 min)     | 0.07x   | **Full Phase 1 (audio chain parallel ✓)**                                                                  |
| mrbeastflagrant.mp4  | 392.9s   | 104   | **251s (4m11s)**     | 0.07x   | **Full Phase 1–4 end-to-end ✓** (Phase 1: 153s, Phases 2–4: 98s)                                           |
| mrbeastflagrant.mp4  | 392.9s   | 104   | **273.5s (4.6 min)** | 0.07x   | **Full Phase 1–4 end-to-end ✓** (fresh `ai/ml` redeploy; Phases 2–4: 271.8s)                               |
| mrbeastflagrant.mp4  | 392.9s   | 104   | **820.8s (13m41s)**  | —       | **Phase 2–4 queue worker ✓** (Cloud Run CPU-encoder fallback baseline)                                     |
| mrbeastflagrant.mp4  | 392.9s   | 104   | **143.8s (2m24s)**   | —       | **Phase 2–4 queue worker ✓** (verified Phase 1 handoff replay; us-east4 L4 + tuned Flash thinking profile) |


Audio chain parallel = NFA → emotion2vec+ → YAMNet start immediately after ASR, concurrent with RF-DETR. Audio artifacts ready ~234s earlier on the 13-min clip. Current production mode.

Historical-run labeling:
- Rows marked `audio chain sequential` and `CPU-encoder fallback baseline` are historical baselines, not current default production behavior.

Immediately after a successful Phase 1 run, validate queue handoff and worker execution (Cloud Task enqueue + Cloud Run logs) before treating the run as complete.

## Droplet Shape

**Current live droplet:**

- region: `atl1`
- GPU: `NVIDIA H200` (`143771 MiB` from `nvidia-smi`)
- image family: Ubuntu 22.04 + CUDA 12 GPU base image
- hostname: `clypt-v3-phase1-worker`
- SSH key: `~/.ssh/clypt_do_ed25519`

**Acceptable alternatives (in order of preference):**

1. `atl1` — `gpu-h200x1-141gb` (currently validated)
2. `nyc2` — H100/H200 equivalent if available
3. `ams3` — H100 equivalent if available

Do **not** silently fall back to a non-GPU droplet shape. If no H100/H200 slot is available, stop and check.

**Note:** GPU snapshots are GPU-family-specific. A snapshot taken from an H200 droplet will NOT work on an H100 droplet and vice versa. Always use the matching base image when creating a new droplet.

**Performance observed on validated GPU droplets:**

- RF-DETR Small (PyTorch CUDA FP16, `torch.compile`): ~109–114 fps
- VibeVoice vLLM: RTF ~0.07x (392.9s clip in 30.8s; 788.7s clip in 64.3s)

## DO Project

Use the DigitalOcean project: `Clypt-V3`

This is separate from the GCP project: `clypt-v3`

## Deploy Method

Use `rsync` from the local machine. The droplet has no GitHub SSH key.

```bash
rsync -az --delete \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='outputs' \
  --exclude='*.egg-info' \
  -e "ssh -i ~/.ssh/clypt_do_ed25519" \
  /Users/rithvik/Clypt-V3/ \
  root@<DROPLET_IP>:/opt/clypt-phase1/repo/
```

The `.env.local` file is **not** synced (gitignored). The active systemd services read `/etc/clypt-phase1/v3_1_phase1.env`, which must be populated manually.

## vLLM Service Deployment

Run once on a fresh droplet:

```bash
ssh -i ~/.ssh/clypt_do_ed25519 root@<DROPLET_IP>
cd /opt/clypt-phase1/repo
bash scripts/do_phase1/deploy_vllm_service.sh
```

What `deploy_vllm_service.sh` does:

1. Installs host prerequisites (`ffmpeg`, `git`, `python3-venv`, `curl`, `unzip`, etc.)
2. Installs Docker CE if absent
3. Installs `nvidia-container-toolkit` and configures the nvidia Docker runtime (`nvidia-ctk runtime configure --runtime=docker && systemctl restart docker`) — **required** for `--gpus all` to work
4. Validates `/etc/clypt-phase1/v3_1_phase1.env` and auto-quotes `VIBEVOICE_HOTWORDS_CONTEXT` if it contains shell-unsafe spaces
5. Installs main worker pip deps from `requirements-do-phase1.txt` into `.venv` (**no** native VibeVoice venv — the vLLM path does not need one), including guarded handling for the `youtokentome` build gotcha and resolver fallback
6. Validates `torchaudio` import in the main venv
7. Stops Phase 1 API/worker, then prewarms emotion2vec+ and NFA into persistent host cache (`/opt/clypt-phase1/.cache`) with timeout+retries (default enabled; keep `PREWARM_PHASE1_MODELS=1`), so the first real Phase 1 job does not block on large downloads
8. Installs Phase 1 API + worker systemd units
9. Clones/updates the VibeVoice plugin repo at `/opt/clypt-phase1/vibevoice-repo`
10. Builds the Docker image from `docker/vibevoice-vllm/Dockerfile` (based on `vllm/vllm-openai:v0.14.1`)
11. Creates `/opt/clypt-phase1/hf-cache` (mounted into the container for model weight persistence)
12. Installs and starts `clypt-vllm-vibevoice.service`
13. Polls `http://127.0.0.1:8000/health` until healthy (default timeout: 2400s)
14. Verifies `/v1/models` includes model id `vibevoice`
15. Restarts the Phase 1 API + worker and confirms both are active

### What happens on first container start

When the `clypt-vllm-vibevoice.service` container starts for the first time, `start_server.py` runs these steps inside the container (takes 15–30 minutes):

1. `apt-get install -y ffmpeg libsndfile1` — installs audio libraries
2. `pip install -e /app[vllm]` — installs the VibeVoice plugin as an editable package from the cloned repo (this is what registers the `vibevoice` model architecture with vLLM — **not** from PyPI)
3. `snapshot_download("microsoft/VibeVoice-ASR")` — downloads ~17GB of model weights to `/root/.cache/huggingface/` (mounted at `/opt/clypt-phase1/hf-cache`)
4. Generates tokenizer files
5. Starts vLLM: `vllm serve <model_path> --served-model-name vibevoice --trust-remote-code ...`

Steps 1–2 run on every container start (~1–2 min). The model download (step 3) only happens once because the HF cache volume is persistent. Subsequent starts skip the download and reach `/health` in ~2–3 minutes.

**Critical: do not install the PyPI `vibevoice` package.** There is an unrelated package on PyPI named `vibevoice`. Installing it breaks the main venv's `transformers` installation. The VibeVoice plugin is installed as an editable install from the cloned repo inside the container only.

### Monitoring startup

```bash
# Watch container logs
docker logs -f clypt-vllm-vibevoice

# Check systemd service status
systemctl status clypt-vllm-vibevoice.service

# Poll health manually
curl http://127.0.0.1:8000/health
```

The container is healthy when `/health` returns HTTP 200. Check available models:

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
# Should show: "id": "vibevoice"
```

### Model ID gotcha

The vLLM service registers the model as `vibevoice` via `--served-model-name vibevoice` in `start_server.py`. Using `microsoft/VibeVoice-ASR` as the model ID in API requests returns `HTTP 404 NotFoundError`. Always set:

```bash
VIBEVOICE_VLLM_MODEL=vibevoice
```

### Post-deploy first-run checklist

Run these immediately after `deploy_vllm_service.sh`:

```bash
systemctl is-active clypt-vllm-vibevoice clypt-v31-phase1-api clypt-v31-phase1-worker
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models | python3 -m json.tool
set -a; source /etc/clypt-phase1/v3_1_phase1.env; set +a
python - <<'PY'
import os
for k in ["CLYPT_PHASE1_CACHE_HOME","XDG_CACHE_HOME","TORCH_HOME","HF_HOME","FUNASR_MODEL_SOURCE"]:
    print(f"{k}={os.getenv(k)}")
PY
```

Expected:

- all three services are `active`
- `/health` returns HTTP 200
- `/v1/models` includes `"id": "vibevoice"`
- cache env resolves to `/opt/clypt-phase1/.cache/...` paths

### nvidia-container-toolkit requirement

The `docker run --gpus all` flag requires the nvidia Docker runtime. On fresh droplets, Docker may be installed but the nvidia runtime not configured. Fix:

```bash
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
# Verify:
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

### Stalled model download recovery

If the model download stalls (no `.incomplete` file size changes, 0.01% CPU, no network connections), restart the service. The HF cache is mounted as a volume so the download resumes from where it left off:

```bash
systemctl restart clypt-vllm-vibevoice.service
# Watch for "0 incomplete" in hf-cache:
find /opt/clypt-phase1/hf-cache/hub -name '*.incomplete' | wc -l
```

## Env / Secrets Required On The Droplet

Write these into `/etc/clypt-phase1/v3_1_phase1.env`. They are not synced by rsync.

### Core runtime

```bash
CLYPT_V31_OUTPUT_ROOT=backend/outputs/v3_1
CLYPT_PHASE1_WORK_ROOT=backend/outputs/v3_1_phase1_work
CLYPT_PHASE1_KEEP_WORKDIR=0
CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT=1
CLYPT_PHASE1_YAMNET_DEVICE=cpu
```

### VibeVoice vLLM backend

```bash
VIBEVOICE_BACKEND=vllm
VIBEVOICE_VLLM_BASE_URL=http://127.0.0.1:8000
VIBEVOICE_VLLM_MODEL=vibevoice
VIBEVOICE_VLLM_TIMEOUT_S=7200
VIBEVOICE_VLLM_HEALTHCHECK_PATH=/health
VIBEVOICE_VLLM_MAX_RETRIES=1
VIBEVOICE_MAX_NEW_TOKENS=32768
VIBEVOICE_TEMPERATURE=0
VIBEVOICE_TOP_P=1
VIBEVOICE_HOTWORDS_CONTEXT="I, you, he, she, it, we, they, me, him, her, us, them, my, your, his, hers, its, our, their, mine, yours, ours, theirs, this, that, these, those, who, whom, whose, which, what, and, but, or, nor, for, so, yet, after, although, as, because, before, if, since, that, though, unless, until, when, whenever, where, whereas, while, however, therefore, moreover, furthermore, also, additionally, meanwhile, consequently, otherwise, nevertheless, for example, in addition, on the other hand, similarly, likewise, in contrast, thus, hence, indeed, finally, first, second, third"
```

### GCP / Gemini

**IMPORTANT:** `GOOGLE_CLOUD_PROJECT` and `GCS_BUCKET` are required by `load_provider_settings()` even when running VibeVoice-only. The config loader validates them unconditionally — they are not optional even if Vertex AI is not being called.

```bash
GOOGLE_CLOUD_PROJECT=clypt-v3
GOOGLE_CLOUD_LOCATION=global              # retained for compatibility; generation is Developer API-only
GENAI_GENERATION_LOCATION=global
VERTEX_EMBEDDING_LOCATION=us-central1    # Embedding endpoint — cannot use global
GENAI_GENERATION_BACKEND=developer       # required generation path (AI Studio/Developer API only)
VERTEX_EMBEDDING_BACKEND=vertex          # keep embeddings on Vertex
GEMINI_API_KEY=<from secret manager/env>
GENAI_GENERATION_MODEL=gemini-3-flash-preview
GENAI_FLASH_MODEL=gemini-3-flash-preview # Current default model for all Phase 2-4 generation calls
VERTEX_EMBEDDING_MODEL=gemini-embedding-2-preview
GCS_BUCKET=clypt-storage-v3
```

Set `GENAI_FLASH_MODEL` explicitly for reproducibility. Current generation profile uses Flash for all Phase 2-4 calls with per-stage thinking levels.

### Phase 1 -> Phase 2-4 handoff contract (required immediately after Phase 1)

This deployment guide keeps only the interface contract needed after Phase 1 completes:

1. Run Phase 1 with `--run-phase14` so the queue handoff is enabled.
2. Verify handoff log marker:
   - `[phase24] queue-mode handoff complete — Cloud Task enqueued while RF-DETR finishes`
3. Confirm Cloud Run worker receives the task:
   - `gcloud run services logs read clypt-phase24-worker --region=us-east4 --project=clypt-v3 --follow`
4. Ensure Spanner schema is current before replay/benchmark runs that include comments/trends + provenance writes:
   - `python3 scripts/spanner/ensure_phase24_signal_schema.py --project clypt-v3 --instance clypt-spanner-v3 --database clypt-graph-db-v3`
5. Confirm run status in Spanner transitions to `PHASE24_DONE` (or terminal failure requiring rerun).

Phase 2-4 worker defaults, comments/trends signal defaults, and Phase 2-4 caveats are documented in the runtime guide.

Authoritative runtime reference:

- [RUNTIME_GUIDE.md - Phase 2-4 worker defaults](../runtime/RUNTIME_GUIDE.md#51-phase-2-4-worker-defaults-future-runs)
- [RUNTIME_GUIDE.md - Comments/Trends signal defaults](../runtime/RUNTIME_GUIDE.md#52-commentstrends-signal-defaults-future-runs)
- [RUNTIME_GUIDE.md - Detailed Phase 2-4 worker caveats](../runtime/RUNTIME_GUIDE.md#71-detailed-phase-2-4-worker-caveats)

### Cache env (must be consistent across prewarm + runtime)

```bash
CLYPT_PHASE1_CACHE_HOME=/opt/clypt-phase1/.cache
XDG_CACHE_HOME=/opt/clypt-phase1/.cache
TORCH_HOME=/opt/clypt-phase1/.cache/torch
HF_HOME=/opt/clypt-phase1/.cache/huggingface
FUNASR_MODEL_SOURCE=hf
```

`deploy_vllm_service.sh` now sources the env file before prewarm so these paths are honored consistently.

### Reference env template (current defaults)

Use these as the current baseline for `/etc/clypt-phase1/v3_1_phase1.env`:

```bash
CLYPT_V31_OUTPUT_ROOT=backend/outputs/v3_1
CLYPT_PHASE1_WORK_ROOT=backend/outputs/v3_1_phase1_work
CLYPT_PHASE1_KEEP_WORKDIR=0
CLYPT_PHASE1_YAMNET_DEVICE=cpu

CLYPT_PHASE1_VISUAL_BACKEND=pytorch_cuda_fp16
CLYPT_PHASE1_VISUAL_BATCH_SIZE=4
CLYPT_PHASE1_VISUAL_THRESHOLD=0.35
CLYPT_PHASE1_VISUAL_SHAPE=640
CLYPT_PHASE1_VISUAL_TRACKER=bytetrack
CLYPT_PHASE1_VISUAL_TRACKER_BUFFER=30
CLYPT_PHASE1_VISUAL_TRACKER_MATCH_THRESH=0.7
CLYPT_PHASE1_VISUAL_DECODE=cpu

VIBEVOICE_BACKEND=vllm
VIBEVOICE_VLLM_BASE_URL=http://127.0.0.1:8000
VIBEVOICE_VLLM_MODEL=vibevoice
VIBEVOICE_VLLM_TIMEOUT_S=7200
VIBEVOICE_VLLM_HEALTHCHECK_PATH=/health
VIBEVOICE_VLLM_MAX_RETRIES=1
VIBEVOICE_MAX_NEW_TOKENS=32768
VIBEVOICE_TEMPERATURE=0
VIBEVOICE_TOP_P=1
VIBEVOICE_HOTWORDS_CONTEXT=I,you,he,she,it,we,they,me,him,her,us,them,my,your,his,hers,its,our,their,mine,yours,ours,theirs,this,that,these,those,who,whom,whose,which,what,and,but,or,nor,for,so,yet,after,although,as,because,before,if,since,that,though,unless,until,when,whenever,where,whereas,while,however,therefore,moreover,furthermore,also,additionally,meanwhile,consequently,otherwise,nevertheless,forexample,inaddition,ontheotherhand,similarly,likewise,incontrast,thus,hence,indeed,finally,first,second,third

GOOGLE_CLOUD_PROJECT=clypt-v3
GOOGLE_CLOUD_LOCATION=global
GENAI_GENERATION_LOCATION=global
VERTEX_EMBEDDING_LOCATION=us-central1
GENAI_GENERATION_BACKEND=developer
VERTEX_EMBEDDING_BACKEND=vertex
GENAI_GENERATION_MODEL=gemini-3-flash-preview
GENAI_FLASH_MODEL=gemini-3-flash-preview
VERTEX_EMBEDDING_MODEL=gemini-embedding-2-preview
GCS_BUCKET=clypt-storage-v3
GOOGLE_APPLICATION_CREDENTIALS=/opt/clypt-phase1/sa-key.json

CLYPT_PHASE1_CACHE_HOME=/opt/clypt-phase1/.cache
XDG_CACHE_HOME=/opt/clypt-phase1/.cache
TORCH_HOME=/opt/clypt-phase1/.cache/torch
HF_HOME=/opt/clypt-phase1/.cache/huggingface
FUNASR_MODEL_SOURCE=hf
```

If `CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT` is omitted, runtime default is `1` (strict mode).
Legacy experiment keys (`VIBEVOICE_OUTPUT_MODE`, `VIBEVOICE_WORD_*`) are ignored by the main pipeline; deploy script warns if they are present.

### GCS service account key (required for video upload + Vertex)

The GCS bucket uses **uniform bucket-level access**. `blob.make_public()` fails with `400 BadRequest`. The only working path is V4 signed URLs via a service account key.

**The key already exists on the local machine** at `.tmp/clypt-phase1-worker-key.json` (gitignored). After provisioning a new droplet, upload it:

```bash
# Upload existing key from local machine
scp -i ~/.ssh/clypt_do_ed25519 \
  /Users/rithvik/Clypt-V3/.tmp/clypt-phase1-worker-key.json \
  root@<DROPLET_IP>:/opt/clypt-phase1/sa-key.json

ssh -i ~/.ssh/clypt_do_ed25519 root@<DROPLET_IP> \
  "chmod 600 /opt/clypt-phase1/sa-key.json"
```

If the key needs to be regenerated:

```bash
gcloud iam service-accounts keys create /tmp/clypt-phase1-worker-key.json \
  --iam-account=clypt-phase1-worker@clypt-v3.iam.gserviceaccount.com \
  --project=clypt-v3
```

Add to `/etc/clypt-phase1/v3_1_phase1.env`:

```bash
GOOGLE_APPLICATION_CREDENTIALS=/opt/clypt-phase1/sa-key.json
```

### Local video upload procedure

To run on a local video file (not YouTube), upload it to the droplet first:

```bash
# Upload video to droplet via SSH alias (preferred — IP is stored in ~/.ssh/config)
scp /path/to/local/video.mp4 clypt-do:/opt/clypt-phase1/videos/video.mp4

# Or with explicit key and IP
scp -i ~/.ssh/clypt_do_ed25519 \
  /path/to/local/video.mp4 \
  root@<DROPLET_IP>:/opt/clypt-phase1/videos/video.mp4
```

Then run with `--source-path` (NOT `--source-url`, NOT `file://`):

```bash
python -m backend.runtime.run_phase1 \
  --source-path /opt/clypt-phase1/videos/video.mp4 \
  --job-id "myvideo_$(date +%Y%m%d_%H%M%S)" \
  --run-phase14
```

The `videos/` directory must exist on the droplet (`mkdir -p /opt/clypt-phase1/videos`). Videos tested so far live at `/opt/clypt-phase1/videos/` — `mrbeastflagrant.mp4` and `joeroganflagrant.mp4` are already present on the current droplet.

### Visual pipeline

```bash
CLYPT_PHASE1_VISUAL_BACKEND=pytorch_cuda_fp16
CLYPT_PHASE1_VISUAL_BATCH_SIZE=4
CLYPT_PHASE1_VISUAL_THRESHOLD=0.35
CLYPT_PHASE1_VISUAL_SHAPE=640
CLYPT_PHASE1_VISUAL_TRACKER=bytetrack
CLYPT_PHASE1_VISUAL_TRACKER_BUFFER=30
CLYPT_PHASE1_VISUAL_TRACKER_MATCH_THRESH=0.7
CLYPT_PHASE1_VISUAL_DECODE=cpu
```

**Critical:** `CLYPT_PHASE1_VISUAL_SHAPE` must be `640`. RF-DETR's backbone requires inputs divisible by 32 with patch size 16 — `560` triggers an `AssertionError` at runtime.

**Critical:** `CLYPT_PHASE1_VISUAL_TRACKER_MATCH_THRESH` must be `0.7` (ByteTrack's upstream default). Higher values delay track activation, losing the beginning of person appearances.

## VibeVoice-Only Smoke Test

Before running a full Phase 1 job, validate the vLLM service with a direct VibeVoice-only run:

```bash
cd /opt/clypt-phase1/repo
source .venv/bin/activate
set -a; source /etc/clypt-phase1/v3_1_phase1.env; set +a
export PYTHONPATH=.
python scripts/run_vibevoice_only.py \
  --audio /path/to/clip.mp4 \
  --output-json /tmp/turns.json
```

The script accepts MP4, WAV, MP3, etc. MP4/video files are automatically extracted to a temporary MP3 before sending to vLLM.

Expected output: `INFO ... done in Xs — N turns (RTF 0.07x)`

## Launch Command (Full Phase 1)

```bash
cd /opt/clypt-phase1/repo
JOB_ID="my-video-$(date +%Y%m%d%H%M%S)"
LOG="/opt/clypt-phase1/logs/phase1-${JOB_ID}.log"
mkdir -p /opt/clypt-phase1/logs
set -a; source /etc/clypt-phase1/v3_1_phase1.env; set +a
source .venv/bin/activate

# YouTube / remote URL:
nohup python -m backend.runtime.run_phase1 \
  --source-url "https://www.youtube.com/watch?v=<VIDEO_ID>" \
  --job-id "$JOB_ID" \
  > "$LOG" 2>&1 &

# Local file:
nohup python -m backend.runtime.run_phase1 \
  --source-path /opt/clypt-phase1/videos/myvideo.mp4 \
  --job-id "$JOB_ID" \
  > "$LOG" 2>&1 &

echo "PID $! — log $LOG"
tail -f "$LOG"
```

## Runtime Dependencies

From `requirements-do-phase1.txt` (main venv — no second venv needed for vLLM):

- `rfdetr[onnx]` — RF-DETR Small + ONNX export for TensorRT path
- `supervision` — detection/annotation primitives
- `trackers` — ByteTrackTracker
- `opencv-python-headless` — frame decoding
- `funasr` — emotion2vec+
- `torchaudio>=2.0.0` — required by emotion2vec+ and NFA
- `tensorflow` + `tensorflow-hub` — YAMNet
- `torch` with CUDA — RF-DETR, emotion2vec+, NFA
- `transformers>=5.3.0` — RF-DETR runtime
- `nemo-toolkit[asr]` — NeMo Forced Aligner (~1GB model auto-downloads from NGC on first run)
- `protobuf<7` (with `!=5.28.0,!=5.29.0`) — required for `wandb` compatibility in the NeMo import path
- `httpx>=0.27.0` — HTTP client

System packages (on the droplet, not the container):

- `ffmpeg` — audio extraction, scene detection

For the vLLM container, `ffmpeg` and audio libraries are installed automatically by `start_server.py` inside the container on each start.

## Known Gotchas

### Model ID is `vibevoice` not `microsoft/VibeVoice-ASR`

The vLLM service registers the model under the name `vibevoice` via `--served-model-name vibevoice`. Using the HuggingFace repo ID as the model name returns `HTTP 404 NotFoundError`. Always set `VIBEVOICE_VLLM_MODEL=vibevoice`.

Confirm the served model name at any time:

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -c "import sys,json; [print(m['id']) for m in json.load(sys.stdin)['data']]"
```

### GOOGLE_CLOUD_PROJECT required even for vLLM-only runs

`load_provider_settings()` unconditionally validates `GOOGLE_CLOUD_PROJECT` regardless of which backend is active. The error `ValueError: GOOGLE_CLOUD_PROJECT is required` will appear even in a VibeVoice-only smoke test unless this env var is set.

### vLLM version must be v0.14.1

Using `vllm/vllm-openai:latest` fails. The VibeVoice plugin is only tested against `v0.14.1`. The `Dockerfile` is pinned to this version — do not change it.

The issue with other versions: vLLM's bundled `transformers` changes between releases, and the VibeVoice model architecture registration relies on the exact plugin hook API in `v0.14.1`.

### PyPI `vibevoice` package must NOT be installed

There is a completely unrelated package on PyPI named `vibevoice`. Installing it (e.g. `pip install vibevoice`) downgrades `transformers` from `>=5` to `4.51.x` and breaks vLLM's internal `ALLOWED_LAYER_TYPES` import with an `AttributeError`. The correct VibeVoice is installed as an editable package from the cloned repo (`pip install -e /app[vllm]`) inside the container, NOT from PyPI.

### nvidia Docker runtime must be explicitly configured

Docker may be installed on the base image without the nvidia runtime registered. `docker run --gpus all` silently fails or returns "unknown runtime" without it. Always run:

```bash
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

Verify before starting the service:

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

### start_server.py runs apt-get + pip install on every container start

This is expected and normal — the VibeVoice plugin's one-click setup. It takes ~1–2 minutes on each container start before the vLLM server begins. The model weights are cached so no re-download occurs on restarts.

### First full run may be slow if Phase 1 caches are cold

Phase 1 loads large model families outside the vLLM container:

- emotion2vec+ model (`emotion2vec/emotion2vec_plus_large` on HuggingFace; mapped from `iic/emotion2vec_plus_large`, ~1.81GB)
- NeMo forced aligner model (~1GB), when forced alignment is available in the current env

`deploy_vllm_service.sh` now prewarms these on deploy by default. If prewarm is skipped (`PREWARM_PHASE1_MODELS=0`), the first real Phase 1 job will spend significant time downloading them.

### Phase 2-4 timing gates in logs

Current live runner logs explicit timing boundaries:

- `[phase14] Phase 2 done in ...`
- `[phase14] Phase 3 done in ...`
- `[phase14] Phase 4 done in ...`
- `[phase14] Phases 2-4 done in ...`

The wrapper runtime also logs a join time for the concurrent branch:

- `[phase24] queue-mode handoff complete — Cloud Task enqueued while RF-DETR finishes`

### emotion2vec top-label log artifact (fixed)

Older builds logged `labels[0]/scores[0]` directly, which could print misleading lines such as `top: angry 0.00` when scores were unsorted. Current builds log the true top class by score argmax.

### Forced aligner returns `0 words`

If NeMo import fails due transitive dependency mismatch (common case: `wandb` with `protobuf>=7`), forced alignment can fail. Ensure:

```bash
pip show protobuf wandb nemo-toolkit
# protobuf must be <7 and not 5.28.0/5.29.0
```

`requirements-do-phase1.txt` now pins a compatible protobuf range to prevent this.

By default, the runtime now treats `0 words` on non-empty ASR turns as a hard error (`CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT=1`). Set it to `0` only as a temporary debugging bypass.

### Stalled model download

On the first start, the ~17GB model download may stall with no network activity (0.01% CPU, no open connections). The `Restart=always` policy will NOT auto-recover this — systemd only restarts if the process exits. Detect and recover manually:

```bash
# Check if download is stuck
find /opt/clypt-phase1/hf-cache/hub -name '*.incomplete' -exec ls -la {} \;
# Compare sizes 5 seconds apart — if unchanged, it's stalled
```

Recovery: `systemctl restart clypt-vllm-vibevoice.service`. The download resumes from the last byte.

### MP4 files are automatically extracted to MP3

The vLLM provider (`vibevoice_vllm.py`) detects video file extensions (`.mp4`, `.mov`, `.mkv`, etc.) and runs `ffmpeg -vn -acodec libmp3lame -q:a 2` to extract audio before sending. This is required — sending a raw 68MB MP4 as base64 to vLLM causes response truncation. The extracted MP3 is cleaned up after the request.

### Streaming is required to avoid response truncation

The provider uses `stream=True` in the vLLM API request and reads SSE chunks. Using non-streaming caused responses to be truncated mid-JSON for long audio (e.g. at character 13216 for a 392s clip). Always use the streaming path.

### Prompt format matters

VibeVoice was trained to respond to a specific prompt format. The provider sends:

```
This is a {duration:.2f} seconds audio, [with extra info: {hotwords}\n\n]please transcribe it with these keys: Start time, End time, Speaker ID, Content
```

Along with a system message: `"You are a helpful assistant that transcribes audio input into text output in JSON format."`

The duration in the prompt must be the actual audio duration (probed via `ffprobe`). An incorrect duration (e.g. `0.4 s` from a failed torchaudio probe on an MP4) causes the model to produce incomplete output.

### GCS bucket ACL vs uniform access

The bucket `clypt-storage-v3` has uniform bucket-level access. `blob.make_public()` fails with `400 BadRequest`. V4 signed URLs via a service account key are the only working path.

### RF-DETR COCO person class ID is 1, not 0

COCO class 0 is background in RF-DETR output. Person is class 1. Filtering on class 0 returns zero detections.

### ByteTrackTracker must be configured explicitly

The `ByteTrackTracker` constructor silently ignores config if arguments are not passed explicitly. Always pass `lost_track_buffer`, `frame_rate`, and `track_activation_threshold` directly.

### ByteTrackTracker frame rate must match actual video FPS

`lost_track_buffer` timeout = `buffer / frame_rate`. Passing `frame_rate=30` for a 23.98 fps video gives a wrong timeout. The actual FPS is detected via `cv2.VideoCapture` at tracker initialization.

### `sv.Detections` boolean indexing fails with non-uniform data fields

RF-DETR's `sv.Detections` output may contain `data` dict fields with shapes that don't match the primary detection array. Filtering via `det[person_mask]` triggers an `IndexError`. Reconstruct `sv.Detections` manually without carrying `det.data`.

### setuptools must be pinned to 69.5.1

`tensorflow-hub` 0.16.x uses `pkg_resources` from `setuptools`. Starting with setuptools ≥ 70, `pkg_resources` was removed from the default install. On a fresh droplet the venv may come with setuptools 80+ (via pip bootstrap), causing:

```
ModuleNotFoundError: No module named 'pkg_resources'
RuntimeError: tensorflow and tensorflow-hub are required for live YAMNet execution.
```

Fix is already in `requirements-do-phase1.txt`:

```
setuptools==69.5.1
```

If you hit this on an existing droplet: `pip install 'setuptools==69.5.1'`. NeMo reports a conflict warning but works fine at this version.

### YAMNet must run on CPU (CLYPT_PHASE1_YAMNET_DEVICE=cpu)

TensorFlow GPU init is unreliable on current GPU droplets for this stack — the error is:

```
Cannot dlopen some GPU libraries.
RuntimeError: YAMNet was configured for GPU but no TensorFlow GPU device is available.
```

PyTorch (RF-DETR, emotion2vec+, NFA) uses CUDA correctly; only TensorFlow's GPU runtime fails. YAMNet runs in ~0.5s on CPU for a 6-minute clip, so keep it on CPU:

```
CLYPT_PHASE1_YAMNET_DEVICE=cpu
```

This is included in the reference `.env` below.

### `youtokentome` requires Cython at build time

This is now handled directly by `deploy_vllm_service.sh` (it preinstalls Cython and installs `youtokentome` without build isolation before the full requirements install).

If you are installing dependencies manually, use:

```bash
pip install Cython setuptools
pip install --no-build-isolation youtokentome
pip install -r requirements-do-phase1.txt
```

### Pip resolver backtracking can fail on `datasets/pyarrow` (auto-recovered)

On some runs, the standard pip resolver backtracks into old `datasets` constraints and attempts a source build of legacy `pyarrow`/`numpy`, which fails on Python 3.10. `deploy_vllm_service.sh` handles this automatically:

1. tries standard resolver first
2. if it fails, retries with `PIP_USE_DEPRECATED=legacy-resolver`

This fallback path is expected and was validated on 2026-04-08; deployment still completed successfully end-to-end.

### Phase 2-4 worker caveats (moved)

Phase 2-4 worker caveats now live in:

- [RUNTIME_GUIDE.md - Detailed Phase 2-4 worker caveats](../runtime/RUNTIME_GUIDE.md#71-detailed-phase-2-4-worker-caveats)

### `as_completed` loop blocked the audio-chain callback (fixed)

A prior version of `backend/phase1_runtime/extract.py` used:

```python
for completed_fut in as_completed([visual_future, asr_future]):
    if completed_fut is asr_future:
        audio_chain_future = pool.submit(_run_audio_chain, ...)
```

This is wrong — `as_completed` exits the `for` loop iteration per completion but the callback submission was inside the loop, then continued to wait for the next future. The callback fired at the same time RF-DETR finished (~33s late), not when ASR finished.

**Fix (current):** Replace with `asr_future.result()` directly, then submit the audio chain and fire the callback while RF-DETR still runs:

```python
vibevoice_turns = asr_future.result()   # blocks until ASR only
audio_chain_future = pool.submit(_run_audio_chain, ...)
diarization_payload, emotion2vec_payload, yamnet_payload = audio_chain_future.result()
if on_audio_chain_complete is not None:
    on_audio_chain_complete(_partial)   # fires while RF-DETR still running
phase1_visual = visual_future.result()  # THEN wait for RF-DETR
```

### Phase 1 artifacts are local-first; queue handoff payload is uploaded to GCS

Only the **source video** (`phase1/{job_id}/source_video.mp4`) and the **Phase 2-4 handoff payload**
(`phase1/{job_id}/phase24_inputs/phase1_outputs.json`) are uploaded to GCS for queue-mode execution.
Phase 1 sidecar artifacts (`canonical_timeline.json`, `tracklet_geometry.json`, `speech_emotion_timeline.json`,
`audio_event_timeline.json`, etc.) are still written locally under `backend/outputs/v3_1/{job_id}/`.

Implications:

- If the droplet is destroyed after Phase 1 completes, local sidecar artifacts are lost unless separately exported.
- Queue-mode Phase 2-4 executes on Cloud Run and reads the handoff payload from GCS; it does not require the Phase 1 host.
- Keep queue handoff healthy and verify Cloud Task dispatch logs immediately after Phase 1 audio-chain completion.

### The systemd env file is not synced by rsync

The systemd env file lives outside the repo directory and must be managed manually after each new droplet provisioning.

## Benchmarking (Pending)

Before making TensorRT the production-default visual backend, benchmark on the target GPU:

1. `CLYPT_PHASE1_VISUAL_BACKEND=pytorch_cuda_fp16` — baseline
2. `CLYPT_PHASE1_VISUAL_BACKEND=tensorrt_fp16` — optimized (ONNX export + engine build on first run)

Success condition: TensorRT preserves or improves detection/tracking quality vs PyTorch CUDA and is faster in effective FPS.

---

*For the runtime provider surface and execution model, see `docs/runtime/RUNTIME_GUIDE.md`.*