# Clypt V3.1 Backend

This branch contains the V3.1 backend implementation for Phases 1–4 of the Clypt pipeline, with the vLLM VibeVoice ASR sidecar.

**Branch:** `v3.1-refactor-vLLM`

Phases implemented:

1. `Phase 1: Timeline Foundation` — visual extraction, ASR, forced alignment, emotion, audio events
2. `Phase 2: Node Construction, Classification, And Embedding`
3. `Phase 3: Graph Construction`
4. `Phase 4: Candidate Retrieval, Subgraph Selection, And Ranking`

Phases 5–6 (speaker participation grounding, render/9:16 output) are not yet implemented.

## Current Working Setup

Phase 1 runs visual extraction and ASR **concurrently**. The audio chain (NFA → emotion2vec+ → YAMNet) starts immediately when ASR finishes — without waiting for RF-DETR:

```
visual (RF-DETR + ByteTrack) ───────────────────────┐
                               ThreadPoolExecutor    ├── both done
ASR (VibeVoice vLLM HTTP) ───┘  (max_workers=3)     │
    ↓ immediately (RF-DETR still running)            │
NFA → emotion2vec+ → YAMNet ────────────────────────┘
    (serial with each other; concurrent with RF-DETR)
```

Audio artifacts ready ~230s before RF-DETR finishes on a 13-min clip.

- **ASR:** `VibeVoiceVLLMProvider` — HTTP calls to `clypt-vllm-vibevoice.service` (Docker-managed vLLM container)
- **Visual:** RF-DETR Small + ByteTrack (`pytorch_cuda_fp16` default)
- **vLLM version:** `v0.14.1` (pinned — tested compatible with the VibeVoice plugin)
- **Served model name:** `vibevoice` (not `microsoft/VibeVoice-ASR`)
- **No second venv** — the vLLM path does not require a native VibeVoice subprocess venv

### Validated runs (H100 80GB, 2026-04-07)

| Clip | Duration | Turns | Wall time | RTF |
|------|----------|-------|-----------|-----|
| mrbeastflagrant.mp4 | 392.9s | 102 | 30.8s | 0.07x |
| joeroganflagrant.mp4 | 788.7s | 200 | 64.3s | 0.07x |

## Environment

Core env vars:

```bash
CLYPT_V31_OUTPUT_ROOT=backend/outputs/v3_1
VIBEVOICE_BACKEND=vllm
VIBEVOICE_VLLM_BASE_URL=http://127.0.0.1:8000
VIBEVOICE_VLLM_MODEL=vibevoice          # NOT "microsoft/VibeVoice-ASR" — returns 404
GOOGLE_CLOUD_PROJECT=clypt-v3           # required even for VibeVoice-only runs
GCS_BUCKET=clypt-storage-v3            # required even for VibeVoice-only runs
```

Full config in [.env.example](/Users/rithvik/Clypt-V3/.env.example).

## Setup

```bash
# Primary venv (all backends including vLLM)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# For Phase 1 GPU deployment
pip install -r requirements-do-phase1.txt
```

A second venv is only needed for `VIBEVOICE_BACKEND=native` (legacy). The vLLM path needs only the primary venv.

## Tests

```bash
source .venv/bin/activate
python -m pytest tests/backend/pipeline -q
```

Pipeline tests (Phases 1–4) run fully offline — no API keys, GPU, or vLLM service required.

## VibeVoice-Only Smoke Test

```bash
source .venv/bin/activate
export PYTHONPATH=.
export VIBEVOICE_BACKEND=vllm
export VIBEVOICE_VLLM_BASE_URL=http://127.0.0.1:8000
export VIBEVOICE_VLLM_MODEL=vibevoice
export GOOGLE_CLOUD_PROJECT=clypt-v3
export GCS_BUCKET=clypt-storage-v3
python scripts/run_vibevoice_only.py --audio /path/to/clip.mp4
```

Accepts MP4, WAV, MP3, etc. MP4 files are automatically extracted to MP3 before sending.

## Production Deployment

Target: DigitalOcean GPU droplet (`gpu-h100x1-base`, Ubuntu 22.04 + CUDA 12).

```bash
# Sync repo to droplet
rsync -az --delete --exclude='.git' --exclude='.venv' --exclude='outputs' \
  -e "ssh -i ~/.ssh/clypt_do_ed25519" \
  /Users/rithvik/Clypt-V3/ root@<DROPLET_IP>:/opt/clypt-phase1/repo/

# Deploy vLLM service (run once on fresh droplet)
ssh -i ~/.ssh/clypt_do_ed25519 root@<DROPLET_IP>
bash /opt/clypt-phase1/repo/scripts/do_phase1/deploy_vllm_service.sh
```

First start downloads the ~17GB model — takes 15–30 minutes. Subsequent starts complete in ~2–3 minutes (model is cached).

See [docs/deployment/v3.1_phase1_digitalocean.md](/Users/rithvik/Clypt-V3/docs/deployment/v3.1_phase1_digitalocean.md) for the full deployment reference including all gotchas.

## Runtime Paths

### Injected / test path

- [backend/pipeline/orchestrator.py](/Users/rithvik/Clypt-V3/backend/pipeline/orchestrator.py) — `V31Phase14Orchestrator` / `V31Phase14RunInputs`

### Live / provider-backed path

- [backend/runtime/phase14_live.py](/Users/rithvik/Clypt-V3/backend/runtime/phase14_live.py) — `V31LivePhase14Runner`

### Phase 1 sidecar orchestration

- [backend/phase1_runtime/extract.py](/Users/rithvik/Clypt-V3/backend/phase1_runtime/extract.py) — concurrent or serial depending on backend

### Phase 1 CLI

```bash
source .venv/bin/activate
set -a; source /etc/clypt-phase1/v3_1_phase1.env; set +a
python -m backend.runtime.run_phase1 \
  --job-id demo_run \
  --source-url "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Key Docs

- [docs/deployment/v3.1_phase1_digitalocean.md](/Users/rithvik/Clypt-V3/docs/deployment/v3.1_phase1_digitalocean.md) — deployment, env setup, all gotchas
- [docs/runtime/v3.1_runtime_guide.md](/Users/rithvik/Clypt-V3/docs/runtime/v3.1_runtime_guide.md) — provider surface, execution model
- [docs/superpowers/specs/2026-04-06-vllm-vibevoice-phase1-design.md](/Users/rithvik/Clypt-V3/docs/superpowers/specs/2026-04-06-vllm-vibevoice-phase1-design.md) — vLLM migration design spec

## Repository Structure

```
backend/pipeline/          — Phase 1-4 contracts, transforms, orchestrator
backend/providers/         — VibeVoice (native/hf/vllm), Vertex AI, GCS, emotion2vec+, YAMNet
backend/phase1_runtime/    — Phase 1 sidecar orchestration, visual pipeline, job store
backend/runtime/           — Live Phase 1-4 runner, CLI entrypoints, native VibeVoice worker
docker/vibevoice-vllm/     — Dockerfile for the vLLM sidecar (based on vllm/vllm-openai:v0.14.1)
scripts/do_phase1/         — Deployment scripts and systemd units
docs/                      — Deployment reference, runtime guide, specs
```
