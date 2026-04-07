# Clypt V3.1 Backend Refactor

This branch contains only the refactored V3.1 backend implementation for:

1. `Phase 1: Timeline Foundation`
2. `Phase 2: Node Construction, Classification, And Embedding`
3. `Phase 3: Graph Construction`
4. `Phase 4: Candidate Retrieval, Subgraph Selection, And Ranking`

The implementation source of truth is:

- [docs/planning/v3.1_refactor_spec.md](/Users/rithvik/Clypt-V3/docs/planning/v3.1_refactor_spec.md)
- [docs/planning/v3.1_backend_implementation_spec_phases_1_4.md](/Users/rithvik/Clypt-V3/docs/planning/v3.1_backend_implementation_spec_phases_1_4.md)

## Repository Scope

This repository intentionally excludes the previous backend runtime, LR-ASD paths, old semantic pipeline scripts, old render code, and old frontend code.

What remains is the clean V3.1 Phase 1-4 backend under:

- [backend/pipeline](/Users/rithvik/Clypt-V3/backend/pipeline)

Primary module groups:

- [backend/pipeline/timeline](/Users/rithvik/Clypt-V3/backend/pipeline/timeline)
- [backend/pipeline/semantics](/Users/rithvik/Clypt-V3/backend/pipeline/semantics)
- [backend/pipeline/graph](/Users/rithvik/Clypt-V3/backend/pipeline/graph)
- [backend/pipeline/candidates](/Users/rithvik/Clypt-V3/backend/pipeline/candidates)
- [backend/pipeline/orchestrator.py](/Users/rithvik/Clypt-V3/backend/pipeline/orchestrator.py)
- [backend/providers](/Users/rithvik/Clypt-V3/backend/providers)
- [backend/phase1_runtime](/Users/rithvik/Clypt-V3/backend/phase1_runtime)
- [backend/runtime](/Users/rithvik/Clypt-V3/backend/runtime)

## Current Status

Implemented:

- typed Phase 1-4 contracts
- run-scoped artifact helpers
- Phase 1 timeline transforms
- Phase 1 sidecar orchestration for local GPU-backed audio + visual tasks
- Phase 1 operational runtime shell:
  - media preparation
  - ffmpeg shot detection + RF-DETR Small + ByteTrack visual extraction
  - shot-local track splitting
  - person detection ledger generation
  - SQLite-backed job store
  - FastAPI Phase 1 service shell
  - local Phase 1 job runner
  - remote job submission/log tailing shell
- live provider config for VibeVoice, Vertex AI, and GCS
- provider adapters for native / HF VibeVoice, NeMo Forced Aligner, Vertex Gemini, Vertex embeddings, emotion2vec+, and YAMNet
- Phase 2 turn-neighborhood batching, merge/classify adaptation, and boundary reconciliation
- Phase 3 structural edges, local semantic edges, long-range edge adjudication, and deterministic reconciliation
- Phase 4 prompt generation, seed retrieval, local subgraph construction, subgraph review adaptation, candidate dedupe, and pooled final review
- a Phase 1-4 orchestrator with provider-injected inputs
- a live Phase 1-4 runner for provider-backed Phases 2-4
- first validated native VibeVoice GPU path on DigitalOcean using a second venv, `flash_attention_2`, and `microsoft/VibeVoice-ASR`
- serial worker logging that streams app logs, native-worker stderr, and per-job output into live job log files
- deterministic native VibeVoice deploy path on DO:
  - main worker env pinned to CUDA 12.4 PyTorch wheels
  - native VibeVoice env built separately
  - `flash-attn==2.8.3` rebuilt from source in the native env

Not implemented yet on this branch:

- Phase 5 participation grounding
- Phase 6 camera intent and render planning
- face/identity-oriented visual ledgers beyond person tracking
- comment/trend/onboarding integrations
- frontend integration

## Current Working Setup

The current working setup on this branch is the **serial** Phase 1 runtime:

1. download or copy media
2. extract WAV audio locally
3. upload the source video to GCS
4. visual extraction with **RF-DETR Small + ByteTrack**
5. native VibeVoice ASR in a **second venv**
6. NeMo forced alignment
7. `emotion2vec+`
8. `YAMNet`

Important runtime facts:

- default visual backend: `pytorch_cuda_fp16`
- optimized visual backend: `tensorrt_fp16`
- default ASR backend: `VIBEVOICE_BACKEND=native`
- main worker env is pinned to:
  - `torch==2.6.0+cu124`
  - `torchvision==0.21.0+cu124`
  - `torchaudio==2.6.0+cu124`
- native VibeVoice env uses:
  - `microsoft/VibeVoice-ASR`
  - `flash_attention_2`
  - Liger enabled
- API and worker share the same SQLite DB and log root on the droplet
- worker systemd env sets:
  - `HOME=/opt/clypt-phase1`
  - `PYTORCH_KERNEL_CACHE_PATH=/opt/clypt-phase1/.cache/torch/kernels`

Validated on this branch:

- native VibeVoice-only runs succeeded on:
  - `60s` Joe Rogan slice
  - `300s` Joe Rogan slice
  - `540s` Joe Rogan slice
  - full `392.9s` MrBeast clip

Not yet freshly revalidated after the latest subprocess and logging fixes:

- one full end-to-end serial Phase 1 rerun on the full `788.7s` Joe Rogan clip

## Environment

Minimal Python dependencies are listed in [requirements.txt](/Users/rithvik/Clypt-V3/requirements.txt).

The Phase 1 worker now uses two Python environments in production:

- primary worker env: RF-DETR, runtime orchestration, Phases 2-4
- secondary native VibeVoice env: `microsoft/VibeVoice-ASR` subprocess worker

See [requirements-vibevoice-native.txt](/Users/rithvik/Clypt-V3/requirements-vibevoice-native.txt) and [docs/deployment/v3.1_phase1_digitalocean.md](/Users/rithvik/Clypt-V3/docs/deployment/v3.1_phase1_digitalocean.md) for the validated GPU bring-up path.

Core environment variables:

- `CLYPT_V31_OUTPUT_ROOT`
- `VIBEVOICE_BACKEND`
- `VIBEVOICE_NATIVE_VENV_PYTHON`
- `VIBEVOICE_MODEL_ID`
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`
- `GCS_BUCKET`

Additional defaults and runtime knobs live in [.env.example](/Users/rithvik/Clypt-V3/.env.example).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the native production ASR path, create the second venv separately:

```bash
bash scripts/do_phase1/install_native_vibevoice_env.sh
```

## Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/backend/pipeline -q
```

## Runtime Paths

The injected/original implementation entrypoint is:

- [backend/pipeline/orchestrator.py](/Users/rithvik/Clypt-V3/backend/pipeline/orchestrator.py)

Key entrypoints:

- `V31Phase14RunInputs`
- `V31Phase14Orchestrator`

The live/provider-backed entrypoint is:

- [backend/runtime/phase14_live.py](/Users/rithvik/Clypt-V3/backend/runtime/phase14_live.py)

Key entrypoint:

- `V31LivePhase14Runner`

The Phase 1 sidecar runtime/orchestration lives in:

- [backend/phase1_runtime/extract.py](/Users/rithvik/Clypt-V3/backend/phase1_runtime/extract.py)

Useful docs:

- [docs/planning/v3.1_backend_completion_spec.md](/Users/rithvik/Clypt-V3/docs/planning/v3.1_backend_completion_spec.md)
- [docs/deployment/v3.1_phase1_digitalocean.md](/Users/rithvik/Clypt-V3/docs/deployment/v3.1_phase1_digitalocean.md)
- [docs/runtime/v3.1_runtime_guide.md](/Users/rithvik/Clypt-V3/docs/runtime/v3.1_runtime_guide.md)

Local Phase 1 CLI:

```bash
source .venv/bin/activate
python -m backend.runtime.run_phase1 --job-id demo_run --source-url "https://www.youtube.com/watch?v=VIDEO_ID"
```

Native VibeVoice-only smoke test:

```bash
source .venv/bin/activate
export PYTHONPATH=.
python scripts/run_vibevoice_only.py --audio /path/to/file.wav
```
