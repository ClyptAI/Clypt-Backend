# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Skills

Use the `gstack` skill whenever plausible — it provides a fast headless browser for QA testing, navigating pages, interacting with elements, and verifying UI behavior.

## Project Overview

Clypt V3.1 is an AI-powered video analysis and clip generation system that processes long-form content (podcasts, interviews) into ranked short-form clip candidates via a six-phase pipeline. Phases 1–4 are implemented; Phases 5–6 (speaker participation grounding, render/9:16 output) are not yet implemented.

## Commands

### Environment Setup

```bash
# Primary venv (used for all backends including vLLM)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Native VibeVoice venv (only needed when VIBEVOICE_BACKEND=native)
python3 -m venv .venv-vibevoice-native && source .venv-vibevoice-native/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-vibevoice-native.txt

# Phase 1 GPU deployment dependencies
pip install -r requirements-do-phase1.txt
```

### Tests

Pipeline tests (Phases 1–4) run offline using pre-computed fixture responses — no real API keys or GPU needed. Provider and Phase 1 runtime tests may require credentials/hardware.

```bash
# All pipeline tests (Phases 1–4)
python -m pytest tests/backend/pipeline -q

# Provider integration tests
python -m pytest tests/backend/providers/ -v

# Phase 1 runtime tests
python -m pytest tests/backend/phase1_runtime/ -v

# Single test
python -m pytest tests/backend/pipeline/test_orchestrator.py::test_orchestrator_runs_phases_1_to_4_and_writes_artifacts -v
```

### Running the Pipeline

```bash
# Phase 1 locally (optionally continues to Phases 2–4)
python -m backend.runtime.run_phase1 \
  --job-id <run_id> \
  --source-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --run-phase14

# Phase 1 FastAPI service
python -m backend.runtime.run_phase1_service

# Phase 1 remote worker
python -m backend.runtime.run_phase1_worker

# Smoke test — VibeVoice ASR only (vLLM backend)
export PYTHONPATH=.
export VIBEVOICE_BACKEND=vllm
export VIBEVOICE_VLLM_BASE_URL=http://127.0.0.1:8000
export VIBEVOICE_VLLM_MODEL=vibevoice
export GOOGLE_CLOUD_PROJECT=clypt-v3
export GCS_BUCKET=clypt-storage-v3
python scripts/run_vibevoice_only.py --audio /path/to/file.mp4
```

## Architecture

### Six-Phase Pipeline

All phases are orchestrated by `V31Phase14Orchestrator` (`backend/pipeline/orchestrator.py`). The live provider-backed path uses `V31LivePhase14Runner` (`backend/runtime/phase14_live.py`).

**Phase 1 — Timeline Foundation** (GPU-only sidecar tasks):
- RF-DETR Small + ByteTrack → `shot_tracklet_index.json`, `tracklet_geometry.json`
- VibeVoice ASR (diarization + transcription) → `canonical_timeline.json`
- NeMo Forced Aligner (word timestamps per turn) → appended to timeline
- emotion2vec+ (speech emotion per turn) → `speech_emotion_timeline.json`
- YAMNet (scene audio events) → `audio_event_timeline.json`

Entry: `backend/phase1_runtime/extract.py:run_phase1_sidecars()`

**Phase 2 — Node Construction & Classification**:
Groups adjacent speaker turns into neighborhoods (~8 turns + 2 halo), calls Gemini to merge and classify into semantic node types (`claim`, `anecdote`, `qa_exchange`, `challenge_exchange`, `setup_payoff`, `reveal`, `transition`, etc.), then computes embeddings.

**Phase 3 — Graph Construction**:
Builds structural edges (turn adjacency), local semantic edges (answers, challenges, elaborates, setup_for, payoff_of, etc.), and long-range edges (topic_recurrence, escalates) using embedding similarity.

**Phase 4 — Candidate Retrieval & Ranking**:
Generates meta-prompts → embeds → retrieves seed nodes via similarity → expands local subgraphs (BFS, bounded by `Phase4SubgraphConfig`) → Gemini validates/ranks subgraph candidates → deduplicates by span IoU and node overlap → final pool review.

### Key Source Files

| Path | Role |
|------|------|
| `backend/pipeline/contracts.py` | All Pydantic data models (canonical types for every phase output) |
| `backend/pipeline/artifacts.py` | Artifact path layout per run_id |
| `backend/pipeline/config.py` | `V31Config` + `Phase4SubgraphConfig` (duration/node/hop budgets) |
| `backend/pipeline/orchestrator.py` | `V31Phase14Orchestrator` — master coordinator |
| `backend/pipeline/_embedding_utils.py` | Fallback SHA256-based deterministic embeddings |
| `backend/providers/config.py` | All provider env loading |
| `backend/providers/vertex.py` | Gemini LLM + embedding calls |
| `backend/providers/vibevoice.py` | VibeVoice ASR (native subprocess + HF in-process paths) |
| `backend/providers/vibevoice_vllm.py` | VibeVoice ASR via persistent vLLM HTTP sidecar |
| `backend/runtime/phase14_live.py` | `V31LivePhase14Runner` — production entry point |
| `backend/runtime/vibevoice_native_worker.py` | Subprocess worker for native VibeVoice (flash-attn, Liger, CUDA 12.4) |
| `backend/phase1_runtime/extract.py` | Phase 1 sidecar orchestration — serial or concurrent depending on backend |
| `backend/phase1_runtime/factory.py` | Wires the correct VibeVoice provider based on `VIBEVOICE_BACKEND` |
| `backend/phase1_runtime/visual.py` | RF-DETR + ByteTrack visual extraction pipeline |
| `backend/phase1_runtime/state_store.py` | SQLite-backed job persistence for Phase 1 worker loop |

### Artifact Layout

All outputs land under `{CLYPT_V31_OUTPUT_ROOT}/{run_id}/`:
```
timeline/     → Phase 1 sidecar outputs
semantics/    → Phase 2 nodes + debug
graph/        → Phase 3 edges
candidates/   → Phase 4 clip candidates + debug
```

### Two Execution Paths

1. **Injected/Test path** — `V31Phase14Orchestrator.from_env()` accepts pre-loaded sidecar payloads via `V31Phase14RunInputs`. Used in all tests.
2. **Live/Production path** — `V31LivePhase14Runner.from_env(...)` wires real provider clients (Vertex, GCS, VibeVoice, etc.) and accepts `Phase1SidecarOutputs`.

### Phase 1 Service Infrastructure

The Phase 1 runtime has three runtime modes that share the same extraction logic:
- **CLI** (`run_phase1.py`) — single job, local execution
- **FastAPI service** (`app.py` + `run_phase1_service.py`) — HTTP job submission endpoint
- **Worker loop** (`worker.py` + `run_phase1_worker.py`) — polls SQLite for pending jobs

Job state is persisted via `backend/phase1_runtime/state_store.py` (SQLite). `backend/phase1_runtime/factory.py` wires provider instances; `backend/phase1_runtime/runner.py` executes a job from a `Phase1Workspace`.

### VibeVoice Backends

Controlled by `VIBEVOICE_BACKEND` env var:
- `vllm` — **current production backend**. Sends audio to a persistent Docker-managed vLLM service (`clypt-vllm-vibevoice.service`) over localhost HTTP. Uses OpenAI-compatible `/v1/chat/completions` with streaming. MP4/video inputs are automatically extracted to MP3 via ffmpeg before sending. The served model name is `vibevoice` (registered by `start_server.py --served-model-name vibevoice` — NOT the HuggingFace repo ID). Because ASR is an HTTP call and not a CUDA operation, `VibeVoiceVLLMProvider.supports_concurrent_visual = True` allows `extract.py` to run visual extraction and ASR in parallel via `ThreadPoolExecutor`.
- `native` — runs VibeVoice in a subprocess using a separate venv (`.venv-vibevoice-native`) to isolate `transformers<5` from the main env. Path configured via `VIBEVOICE_NATIVE_VENV_PYTHON`. Runs serially after visual.
- `hf` — runs VibeVoice directly in the main venv (in-process). Runs serially after visual.

### Phase 1 Execution Schedule

**vLLM backend (concurrent):**
```
visual extraction ──────────────────────────────────┐
                                                     ├── both done → complete
ASR (vLLM HTTP) ────┘                               │
    ↓ immediately (RF-DETR still running)            │
NeMo Forced Aligner → emotion2vec+ → YAMNet ────────┘
    (serial with each other; concurrent with RF-DETR)
```

ASR typically finishes in ~30–60s (RTF 0.07–0.08x). NFA+emotion2vec++YAMNet
complete ~35s later. RF-DETR takes ~150–300s. Audio chain artifacts are ready
well before visual finishes, allowing Phases 2–4 to start earlier.

**native / hf backends (serial):**
```
visual → ASR → NFA → emotion2vec+ → YAMNet
```

NFA, emotion2vec+, and YAMNet are always serial with each other — they share
the GPU and running them concurrently causes CUDA memory contention and graph
conflicts. Concurrency with RF-DETR (vLLM path only) is safe because RF-DETR's
torch.compile CUDA graphs are captured during warmup, not inference replay.

## Environment

Copy `.env.example` for all config keys. Critical ones for the vLLM backend:

```bash
CLYPT_V31_OUTPUT_ROOT=backend/outputs/v3_1
GOOGLE_CLOUD_PROJECT=clypt-v3          # REQUIRED even for VibeVoice-only runs
GCS_BUCKET=clypt-storage-v3            # REQUIRED even for VibeVoice-only runs
VIBEVOICE_BACKEND=vllm
VIBEVOICE_VLLM_BASE_URL=http://127.0.0.1:8000
VIBEVOICE_VLLM_MODEL=vibevoice         # NOT "microsoft/VibeVoice-ASR" — that returns 404
VERTEX_GEMINI_MODEL=gemini-3.1-pro-preview
VERTEX_EMBEDDING_MODEL=gemini-embedding-2-preview
CLYPT_PHASE1_VISUAL_BACKEND=pytorch_cuda_fp16  # or tensorrt_fp16
```

`GOOGLE_CLOUD_PROJECT` and `GCS_BUCKET` are validated unconditionally by `load_provider_settings()` even when only running VibeVoice. They must be set.

## Production Deployment

Target: DigitalOcean GPU droplet (`gpu-h100x1-base`, Ubuntu 22.04 + CUDA 12, region `ams3` or `nyc2`).
Deploy via `rsync` (not git). See `docs/deployment/v3.1_phase1_digitalocean.md`.
Deployment scripts live in `scripts/do_phase1/` (bootstrap, install, deploy, systemd units).

The vLLM service is managed by `scripts/do_phase1/deploy_vllm_service.sh`, which:
1. Installs Docker + nvidia-container-toolkit (if absent)
2. Configures the nvidia Docker runtime (`nvidia-ctk runtime configure --runtime=docker`)
3. Clones the VibeVoice plugin repo to `/opt/clypt-phase1/vibevoice-repo`
4. Builds the Docker image from `docker/vibevoice-vllm/Dockerfile` (based on `vllm/vllm-openai:v0.14.1`)
5. Installs and starts `clypt-vllm-vibevoice.service`
6. Polls `/health` until ready (first start downloads the ~17GB model)

Expected Phase 1 throughput on H100 80GB: RF-DETR ~109–114 fps; VibeVoice vLLM RTF ~0.07x.
