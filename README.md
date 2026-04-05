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

This repository intentionally excludes the previous backend runtime, LR-ASD paths, DigitalOcean worker/service code, old semantic pipeline scripts, old render code, and old frontend code.

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
- Phase 1 sidecar orchestration for pyannote + local worker tasks
- Phase 1 operational runtime shell:
  - media preparation
  - ffmpeg shot detection + Ultralytics/ByteTrack visual extraction
  - shot-local track splitting
  - person detection ledger generation
  - SQLite-backed job store
  - FastAPI Phase 1 service shell
  - local Phase 1 job runner
  - remote job submission/log tailing shell
- live provider config for pyannote cloud, Vertex AI, and GCS
- provider adapters for pyannote cloud, Vertex Gemini, Vertex embeddings, emotion2vec+, and YAMNet
- Phase 2 turn-neighborhood batching, merge/classify adaptation, and boundary reconciliation
- Phase 3 structural edges, local semantic edges, long-range edge adjudication, and deterministic reconciliation
- Phase 4 prompt generation, seed retrieval, local subgraph construction, subgraph review adaptation, candidate dedupe, and pooled final review
- a Phase 1-4 orchestrator with provider-injected inputs
- a live Phase 1-4 runner for provider-backed Phases 2-4

Not implemented yet on this branch:

- Phase 5 participation grounding
- Phase 6 camera intent and render planning
- face/identity-oriented visual ledgers beyond person tracking
- first live DigitalOcean deployment and bring-up of the new worker/service
- comment/trend/onboarding integrations
- frontend integration

## Environment

Minimal Python dependencies are listed in [requirements.txt](/Users/rithvik/Clypt-V3/requirements.txt).

Core environment variables:

- `CLYPT_V31_OUTPUT_ROOT`
- `PYANNOTE_API_KEY`
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
