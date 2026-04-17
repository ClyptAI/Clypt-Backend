# Clypt V3.1 Backend

Clypt V3.1 is a long-form video understanding and clip-candidate system.

- Implemented today: **Phases 1-4**
- Planned next: **Phases 5-6** (participation grounding + final render/export)
- Active production topology: **two H200 hosts + Modal L4 media prep**

## Topology

- **Phase 1 host (`phase1`, H200 by default)**
  - Phase 1 runner/orchestrator
  - persistent local VibeVoice service at `POST /tasks/vibevoice-asr`
  - persistent local visual service at `POST /tasks/visual-extract`
  - co-located VibeVoice vLLM sidecar
  - in-process NFA -> emotion2vec+ -> YAMNet
- **Downstream host (`phase26`, H200)**
  - `POST /tasks/phase26-enqueue`
  - local SQLite queue + local worker
  - SGLang Qwen on `:8001`
  - current Phase 2-4 logic, with Phase 5-6 orchestration planned here later
- **Modal**
  - `POST /tasks/node-media-prep` on `L4`, `min_containers=1`
  - future `render-video` endpoint lives here later

There is **no local fallback** for remote boundaries. Phase 1 requires the local VibeVoice service, the local visual service, and the remote Phase26 dispatch service. Phase26 requires the remote node-media-prep service.

## Current Pipeline State

1. **Phase 1**
   - VibeVoice ASR served locally on the Phase 1 H200
   - NFA -> emotion2vec+ -> YAMNet run in-process on the Phase 1 H200
   - RF-DETR + ByteTrack served locally on the Phase 1 H200 with the existing fast settings preserved
2. **Phase 2**
   - semantic node construction on the Phase26 host
   - node-media-prep delegated to Modal after node creation
3. **Phase 3**
   - graph construction on the Phase26 host
4. **Phase 4**
   - candidate retrieval, review, ranking on the Phase26 host
5. **Phases 5-6**
   - planned on the Phase26 host, with final render/export expected to call Modal later

## Canonical Docs

- Runtime behavior: [docs/runtime/RUNTIME_GUIDE.md](/Users/rithvik/Clypt-Backend/docs/runtime/RUNTIME_GUIDE.md)
- Environment catalog: [docs/runtime/ENV_REFERENCE.md](/Users/rithvik/Clypt-Backend/docs/runtime/ENV_REFERENCE.md)
- Run baselines: [docs/runtime/RUN_REFERENCE.md](/Users/rithvik/Clypt-Backend/docs/runtime/RUN_REFERENCE.md)
- Phase 1 host deploy: [docs/deployment/PHASE1_HOST_DEPLOY.md](/Users/rithvik/Clypt-Backend/docs/deployment/PHASE1_HOST_DEPLOY.md)
- Phase26 host deploy: [docs/deployment/PHASE26_HOST_DEPLOY.md](/Users/rithvik/Clypt-Backend/docs/deployment/PHASE26_HOST_DEPLOY.md)
- Modal media-prep deploy: [docs/deployment/MODAL_NODE_MEDIA_PREP_DEPLOY.md](/Users/rithvik/Clypt-Backend/docs/deployment/MODAL_NODE_MEDIA_PREP_DEPLOY.md)
- Architecture: [docs/ARCHITECTURE.md](/Users/rithvik/Clypt-Backend/docs/ARCHITECTURE.md)
- Specs index: [docs/specs/SPEC_INDEX.md](/Users/rithvik/Clypt-Backend/docs/specs/SPEC_INDEX.md)
- Operator/agent rules: [AGENTS.md](/Users/rithvik/Clypt-Backend/AGENTS.md)
- Incident log: [docs/ERROR_LOG.md](/Users/rithvik/Clypt-Backend/docs/ERROR_LOG.md)

## Setup

Local/dev:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-local.txt
```

Phase 1 host runtime:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-do-phase1-h200.txt
```

Phase26 host runtime:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-do-phase26-h200.txt
```

## Tests

```bash
source .venv/bin/activate
python -m pytest tests/backend/pipeline -q
```

## Repository Structure

```text
backend/phase1_runtime/                    Phase 1 orchestration, in-process audio-post chain
backend/providers/                         Remote clients, config, storage, LLM, embeddings
backend/runtime/phase1_vibevoice_service/ Local Phase 1 VibeVoice service
backend/runtime/phase1_visual_service/    Local Phase 1 visual service
backend/runtime/phase26_dispatch_service/ Downstream enqueue API
backend/runtime/                          Phase 1/Phase26 runners and worker entrypoints
docker/vibevoice-vllm/                    VibeVoice vLLM image used on the Phase 1 host
scripts/do_phase1/                        Phase 1 host bootstrap, deploy, systemd units
scripts/do_phase26/                       Phase26 host bootstrap, deploy, systemd units
scripts/modal/                            Modal node-media-prep and future render services
docs/                                     Runtime, deployment, architecture, specs, outputs
```
