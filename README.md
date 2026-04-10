# Clypt V3.1 Backend

Clypt V3.1 is a long-form video understanding and clip-candidate system.

- Implemented today: **Phases 1-4**
- Planned next: **Phases 5-6** (speaker participation grounding + final 9:16 render)
- Current ASR path: **vLLM VibeVoice only**

## Current Pipeline State

1. **Phase 1 — Timeline Foundation**
   - RF-DETR + ByteTrack visual extraction
   - VibeVoice ASR via local vLLM sidecar
   - NeMo forced alignment
   - emotion2vec+ and YAMNet sidecars
2. **Phase 2 — Semantic Node Construction**
3. **Phase 3 — Graph Construction**
4. **Phase 4 — Candidate Retrieval, Review, Ranking**
   - includes comments/trends signal augmentation in hard-join, fail-fast mode

## Comments + Trends Augment Status

The comments/trends augmentation spec is implemented in the Phase 2-4 flow:

- signal futures run in parallel with core semantic phases
- Phase 4 waits on a hard join before prompt seeding
- enabled signal failures are terminal for the run
- attribution/provenance is persisted in Spanner and propagated to candidate scoring

See [2026-04-09_comments_trends_augment_spec.md](docs/specs/2026-04-09_comments_trends_augment_spec.md) for the canonical spec.

## Runtime Highlights

- Phase 1 visual and ASR execute concurrently.
- Audio chain starts immediately after ASR (`asr_future.result()` path), not after RF-DETR.
- Default Phase 2-4 route is Cloud Tasks -> Cloud Run worker on `us-east4` L4 GPU-accelerated profile.
- Generation backend is Developer API; embeddings remain Vertex.

## Canonical Docs

- Runtime execution and env contract: [RUNTIME_GUIDE.md](docs/runtime/RUNTIME_GUIDE.md)
- Baseline runs and reference outputs: [RUN_REFERENCE.md](docs/runtime/RUN_REFERENCE.md)
- Deployment runbook: [PHASE_1_DEPLOYMENT.md](docs/deployment/PHASE_1_DEPLOYMENT.md)
- Architecture (implemented + planned): [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Active specs index: [docs/specs/SPEC_INDEX.md](docs/specs/SPEC_INDEX.md)
- Agent/operator startup + maintenance rules: [AGENTS.md](AGENTS.md)
- Historical incident/recovery log: [ERROR_LOG.md](docs/ERROR_LOG.md)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-do-phase1.txt
```

## Tests

```bash
source .venv/bin/activate
python -m pytest tests/backend/pipeline -q
```

## Repository Structure

```text
backend/pipeline/          Phase 1-4 contracts, transforms, orchestrator
backend/providers/         VibeVoice vLLM, Vertex AI, GCS, emotion2vec+, YAMNet
backend/phase1_runtime/    Phase 1 sidecar orchestration, visual pipeline, job store
backend/runtime/           Live Phase 1-4 runner and entrypoints
docker/vibevoice-vllm/     vLLM sidecar image
scripts/do_phase1/         Deployment scripts and systemd units
docs/                      Runtime, deployment, architecture, specs, outputs
```
