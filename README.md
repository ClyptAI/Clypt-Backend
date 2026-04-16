# Clypt V3.1 Backend

Clypt V3.1 is a long-form video understanding and clip-candidate system.

- Implemented today: **Phases 1-4**
- Planned next: **Phases 5-6** (speaker participation grounding + final 9:16 render)
- Current Phase 1 ASR modes: **local vLLM VibeVoice** or **remote Cloud Run L4 combined service**

## Current Pipeline State

1. **Phase 1 — Timeline Foundation**
   - RF-DETR + ByteTrack visual extraction
   - VibeVoice ASR via `CLYPT_PHASE1_ASR_BACKEND` (`vllm` or `cloud_run_l4`)
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
- Default local Phase 2-4 route is SQLite queue -> local worker loop.
- Local worker generation is hard-gated to `GENAI_GENERATION_BACKEND=local_openai`.
- Embeddings remain Vertex-backed by default.
- Optional Cloud Run L4 offload can handle both Phase 1 ASR and Phase 2 node-media prep.
- Per-stage concurrency is explicit. `CLYPT_GEMINI_MAX_CONCURRENT` has been removed.

## Canonical Docs

- Runtime execution and env contract: [RUNTIME_GUIDE.md](docs/runtime/RUNTIME_GUIDE.md)
- Full environment catalog: [ENV_REFERENCE.md](docs/runtime/ENV_REFERENCE.md)
- Baseline runs and reference outputs: [RUN_REFERENCE.md](docs/runtime/RUN_REFERENCE.md)
- Deployment runbook: [P1_DEPLOY.md](docs/deployment/P1_DEPLOY.md)
- Architecture (implemented + planned): [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Active specs index: [docs/specs/SPEC_INDEX.md](docs/specs/SPEC_INDEX.md)
- Agent/operator startup + maintenance rules: [AGENTS.md](AGENTS.md)
- Historical incident/recovery log: [ERROR_LOG.md](docs/ERROR_LOG.md)

## Setup (Local Dev / Tests)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-local.txt
```

`requirements-local.txt` is for local repo work (tests, API/runtime code paths, tooling).  
`requirements-do-phase1.txt` is the standalone Phase 1 runtime dependency set for DO GPU hosts.

## Tests

```bash
source .venv/bin/activate
python -m pytest tests/backend/pipeline -q
```

## Phase 1 Runtime Deps (DO GPU Host)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-do-phase1.txt
```

## Repository Structure

```text
backend/pipeline/          Phase 1-4 contracts, transforms, orchestrator
backend/providers/         ASR, local OpenAI, Vertex, GCS, emotion2vec+, YAMNet
backend/phase1_runtime/    Phase 1 sidecar orchestration, visual pipeline, job store
backend/runtime/           Live Phase 1-4 runner, local worker, combined L4 service entrypoints
docker/vibevoice-vllm/     Local VibeVoice vLLM image
docker/phase24-media-prep/ Combined Cloud Run L4 ASR + node-media-prep image
scripts/do_phase1/         Deployment scripts and systemd units
docs/                      Runtime, deployment, architecture, specs, outputs
```
