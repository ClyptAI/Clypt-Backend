# Clypt V3.1 Backend

Clypt V3.1 is a long-form video understanding and clip-candidate system.

- Implemented today: **Phases 1-4**
- Planned next: **Phases 5-6** (speaker participation grounding + final 9:16 render)
- Two-host Phase 1 topology — **no local fallback**:
  - **RTX 6000 Ada (48 GB, sole tenant)**: VibeVoice vLLM ASR and ffmpeg
    NVENC/NVDEC node-media prep. Exposed as one FastAPI service with
    `POST /tasks/vibevoice-asr` (VibeVoice ASR only — no NFA / emotion2vec+
    / YAMNet on this host) and `POST /tasks/node-media-prep`.
  - **H200**: Phase 1 orchestrator, RF-DETR + ByteTrack visual chain, the
    post-ASR audio chain in-process (**NFA → emotion2vec+ → YAMNet (CPU)**),
    SGLang Qwen3.6-35B-A3B on `:8001`, Phase 2-4 local SQLite queue and
    worker, Spanner/GCS I/O. Calls the RTX host over HTTP for ASR and
    node-media prep.

## Current Pipeline State

1. **Phase 1 — Timeline Foundation**
   - Phase 1 **visual chain** (H200): RF-DETR + ByteTrack
   - Phase 1 **audio chain** (split across hosts):
     - VibeVoice ASR on the RTX 6000 Ada via `POST /tasks/vibevoice-asr`
     - NFA forced alignment → emotion2vec+ → YAMNet (CPU), in-process on
       the H200, seeded with the VibeVoice turns
2. **Phase 2 — Semantic Node Construction** (node-media prep delegated to RTX host)
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

- Phase 1 visual chain (H200) and audio chain (split RTX + H200) execute
  concurrently. The H200 dispatches the ASR step via
  `RemoteVibeVoiceAsrClient` (legacy alias: `RemoteAudioChainClient`) and,
  as soon as VibeVoice turns return, runs NFA / emotion2vec+ / YAMNet
  in-process on the H200.
- Audio chain enqueue to Phase 2-4 fires immediately when the VibeVoice HTTP
  call returns and the in-process post-processing completes, not after
  RF-DETR finishes.
- Default Phase 2-4 route is SQLite queue → local worker loop on the H200.
- Local worker generation is hard-gated to `GENAI_GENERATION_BACKEND=local_openai`.
- Embeddings remain Vertex-backed by default.
- Node-media prep is always delegated to the RTX host via
  `RemoteNodeMediaPrepClient`; the H200 does not ship a local ffmpeg path.
- Config load fails fast if `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL`,
  `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN`,
  `CLYPT_PHASE24_NODE_MEDIA_PREP_URL`, or
  `CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN` is missing on the H200. The legacy
  `CLYPT_PHASE1_AUDIO_HOST_URL` / `CLYPT_PHASE1_AUDIO_HOST_TOKEN` names are
  still accepted as deprecated aliases for one release.
- Per-stage concurrency is explicit. `CLYPT_GEMINI_MAX_CONCURRENT` has been removed.

## Canonical Docs

- Runtime execution and env contract: [RUNTIME_GUIDE.md](docs/runtime/RUNTIME_GUIDE.md)
- Full environment catalog: [ENV_REFERENCE.md](docs/runtime/ENV_REFERENCE.md)
- Baseline runs and reference outputs: [RUN_REFERENCE.md](docs/runtime/RUN_REFERENCE.md)
- H200 deployment runbook: [P1_DEPLOY.md](docs/deployment/P1_DEPLOY.md)
- RTX 6000 Ada audio host runbook: [P1_AUDIO_HOST_DEPLOY.md](docs/deployment/P1_AUDIO_HOST_DEPLOY.md)
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
`requirements-do-phase1-visual.txt` is the H200 Phase 1 runtime dependency set. It now
includes the post-ASR audio chain deps (NFA, emotion2vec+, YAMNet).
`requirements-do-phase1-audio.txt` is the RTX 6000 Ada dependency set — VibeVoice vLLM
plus FastAPI plus ffmpeg. It explicitly does **not** include NeMo / FunASR / TensorFlow
/ librosa / resampy (those moved to the H200).

## Tests

```bash
source .venv/bin/activate
python -m pytest tests/backend/pipeline -q
```

## Phase 1 Runtime Deps

H200 (visual + audio-post + Phase 2-4 worker):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-do-phase1-visual.txt
```

RTX 6000 Ada (VibeVoice ASR + node-media prep, sole tenant):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-do-phase1-audio.txt
```

## Repository Structure

```text
backend/pipeline/                          Phase 1-4 contracts, transforms, orchestrator
backend/providers/                         Remote VibeVoice ASR / media clients, local OpenAI, Vertex, GCS
backend/phase1_runtime/                    Phase 1 orchestration, visual pipeline, in-process audio-post chain, job store (H200)
backend/runtime/phase1_audio_service/      RTX 6000 Ada FastAPI service (VibeVoice ASR + node-media prep)
backend/runtime/                           Phase 1 + Phase 2-4 runners, local SQLite worker (H200)
docker/vibevoice-vllm/                     VibeVoice vLLM image (RTX 6000 Ada)
scripts/do_phase1_visual/                  Deployment scripts + systemd units for the H200
scripts/do_phase1_audio/                   Deployment scripts + systemd units for the RTX 6000 Ada
docs/                                      Runtime, deployment, architecture, specs, outputs
```
