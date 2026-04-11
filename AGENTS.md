# AGENTS

Operational startup and maintenance guide for coding agents and maintainers.

## Project Snapshot

- Product: Clypt V3.1 backend
- Implemented: Phases 1-4
- Planned: Phases 5-6
- Current ASR backend: vLLM VibeVoice only

## Read Order (Required)

1. [README.md](README.md)
2. [docs/runtime/RUNTIME_GUIDE.md](docs/runtime/RUNTIME_GUIDE.md)
3. [docs/deployment/PHASE_1_DEPLOYMENT.md](docs/deployment/PHASE_1_DEPLOYMENT.md)
4. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
5. [docs/specs/SPEC_INDEX.md](docs/specs/SPEC_INDEX.md)

## Documentation-First Rule (Required)

When in doubt, read documentation before making changes or running operational commands.

- Start with the required read order above.
- Treat docs as the source of truth for runtime/deploy behavior unless code has clearly diverged.
- If behavior is unclear, re-check `RUNTIME_GUIDE.md` and `PHASE_1_DEPLOYMENT.md` before proceeding.
- If you discover a docs/code mismatch, call it out and fix docs or code intentionally (do not guess).

## Canonical Run Commands

### Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-do-phase1.txt
```

### Pipeline tests (offline)

```bash
python -m pytest tests/backend/pipeline -q
```

### Phase 1 + Phase 2-4 queue mode

```bash
python -m backend.runtime.run_phase1 \
  --job-id "run_$(date +%Y%m%d_%H%M%S)" \
  --source-path /opt/clypt-phase1/videos/<video>.mp4 \
  --run-phase14
```

## Runtime Truths To Preserve

- `VIBEVOICE_VLLM_MODEL` must be `vibevoice`.
- `GOOGLE_CLOUD_PROJECT` and `GCS_BUCKET` are required even for VibeVoice-only runs.
- Phase 1 audio chain must launch immediately after ASR completion.
- Phase 2-4 production worker profile defaults to `us-east4` L4 GPU-accelerated.
- Comments/trends augmentation is hard-join + fail-fast before Phase 4.
- Phase 2-4 worker runtime requires `ffmpeg`; deploy from `docker/phase24-worker/Dockerfile` (use `scripts/deploy_phase24_worker.sh`, not generic source buildpacks).

## Critical Maintenance Rule

Whenever a major runtime/deploy/pipeline error is diagnosed and resolved, update:

- [docs/ERROR_LOG.md](docs/ERROR_LOG.md)

Each entry must include:

- date/time
- affected subsystem
- error signature
- root cause
- fix
- verification evidence

