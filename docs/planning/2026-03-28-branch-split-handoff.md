# 2026-03-28 Branch Split Handoff

This note documents the branch split between mixed Phase 1 experimentation and
Phase 2-5 inference work so future agents do not accidentally merge the wrong
changes together.

## Branches

- `codex/2026-03-28-mixed-phase1-inference-snapshot`
  - Purpose: full snapshot of the user's mixed work from March 28, 2026.
  - Status: pushed to `origin`.
  - Contains:
    - DigitalOcean / Phase 1 service edits
    - `phase_1_do_pipeline.py` experiments
    - `phase_1_transcript_fast.py`
    - `run_pipeline.py` library/test-harness changes
    - downstream inference edits

- `codex/2026-03-28-inference-clean`
  - Purpose: clean branch for the user's Phase 2-5 inference lane.
  - Status: created from pre-split base commit `b9c3575b014c0f1c615e6b5b647022927a841392`.
  - Contains only inference-safe carryovers plus local repo hygiene.

## Ownership Boundary

The user's teammate is working primarily on Phase 1. To reduce conflict risk,
the following files were intentionally *not* carried into the clean inference
branch:

- `backend/do_phase1_service/.env.example`
- `backend/do_phase1_service/app.py`
- `backend/do_phase1_service/extract.py`
- `backend/pipeline/do_phase1_client.py`
- `backend/pipeline/phase_1_do_pipeline.py`
- `backend/pipeline/phase_1_transcript_fast.py`
- `backend/pipeline/run_pipeline.py`
- `docs/deployment/do-phase1-digitalocean.md`
- `requirements-do-phase1.txt`
- `scripts/do_phase1/deploy_phase1_service.sh`
- `README.md`

Those files still exist on:

- `codex/2026-03-28-mixed-phase1-inference-snapshot`

If any of that work is later needed, cherry-pick or selectively restore only
after coordinating with the Phase 1 owner.

## Files Carried Into The Clean Inference Branch

- `.gitignore`
- `backend/pipeline/phase_2a_make_nodes.py`
- `backend/pipeline/phase_3_multimodal_embeddings.py`
- `backend/pipeline/phase_4_store_graph.py`
- `backend/pipeline/phase_5_auto_curate.py`

## Why `run_pipeline.py` Was Excluded

`run_pipeline.py` contained a mix of:

- inference/test-harness convenience features for video-library playback
- changes that interact with Phase 1 execution flow and transcript-fast entry
  points

Because it crossed the ownership boundary, it was left on the snapshot branch
instead of being carried automatically into the inference branch.

## Recommended Workflow Going Forward

- Use `codex/2026-03-28-inference-clean` for Phase 2-5 work and Crowd Clip.
- Treat `codex/2026-03-28-mixed-phase1-inference-snapshot` as a recovery or
  reference branch, not the default implementation branch.
- If a future agent needs a Phase 1 experiment from the snapshot branch, pull
  only the smallest necessary diff and document it.
