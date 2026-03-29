# LR-ASD Prep Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize LR-ASD preparation work so frame fetch, crop extraction, and subchunk tensor prep no longer bottleneck a single inference lane.

**Architecture:** Keep LR-ASD model weights, scoring, bucket logic, and alignment semantics unchanged. Refactor `_run_lrasd_binding` into a staged pipeline: enumerate chunk jobs, prepare subchunks in a bounded prep-worker pool, and feed ready pending items into the existing bucketed inference scheduler.

**Tech Stack:** Python, concurrent.futures, NumPy, Torch, Decord, pytest

## Status Update (2026-03-28)

- Status: Implemented and deployed.
- Shipped outcomes:
  - LR-ASD prep-worker pool
  - CUDA-enabled Decord / GPU decode path
  - GPU-oriented decode/crop/preprocess pipeline
  - drain/progress logging improvements
  - follow/debug renderer rescue rules for lone visible boxes, dominant-vs-fragment cases, and fragment-to-full-body cross-track rescue
- Current branch lineage: this plan is the active state of `codex/lrasd-prep-pool`, which is being promoted to `main`.

---

### Task 1: Add prep-pool helpers and configuration

**Files:**
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] Add small helper methods for LR-ASD prep-worker configuration and job/result envelopes.
- [ ] Add tests for default and bounded env parsing for prep workers / prep queue.
- [ ] Keep defaults conservative and single-GPU safe.

### Task 2: Extract serial prep logic into reusable helpers

**Files:**
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] Extract inline chunk/subchunk preparation into helper functions that preserve current crop and padding behavior.
- [ ] Add tests for helper output equivalence with current `_lrasd_build_pending_subchunk` semantics.
- [ ] Ensure helpers are pure enough to run inside a prep executor.

### Task 3: Introduce bounded prep-worker pipeline

**Files:**
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] Add a bounded prep executor and queue-drain loop inside `_run_lrasd_binding`.
- [ ] Enumerate chunk jobs in order, submit prep tasks up to a bounded queue size, and commit prepared items back into `pending_by_t` in arrival order.
- [ ] Preserve existing inference batching and score commit behavior.
- [ ] Add tests that multiple prep tasks can complete and still flush into the same bucketed inference path.

### Task 4: Add progress/debug logging for prep stage

**Files:**
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] Extend the LR-ASD pipeline log line to include prep worker and queue settings.
- [ ] Add progress logs for prepared vs inferred subchunks and bounded queue depth.
- [ ] Keep log format compatible with current extract parser expectations.

### Task 5: Verify no-regression and deploy

**Files:**
- Modify: `scripts/do_phase1/deploy_phase1_service.sh` only if required
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] Run focused LR-ASD tests locally.
- [ ] Run a syntax/import check on the touched runtime file.
- [ ] Deploy the branch to `clypt-MFH-1`.
- [ ] Verify envs for GPU/runtime path before restart.
- [ ] Verify service health after deployment.
