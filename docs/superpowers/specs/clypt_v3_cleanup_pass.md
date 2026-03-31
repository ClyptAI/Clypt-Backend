# Clypt v3 Cleanup Pass (Phase 1)

This document records the v3 cleanup pass requested after final review findings. **Code on [github.com/rithm84/Clypt-V3](https://github.com/rithm84/Clypt-V3)** is authoritative if this note drifts.

## Scope

- Phase 1 worker/runtime (`backend/do_phase1_worker.py`)
- Phase 1 contract + persistence (`backend/pipeline/phase1_contract.py`, `backend/do_phase1_service/storage.py`)
- Phase 1 scorecard/benchmark reporting (`backend/pipeline/phase1/metrics_scorecard.py`, `backend/pipeline/phase1/benchmark_corpus.py`)
- Phase 1 extraction/service interfaces (`backend/do_phase1_service/extract.py`)
- Phase 1 clustering signal metadata (`backend/pipeline/phase1/clustering.py`)
- Phase 1/DO env + docs + tests

## Fixes Applied From Review Findings

### 1) Runtime log `job_id` leak on early exceptions

- `finalize_extraction` now restores runtime log context in an outer `try/finally`, even if exceptions occur before the inner finalize block.
- Added robust cleanup using guarded locals access so cleanup and context-restore are safe under partial initialization.

## 2) Structured logging migration gap in LR-ASD/binding

- Replaced remaining non-debug LR-ASD video open warning print path with structured stage logging:
  - `reason_code=lrasd_video_open_failed`
  - `event=stage_warning`
  - includes structured `error_type` and `error`.

## 3) Aggregation policy side effect

- `aggregate_scorecard_summary` now:
  - computes ratio aggregates from all successful scorecards,
  - computes wallclock/decode timing aggregates from wallclock-eligible scorecards only.
- `aggregation_inputs.*` metadata remains to surface excluded-row counts explicitly.

## v3 Boundary Migration Updates

### Contract and schema

- `Phase1Manifest.contract_version` migrated to `v3`.
- `PHASE1_SCHEMA_VERSION` migrated to `3.0.0`.
- NDJSON default schema header in phase pipeline migrated to `3.0.0`.

### Persistence

- Service persistence payload now emits `"contract_version": "v3"`.
- Default Phase 1 bucket updated to `clypt-storage-v3` in service storage defaults and DO env template.

### Legacy alias removal

- Removed legacy response aliases from worker output:
  - removed legacy Phase 1A visual/audio alias keys from worker responses
- Removed legacy extraction fallback reads for those aliases in DO extraction service.
- Removed legacy Phase 1A stale-output cleanup names in `phase_1_do_pipeline.py`.

### Signal/version metadata normalization

- `worker_bbox` signal version migrated to `worker_bbox_v3`.
- `cluster_attach_bbox` signal version migrated to `cluster_attach_bbox_v3`.
- scorecard/report/comparison versions:
  - `SCORECARD_VERSION`: `2` -> `3`
  - `BENCHMARK_REPORT_VERSION`: `2` -> `3`
  - `BENCHMARK_COMPARISON_VERSION`: `2` -> `3`
  - supported benchmark report versions tightened to `(3,)`.

### v3 contract shape completion

Added v3 contract fields expected by strict contract tests/spec work:

- `Phase1Word`:
  - `speaker_track_ids`
  - `offscreen_audio_speaker_ids`
  - `speaker_assignment_source`
  - `requires_hard_disambiguation`
- `Phase1TranscriptArtifact`:
  - `audio_visual_mappings`
  - `span_assignments`
- `Phase1VisualArtifact`:
  - `visual_identities` (typed model list)

## Verification

Targeted verification run after this pass:

- Phase 1 contract + storage + client + scorecard/benchmark + key worker/extract paths:
  - **54 passed**
  - **0 failed** (targeted selection)

This pass intentionally focuses on v3 compliance and cleanup in the Phase 1 path.
