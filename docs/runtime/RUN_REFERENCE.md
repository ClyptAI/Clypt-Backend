# RUN REFERENCE

**Status:** Active  
**Last updated:** 2026-04-17

This document records known baseline runs and recent migration test attempts.

## 1) Canonical Reference Runs

| Context | Video | Duration | Turns | Phase 1 | Phases 2-4 | Total | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| Full Phase 1-4 validated | `mrbeastflagrant.mp4` | 392.9s | 104 | 153s | 98s | 251s | End-to-end baseline |
| Full Phase 1-4 fresh redeploy | `mrbeastflagrant.mp4` | 392.9s | 104 | n/a | 271.8s | 273.5s | `run_20260408_095543_mrbeastflagrant` |
| Queue worker baseline | `mrbeastflagrant.mp4` | 392.9s | 104 | n/a | 820.8s | n/a | CPU ffmpeg fallback path (`job_430a676441ab42af8a43a25d190dbc13`) |
| Queue worker benchmark rerun | `mrbeastflagrant.mp4` | 392.9s | 104 | n/a | 421.2s | n/a | `run_20260409_1517_phase24_bench` |
| Queue worker tuned replay | `mrbeastflagrant.mp4` | 392.9s | 104 | n/a | 143.8s | n/a | `run_20260409_175717_phase24_dispatch` |
| Longer clip stress reference | `joeroganflagrant.mp4` | 788.7s | 201 | 285s | incomplete | incomplete | historical quota/compute pressure |

## 2) Supplemental Historical Runs

| Context | Video | Duration | Turns | Wall time | Notes |
|---|---|---:|---:|---:|---|
| ASR only | `mrbeastflagrant.mp4` | 392.9s | 102 | 30.8s | vLLM RTF ~0.07x |
| ASR only | `joeroganflagrant.mp4` | 788.7s | 200 | 64.3s | vLLM RTF ~0.07x |
| Full Phase 1 (sequential audio chain) | `mrbeastflagrant.mp4` | 392.9s | 104 | 179.2s | pre-concurrency fix |
| Full Phase 1 (sequential audio chain) | `joeroganflagrant.mp4` | 788.7s | 201 | 342.4s | pre-concurrency fix |
| Full Phase 1 (parallel audio chain) | `joeroganflagrant.mp4` | 788.7s | 201 | 299.6s | concurrency validated |

## 3) Current Validation Focus

- Current code paths to validate against these historical baselines:
  - local SQLite queue + local Phase 2-4 worker
  - SGLang Qwen on `:8001`
  - Phase 1 ASR: local VibeVoice vLLM on the Phase 1 GPU host
  - node-media prep: in-process on the Phase 2-4 worker host
- Compare new measurements against Sections 1, 2, and 5, then append the new baseline here.

## 4) Recorded Phase 2-4 Timing Snapshots

### 5.1 Queue baseline (`job_430a676441ab42af8a43a25d190dbc13`)

- Phase 2: `692,564.245 ms`
- Phase 3: `75,776.984 ms`
- Phase 4: `52,392.827 ms`
- Total: `820,842.256 ms`

### 5.2 Queue benchmark rerun (`run_20260409_1517_phase24_bench`)

- Phase 2: `207,224.693 ms`
- Phase 3: `164,671.637 ms`
- Phase 4: `49,197.826 ms`
- Total: `421,153.311 ms`
- Spanner summary: `node_count=14`, `edge_count=67`, `candidate_count=7`

### 5.3 Tuned queue replay (`run_20260409_175717_phase24_dispatch`)

- Phase 2: `72,685.086 ms`
- Phase 3: `8,893.872 ms`
- Phase 4: `62,178.757 ms`
- Total: `143,816.557 ms`
- Spanner summary: `node_count=23`, `edge_count=109`, `candidate_count=11`

## 5) Candidate Snapshot References

### 6.1 Successful mrbeast run (ranked excerpt)

1. `02:08.96-03:08.97`
2. `00:00.00-00:32.20`
3. `05:36.15-06:17.91`
4. `01:22.36-02:01.36`
5. `05:02.13-05:36.15`

### 6.2 Tuned queue replay (`run_20260409_175717_phase24_dispatch`)

1. `05:15.32-05:47.95`
2. `02:08.96-03:08.97`
3. `04:17.27-05:02.13`
4. `00:00.00-00:32.20`
5. `05:47.95-06:17.91`

## 6) How To Use This Reference

- Use this file as the baseline target when validating runtime/deploy changes.
- Compare new runs against these numbers before claiming improvements.
- Log major deviations and recoveries in [ERROR_LOG.md](../ERROR_LOG.md).
