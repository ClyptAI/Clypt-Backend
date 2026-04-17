# RUN REFERENCE

**Status:** Active  
**Last updated:** 2026-04-17

This document tracks reference runs and migration validation targets.

## 1) Historical Baselines

The entries below are still useful for relative performance comparison, but many were gathered under the earlier H200 + RTX topology.

| Context | Video | Duration | Turns | Phase 1 | Phases 2-4 | Total | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Full Phase 1-4 validated | `mrbeastflagrant.mp4` | 392.9s | 104 | 153s | 98s | 251s | Historical full-pipeline baseline |
| Full Phase 1-4 fresh redeploy | `mrbeastflagrant.mp4` | 392.9s | 104 | n/a | 271.8s | 273.5s | Historical redeploy baseline |
| DO-speedup-and-OSS-swap baseline | `openclawyc.mp4` | 1356s | — | ~200s | ~123-154s | ~322-354s | Historical pre-split benchmark |

## 2) New Validation Focus

The current topology to validate is:

- Phase1 H200:
  - local VibeVoice service
  - local visual service
  - in-process NFA + emotion2vec+ + YAMNet
- Phase26 H200:
  - remote enqueue API
  - local SQLite queue + local worker
  - SGLang Qwen on `:8001`
- Modal:
  - node-media-prep after node creation

For every new benchmark, record:

- Phase 1 total wall time
- local VibeVoice service wall time
- audio-post wall time
- local visual service wall time
- dispatch latency to Phase26
- Phase 2-4 total wall time
- node-media-prep wall time on Modal
- end-to-end total wall time

## 3) Migration Acceptance Targets

For this split, validate:

1. Phase 1 can run with both local services hot and healthy.
2. Phase 1 handoff reaches `POST /tasks/phase26-enqueue` successfully.
3. Phase26 worker calls Modal for media prep, not the Phase1 host.
4. RF-DETR output parity stays within expected bounds with the preserved fast settings.
5. The H100 overlay changes only memory-sensitive VibeVoice knobs.

## 4) Notes

- Until new measurements are recorded, use the historical table above only as directional context.
- Add new entries here after the first successful two-H200 + Modal benchmark pass.
- Log any regressions or deployment recoveries in [docs/ERROR_LOG.md](/Users/rithvik/Clypt-Backend/docs/ERROR_LOG.md).
