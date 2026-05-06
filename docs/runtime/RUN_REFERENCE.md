# RUN REFERENCE

**Status:** Active  
**Last updated:** 2026-05-04

This document tracks reference runs and validation targets for the Scribe/Modal + Phase26 MI300X topology.

## 1) Historical Baselines

The entries below are useful for relative comparison only. They were gathered under older GPU topologies and should be compared against Spanner telemetry, not treated as deployment instructions.

| Context | Video | Duration | Turns | Phase 1 | Phases 2-4 | Total | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Full Phase 1-4 validated | `mrbeastflagrant.mp4` | 392.9s | 104 | 153s | 98s | 251s | Historical full-pipeline baseline |
| Full Phase 1-4 fresh redeploy | `mrbeastflagrant.mp4` | 392.9s | 104 | n/a | 271.8s | 273.5s | Historical redeploy baseline |
| DO-speedup-and-OSS-swap baseline | `openclawyc.mp4` | 1356s | - | ~200s | ~123-154s | ~322-354s | Historical pre-split benchmark |

## 2) Current Validation Focus

The current topology to validate is:

- Phase1 orchestrator:
  - canonical media preparation
  - signed HTTPS GCS URL generation
  - synchronous ElevenLabs Scribe v2
  - Modal RF-DETR-Seg visual future submit
  - immediate Phase26 dispatch after Scribe adaptation
- Phase26 MI300X:
  - remote enqueue API
  - local SQLite queue + worker
  - SGLang ROCm Qwen on `:8001`
- Modal:
  - dedicated L40S RF-DETR-Seg visual worker
  - shared L40S media-prep/render worker

For every new benchmark, record:

- Phase1 total wall time
- Scribe request wall time
- Scribe adapter wall time
- Modal visual queue wait and extraction wall time
- dispatch latency to Phase26
- Phase2-4 total wall time
- node-media-prep wall time on Modal
- multimodal embedding wall time
- end-to-end total wall time

## 3) Acceptance Targets

1. Phase1 dispatches to Phase26 as soon as Scribe audio artifacts are adapted.
2. Phase26 can advance Phase2-4 while the visual future is still pending.
3. Phase26 joins/fails-hard on the visual future before Phase5/frontend grounding or Phase6 visual use.
4. Modal visual uses TensorRT FP16 RF-DETR-Seg Nano with CUDA/NVDEC decode, box-only ByteTrack, retained masks, and no software/CPU/detection-only fallback.
5. Phase26 worker calls Modal for node-media-prep and render/export, not local ffmpeg fallbacks.
6. SGLang staged profiles pass on the MI300X host before the worker starts.
7. Phase6 render outputs must pass human visual review for tracking and crop smoothness, not only `1080x1920` ffprobe validation. The current Phase5-less auto-follow fallback failed that review on 2026-05-04 and remains experimental.

## 4) Notes

- Add new entries here after the first successful Scribe/Modal + Phase26 MI300X benchmark pass.
- Log major deployment or runtime recoveries in [ERROR_LOG.md](/Users/rithvik/Clypt-Backend/docs/ERROR_LOG.md).
- When testing persistent Modal workers ad hoc, stop `clypt-visual-l40s` and `clypt-media-l40s` after the session if they should not stay warm.
