# Clypt V3.1 Spec: Phase1 Visual Zero-Copy Decode, Decode/Infer Overlap, and Static TensorRT Batch Retune

## Summary

This spec defines the next speed-focused Phase1 visual optimization pass for the
current H200 visual pipeline:

1. eliminate the current GPU -> CPU -> GPU frame path before RF-DETR inference,
2. overlap decode and detector inference with staged queues / CUDA streams, and
3. benchmark larger **static** TensorRT engine batches (`16`, `24`, `32`) while
   preserving the current semantic visual behavior.

The optimization target is the hot Phase1 visual service on the Phase1 H200
host. The current visual path is already correct and reasonably fast, but it
still performs host-side frame movement and Python-side staging that are likely
leaving throughput on the table even after TensorRT FP16 acceleration.

This spec is intentionally **speed-only**. It does **not** authorize changes to:

- RF-DETR model family
- detection threshold
- tracker settings
- frame stride / frame dropping
- output schema
- shot boundary semantics

The current documented visual semantics remain locked:

- `CLYPT_PHASE1_VISUAL_BACKEND=tensorrt_fp16`
- `CLYPT_PHASE1_VISUAL_BATCH_SIZE=16` as the current baseline
- `CLYPT_PHASE1_VISUAL_THRESHOLD=0.35`
- `CLYPT_PHASE1_VISUAL_SHAPE=640`
- `CLYPT_PHASE1_VISUAL_TRACKER=bytetrack`
- `CLYPT_PHASE1_VISUAL_TRACKER_BUFFER=30`
- `CLYPT_PHASE1_VISUAL_TRACKER_MATCH_THRESH=0.7`
- `CLYPT_PHASE1_VISUAL_DECODE=gpu`

## Motivation

The current code-backed visual fast path is:

```text
NVDEC -> scale_cuda -> hwdownload -> CUDA tensor normalize -> TensorRT -> ByteTrack
```

That path still downloads decoded frames to host memory as raw RGB and then
reconstructs CUDA tensors inside Python before feeding TensorRT. The current
implementation details are in:

- `backend/phase1_runtime/frame_decode.py`
- `backend/phase1_runtime/visual.py`
- `backend/phase1_runtime/tensorrt_detector.py`
- `backend/phase1_runtime/tracker_runtime.py`

The last validated long-form replay on the H200 reached about `211 fps`
(`~7.05x` realtime) on the visual branch. That is good, but the remaining
overhead is likely dominated by data movement and host-side scheduling rather
than by the RF-DETR engine alone.

This spec therefore focuses on reducing non-semantic overhead without reducing
visual fidelity or changing the downstream artifact contract.

## Goals

1. Increase end-to-end Phase1 visual throughput on the H200 without changing
   detection / tracking semantics.
2. Remove avoidable host-device copies on the detector input path.
3. Keep ByteTrack semantics unchanged while reducing detector starvation.
4. Retune static TensorRT batch size only through explicit benchmarking and only
   if output parity remains acceptable.
5. Preserve fresh-host deployability: a new H200 droplet must inherit the same
   fast path without manual tuning.

## Non-Goals

This spec does **not** include:

- detector threshold tuning
- resolution changes below `640`
- tracker threshold tuning
- frame skipping / sampling
- detector model replacement
- INT8 / quantization changes
- dynamic-batch TensorRT export
- Phase1 visual output schema changes
- ByteTrack replacement

## Locked Decisions

1. **No semantic visual knob changes in this pass.** Thresholds, resolution,
   decode mode, and ByteTrack settings remain exactly as they are today.
2. **Static TensorRT engines remain the deployment shape.** Batch retuning is
   allowed only via static engine builds for candidate batch sizes.
3. **Dynamic-batch export is out of scope.** The deployed path continues to use
   a fixed batch baked into the engine.
4. **GPU decode remains required.** No CPU decode fallback is reintroduced.
5. **The tracker stays sequential and semantically unchanged.** Any overlap work
   must preserve chronological tracker updates.
6. **The canonical artifact contract does not change.** Track rows, shot
   boundary behavior, and downstream visual payload shapes must remain stable.

## Background and External References

The design is based on the current repository implementation plus the following
upstream references:

- RF-DETR export docs expose static `batch_size` and optional `dynamic_batch`,
  but the production recommendation in this spec stays with static batch builds:
  [RF-DETR export docs](https://rfdetr.roboflow.com/latest/learn/export/)
- RF-DETR reference docs explicitly describe `batch_size` as the static batch
  baked into the ONNX graph:
  [RF-DETR reference](https://rfdetr.roboflow.com/develop/reference/nano/)
- RF-DETR benchmark tables show that latency/accuracy trade-offs are strongly
  tied to model/resolution choices, which is why this spec does not authorize
  silent resolution cuts:
  [RF-DETR repo benchmarks](https://github.com/roboflow/rf-detr)
- NVIDIA TensorRT docs emphasize that host/device transfers and enqueue overhead
  can materially reduce realized throughput even when GPU compute is fast:
  [TensorRT best practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
  and
  [TensorRT command-line docs](https://docs.nvidia.com/deeplearning/tensorrt/10.12.0/reference/command-line-programs.html)
- ByteTrack / supervision docs treat track activation and lost-track buffering
  as accuracy/stability knobs, not free speed knobs:
  [ByteTrack parameter docs](https://hirai-labs.github.io/supertracker/bytetrack/)

## Current Pipeline Constraints

### 1. Decode path

Current decode is performed by ffmpeg with:

- `-hwaccel cuda`
- `-hwaccel_output_format cuda`
- optional `scale_cuda`
- `hwdownload`
- host-side `rgb24` piping into Python

This means decoded frames are still copied back to host memory before the
TensorRT detector preprocesses them on GPU again.

### 2. Detector path

Current TensorRT preprocessing in `TensorRTDetector._preprocess_batch()`:

- stacks host numpy arrays,
- copies them to CUDA,
- converts to NCHW,
- normalizes on GPU,
- writes into TensorRT input bindings.

The TensorRT engine is currently keyed by:

- detector resolution
- detector batch size
- precision (`fp16`)

### 3. Tracking path

ByteTrack is fed chronologically frame-by-frame after detector output is
available. The tracker interface is sequential and must stay sequential in this
spec.

## Proposed Design

### A. Eliminate GPU -> CPU -> GPU frame movement on the detector input path

#### Proposed runtime shape

The new target path is:

```text
NVDEC -> scale_cuda -> CUDA staging buffer -> TensorRT -> host-side detection rows -> ByteTrack
```

The critical change is that decoded / resized frames should stay on device until
the TensorRT engine consumes them.

#### Implementation direction

1. Replace the current raw `rgb24` host-pipe decoder path with a GPU-resident
   frame staging path.
2. Introduce a CUDA-aware frame container in `frame_decode.py` (or a sibling
   module) that can carry:
   - frame index
   - source dimensions
   - device-resident tensor / buffer handle
3. Update `TensorRTDetector.detect_batch(...)` to accept GPU-resident inputs
   without reconstructing the batch from host numpy arrays.
4. Keep the current output post-processing contract unchanged: detector outputs
   may still be materialized on host after inference because ByteTrack and the
   canonical track rows remain host/Python-side in this pass.

#### Acceptable implementation shapes

Any one of the following is acceptable, provided deployability is preserved:

- PyTorch CUDA tensors emitted directly by the decoder stage
- CuPy / DLPack handoff into PyTorch tensors
- an ffmpeg + CUDA staging mechanism that fills pre-allocated device buffers

The exact mechanism is left open to implementation as long as:

- input frames are not round-tripped through host RGB arrays before TensorRT,
- the detector still sees the same resized / normalized data shape,
- the service remains self-contained on a fresh droplet.

### B. Add decode / infer overlap and staging queues

#### Current problem

The current loop is effectively:

1. decode a batch
2. infer a batch
3. track each frame
4. repeat

This causes detector idle time whenever decode or host-side staging stalls, and
decode idle time whenever the detector is busy.

#### Proposed runtime shape

Introduce a bounded producer/consumer pipeline with three conceptual stages:

1. **decode stage**
   - NVDEC + resize into device-resident frame batches
2. **infer stage**
   - TensorRT consumes ready batches from a bounded queue
3. **tracking stage**
   - tracker consumes completed detections in strict frame order

#### Requirements

- Tracking must preserve chronological order exactly.
- A bounded queue must be used to avoid unbounded VRAM growth.
- CUDA streams may be used to overlap decode staging and inference.
- The detector and tracker metrics must be extended so queueing / overlap gains
  are visible in logs and diagnostics.

#### Proposed defaults

- initial queue depth target: `2` detector batches
- one decode producer
- one detector consumer
- one tracking consumer

These values are initial implementation defaults, not locked user-facing envs.

### C. Benchmark static TensorRT batch sizes `16`, `24`, `32`

#### Why static only

This spec keeps static engines because the deployed RF-DETR TensorRT path is
already static and because dynamic-batch export remains operationally riskier
than static profile benchmarking for this project.

#### Benchmark requirements

The implementation must support explicit benchmarking of:

- batch `16` (current baseline)
- batch `24`
- batch `32`

at the same:

- model
- resolution (`640`)
- threshold (`0.35`)
- tracker settings
- decode mode (`gpu`)

#### Promotion rule

A larger batch may become the new default only if it satisfies all of:

1. no detector output regressions outside the allowed parity band,
2. no tracker instability attributable to batching,
3. no Phase1 co-tenancy regressions with the current H200 runtime shape,
4. a measurable end-to-end throughput gain on the reference videos.

If batch `24` and `32` both fail those gates, `16` remains the default.

## Config and Surface Changes

### New / updated envs

This spec allows the following new visual runtime envs to be introduced if
helpful:

- `CLYPT_PHASE1_VISUAL_OVERLAP_ENABLED`
- `CLYPT_PHASE1_VISUAL_PREFETCH_BATCHES`
- `CLYPT_PHASE1_VISUAL_TRT_BATCH_CANDIDATE`

However:

- no semantic env defaults may change in this spec,
- the committed known-good baseline may only change the detector batch default
  after benchmark acceptance.

### Engine cache naming

TensorRT engine cache keys must continue to include at least:

- model family
- resolution
- batch size
- precision

so `16`, `24`, and `32` engines can coexist safely.

## Logging and Diagnostics Requirements

The visual service must log enough to compare before/after throughput cleanly.

At minimum, add or preserve:

- total visual wall time
- frames processed
- effective fps
- detector-only wall time
- tracker-only wall time
- queue wait / stall metrics for overlap mode
- decode throughput metrics
- active detector batch size
- engine path / batch key selected

The goal is that future debugging does not require ad-hoc reconstruction from
raw progress-bar output.

## Test Plan

### Unit / component tests

1. Decoder component tests:
   - GPU-resident frame staging path returns correct frame count and metadata
   - source dimensions and frame indices are preserved
2. Detector input tests:
   - GPU-resident path and current host path produce equivalent input tensor
     shapes / normalization semantics
3. Queue / overlap tests:
   - bounded queue respects backpressure
   - completed detection batches are emitted in correct frame order
4. Engine selection tests:
   - engine path changes correctly for `16`, `24`, `32`
   - cached engines do not collide across batch sizes

### Integration tests

1. End-to-end visual extraction parity on a representative test-bank video:
   - shot count parity
   - track-row count parity within expected tolerance
   - no catastrophic ID fragmentation increase
2. Batch benchmark replay:
   - compare `16`, `24`, `32` on the same source
   - log throughput and stability

## Acceptance Criteria

This work is considered successful only if:

1. The visual branch is faster in end-to-end wall clock on the H200 than the
   current baseline.
2. Visual output remains semantically equivalent for the current use case.
3. The default visual settings recorded in runtime docs remain truthful.
4. A fresh H200 host can still build / load the correct TensorRT engine and run
   the optimized path without manual intervention.

## Rollout Plan

1. Land zero-copy input path behind an internal runtime flag if needed.
2. Land decode/infer overlap with bounded queueing.
3. Benchmark static TensorRT batch sizes `16`, `24`, `32`.
4. Promote a new default batch only if the acceptance criteria pass.
5. Update:
   - `docs/runtime/RUNTIME_GUIDE.md`
   - `docs/runtime/ENV_REFERENCE.md`
   - `docs/runtime/known-good-phase1-h200.env`
   - `docs/ARCHITECTURE.md`
   - `docs/ERROR_LOG.md`
   after implementation and validation.

## Risks

1. **Driver / ffmpeg integration complexity**
   - GPU-resident decode handoff may be trickier than the current raw RGB pipe.
2. **VRAM pressure**
   - overlap queues and larger static batches can increase transient memory use.
3. **Tracker starvation bugs**
   - incorrect ordering in the overlap pipeline would silently damage track
     quality even if detector throughput improves.
4. **Benchmark misleadingness**
   - detector-only gains do not matter if end-to-end visual wall clock does not
     improve.

## Open Questions Left Intentionally Deferred

These are explicitly not decided in this spec:

- whether batch `24` or `32` becomes the new default
- whether future work should move any tracker components off CPU
- whether future work should use `trtexec` layer profiling as a formal gate

That last item is intentionally deferred because this spec is limited to the
three implementation directions above and does not require a TensorRT
profiling-first gate to proceed.
