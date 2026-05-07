# ARCHITECTURE

**Status:** Active Scribe/Modal + Phase26 MI300X topology
**Last updated:** 2026-05-07

This document describes the current `AMD-refactor` topology: Phase1 orchestration colocated on the DigitalOcean MI300X host's vCPUs, Phase26/Qwen on the same MI300X GPU host, and two persistent Modal L40S GPU workers.

## 1) End-to-End Flow

```mermaid
flowchart TD
  source["Source URL or local path"]
  gcs["GCS canonical audio/video"]
  scribe["ElevenLabs Scribe v2"]
  vertex["Vertex embeddings"]
  spanner["Spanner persistence"]
  terminal["Ranked artifacts + terminal state"]

  subgraph p1host["MI300X host vCPUs: Phase1"]
    ingest["Resolve test-bank/local input"]
    upload["Upload canonical media"]
    sign["Create signed audio URL"]
    submitVisual["Submit visual future"]
    adapt["Adapt Scribe response"]
    enqueue["Dispatch Phase26 job"]
  end

  subgraph visual["Modal visual L40S"]
    visualApi["POST /tasks/visual-extract"]
    visualJob["visual_extract_job"]
    decode["NVDEC/CUDA decode + resize"]
    detect["TensorRT RF-DETR-Seg Nano boxes + masks"]
    track["ByteTrack on boxes"]
    maskAssoc["same-frame mask association"]
    pose["YOLO pose validation"]
    maskArtifact["visual_masks_lowres_v1.npz"]
    visualArtifact["phase1_visual.json.gz"]
    visualPointer["small poll result with GCS pointer"]
  end

  subgraph p26["MI300X host: Phase26"]
    dispatch["POST /tasks/phase26-enqueue"]
    queue["Local SQLite queue"]
    worker["Phase26 local worker"]
    qwen["SGLang Qwen :8001"]
    phase2["Phase2 node construction"]
    visualJoin["Poll/join visual future"]
    hydrate["download + inflate phase1_visual"]
    phase3["Phase3 graph construction"]
    phase4["Phase4 retrieval + ranking"]
    phase5["Phase5/frontend grounding gate"]
    phase6["Phase6 render orchestration"]
  end

  subgraph media["Modal media L40S"]
    mediaApi["CPU submit/poll API"]
    mediaLease["exclusive media_gpu_job lease"]
    nodePrep["node-media-prep batches"]
    render["render/export"]
  end

  source --> ingest --> upload --> gcs
  sign --> scribe --> adapt
  gcs --> sign
  gcs --> submitVisual --> visualApi --> visualJob --> decode --> detect --> track --> maskAssoc --> pose
  pose --> maskArtifact
  pose --> visualArtifact --> visualPointer
  adapt --> enqueue --> dispatch --> queue --> worker
  worker --> qwen --> phase2
  phase2 --> mediaApi --> mediaLease --> nodePrep --> worker
  worker --> vertex --> phase3 --> phase4
  visualPointer --> visualJoin --> hydrate
  maskArtifact --> hydrate
  hydrate --> phase5
  phase4 --> visualJoin --> phase5 --> phase6
  phase6 --> mediaApi
  mediaLease --> render --> spanner
  phase4 --> spanner --> terminal
```

Phase26 starts as soon as Scribe audio artifacts are adapted. It does not wait for RF-DETR-Seg before Phase2-4, but it must join and fail hard on the visual future before Phase5/frontend grounding or any Phase6 visual use.

## 2) Surfaces

| Surface | Runs |
| --- | --- |
| **Phase1 orchestrator** | Runs on the MI300X host's vCPUs. Test-bank ingress, canonical media upload, signed HTTPS GCS URL creation, synchronous ElevenLabs Scribe v2 call, Modal visual future submission, and Phase26 dispatch. No local GPU service. |
| **Modal visual L40S** | `POST /tasks/visual-extract` submit/poll API plus one warm `visual_extract_job` using CUDA/NVDEC decode, TensorRT FP16 RF-DETR-Seg Nano, ByteTrack on boxes, mask association, and YOLO pose validation. |
| **Phase26 MI300X** | `POST /tasks/phase26-enqueue`, local SQLite queue, Phase2-4 worker/runtime, SGLang ROCm Qwen on `127.0.0.1:8001`, future Phase5-6 orchestration boundary. |
| **Modal media L40S** | `POST /tasks/node-media-prep` and `POST /tasks/render-video` submit/poll APIs, both backed by one warm `media_gpu_job` worker. |

## 3) Phase1

Phase1 is orchestration-only and colocated with Phase26:

- resolves test-bank source URLs or accepts local source paths
- prepares canonical audio/video artifacts and uploads them to GCS
- signs the audio GCS object as an HTTPS URL for Scribe v2
- submits the source video GCS URI to Modal visual extraction
- adapts Scribe words, speakers, and audio-event tags into the canonical Phase1 handoff payload
- enqueues Phase26 immediately after audio adaptation

There is no VibeVoice, local NFA, emotion2vec+, YAMNet, local RF-DETR, local vLLM, local SGLang, VAAPI, or ROCm requirement on Phase1.

## 4) Visual

The active visual fast path is Modal L40S only:

- Phase1 orchestrator route: `CLYPT_PHASE1_VISUAL_BACKEND=modal_rfdetr`
- `CLYPT_PHASE1_VISUAL_MODEL=seg_nano`
- `CLYPT_PHASE1_VISUAL_BATCH_SIZE=16`
- `CLYPT_PHASE1_VISUAL_THRESHOLD=0.85`
- `CLYPT_PHASE1_VISUAL_SHAPE=648`
- `CLYPT_PHASE1_VISUAL_GPU_DECODE_BACKEND=nvdec`
- RF-DETR-Seg masks are retained once in a compressed low-resolution `.npz` sidecar artifact. Raw detections, tracked rows, person detections, and tracklet geometry carry `mask_ref` pointers using `lowres_mask_ref_v1`; the active path does not emit full-frame inline `mask_rle` blobs.
- The active TensorRT RF-DETR-Seg path stays close to upstream RF-DETR postprocess semantics: decode logits to per-query scores/labels, threshold, filter to `person`, and retain the surviving queries. It does not add a separate hard box-IoU NMS stage on top of that active path.
- Modal visual does not return the full `phase1_visual` JSON inline. The worker uploads `phase1_visual.json.gz` to GCS, returns a small pointer-bearing poll result, and the colocated host hydrates the artifact before Phase26 uses it. This avoids oversized HTTP result payloads and keeps the Modal poll surface bounded.
- Segmentation is present to enable future person-aware captions, motion graphics/overlays inside the short/reel frame, and better crop/negative-space decisions. Phase6 crop math and caption placement do not consume masks yet.
- sampled YOLO11s-pose TensorRT validation marks `auto_follow_eligible` tracklets and stores source-space pose anchors for Phase5-less render auto-follow
- Phase5-less render auto-follow uses the two-step subject model: manual/frontend `primary_tracklet_id` wins when present; otherwise the compiler locks one pose-qualified subject tracklet per shot
- the active auto-follow crop mode is `tracklet_follow_9x16_pose_x_dynamic_inside_person`: the compiler computes per-keyframe inside-person 9:16 crops, pose controls horizontal head/face anchoring only, and vertical placement is bbox-top anchored
- the active Modal FFmpeg renderer does not change crop `w/h` inside one ffmpeg pass; it renders per-run/per-tracklet fixed-size cropped video pieces, stitches them back into one clip, and applies subtitles in a final pass. Within each piece it still drives dynamic `x/y` through `sendcmd`
- shot or primary-tracklet changes are hard crop cuts with run-local interpolation only, so the new shot starts already framed on the selected subject instead of animating from the previous shot crop
- Modal worker detector route: `CLYPT_MODAL_VISUAL_BACKEND=tensorrt`; the worker sets internal `CLYPT_PHASE1_VISUAL_BACKEND=tensorrt_fp16`.
- ByteTrack buffer `30`
- ByteTrack match threshold `0.7`

The worker fails hard if CUDA ffmpeg hwaccel, `scale_cuda`, TensorRT, `trtexec`, CUDA PyTorch, RF-DETR-Seg dependencies, or a usable mask output binding are unavailable. There is no software decode, CPU detector, detection-only RF-DETR, VAAPI, or PyTorch ROCm fallback path.

### Current Render Quality Caveat

The Phase5-less auto-follow render path is implemented but **not accepted as production-quality**. The latest Modal render replay proved that the technical crop/render contract runs end-to-end and emits valid `1080x1920` MP4s, but the clips still looked terrible in review: crop movement was not smooth enough and subject tracking/selection was visibly wrong in places. Treat `tracklet_follow_9x16_pose_x_dynamic_inside_person` as an experimental fallback for Phase5-less demos only until the next human render review accepts it. Manual Phase5 grounding remains the expected production-quality route.

## 5) Phase26

Phase26 owns the downstream queue and graph pipeline:

- local SQLite queue with `CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite`
- worker entrypoint: `python -m backend.runtime.run_phase26_worker`
- local generation through SGLang ROCm Qwen at `http://127.0.0.1:8001/v1`
- `GENAI_GENERATION_BACKEND=local_openai`
- Vertex-backed embeddings
- Modal-backed media prep/render
- Spanner persistence

The worker may process Phase2-4 while the visual future is pending. Pending visual payloads require a configured Modal visual client and malformed or failed visual results fail hard. If a run resumes after a visual-join failure, it must not short-circuit to success merely because persisted Phase4 metrics exist; it must rerun Phase4 and then re-attempt the visual join.

## 6) Removed Paths

The atomic refactor deletes, rather than keeps, the old active runtime families:

- H200 Phase1/Phase26 env baselines and deploy scripts
- Phase1 MI300X/VibeVoice env baselines and deploy scripts
- local VibeVoice service and vLLM Docker images
- local NFA, emotion2vec+, YAMNet, and speaker-verification providers
- local Phase1 visual FastAPI service
- PyTorch ROCm/VAAPI RF-DETR path
- VibeVoice transcript output references

Historical incidents remain in [ERROR_LOG.md](/Users/rithvik/Clypt-Backend/docs/ERROR_LOG.md) because they are useful debugging context, but they are not current operator entrypoints.
