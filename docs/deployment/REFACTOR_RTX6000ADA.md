# Refactor: Split Phase 1 Audio Chain Onto RTX 6000 Ada

**Status:** Planned  
**Last updated:** 2026-04-17  
**Scope:** Phase 1 execution topology, node-media prep placement, cross-host coordination.

This document captures the agreed target topology and the refactor work required
to get there. It is the single source of truth for the in-progress split. Do
**not** treat sections 1-3 as describing current code — see "Current state
(pre-refactor)" for what mainline actually runs today.

## 1. Target Topology

Two single-GPU DigitalOcean droplets, possibly on different DO teams (the RTX
6000 Ada host is expected to live under a separate team account; credentials
are shared explicitly through a dedicated DO token).

### 1.1 Phase 1 Audio Host — RTX 6000 Ada (48 GB VRAM)

Runs the **entire Phase 1 audio chain** plus all GPU-bound **node-media prep**:

- VibeVoice vLLM ASR (GPU, native dtype — no bf16 patch required at 48 GB)
- NeMo Forced Aligner (GPU)
- emotion2vec+ (GPU)
- YAMNet (CPU, torch/tensorflow)
- FFmpeg with NVENC/NVDEC for Phase 2 node-clip extraction

NVENC/NVDEC placement is **not a performance preference** — the H200 NVENC
support is broken in practice (`h264_nvenc` returns `unsupported device (2)`),
so GPU encode/decode must happen on a non-H200 GPU. The RTX 6000 Ada has a
working NVENC/NVDEC pipeline.

### 1.2 Phase 1 Visual + Phase 2-4 Host — H200 (141 GB VRAM)

Runs everything H200 is good at:

- Phase 1 visual chain: RF-DETR + ByteTrack on the TensorRT FP16 fast path
- SGLang Qwen3.6-35B-A3B service on `:8001`
- Phase 2-4 local SQLite queue + local worker loop
- Vertex embedding calls, Spanner writes, GCS I/O

### 1.3 Canonical responsibilities by host

| Responsibility                          | Host             | Notes                                                            |
| --------------------------------------- | ---------------- | ---------------------------------------------------------------- |
| VibeVoice vLLM ASR (`:8000`)            | RTX 6000 Ada     | Native dtype, concurrent with RF-DETR on H200                    |
| NFA / emotion2vec+ / YAMNet             | RTX 6000 Ada     | Driven from the audio-chain orchestrator on the audio host       |
| Phase 1 visual (RF-DETR + ByteTrack)    | H200             | TensorRT FP16 fast path, unchanged                               |
| Node-media prep (NVENC clip extraction) | RTX 6000 Ada     | Worker host requests GPU clip extraction from audio host         |
| SGLang Qwen service (`:8001`)           | H200             | Unchanged                                                        |
| Phase 2-4 local worker + SQLite queue   | H200             | Unchanged generation backend gate                                |
| Spanner persistence / GCS I/O           | both as needed   | No change                                                        |

## 2. Why The Split

1. **VRAM + dtype sanity.** VibeVoice on L4 required a `bfloat16` audio
   encoder patch to fit into 24 GB. RTX 6000 Ada has 48 GB and runs the audio
   encoder natively without patching.
2. **NVENC placement forced.** H200 NVENC is not usable for ffmpeg clip
   extraction in our environment. The node-clip hot loop must run on a GPU
   with working NVENC/NVDEC.
3. **H200 headroom.** Moving VibeVoice + node-media prep off H200 frees
   activations and SM time for RF-DETR on the TensorRT FP16 fast path, the
   SGLang Qwen service, and concurrent Phase 2-4 work.
4. **Single-tenant ASR.** Phase 1 ASR no longer competes with SGLang for
   memory fraction or scheduler headroom.

## 3. Cross-Host Coordination Contract

### 3.1 Phase 1 audio chain invocation

The Phase 1 orchestrator is conceptually the job driver. Under the split:

- The orchestrator can live on either host. Simpler to colocate with the
  H200/Phase 2-4 worker so queue bookkeeping and SGLang call targets stay on
  one host.
- The Phase 1 audio chain on RTX 6000 Ada exposes a small HTTP service (or
  SSH-invoked CLI) that accepts `{source_video_gcs_uri, audio_gcs_uri,
  run_id}` and returns `Phase1SidecarOutputs` audio-only payload.
- The orchestrator still launches the RF-DETR / visual branch on H200
  locally and fans the audio branch out to RTX 6000 Ada.

### 3.2 Node-media prep contract

The `node_media_preparer` callable hook on `Phase24WorkerService` is already
a pluggable seam (currently wired to `None`, which falls back to in-process
ffmpeg + GCS upload from the worker host). The refactor re-wires that hook
to an HTTP client that requests NVENC clip extraction from the RTX 6000 Ada
host.

- Client request: `{node_id, start_ms, end_ms, source_video_gcs_uri}`
- Server response: `{node_id, multimodal_gcs_uri}` after GCS upload.
- Auth: shared symmetric token on a private DO VPC / firewall allowlist.

### 3.3 Credentials

- Separate DO team → separate DO API token. Stored in the H200 host's
  `/etc/clypt-phase1/v3_1_phase1.env` under a named var (e.g.
  `CLYPT_RTX6000ADA_DO_TOKEN`) and not checked into git.
- The RTX 6000 Ada host still needs GCP service-account access for GCS reads
  (signed URLs for source video/audio) and writes (multimodal clip uploads,
  audio-chain sidecar uploads).

## 4. Required Code Changes

### 4.1 Phase 1 audio/visual host split

| Path                                             | Needed change                                                                                                    |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `backend/phase1_runtime/extract.py`              | Replace in-process `audio_chain_future` with a remote-call stub when `CLYPT_PHASE1_AUDIO_HOST_MODE=remote`.      |
| `backend/phase1_runtime/runner.py`               | Add `audio_host_client` dependency alongside `vibevoice_provider`.                                               |
| `backend/phase1_runtime/factory.py`              | Branch on `CLYPT_PHASE1_AUDIO_HOST_MODE` to build either the in-process providers or a `RemoteAudioHostClient`. |
| `backend/providers/` (new)                       | Add `phase1_audio_host_client.py` implementing the HTTP/SSH contract to the RTX 6000 Ada audio service.          |
| `backend/runtime/` (new)                         | Add a small `run_phase1_audio_service.py` entrypoint + FastAPI app hosted on RTX 6000 Ada.                       |

### 4.2 Node-media prep remote client

| Path                                             | Needed change                                                                                                    |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `backend/runtime/phase24_worker_app.py`          | Replace the unconditional `node_media_preparer=None` with a config-gated remote client.                         |
| `backend/providers/` (new)                       | Add `node_media_prep_client.py` (HTTP client to RTX 6000 Ada ffmpeg service).                                   |
| `backend/runtime/` (new)                         | Add `run_node_media_prep_service.py` hosted on RTX 6000 Ada.                                                    |
| `backend/providers/config.py`                    | Add `NodeMediaPrepSettings { backend, service_url, auth_token_env }`; default `backend=local` to keep mainline. |

### 4.3 Deploy scripts

| Path                                              | Needed change                                                                                                                          |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/do_phase1_audio/bootstrap_rtx6000ada.sh` | New: provision RTX 6000 Ada host (CUDA + NVENC check, venv, VibeVoice repo, NFA/Emotion2Vec/YAMNet models).                            |
| `scripts/do_phase1_audio/deploy_vllm_service.sh`  | New: systemd unit for VibeVoice vLLM at `:8000` on RTX 6000 Ada (drop L4 bf16 patch; use native dtype).                                |
| `scripts/do_phase1_audio/deploy_audio_service.sh` | New: systemd unit for the Phase 1 audio orchestrator + node-media prep HTTP service at a fixed private port.                           |
| `scripts/do_phase1/deploy_vllm_service.sh`        | Strip VibeVoice provisioning block (visual-only deploy now); keep RF-DETR TensorRT bootstrap.                                          |
| `scripts/do_phase1/deploy_sglang_qwen_service.sh` | Keep as-is.                                                                                                                            |

### 4.4 Config surface additions

New env vars (all default to in-process mode for backward compatibility):

- `CLYPT_PHASE1_AUDIO_HOST_MODE` = `local` (default) | `remote`
- `CLYPT_PHASE1_AUDIO_HOST_URL` = `http://<rtx6000ada-private-ip>:9100`
- `CLYPT_PHASE1_AUDIO_HOST_AUTH_TOKEN_ENV` = `CLYPT_PHASE1_AUDIO_HOST_TOKEN`
- `CLYPT_PHASE24_NODE_MEDIA_PREP_BACKEND` = `local` (default) | `remote`
- `CLYPT_PHASE24_NODE_MEDIA_PREP_URL` = `http://<rtx6000ada-private-ip>:9101`
- `CLYPT_PHASE24_NODE_MEDIA_PREP_AUTH_TOKEN_ENV` = `CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN`

## 5. Migration Plan

1. **Provision RTX 6000 Ada droplet** in the sibling DO team. Verify NVENC
   (`ffmpeg -hide_banner -init_hw_device cuda=cu -c:v h264_nvenc -f null - </dev/null`)
   and stream driver ABI (`nvidia-smi`, `libnvidia-ml.so.1`).
2. **Land the seam code (backward compatible).** All new config envs default
   to `local`; pipeline tests still pass with single-host in-process flow.
3. **Bring up RTX 6000 Ada audio service.** Deploy VibeVoice vLLM + audio
   orchestrator. Run a single `run_phase1` with `CLYPT_PHASE1_AUDIO_HOST_MODE=remote`
   against a known-good test-bank asset and diff sidecar outputs against the
   2026-04-15 H200 reference run.
4. **Bring up RTX 6000 Ada node-media prep service.** Deploy ffmpeg NVENC
   extraction service. Enable `CLYPT_PHASE24_NODE_MEDIA_PREP_BACKEND=remote`
   on the H200 host and run one Phase 2-4 job end-to-end. Compare multimodal
   embedding latency + generated clip md5s against the in-process baseline.
5. **Flip default once both paths are validated.** Update
   `docs/runtime/known-good.env` to enable remote mode, and mark the
   single-host mode as a dev fallback.
6. **Stale-config sweep.** Remove any lingering references to "in-process
   on the Phase 2-4 worker host" from docs once the flip is real.

## 6. Non-Goals For This Refactor

- No Phase 5/6 work.
- No concurrency cap changes (continue to use the 2026-04-16 Qwen3.6 bench
  defaults on H200).
- No SGLang tuning changes — the H200 headroom gain from moving VibeVoice
  off-box is captured in the existing `mem-fraction-static` ladder in
  `docs/specs/2026-04-16_qwen36_swap_and_sglang_tuning_spec.md`.
- No cross-cloud. Both hosts stay on DigitalOcean. GCP stays as the storage
  and graph plane (GCS + Spanner + Vertex embeddings).

## 7. Current State (Pre-Refactor)

Mainline today is still **single-host**:

- `run_phase1_sidecars()` runs visual + audio chains in the **same process**
  via a `ThreadPoolExecutor(max_workers=3)`.
- `Phase24WorkerService` runs `node_media_preparer=None`, which falls back
  to in-process ffmpeg + GCS upload from the worker host.
- There is no `RemoteAudioHostClient` or node-media-prep HTTP client in
  `backend/providers/`.

Tests (`tests/backend/pipeline`, `tests/backend/providers`, and related
runtime tests) pass against this single-host topology.

## 8. References

- `docs/ARCHITECTURE.md` — high-level pipeline + invariants (updated to call
  out the target split).
- `docs/runtime/RUNTIME_GUIDE.md` — runtime behavior and current host
  topology.
- `docs/deployment/P1_DEPLOY.md` — current single-host deploy runbook.
- `docs/ERROR_LOG.md` — historical context for why L4 was abandoned and why
  RTX 6000 Ada is the replacement.
- `docs/specs/2026-04-16_qwen36_swap_and_sglang_tuning_spec.md` — SGLang
  sizing assumptions that depend on VibeVoice being off-H200.
