# RTX 6000 Ada VibeVoice ASR + Node-Media Prep Host Deployment

**Status:** Active
**Last updated:** 2026-04-17

This runbook covers the **RTX 6000 Ada** host. Pair it with the H200
runbook at [`docs/deployment/P1_DEPLOY.md`](P1_DEPLOY.md).

The service name on this host is still `clypt-audio-host` (FastAPI +
systemd) for backward compatibility, but its responsibilities have
narrowed to VibeVoice ASR and GPU ffmpeg for node-media prep. The
Phase 1 audio post-processing chain (NFA / emotion2vec+ / YAMNet) no
longer runs here — it was moved back to the H200.

## 0) Why This Host Exists

- **NVENC placement is forced.** H200 NVENC is not usable for ffmpeg clip
  extraction (`h264_nvenc` returns `unsupported device (2)`). Node-media
  prep has to run on a non-H200 GPU with working NVENC/NVDEC.
- **VibeVoice sole tenancy.** 48 GB VRAM lets VibeVoice run at native
  dtype and without co-tenancy throttling. There are no more
  `--max-num-seqs 1 / --max-model-len 32768 /
  --gpu-memory-utilization 0.60 / --enforce-eager /
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` flags — those were
  co-tenancy hacks required when NFA ran on the same card.
- **H200 headroom.** Keeping VibeVoice + ffmpeg off the H200 frees SM
  time for RF-DETR and SGLang; NFA / emotion2vec+ / YAMNet have plenty
  of room back on the H200's 141 GiB next to RF-DETR.

## 1) Endpoints Served

One FastAPI process, one GPU kept hot:

- `POST /tasks/vibevoice-asr` — takes an audio GCS URI and returns
  `{turns, stage_events}` from VibeVoice vLLM ASR. ASR calls are
  serialized on the GPU via an `asyncio.Lock`. This endpoint does **not**
  run NFA, emotion2vec+, or YAMNet; those stages execute in-process on
  the H200 after the ASR response returns.
- `POST /tasks/node-media-prep` — takes a source video GCS URI and a list
  of `{node_id, start_ms, end_ms}`, runs NVENC clip extraction, uploads
  each clip to GCS, and returns the descriptors. Concurrency is bounded by
  `CLYPT_PHASE24_NODE_MEDIA_PREP_MAX_CONCURRENCY` on the caller side.
- `GET /health` — unauthenticated readiness probe.

All POST routes require
`Authorization: Bearer ${CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN}`.

> **Compat note:** a request that presents the legacy
> `CLYPT_PHASE1_AUDIO_HOST_TOKEN` value will continue to authenticate
> for one release (the service reads both names as fallbacks), but new
> deployments should configure the `_VIBEVOICE_ASR_SERVICE_*` names
> everywhere.

The matching H200-side clients live at:

- `backend/providers/audio_host_client.py` — `RemoteVibeVoiceAsrClient`
  (legacy import alias: `RemoteAudioChainClient`)
- `backend/providers/node_media_prep_client.py` — `RemoteNodeMediaPrepClient`

Both clients are wired unconditionally; there is no local fallback.

## 2) Provisioning Requirements

### 2.1 Image

DigitalOcean **NVIDIA AI/ML** base image on an RTX 6000 Ada droplet. Verify
NVENC before running anything else:

```bash
ffmpeg -hide_banner -init_hw_device cuda=cu -c:v h264_nvenc -f null - </dev/null
```

If that prints `unsupported device (2)`, you are still on an H200-class
GPU — stop and check your droplet type.

### 2.2 Required host paths

- repo: `/opt/clypt-audio-host/repo`
- env file: `/etc/clypt-audio-host/audio_host.env`
- service account key: `/opt/clypt-audio-host/sa-key.json`
- audio venv: `/opt/clypt-audio-host/venvs/audio`
- scratch workspace: `/opt/clypt-audio-host/scratch` (ephemeral per-request dirs)
- HF cache for VibeVoice weights: `/opt/clypt-audio-host/hf-cache`

## 3) Sync and Bootstrap

```bash
rsync -az --delete \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='outputs' \
  --exclude='*.egg-info' \
  -e "ssh -i ~/.ssh/clypt_audio_host_ed25519 -o IdentitiesOnly=yes" \
  /Users/rithvik/Clypt-Backend/ \
  root@<RTX_IP>:/opt/clypt-audio-host/repo/
```

```bash
ssh -i ~/.ssh/clypt_audio_host_ed25519 root@<RTX_IP>
cd /opt/clypt-audio-host/repo
bash scripts/do_phase1_audio/bootstrap_rtx6000ada.sh
```

## 4) Environment File

Copy [`docs/runtime/known-good-audio-host.env`](../runtime/known-good-audio-host.env)
to `/etc/clypt-audio-host/audio_host.env` and fill in the real values. At a
minimum the following must be set (deploy fails fast if any are missing):

```bash
CLYPT_PHASE1_AUDIO_HOST_BIND=0.0.0.0
CLYPT_PHASE1_AUDIO_HOST_PORT=9100
CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN=<shared-bearer-token>

VIBEVOICE_BACKEND=vllm
VIBEVOICE_VLLM_BASE_URL=http://127.0.0.1:8000
VIBEVOICE_VLLM_MODEL=vibevoice

GOOGLE_CLOUD_PROJECT=<project>
GCS_BUCKET=<bucket>
GOOGLE_APPLICATION_CREDENTIALS=/opt/clypt-audio-host/sa-key.json
```

The bearer token must match the H200's
`CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN` and
`CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN` (the same value can be reused for both).

> **Compat note:** `CLYPT_PHASE1_AUDIO_HOST_TOKEN` is still accepted as
> a deprecated alias for `_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN` for one
> release. Do not ship new deployments on the old name.

## 5) Deploy Runtime Services

### 5.1 VibeVoice vLLM (sole tenant)

```bash
cd /opt/clypt-audio-host/repo
bash scripts/do_phase1_audio/deploy_vllm_service.sh
```

Installs Docker + NVIDIA runtime if missing, builds
`clypt-vllm-vibevoice:latest` (VibeVoice and all Python deps are **baked
into the image** — 767 MB layer: ffmpeg, libsndfile1, vibevoice[vllm]
deps), installs `clypt-vllm-vibevoice.service`, and waits for
`http://127.0.0.1:8000/health` to go green. Model weights live on the
host mount at `/root/.cache/huggingface`; cold restart takes ~45 s (vs
~5 min before baking).

Current systemd unit vLLM flags (sole-tenant tuned):
- `--gpu-memory-utilization 0.77` — leaves ~8 GiB for concurrent NVDEC
  contexts during node-media prep
- `--max-num-seqs 2`
- `--dtype bfloat16`
- CUDA graph capture enabled (`enforce_eager=False`, the default)
- No speculative decoding — VibeVoice is a Whisper encoder-decoder, not
  decoder-only; MTP/speculative heads do not apply.

The earlier co-tenancy workarounds (`--max-num-seqs 1`,
`--max-model-len 32768`, `--enforce-eager`,
`--gpu-memory-utilization 0.60`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) are gone — they were
required when NFA shared the card.

### 5.2 Audio host FastAPI

```bash
bash scripts/do_phase1_audio/deploy_audio_service.sh
```

Installs `requirements-do-phase1-audio.txt` into the audio venv — now a
minimal set: VibeVoice vLLM runtime dependencies, the FastAPI service,
and GPU ffmpeg only. NeMo, FunASR, TensorFlow/TensorFlow-Hub, librosa,
and resampy are **no longer installed on this host**; they live in
`requirements-do-phase1-visual.txt` on the H200. Installs
`clypt-audio-host.service` and starts it. The unit depends on
`clypt-vllm-vibevoice.service`.

## 6) Service Verification

From the RTX host itself:

```bash
systemctl is-active clypt-vllm-vibevoice clypt-audio-host

curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models | python3 -m json.tool

curl -fsS "http://127.0.0.1:${CLYPT_PHASE1_AUDIO_HOST_PORT}/health"
```

From the H200, using its env file:

```bash
curl -fsS \
  -H "Authorization: Bearer ${CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN}" \
  "${CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL%/}/health"
```

## 7) Operational Notes

- **GPU tenancy.** This box is a **sole tenant** for VibeVoice vLLM +
  ffmpeg NVENC. `POST /tasks/vibevoice-asr` serializes the GPU for ASR
  via an `asyncio.Lock`. Node-media prep runs concurrently with ASR via
  a bounded semaphore but competes only for encoder slices, not general
  SM time.
- **NVDEC/NVENC concurrency cap.** Clip extraction uses explicit
  `-c:v h264_cuvid` (NVDEC hardware decode) + `h264_nvenc` (NVENC
  encode). Max concurrency is **8** (`CLYPT_PHASE24_NODE_MEDIA_PREP_MAX_CONCURRENCY`).
  16 OOMs the NVENC input buffers with vLLM seated at 0.77 utilization
  (~8 GiB free). Do not raise the cap above 8 without re-profiling VRAM.
  The `-ss` seek is output-side (after `-i`) — intentional; input seek
  was evaluated and rejected.
- **NVDEC history.** Originally `-hwaccel_output_format cuda` + seek
  caused a filter reinit error that silently fell back to CPU decode. The
  interim fix (`-hwaccel cuda` only, without `-hwaccel_output_format`)
  also fell through to CPU decode on ffmpeg 4.4 (that version does not
  auto-select NVDEC with just `-hwaccel cuda`). The current fix uses
  explicit `-c:v h264_cuvid` to force NVDEC. See `docs/ERROR_LOG.md`
  for the full fix chain.
- **Scratch lifecycle.** Every request writes to `tempfile.mkdtemp(dir=...)`
  under `CLYPT_PHASE1_AUDIO_HOST_SCRATCH_ROOT` and cleans up on exit. If
  the droplet reboots mid-request, stale dirs under `scratch/` can be
  removed safely.
- **GCS scope.** The service account key needs read on the source bucket
  (for audio/video downloads) and write on the destination bucket (for
  node-clip uploads). Typically the same bucket.
- **No pipeline code outside the two endpoints.** This host does not run
  NFA, emotion2vec+, or YAMNet, does not enqueue into the Phase 2-4
  queue, and does not talk to Spanner. All audio post-processing and
  graph/state writes happen on the H200.
- **Failure modes:** see the "Remote VibeVoice ASR" entries in
  [`docs/ERROR_LOG.md`](../ERROR_LOG.md). A 5xx from this host typically
  means either `systemctl status clypt-vllm-vibevoice` is degraded, the
  HF cache filled the disk, or the GCP service account lost a bucket IAM
  binding.
