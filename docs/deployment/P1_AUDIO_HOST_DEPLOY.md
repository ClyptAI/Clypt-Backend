# RTX 6000 Ada Phase 1 Audio Host Deployment

**Status:** Active
**Last updated:** 2026-04-17

This runbook covers the **RTX 6000 Ada** audio host. Pair it with the H200
runbook at [`docs/deployment/P1_DEPLOY.md`](P1_DEPLOY.md).

## 0) Why This Host Exists

- **NVENC placement is forced.** H200 NVENC is not usable for ffmpeg clip
  extraction (`h264_nvenc` returns `unsupported device (2)`). Node-media
  prep has to run on a non-H200 GPU with working NVENC/NVDEC.
- **VibeVoice dtype sanity.** 48 GB VRAM lets VibeVoice run at native dtype;
  the L4-era bf16 encoder patch is unnecessary.
- **H200 headroom.** Moving VibeVoice + ffmpeg off the H200 frees SM time
  and memory fraction for RF-DETR and SGLang.

## 1) Endpoints Served

One FastAPI process, one GPU kept hot:

- `POST /tasks/phase1-audio` — takes an audio GCS URI and returns
  `{turns, alignments, emotions, yamnet_tags, stage_events}` in a single
  round trip. Serialized on the GPU via an `asyncio.Lock`.
- `POST /tasks/node-media-prep` — takes a source video GCS URI and a list
  of `{node_id, start_ms, end_ms}`, runs NVENC clip extraction, uploads
  each clip to GCS, and returns the descriptors. Concurrency is bounded by
  `CLYPT_PHASE24_NODE_MEDIA_PREP_MAX_CONCURRENCY` on the caller side.
- `GET /health` — unauthenticated readiness probe.

All POST routes require `Authorization: Bearer ${CLYPT_PHASE1_AUDIO_HOST_TOKEN}`.

The matching H200-side clients live at:

- `backend/providers/audio_host_client.py` — `RemoteAudioChainClient`
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
CLYPT_PHASE1_AUDIO_HOST_TOKEN=<shared-bearer-token>

VIBEVOICE_BACKEND=vllm
VIBEVOICE_VLLM_BASE_URL=http://127.0.0.1:8000
VIBEVOICE_VLLM_MODEL=vibevoice

GOOGLE_CLOUD_PROJECT=<project>
GCS_BUCKET=<bucket>
GOOGLE_APPLICATION_CREDENTIALS=/opt/clypt-audio-host/sa-key.json
```

The bearer token must match the H200's `CLYPT_PHASE1_AUDIO_HOST_TOKEN` and
`CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN` (the same value can be reused for both).

## 5) Deploy Runtime Services

### 5.1 VibeVoice vLLM

```bash
cd /opt/clypt-audio-host/repo
bash scripts/do_phase1_audio/deploy_vllm_service.sh
```

Installs Docker + NVIDIA runtime if missing, clones the VibeVoice repo mount,
builds `clypt-vllm-vibevoice:latest`, installs `clypt-vllm-vibevoice.service`,
and waits for `http://127.0.0.1:8000/health` to go green. The systemd unit
runs with `--gpu-memory-utilization 0.45` (leaving room for NFA/emotion2vec+
+ ffmpeg concurrently on the same card).

### 5.2 Audio host FastAPI

```bash
bash scripts/do_phase1_audio/deploy_audio_service.sh
```

Installs `requirements-do-phase1-audio.txt` into the audio venv, pre-warms
NFA + emotion2vec+ model caches, installs `clypt-audio-host.service`, and
starts it. The unit depends on `clypt-vllm-vibevoice.service`.

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
  -H "Authorization: Bearer ${CLYPT_PHASE1_AUDIO_HOST_TOKEN}" \
  "${CLYPT_PHASE1_AUDIO_HOST_URL%/}/health"
```

## 7) Operational Notes

- **GPU contention.** VibeVoice, NFA, emotion2vec+, and ffmpeg NVENC all
  live on the same card. `POST /tasks/phase1-audio` is serialized via an
  `asyncio.Lock` in the app so the audio chain owns the GPU end-to-end for
  one run at a time. Node-media prep runs concurrently with the audio
  chain via a bounded semaphore but competes for encoder slices.
- **Scratch lifecycle.** Every request writes to `tempfile.mkdtemp(dir=...)`
  under `CLYPT_PHASE1_AUDIO_HOST_SCRATCH_ROOT` and cleans up on exit. If
  the droplet reboots mid-request, stale dirs under `scratch/` can be
  removed safely.
- **GCS scope.** The service account key needs read on the source bucket
  (for audio/video downloads) and write on the destination bucket (for
  node-clip uploads). Typically the same bucket.
- **No pipeline code outside the two endpoints.** This host does not enqueue
  into the Phase 2-4 queue and does not talk to Spanner. All graph/state
  writes happen on the H200.
- **Failure modes:** see the "Remote audio host" entries in
  [`docs/ERROR_LOG.md`](../ERROR_LOG.md). A 5xx from this host typically
  means either `systemctl status clypt-vllm-vibevoice` is degraded, the
  HF cache filled the disk, or the GCP service account lost a bucket IAM
  binding.
