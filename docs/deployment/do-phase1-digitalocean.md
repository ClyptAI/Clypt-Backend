# DigitalOcean Phase 1 Deployment

This is the active deployment path for Phase 1 extraction.

## Current DO Facts To Verify Before Provisioning

- Project: `Clypt-V2`
- Droplet name: `clypt-phase1-gpu-1`
- SSH key name: `clypt-do-phase1`
- SSH key id: `55082300`
- Preferred region: `atl1`
- Preferred GPU size slug: `gpu-h200x1-141gb`
- Preferred image slug: `gpu-h100x1-base`
- Image name in DO UI: `NVIDIA AI/ML Ready`
- Last known good fallback region after explicit confirmation: `nyc2`
- Last known good deployed droplet shape: `nyc2 + gpu-h200x1-141gb + gpu-h100x1-base`

Important:
- Do **not** silently fall back to another region, another GPU type, or plain Ubuntu.
- Re-check GPU availability immediately before creating the droplet. `atl1` may be temporarily out of GPU capacity.
- If `atl1` is blocked and you intentionally fall back, use `nyc2` with the exact same `H200 + NVIDIA AI/ML Ready` shape.

## Preflight Checks

List regions with currently available GPU sizes:

```bash
doctl compute region list --output json | jq -r '
  .[]
  | select(.available == true)
  | {slug, gpu_sizes: ((.sizes // []) | map(select(startswith("gpu-"))))}
  | select(.gpu_sizes | length > 0)
'
```

Confirm the target image exists:

```bash
doctl compute image list --public --output json | jq -r '
  .[]
  | select(.slug == "gpu-h100x1-base" or .name == "NVIDIA AI/ML Ready")
  | {id, name, slug, distribution}
'
```

Confirm the SSH key:

```bash
doctl compute ssh-key list --output json | jq -r '
  .[]
  | select(.name == "clypt-do-phase1" or .id == 55082300)
  | {id, name, fingerprint}
'
```

## Create the Droplet

Only run this after the preflight checks confirm the exact target shape is available:

```bash
doctl compute droplet create clypt-phase1-gpu-1 \
  --project-id b306872c-0eb7-4c6f-9658-32188ac4a642 \
  --region atl1 \
  --size gpu-h200x1-141gb \
  --image gpu-h100x1-base \
  --ssh-keys 55082300 \
  --enable-monitoring \
  --enable-private-networking \
  --tag-names clypt,phase1,digitalocean \
  --wait
```

## Bootstrap the Droplet

Get the repo onto the droplet at `/opt/clypt-phase1/repo`, then run:

```bash
sudo bash scripts/do_phase1/bootstrap_gpu_droplet.sh
```

## Wire Secrets and Env

Copy the env template:

```bash
sudo cp backend/do_phase1_service/.env.example /etc/clypt-phase1/do-phase1.env
sudo chmod 600 /etc/clypt-phase1/do-phase1.env
```

Copy the GCP service account JSON:

```bash
sudo cp /path/to/gcp-sa.json /etc/clypt-phase1/gcp-sa.json
sudo chmod 600 /etc/clypt-phase1/gcp-sa.json
```

Minimum env values to fill in:
- `DO_REGION`
- `DO_PHASE1_WORKER_ID`
- `DO_PHASE1_WORKER_CONCURRENCY`
- `DO_PHASE1_GPU_SLOTS`
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GCS_BUCKET`
- `GOOGLE_CLOUD_PROJECT`

Recommended starting values:
- `DO_REGION=atl1` (or `nyc2` only when explicitly choosing the fallback region)
- `DO_PHASE1_WORKER_ID=clypt-phase1-gpu-1`
- `DO_PHASE1_WORKER_CONCURRENCY=3`
- `DO_PHASE1_GPU_SLOTS=1`
- `CLYPT_SPEAKER_BINDING_MODE=auto`
- `CLYPT_TRACKING_MODE=auto`
- `CLYPT_TRACK_CHUNK_WORKERS=1`
- `CLYPT_ANALYSIS_PROXY_ENABLE=1`
- `CLYPT_ANALYSIS_PROXY_MAX_LONG_EDGE=1920`
- `CLYPT_SPEAKER_BINDING_PROXY_ENABLE=1`
- `CLYPT_SPEAKER_BINDING_PROXY_MAX_LONG_EDGE=1920`
- `CLYPT_ASD_PRECOMPUTED_FACE=1`
- `CLYPT_ASD_FACE_FPS=1.0`
- `CLYPT_ASD_PRECOMPUTED_MIN_COVERAGE=0.80`
- `CLYPT_LRASD_BATCH_SIZE=32`
- `CLYPT_LRASD_PIPELINE_OVERLAP=1`
- `CLYPT_LRASD_MAX_INFLIGHT=4`

Pyannote diarization is available but stays off until we turn it on explicitly:
- `CLYPT_AUDIO_DIARIZATION_ENABLE=0`
- `CLYPT_AUDIO_DIARIZATION_MODEL=pyannote/speaker-diarization-community-1`
- `CLYPT_AUDIO_DIARIZATION_MIN_SEGMENT_MS=400`

When you enable diarization, set a Hugging Face token in the droplet environment:
- `HF_TOKEN` is the preferred variable
- `HUGGINGFACE_HUB_TOKEN` is also commonly accepted by Hugging Face tooling
- The token needs access to the pyannote model and will be used when the worker first downloads the diarization pipeline

The first diarization-enabled boot will cache the model on the droplet, so plan for the extra download time and disk usage.

Why these defaults:
- `DO_PHASE1_WORKER_CONCURRENCY=3` allows multiple jobs to be claimed and managed concurrently.
- `DO_PHASE1_GPU_SLOTS=1` keeps the GPU-heavy extraction section serialized until higher overlap is validated.
- `CLYPT_TRACKING_MODE=auto` lets the worker choose between direct and chunked tracking.
- `CLYPT_SPEAKER_BINDING_MODE=auto` keeps LR-ASD for manageable clips while allowing fallback behavior on larger jobs.
- `CLYPT_ANALYSIS_PROXY_ENABLE=1` and `CLYPT_ANALYSIS_PROXY_MAX_LONG_EDGE=1920` keep tracking, face analysis, and LR-ASD on the same shared 1080p proxy instead of the full-resolution source.

## Deploy the Service

From the droplet repo checkout:

```bash
sudo REPO_DIR=/opt/clypt-phase1/repo \
  BRANCH=codex/balanced-hybrid-phase1-contract \
  ENV_FILE=/etc/clypt-phase1/do-phase1.env \
  REQUIREMENTS_FILE=requirements-do-phase1.txt \
  bash scripts/do_phase1/deploy_phase1_service.sh
```

If you synced a working tree without `.git`, add:

```bash
sudo SKIP_GIT_SYNC=1 \
  REPO_DIR=/opt/clypt-phase1/repo \
  ENV_FILE=/etc/clypt-phase1/do-phase1.env \
  REQUIREMENTS_FILE=requirements-do-phase1.txt \
  bash scripts/do_phase1/deploy_phase1_service.sh
```

The deploy script installs the dedicated Phase 1 dependency set and pre-caches the active Phase 1 assets:
- Parakeet ASR
- YOLO26s PyTorch weights
- LR-ASD repo + checkpoint assets
- InsightFace packs

## Watch Progress

API:

```bash
curl http://YOUR_DROPLET_IP:8080/jobs/JOB_ID
curl "http://YOUR_DROPLET_IP:8080/jobs/JOB_ID/logs?tail_lines=200"
```

Systemd:

```bash
sudo journalctl -u clypt-phase1-worker.service -f
sudo journalctl -u clypt-phase1-api.service -f
```

Per-job logs:

```bash
tail -f /var/lib/clypt/do_phase1_service/workdir/logs/JOB_ID.log
```

## Point the Local Pipeline at DO

```bash
export DO_PHASE1_BASE_URL=http://YOUR_DROPLET_IP:8080
```

`backend/pipeline/phase_1_do_pipeline.py` is the active local client for submitting async Phase 1 jobs and materializing local compatibility artifacts from the returned manifest.
