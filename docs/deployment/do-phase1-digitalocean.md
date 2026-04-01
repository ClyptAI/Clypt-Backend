# DigitalOcean Phase 1 Deployment

This is the active deployment path for Phase 1 extraction.

**Organize everything under the DigitalOcean project `Clypt-V3`.** New droplets, and any other DO resources you attach to this product, should be created or moved into that project so billing, access, and the control-plane “project” view match the v3 repo ([`rithm84/Clypt-V3`](https://github.com/rithm84/Clypt-V3)). This is separate from the GCP project `clypt-v3`.

## DigitalOcean project: `Clypt-V3`

Create the project once per account (skip if it already exists — `doctl` will error on duplicate name):

```bash
doctl projects create \
  --name "Clypt-V3" \
  --purpose "Web Application" \
  --description "Clypt v3: Phase 1 GPU extraction, APIs, and related DO resources" \
  --environment Production
```

Resolve its id for CLI use (droplet create, resource assignment, etc.):

```bash
PROJECT_ID=$(doctl projects list -o json | jq -r '.[] | select(.name == "Clypt-V3") | .id')
echo "$PROJECT_ID"
```

To attach an **existing** droplet to this project: DigitalOcean control panel → **Move resources**, or `doctl projects resources assign` with the droplet’s URN.

## Current DO Facts To Verify Before Provisioning

- GitHub repository: [`rithm84/Clypt-V3`](https://github.com/rithm84/Clypt-V3) (canonical remote for this codebase)
- DigitalOcean project: **`Clypt-V3`** (see above; use `PROJECT_ID` when creating droplets)
- Droplet name: `clypt-phase1-gpu-1`
- SSH key name: `clypt-do-phase1`
- SSH key id: `55082300`
- Preferred region: `atl1`
- Preferred GPU size slug: `gpu-h200x1-141gb`
- Preferred image slug: `gpu-h100x1-base`
- Image name in DO UI: `NVIDIA AI/ML Ready`
- Last known good fallback region after explicit confirmation: `nyc2`
- Last known good deployed droplet shape: `nyc2 + gpu-h200x1-141gb + gpu-h100x1-base`
- RTX 6000 Ada fallback size slug: `gpu-6000adax1-48gb` (available in `tor1`; $1.57/hr vs $3.44/hr for H200)

Important:
- Do **not** silently fall back to another region, another GPU type, or plain Ubuntu.
- Re-check GPU availability immediately before creating the droplet. `atl1` may be temporarily out of GPU capacity.
- If `atl1` is blocked and you intentionally fall back, use `nyc2` with the exact same `H200 + NVIDIA AI/ML Ready` shape.
- **RTX 6000 Ada** (`gpu-6000adax1-48gb`, 48 GB GDDR6) is an acceptable secondary fallback for short videos (<10 min). Expect ~2-3x slower wall-clock on LR-ASD-heavy stages due to lower memory bandwidth (~960 GB/s vs ~3.35 TB/s on H200). VRAM is not a concern at this duration. When using the RTX 6000, lower `CLYPT_LRASD_BATCH_SIZE` to 16-24 (default tuned for H200 at 48) to avoid OOM on longer clips.

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
PROJECT_ID=$(doctl projects list -o json | jq -r '.[] | select(.name == "Clypt-V3") | .id')
doctl compute droplet create clypt-phase1-gpu-1 \
  --project-id "$PROJECT_ID" \
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

Defaults and commented overrides are maintained in **`backend/do_phase1_service/.env.example`** (that file is the deployment template shipped with the repo).

Typical production overrides (adjust to your GPU and load):

- `DO_REGION=atl1` (or `nyc2` only when explicitly choosing the fallback region)
- `DO_PHASE1_WORKER_ID=clypt-phase1-gpu-1`
- `DO_PHASE1_WORKER_CONCURRENCY=3` (template default is `1`; raise when validated on your hardware)
- `DO_PHASE1_GPU_SLOTS=1`
- `CLYPT_SPEAKER_BINDING_MODE=lrasd` (template default; `auto` / `heuristic` / `shared_analysis_proxy` are also implemented in `backend/do_phase1_worker.py`)
- `CLYPT_TRACKING_MODE=direct` (worker default in code is `direct`; `chunked` / `auto` / `shared_analysis_proxy` are supported)
- `CLYPT_TRACKER_BACKEND=bytetrack` (only ByteTrack is supported; invalid values fail fast)
- `CLYPT_TRACK_CHUNK_WORKERS=1`
- `CLYPT_SPEAKER_BINDING_PROXY_ENABLE=1`, `CLYPT_ANALYSIS_PROXY_MAX_LONG_EDGE=1280`
- `CLYPT_LRASD_PIPELINE_OVERLAP=1`, and optional `CLYPT_LRASD_BATCH_SIZE` / `CLYPT_LRASD_MAX_INFLIGHT` (see template; `CLYPT_LRASD_PROFILE=auto` resolves defaults in worker)

Pyannote diarization is available but stays off until we turn it on explicitly:
- `CLYPT_AUDIO_DIARIZATION_ENABLE=0`
- `CLYPT_AUDIO_DIARIZATION_MODEL=pyannote/speaker-diarization-3.1`
- `CLYPT_AUDIO_DIARIZATION_MIN_SEGMENT_MS=400`

When you enable diarization, set a Hugging Face token in the droplet environment:
- `HF_TOKEN` is the preferred variable
- `HUGGINGFACE_HUB_TOKEN` is also commonly accepted by Hugging Face tooling
- The token needs access to the pyannote model and will be used when the worker first downloads the diarization pipeline
- `community-1` is deferred until we move to the later pyannote 4.x / newer torch stack

The first diarization-enabled boot will cache the model on the droplet, so plan for the extra download time and disk usage.

Why these defaults:
- `DO_PHASE1_WORKER_CONCURRENCY=3` allows multiple jobs to be claimed and managed concurrently.
- `DO_PHASE1_GPU_SLOTS=1` keeps the GPU-heavy extraction section serialized until higher overlap is validated.
- `CLYPT_TRACKING_MODE=direct` runs full-video tracking in one pass unless you switch to `chunked` or `auto` in code-supported modes.
- `CLYPT_SPEAKER_BINDING_MODE=lrasd` keeps the worker on the LR-ASD path; whole-job heuristic after LR-ASD returns `None` stays off unless `CLYPT_SPEAKER_BINDING_HEURISTIC_FALLBACK=1`.

## Deploy the Service

From the droplet repo checkout:

```bash
sudo REPO_DIR=/opt/clypt-phase1/repo \
  BRANCH=main \
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
- Ultralytics YOLOv26 **segmentation** weights (default `yolo26m-seg.pt` per `YOLO_WEIGHTS_PATH` in `backend/do_phase1_worker.py`)
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
