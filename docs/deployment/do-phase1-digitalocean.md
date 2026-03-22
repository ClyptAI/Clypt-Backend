# DigitalOcean Phase 1 Deployment

This repo now expects the active Phase 1 path to run on a DigitalOcean-hosted service.

## Current DigitalOcean State

- Project: `Clypt-V2`
- SSH key name: `clypt-do-phase1`
- Local key path: `~/.ssh/clypt_do_ed25519`

## Before You Create a GPU Droplet

Check which GPU sizes are actually available to your account:

```bash
doctl compute region list --output json | jq -r '
  .[]
  | select(.available == true)
  | {slug, gpu_sizes: ((.sizes // []) | map(select(startswith("gpu-"))))}
  | select(.gpu_sizes | length > 0)
'
```

Create the droplet with a currently available GPU size:

```bash
doctl compute droplet create clypt-phase1-gpu-1 \
  --project-id b306872c-0eb7-4c6f-9658-32188ac4a642 \
  --region YOUR_REGION \
  --size YOUR_GPU_SIZE \
  --image ubuntu-24-04-x64 \
  --ssh-keys 55082300 \
  --enable-monitoring \
  --enable-private-networking \
  --tag-names clypt,phase1,digitalocean \
  --wait
```

## Bootstrap the Droplet

SSH in and get the repo contents onto the droplet at `/opt/clypt-phase1/repo`.
If the GitHub repo is private, `rsync`/`scp` from a local checkout is fine.
Then run:

```bash
sudo bash scripts/do_phase1/bootstrap_gpu_droplet.sh
```

## Wire Secrets and Env

1. Copy the env template:

```bash
sudo cp backend/do_phase1_service/.env.example /etc/clypt-phase1/do-phase1.env
sudo chmod 600 /etc/clypt-phase1/do-phase1.env
```

2. Copy your GCP service account JSON:

```bash
sudo cp /path/to/gcp-sa.json /etc/clypt-phase1/gcp-sa.json
sudo chmod 600 /etc/clypt-phase1/gcp-sa.json
```

3. Fill in at least:

- `DO_REGION`
- `DO_PHASE1_WORKER_ID`
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GCS_BUCKET`
- `GOOGLE_CLOUD_PROJECT`

## Deploy the Service

Run on the droplet from the repo checkout:

```bash
sudo REPO_DIR=/opt/clypt-phase1/repo \
  BRANCH=codex/balanced-hybrid-phase1-contract \
  ENV_FILE=/etc/clypt-phase1/do-phase1.env \
  REQUIREMENTS_FILE=requirements-do-phase1.txt \
  bash scripts/do_phase1/deploy_phase1_service.sh
```

If you synced a plain working tree instead of a full `.git` checkout, add:

```bash
sudo SKIP_GIT_SYNC=1 \
  REPO_DIR=/opt/clypt-phase1/repo \
  ENV_FILE=/etc/clypt-phase1/do-phase1.env \
  REQUIREMENTS_FILE=requirements-do-phase1.txt \
  bash scripts/do_phase1/deploy_phase1_service.sh
```

The deploy script now installs the dedicated Phase 1 droplet dependency set
from `requirements-do-phase1.txt` and pre-caches the ASR / YOLO / LR-ASD /
InsightFace assets expected by `backend/modal_worker.py`.

## Point the Local Pipeline at DO

On your local machine:

```bash
export DO_PHASE1_BASE_URL=http://YOUR_DROPLET_IP:8080
```

The active pipeline path in `backend/pipeline/phase_1_modal_pipeline.py` now submits async Phase 1 jobs to that base URL.
