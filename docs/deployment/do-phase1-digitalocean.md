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
- `DO_PHASE1_WORKER_CONCURRENCY`
- `DO_PHASE1_GPU_SLOTS`
- `DO_PHASE1_LOG_ROOT`
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GCS_BUCKET`
- `GOOGLE_CLOUD_PROJECT`

Recommended Phase 1 tuning for large creator videos:

- `DO_PHASE1_WORKER_CONCURRENCY=3`
- `DO_PHASE1_GPU_SLOTS=1`
- `CLYPT_SPEAKER_BINDING_MODE=auto`
- `CLYPT_SPEAKER_BINDING_AUTO_MAX_DURATION_S=180`
- `CLYPT_SPEAKER_BINDING_AUTO_MAX_LONG_EDGE=1920`
- `CLYPT_SPEAKER_BINDING_AUTO_MAX_WORDS=450`
- `CLYPT_SPEAKER_BINDING_AUTO_MAX_TRACKS=12`
- `CLYPT_TRACKING_MODE=auto`
- `CLYPT_TRACK_CHUNK_WORKERS=1`
- `CLYPT_SPEAKER_BINDING_PROXY_ENABLE=1`
- `CLYPT_SPEAKER_BINDING_PROXY_MAX_LONG_EDGE=1280`
- `CLYPT_ASD_PRECOMPUTED_FACE=1`
- `CLYPT_ASD_FACE_FPS=1.0`
- `CLYPT_ASD_PRECOMPUTED_MIN_COVERAGE=0.80`
- `CLYPT_LRASD_BATCH_SIZE=32`
- `CLYPT_LRASD_PIPELINE_OVERLAP=1`
- `CLYPT_LRASD_MAX_INFLIGHT=4`

`CLYPT_SPEAKER_BINDING_MODE=auto` keeps a fast heuristic path for large or
long clips, while still using LR-ASD by default on smaller videos. The
speaker-binding proxy keeps LR-ASD off full-resolution 4K frames while
preserving the original video for tracking and downstream artifacts.
`DO_PHASE1_WORKER_CONCURRENCY=3` lets the droplet claim multiple jobs at once
without forcing three GPU-heavy extractions to overlap immediately.
`DO_PHASE1_GPU_SLOTS=1` keeps the expensive tracking / face / speaker-binding
section serialized on a single GPU until higher slot counts are explicitly
validated.
`CLYPT_TRACKING_MODE=auto` prefers the simpler direct full-video tracker when
the droplet is effectively single-worker, which avoids redundant chunk
re-encodes.
`CLYPT_TRACK_CHUNK_WORKERS=1` is the safest default on a single GPU droplet:
it reuses the loaded YOLO model and avoids concurrent GPU tracker calls that
can destabilize long runs.

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
InsightFace assets expected by `backend/do_phase1_worker.py`.

The worker now runs as a persistent Python process instead of a shell loop
that launches a new interpreter every idle poll. That keeps GPU-box overhead
lower and makes logs much easier to follow.

## Watching Progress and Logs

Use the API for job state:

```bash
curl http://YOUR_DROPLET_IP:8080/jobs/JOB_ID
curl "http://YOUR_DROPLET_IP:8080/jobs/JOB_ID/logs?tail_lines=200"
```

The job status payload now includes:

- `current_step`
- `progress_message`
- `progress_pct`
- `log_path`

Use the droplet for service-level logs:

```bash
sudo journalctl -u clypt-phase1-worker.service -f
sudo journalctl -u clypt-phase1-api.service -f
```

Use the per-job log file when you want the exact Phase 1 step stream:

```bash
tail -f /var/lib/clypt/do_phase1_service/workdir/logs/JOB_ID.log
```

Those per-job logs include the underlying `[Phase 1] Step ...` prints from
`backend/do_phase1_worker.py`, so they are the closest DO equivalent to the old
DO progress view.

## Point the Local Pipeline at DO

On your local machine:

```bash
export DO_PHASE1_BASE_URL=http://YOUR_DROPLET_IP:8080
```

The active pipeline path in `backend/pipeline/phase_1_do_pipeline.py` now submits async Phase 1 jobs to that base URL.
