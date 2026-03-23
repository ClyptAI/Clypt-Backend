# Clypt: Creator Intelligence and Clipping

Clypt takes a YouTube URL and produces 9:16 short-form clips by combining:
- Phase 1 DO GPU extraction (ASR + tracking + speaker binding)
- Gemini reasoning and embeddings over semantic nodes/edges
- Spanner + GCS storage
- Remotion rendering

## Pipeline Overview

```text
YouTube URL
  -> Phase 1   DO extraction (ASR + tracking + speaker binding)
  -> Phase 2A  Semantic nodes (Gemini)
  -> Phase 2B  Narrative edges (Gemini)
  -> Phase 3   Gemini Embedding 2 multimodal vectors
  -> Phase 4   Persist nodes/edges/tracking to Spanner + GCS
  -> Phase 5   Clip selection / retrieval
  -> Render    Remotion 9:16 output clips
```

## Tech Stack (Current)

### Phase 1 DO worker (`backend/do_phase1_worker.py`)
- ASR: NVIDIA Parakeet-TDT-1.1B
- Tracking: YOLO26s + BoT-SORT (+ chunk fan-out + stitch)
- Face: InsightFace/ArcFace for ID stabilization
- Active speaker binding: TalkNet
- Runtime: DigitalOcean GPU worker / extraction service

### Local pipeline client (`backend/pipeline/phase_1_do_pipeline.py`)
- Downloads source media with `yt-dlp`
- Converts audio to 16kHz mono WAV
- Calls the DO Phase 1 service
- Writes canonical JSON and NDJSON handoff files

## Prerequisites

- Python 3.11+
- Node.js 18+ and npm
- `ffmpeg` + `ffprobe`
- Google Cloud project access for Vertex AI, Spanner, and GCS

## Team Setup (Shared Project)

This repo and GCP stack are already shared between collaborators.  
Use this README as an in-place runbook from your local checkout of `Clypt-V2` (no cloning/bootstrap steps here).

### Shared cloud resources (already owned by the team)
- Project: `clypt-v2`
- Bucket: `gs://clypt-storage-v2`
- Spanner instance: `clypt-spanner-v2`
- Spanner database: `clypt-graph-db-v2`

## Local One-Time Setup

1. Python + Node dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd remotion-render && npm install && cd ..
```

2. Authenticate CLIs

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project clypt-v2
```

3. Enable required Google APIs (safe to run again)

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  spanner.googleapis.com \
  storage.googleapis.com
```

4. Verify shared resources exist

```bash
gcloud storage buckets describe gs://clypt-storage-v2
gcloud spanner instances describe clypt-spanner-v2
gcloud spanner databases describe clypt-graph-db-v2 --instance=clypt-spanner-v2
```

## Admin-Only: Provision or Repair Infrastructure

Run this section only if resources are missing or intentionally being rebuilt.

`backend/spanner_schema.sql` includes `CREATE PROPERTY GRAPH`, so the Spanner instance must be `ENTERPRISE` edition.

```bash
gcloud spanner instances create clypt-spanner-v2 \
  --config=regional-us-central1 \
  --description="Clypt graph instance" \
  --edition=ENTERPRISE \
  --nodes=1

gcloud spanner databases create clypt-graph-db-v2 \
  --instance=clypt-spanner-v2 \
  --ddl-file=backend/spanner_schema.sql
```

If database exists and schema changed:

```bash
gcloud spanner databases ddl update clypt-graph-db-v2 \
  --instance=clypt-spanner-v2 \
  --ddl-file=backend/spanner_schema.sql
```

## Deploy DO Phase 1 Service

```bash
sudo REPO_DIR=/opt/clypt-phase1/repo \
  BRANCH=codex/balanced-hybrid-phase1-contract \
  ENV_FILE=/etc/clypt-phase1/do-phase1.env \
  REQUIREMENTS_FILE=requirements-do-phase1.txt \
  bash scripts/do_phase1/deploy_phase1_service.sh
```

This installs the droplet-specific dependency bundle, pre-caches the Phase 1
models, and starts both systemd services:
- `clypt-phase1-api.service`
- `clypt-phase1-worker.service`

## Run with a Video URL

### Option A: Run only Phase 1 (recommended for extraction/debug)

```bash
source .venv/bin/activate
DO_PHASE1_BASE_URL=http://<droplet-ip>:8080 \
PHASE1_RUNTIME_PROFILE=podcast_eval \
PHASE1_SPEAKER_BINDING_MODE=lrasd \
PHASE1_TRACKING_MODE=direct \
PHASE1_SHARED_ANALYSIS_PROXY=1 \
.venv/bin/python -c "import asyncio; from backend.pipeline.phase_1_do_pipeline import main; asyncio.run(main('https://www.youtube.com/watch?v=dXUFsDcC0_4'))"
```

This will:
- download source media locally
- upload canonical source video to `gs://clypt-storage-v2/phase_1/video.mp4`
- run DO extraction
- write `backend/outputs/phase_1_visual.json`, `backend/outputs/phase_1_audio.json`, and `backend/outputs/phase_1_visual.ndjson`

### Option B: Run full pipeline (all phases + render)

Interactive:

```bash
source .venv/bin/activate
.venv/bin/python backend/pipeline/run_pipeline.py
```

Then paste a YouTube URL when prompted.

Non-interactive:

```bash
printf '%s\n' 'https://www.youtube.com/watch?v=dXUFsDcC0_4' | .venv/bin/python backend/pipeline/run_pipeline.py
```

## Important Runtime Flags

### Phase 1 client flags (`backend/pipeline/phase_1_do_pipeline.py`)
- `DO_PHASE1_BASE_URL`: base URL for the DO Phase 1 API
- `DO_PHASE1_POLL_INTERVAL_SECONDS`: job polling interval
- `DO_PHASE1_TIMEOUT_SECONDS`: total wait timeout
- `PHASE1_RUNTIME_PROFILE`: `production` or `podcast_eval`
- `PHASE1_SPEAKER_BINDING_MODE`: `auto`, `heuristic`, or `lrasd`
- `PHASE1_TRACKING_MODE`: `direct` or `chunked`
- `PHASE1_SHARED_ANALYSIS_PROXY`: `1`/`0`

### Worker flags (`backend/do_phase1_worker.py`)
- `CLYPT_YOLO_IMGSZ` (default `640`)
- `CLYPT_ENABLE_ROI_DETECT` (`1`/`0`)
- `CLYPT_TRACK_CHUNK_WORKERS` (local thread workers inside one container)
- `CLYPT_ENFORCE_ROLLOUT_GATES` (`1`/`0`)
- `CLYPT_ENABLE_LEGACY_SERVERLESS_SDK` (`1` only when intentionally testing the old optional wrapper shim)
- Gate thresholds:
  - `CLYPT_GATE_MIN_IDF1_PROXY`
  - `CLYPT_GATE_MIN_MOTA_PROXY`
  - `CLYPT_GATE_MAX_FRAGMENTATION`
  - `CLYPT_GATE_MIN_THROUGHPUT_FPS`
  - `CLYPT_GATE_MAX_WALLCLOCK_S`
  - `CLYPT_GATE_MIN_SCHEMA_PASS_RATE`

### DO worker service flags (`backend/do_phase1_service/*.py`)
- `DO_PHASE1_WORKER_CONCURRENCY`: number of worker processes that can claim jobs
- `DO_PHASE1_GPU_SLOTS`: number of extraction jobs allowed into the GPU-heavy critical section at once
- `DO_PHASE1_STATE_ROOT`, `DO_PHASE1_DB_PATH`, `DO_PHASE1_OUTPUT_ROOT`, `DO_PHASE1_LOG_ROOT`
- `DO_PHASE1_HOST_LOCK_PATH`: base path used to derive GPU-slot lock files

## Expected Outputs

### Core artifacts
- `backend/downloads/video.mp4`
- `backend/downloads/audio_16k.wav`
- `backend/outputs/phase_1_visual.json`
- `backend/outputs/phase_1_audio.json`
- `backend/outputs/phase_1_visual.ndjson`
- `backend/outputs/phase_2a_nodes.json`
- `backend/outputs/phase_2b_narrative_edges.json`
- `backend/outputs/phase_3_embeddings.json`
- `backend/outputs/remotion_payloads_array.json`

### Rendered clips
- `remotion-render/out/`

## Script Map

| Phase | Script |
|---|---|
| 1 | `backend/pipeline/phase_1_do_pipeline.py` |
| 2A | `backend/pipeline/phase_2a_make_nodes.py` |
| 2B | `backend/pipeline/phase_2b_draw_edges.py` |
| 3 | `backend/pipeline/phase_3_multimodal_embeddings.py` |
| 4 | `backend/pipeline/phase_4_store_graph.py` |
| 5 (auto-curate) | `backend/pipeline/phase_5_auto_curate.py` |
| 5 (retrieve) | `backend/pipeline/phase_5_retrieve.py` |
| full orchestrator | `backend/pipeline/run_pipeline.py` |

## Metadata Defaults in Code

- `PROJECT_ID`: `clypt-v2`
- `GCS_BUCKET`: `clypt-storage-v2`
- `SPANNER_INSTANCE`: `clypt-spanner-v2`
- `SPANNER_DATABASE`: `clypt-graph-db-v2`

## Planning References

- `docs/planning/01-product-and-demo.md`
- `docs/planning/02-system-architecture.md`
- `docs/planning/03-agents-and-clipping.md`
- `docs/planning/04-data-integrations-and-reference.md`
