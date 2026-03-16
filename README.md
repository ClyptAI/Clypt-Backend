# Clypt: Creator Intelligence and Clipping

Clypt takes a YouTube URL and produces 9:16 short-form clips by combining:
- Phase 1 Modal GPU extraction (ASR + tracking + speaker binding)
- Gemini reasoning and embeddings over semantic nodes/edges
- Spanner + GCS storage
- Remotion rendering

## Pipeline Overview

```text
YouTube URL
  -> Phase 1   Modal extraction (ASR + tracking + speaker binding)
  -> Phase 2A  Semantic nodes (Gemini)
  -> Phase 2B  Narrative edges (Gemini)
  -> Phase 3   Gemini Embedding 2 multimodal vectors
  -> Phase 4   Persist nodes/edges/tracking to Spanner + GCS
  -> Phase 5   Clip selection / retrieval
  -> Render    Remotion 9:16 output clips
```

## Tech Stack (Current)

### Phase 1 Modal worker (`backend/modal_worker.py`)
- ASR: NVIDIA Parakeet-TDT-1.1B
- Tracking: YOLO26s + BoT-SORT (+ chunk fan-out + stitch)
- Face: InsightFace/ArcFace for ID stabilization
- Active speaker binding: TalkNet
- Runtime: Modal `H100`, `max_containers=8`, memory snapshot enabled

### Local pipeline client (`backend/pipeline/phase_1_modal_pipeline.py`)
- Downloads source media with `yt-dlp`
- Converts audio to 16kHz mono WAV
- Calls Modal worker (single-worker or distributed fan-out mode)
- Writes canonical JSON and NDJSON handoff files

## Prerequisites

- Python 3.11+
- Node.js 18+ and npm
- `ffmpeg` + `ffprobe`
- Modal account + CLI auth
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
.venv/bin/modal token new
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

## Deploy Modal Worker

```bash
source .venv/bin/activate
.venv/bin/modal deploy backend/modal_worker.py
```

## Run with a Video URL

### Option A: Run only Phase 1 (recommended for extraction/debug)

```bash
source .venv/bin/activate
CLYPT_DISTRIBUTED_MODAL_FANOUT=1 \
CLYPT_DISTRIBUTED_DETACH=1 \
CLYPT_DISTRIBUTED_RESUME=1 \
CLYPT_MAX_GPU_WORKERS=8 \
.venv/bin/python -c "import asyncio; from backend.pipeline.phase_1_modal_pipeline import main; asyncio.run(main('https://www.youtube.com/watch?v=dXUFsDcC0_4'))"
```

This will:
- download source media locally
- upload canonical source video to `gs://clypt-storage-v2/phase_1/video.mp4`
- run Modal extraction
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

### Phase 1 client flags (`backend/pipeline/phase_1_modal_pipeline.py`)
- `CLYPT_DISTRIBUTED_MODAL_FANOUT` (`1`/`0`): distributed chunk fan-out mode
- `CLYPT_DISTRIBUTED_DETACH` (`1`/`0`): submit jobs detached and collect later
- `CLYPT_DISTRIBUTED_RESUME` (`1`/`0`): resume from `backend/outputs/phase_1_detached_state.json`
- `CLYPT_MAX_GPU_WORKERS` (default `8`, capped to `8`)
- `GCS_BUCKET` (default `clypt-storage-v2`)
- `GCS_VIDEO_OBJECT` (default `phase_1/video.mp4`)

### Worker flags (`backend/modal_worker.py`)
- `CLYPT_YOLO_IMGSZ` (default `640`)
- `CLYPT_ENABLE_ROI_DETECT` (`1`/`0`)
- `CLYPT_TRACK_CHUNK_WORKERS` (local thread workers inside one container)
- `CLYPT_ENFORCE_ROLLOUT_GATES` (`1`/`0`)
- Gate thresholds:
  - `CLYPT_GATE_MIN_IDF1_PROXY`
  - `CLYPT_GATE_MIN_MOTA_PROXY`
  - `CLYPT_GATE_MAX_FRAGMENTATION`
  - `CLYPT_GATE_MIN_THROUGHPUT_FPS`
  - `CLYPT_GATE_MAX_WALLCLOCK_S`
  - `CLYPT_GATE_MIN_SCHEMA_PASS_RATE`

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
| 1 | `backend/pipeline/phase_1_modal_pipeline.py` |
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
