# Clypt: Creator Intelligence and Clipping

Clypt turns a source video into structured multimodal ledgers, graph data, and 9:16 clip renders.

The active extraction path is **DigitalOcean Phase 1**, backed by the code in:
- `backend/do_phase1_worker.py`
- `backend/do_phase1_service/`
- `backend/pipeline/phase_1_do_pipeline.py`

For Phase 1 behavior, those files are the source of truth.

## Pipeline Overview

```text
YouTube URL
  -> Phase 1   DO extraction service (ASR + tracking + face/identity + speaker binding)
  -> Phase 2A  Semantic nodes (Gemini)
  -> Phase 2B  Narrative edges (Gemini)
  -> Phase 3   Multimodal embeddings
  -> Phase 4   Persist to Spanner + GCS
  -> Phase 5   Clip selection / retrieval
  -> Render    Remotion / QA clip renderers
```

## Phase 1 Stack (Current)

### Runtime
- DigitalOcean async extraction API + worker service
- Local pipeline client submits jobs, polls for completion, then materializes compatibility artifacts for downstream phases

### Models and major components
- ASR: `nvidia/parakeet-tdt-1.1b`
- Tracking: `YOLO26s + BoT-SORT`
- Face pipeline: full-frame `SCRFD` face detection on the shared analysis video, then face-track building
- Identity stabilization: `ArcFace/InsightFace` embeddings on face tracks, short-gap face-track propagation, then signature-only attachment for remaining fragments
- Active speaker binding: **LR-ASD primary**, heuristic fallback, `auto` runtime mode for large videos

### Important Phase 1 behavior
- Phase 1 does **not** use TalkNet in the active path.
- Phase 1 does **not** use Google Video Intelligence or `phase_1a_reconcile.py`.
- The worker runs **Parakeet ASR first, then tracking on the same GPU** to avoid CUDA-graph conflicts.
- A shared analysis proxy path is used for tracking, face processing, and LR-ASD when enabled.
- Canonical face observations are built early and reused across clustering, LR-ASD, and final ledgers.
- `face_detections` are real detector-derived face tracks when available; `proxy_face_detections` exist only as a compatibility bridge for older consumers.
- Framing policy metadata currently assumes:
  - `single_person_plus_two_speaker`
  - `shared_two_shot_or_explicit_split`

## Phase 1 Service Flow

### In the DigitalOcean extraction service
1. Download source media from the submitted URL.
2. Convert audio to 16kHz mono WAV.
3. Run the local Phase 1 worker in-process.
4. Inside that worker, run Parakeet ASR, YOLO26s + BoT-SORT tracking, identity clustering, and speaker binding.
5. Build final `phase_1_visual` / `phase_1_audio` payloads.
6. Upload the canonical source video and Phase 1 artifacts to GCS.
7. Persist a contract `v2` manifest and mark the DO job complete.

### In the local pipeline bridge
1. Submit the async DO job.
2. Poll until it succeeds or fails.
3. Fetch the persisted manifest.
4. Re-download the source media locally for downstream compatibility.
5. Materialize:
   - `backend/outputs/phase_1_visual.json`
   - `backend/outputs/phase_1_audio.json`
   - `backend/outputs/phase_1_visual.ndjson`
   - `backend/outputs/phase_1_runtime_controls.json`

That local redownload is still present in code today because later phases and render tooling still expect a local source video alongside the JSON ledgers.

## Prerequisites

- Python 3.11+
- Node.js 18+ and npm
- `ffmpeg` + `ffprobe`
- Google Cloud project access for Vertex AI, Spanner, and GCS
- A running DigitalOcean Phase 1 service if you want to use the active remote extraction path

## Shared Team Resources

- GCP project: `clypt-v2`
- Bucket: `gs://clypt-storage-v2`
- Spanner instance: `clypt-spanner-v2`
- Spanner database: `clypt-graph-db-v2`

## Local Setup

1. Python + Node dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd remotion-render && npm install && cd ..
```

2. Authenticate Google CLIs

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project clypt-v2
```

3. Enable required Google APIs

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  spanner.googleapis.com \
  storage.googleapis.com
```

4. Configure environment variables

Copy [.env.example](/c:/Users/chess/Desktop/Clypt/Clypt-V2/.env.example) into a local untracked env file or export the vars in your shell. The full grouped reference lives in [environment-variables.md](/c:/Users/chess/Desktop/Clypt/Clypt-V2/docs/setup/environment-variables.md).

For the current Senso-backed onboarding flow, the most important local vars are:
- `SENSO_API_KEY`
- `SENSO_CREATOR_PROFILE_PROMPT_ID`
- `YOUTUBE_API_KEY` for richer YouTube lookup/search paths

## Deploy the DO Phase 1 Service

Use the deployment runbook in [docs/deployment/do-phase1-digitalocean.md](docs/deployment/do-phase1-digitalocean.md).

The short version on the droplet is:

```bash
sudo REPO_DIR=/opt/clypt-phase1/repo \
  BRANCH=codex/balanced-hybrid-phase1-contract \
  ENV_FILE=/etc/clypt-phase1/do-phase1.env \
  REQUIREMENTS_FILE=requirements-do-phase1.txt \
  bash scripts/do_phase1/deploy_phase1_service.sh
```

This installs the dedicated Phase 1 dependency bundle, pre-caches the active models, and starts:
- `clypt-phase1-api.service`
- `clypt-phase1-worker.service`

Restart notes that are worth keeping in mind:
- Preferred provisioning target is `atl1 + gpu-h200x1-141gb + gpu-h100x1-base` (`NVIDIA AI/ML Ready`).
- If `atl1` is capacity-blocked, the last known good fallback we actually deployed was `nyc2` with the same `H200 + AI/ML Ready` shape.
- Repo path on droplet: `/opt/clypt-phase1/repo`
- Env file on droplet: `/etc/clypt-phase1/do-phase1.env`
- GCP service account path on droplet: `/etc/clypt-phase1/gcp-sa.json`
- Recommended worker settings for current Phase 1 evals: `DO_PHASE1_WORKER_CONCURRENCY=3`, `DO_PHASE1_GPU_SLOTS=1`

## Run Phase 1 Against a Video URL

```bash
source .venv/bin/activate
DO_PHASE1_BASE_URL=http://<droplet-ip>:8080 \
PHASE1_RUNTIME_PROFILE=podcast_eval \
PHASE1_SPEAKER_BINDING_MODE=lrasd \
PHASE1_TRACKING_MODE=direct \
PHASE1_SHARED_ANALYSIS_PROXY=1 \
.venv/bin/python -c "import asyncio; from backend.pipeline.phase_1_do_pipeline import main; asyncio.run(main('https://www.youtube.com/watch?v=dXUFsDcC0_4'))"
```

This writes:
- `backend/outputs/phase_1_visual.json`
- `backend/outputs/phase_1_audio.json`
- `backend/outputs/phase_1_visual.ndjson`
- `backend/outputs/phase_1_runtime_controls.json`

## Full Pipeline Run

Interactive:

```bash
source .venv/bin/activate
.venv/bin/python backend/pipeline/run_pipeline.py
```

Non-interactive:

```bash
printf '%s\n' 'https://www.youtube.com/watch?v=dXUFsDcC0_4' | .venv/bin/python backend/pipeline/run_pipeline.py
```

## Important Runtime Flags

### Local Phase 1 client
- `DO_PHASE1_BASE_URL`
- `DO_PHASE1_POLL_INTERVAL_SECONDS`
- `DO_PHASE1_TIMEOUT_SECONDS`
- `PHASE1_RUNTIME_PROFILE`
- `PHASE1_SPEAKER_BINDING_MODE`
- `PHASE1_TRACKING_MODE`
- `PHASE1_SHARED_ANALYSIS_PROXY`
- `PHASE1_HEURISTIC_BINDING_ENABLED`

### DO worker / extraction service
- `DO_PHASE1_WORKER_CONCURRENCY`
- `DO_PHASE1_GPU_SLOTS`
- `DO_PHASE1_STATE_ROOT`
- `DO_PHASE1_DB_PATH`
- `DO_PHASE1_OUTPUT_ROOT`
- `DO_PHASE1_LOG_ROOT`
- `CLYPT_SPEAKER_BINDING_MODE`
- `CLYPT_TRACKING_MODE`
- `CLYPT_TRACK_CHUNK_WORKERS`
- `CLYPT_LRASD_BATCH_SIZE`
- `CLYPT_LRASD_PIPELINE_OVERLAP`
- `CLYPT_LRASD_MAX_INFLIGHT`
- rollout gate thresholds in `backend/do_phase1_worker.py`

## Expected Core Artifacts

- `backend/outputs/phase_1_visual.json`
- `backend/outputs/phase_1_audio.json`
- `backend/outputs/phase_1_visual.ndjson`
- `backend/outputs/phase_2a_nodes.json`
- `backend/outputs/phase_2b_narrative_edges.json`
- `backend/outputs/phase_3_embeddings.json`
- `backend/outputs/remotion_payloads_array.json`

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

## Planning Docs

- `docs/planning/01-product-and-demo.md`
- `docs/planning/02-system-architecture.md`
- `docs/planning/03-agents-and-clipping.md`
- `docs/planning/04-data-integrations-and-reference.md`
