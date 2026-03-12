# Clypt: Creator Intelligence and Clipping

Clypt takes a YouTube URL and produces rendered 9:16 short-form clips. It combines a Modal GPU extraction stack (ASR + tracking + speaker binding) with Gemini-powered semantic graph reasoning and Remotion rendering.

## Architecture at a Glance

```text
YouTube URL
  -> Phase 1 (Modal GPU): ASR + tracking + speaker binding
  -> Phase 2A: Gemini mechanism decomposition
  -> Phase 2B: Gemini narrative edges
  -> Phase 3: multimodal embeddings
  -> Phase 4: Spanner + GCS storage
  -> Phase 5: Gemini clip scoring/selection
  -> Remotion render (9:16 clips)
```

### Phase 1 (Modal worker)

- ASR: NVIDIA Parakeet-TDT-1.1B
- Visual tracking: YOLO11 + BoT-SORT
- Face stack: InsightFace
- Active speaker binding: TalkNet

The local client is `pipeline/phase_1_modal_pipeline.py` and the GPU service is `modal_worker.py`.

## Prerequisites

- Python 3.11+
- Node.js 18+ and npm
- FFmpeg
- Modal account/CLI auth
- Google Cloud auth for downstream phases (Spanner, GCS, Vertex)

```bash
gcloud auth login
gcloud auth application-default login
```

## Install

```bash
pip install -r requirements.txt
cd clypt-render-engine && npm install && cd ..
```

Optional UI app:

```bash
cd cortex-ui && npm install && cd ..
```

## Deploy Modal Worker

```bash
modal deploy modal_worker.py
```

## Run

### Run full pipeline

```bash
python3 pipeline/run_pipeline.py
```

### Run only Phase 1 extraction

```bash
python3 pipeline/phase_1_modal_pipeline.py
```

## Outputs

- `downloads/video.mp4` and `downloads/audio_16k.wav`
- `outputs/phase_1_visual.json`
- `outputs/phase_1_audio.json`
- `outputs/phase_2a_nodes.json`
- `outputs/phase_2b_narrative_edges.json`
- `outputs/phase_3_embeddings.json`
- `outputs/remotion_payloads_array.json`
- Rendered clips: `clypt-render-engine/out/`

## Pipeline Scripts

| Phase | Script |
|---|---|
| 1 | `pipeline/phase_1_modal_pipeline.py` |
| 2A | `pipeline/phase_2a_make_nodes.py` |
| 2B | `pipeline/phase_2b_draw_edges.py` |
| 3 | `pipeline/phase_3_multimodal_embeddings.py` |
| 4 | `pipeline/phase_4_store_graph.py` |
| 5 (auto-curate) | `pipeline/phase_5_auto_curate.py` |
| 5 (retrieve) | `pipeline/phase_5_retrieve.py` |
| orchestrator | `pipeline/run_pipeline.py` |

## GCP Defaults Used by Scripts

- Project: `clypt-v2`
- Spanner instance/database: `clypt-spanner-v2` / `clypt-graph-db-v2`
- Bucket: `clypt-storage-v2`

Enable APIs:

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  spanner.googleapis.com \
  storage.googleapis.com
```

Create resources:

```bash
gcloud storage buckets create gs://clypt-storage-v2 --location=us-central1

gcloud spanner instances create clypt-spanner-v2 \
  --config=regional-us-central1 \
  --description="Clypt dev instance" \
  --nodes=1

gcloud spanner databases create clypt-graph-db-v2 \
  --instance=clypt-spanner-v2 \
  --ddl-file=spannerSchema.sql
```

## Planning Docs

- `planning/01-product-and-demo.md`
- `planning/02-system-architecture.md`
- `planning/03-agents-and-clipping.md`
- `planning/04-data-integrations-and-reference.md`
