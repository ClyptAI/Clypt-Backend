# Clypt — AI-Powered Video Clipping Pipeline

Clypt takes a YouTube URL and produces rendered 9:16 short-form clips by building a semantic graph, storing it in Cloud Spanner, and using Gemini for mechanism decomposition, narrative edge mapping, and clip scoring.

For planning docs, see:
- `planning/01-product-and-demo.md`
- `planning/02-system-architecture.md`
- `planning/03-agents-and-clipping.md`
- `planning/04-data-integrations-and-reference.md`

## Current Migration State

Phase 1A is being migrated to a Modal-hosted GPU extraction service.

This repo has already removed deprecated post-extraction stages:
- `phase_1a_reconcile.py`
- `phase_1a_asd_infer.py`
- `phase_1a_asd.py`
- `phase_1a_fuse.py`

## Prerequisites

- Python 3.12+
- Node.js 18+ and npm
- FFmpeg
- Google Cloud SDK (for Spanner, GCS, Vertex)

```bash
gcloud auth login
gcloud auth application-default login
```

## GCP Setup

The defaults in pipeline scripts assume:
- Project: `clypt-preyc`
- Spanner instance/database: `clypt-preyc-db` / `clypt-db`
- Bucket: `clypt-test-bucket`

Enable required APIs for current non-extraction phases:

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  spanner.googleapis.com \
  storage.googleapis.com
```

Legacy extraction-only APIs (to be removed after Modal cutover):

```bash
gcloud services enable \
  videointelligence.googleapis.com \
  speech.googleapis.com
```

Create resources:

```bash
gcloud storage buckets create gs://clypt-test-bucket --location=us-central1

gcloud spanner instances create clypt-preyc-db \
  --config=regional-us-central1 \
  --description="Clypt dev instance" \
  --nodes=1

gcloud spanner databases create clypt-db \
  --instance=clypt-preyc-db \
  --ddl-file=spannerSchema.sql
```

## Installation

```bash
pip install -r requirements.txt
cd clypt-render-engine && npm install && cd ..
```

## Running the Pipeline

```bash
python3 pipeline/run_pipeline.py
```

The orchestrator runs:
1. Phase 1A extraction
2. FFmpeg re-encode
3. Phase 1B
4. Phase 1C
5. Phase 2
6. Phase 3
7. Phase 4
8. Remotion render

Output clips are written to `clypt-render-engine/out/`.

## Pipeline Phases

| Phase | Script | Description |
|---|---|---|
| 1A | `phase_1a_extract.py` | Deterministic extraction ledger generation |
| FFmpeg | `run_pipeline.py` | Re-encode video for Remotion compatibility |
| 1B | `phase_1b_decompose.py` | Content mechanism decomposition (Gemini multimodal) |
| 1C | `phase_1c_edges.py` | Narrative edge mapping (Gemini text-only) |
| 2 | `phase_2_embed.py` | Multimodal embedding (1408-d vectors) |
| 3 | `phase_3_store.py` | Storage in Spanner + GCS tracking upload |
| 4 | `phase_4_auto_curate.py` | Auto-curator (full-graph sweep + Gemini scoring) |
| Retrieve | `phase_4_retrieve.py` | Query-based retrieval (standalone) |

## Project Structure

```text
Clypt-V2/
├── pipeline/
│   ├── run_pipeline.py
│   ├── phase_1a_extract.py
│   ├── phase_1b_decompose.py
│   ├── phase_1c_edges.py
│   ├── phase_2_embed.py
│   ├── phase_3_store.py
│   ├── phase_4_auto_curate.py
│   └── phase_4_retrieve.py
├── clypt-render-engine/
├── planning/
├── spannerSchema.sql
└── requirements.txt
```

## Cost Notes

Main costs come from:
- Vertex AI (Gemini + embeddings)
- Cloud Spanner (continuous while instance is running)
- Cloud Storage

Delete Spanner instance when not in use:

```bash
gcloud spanner instances delete clypt-preyc-db
```
