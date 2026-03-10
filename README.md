# Clypt — AI-Powered Video Clipping Pipeline

Clypt takes a YouTube URL and produces rendered 9:16 short-form video clips by building a semantic graph of the video's content, storing it in Google Cloud Spanner, and using Gemini to identify viral moments. The render engine uses Remotion with spring-animated, speaker-aware camera tracking.

For full architecture documentation, see [`planning/Semantic_Graph_Architecture.md`](planning/Semantic_Graph_Architecture.md).

## Prerequisites

- **Python 3.12+**
- **Node.js 18+** and npm
- **FFmpeg** — `brew install ffmpeg` (macOS) or your system's package manager
- **Google Cloud SDK** — [install](https://cloud.google.com/sdk/docs/install), then authenticate:
  ```bash
  gcloud auth login
  gcloud auth application-default login
  ```

## GCP Project Setup

All services run under a single GCP project. The pipeline expects project ID `clypt-preyc` — update the `PROJECT_ID` constants in the pipeline scripts if you use a different one.

### 1. Create the Project

```bash
gcloud projects create clypt-preyc --name="Clypt PreYC"
gcloud config set project clypt-preyc
```

### 2. Enable Required APIs

```bash
gcloud services enable \
  videointelligence.googleapis.com \
  speech.googleapis.com \
  aiplatform.googleapis.com \
  spanner.googleapis.com \
  storage.googleapis.com
```

### 3. Create a Cloud Storage Bucket

```bash
gcloud storage buckets create gs://clypt-test-bucket --location=us-central1
```

Update `GCS_BUCKET` in the pipeline scripts if you use a different bucket name.

### 4. Create a Spanner Instance and Database

```bash
# Create the instance (1 node — sufficient for dev/testing)
gcloud spanner instances create clypt-preyc-db \
  --config=regional-us-central1 \
  --description="Clypt dev instance" \
  --nodes=1

# Create the database with the schema
gcloud spanner databases create clypt-db \
  --instance=clypt-preyc-db \
  --ddl-file=spannerSchema.sql
```

The schema creates two tables (`SemanticClipNode`, `NarrativeEdge`), a ScaNN vector index, and a Spanner property graph. See `spannerSchema.sql` for the full DDL.

### 5. Verify Gemini Model Access

The pipeline uses `gemini-3.1-pro-preview` via the Vertex AI API. Ensure your project has access to this model in the `global` region (used for Gemini calls) and `us-central1` (used for embeddings).

## Installation

### Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `google-cloud-videointelligence` — Video Intelligence API client
- `google-cloud-speech` — Speech-to-Text v2 (Chirp 3) client
- `google-cloud-storage` — GCS client
- `google-genai` — Gemini SDK (unified google-genai)
- `google-cloud-aiplatform` — Vertex AI (multimodal embeddings)
- `google-cloud-spanner` — Spanner client
- `yt-dlp` — YouTube downloader

Optional diarization tuning (helpful for multi-speaker podcasts/interviews):

```bash
export STT_MIN_SPEAKERS=2
export STT_MAX_SPEAKERS=6
```

### Remotion (Node.js) Dependencies

```bash
cd clypt-render-engine
npm install
cd ..
```

## Running the Pipeline

```bash
python3 pipeline/run_pipeline.py
```

The orchestrator will:

1. Prompt for a YouTube URL
2. Run all phases sequentially (1A → 1A-R → FFmpeg → 1B → 1C → 2 → 3 → 4)
3. Symlink outputs into the Remotion project
4. Fetch spatial tracking data from GCS
5. Render each clip to `clypt-render-engine/out/`

Output clips land in `clypt-render-engine/out/` as `clip-1.mp4`, `clip-2.mp4`, etc.

### Pipeline Phases

| Phase | Script | Description |
|---|---|---|
| 1A | `phase_1a_extract.py` | Deterministic extraction (Video Intelligence + STT) |
| 1A-R | `phase_1a_reconcile.py` | Speaker-to-face reconciliation |
| 1A-ASD-INFER | `phase_1a_asd_infer.py` | True ASD model runner (TalkNet backend, optional) |
| 1A-ASD | `phase_1a_asd.py` | ASD timeline mapping + confidence gating |
| FFmpeg | (orchestrator) | Re-encode video for Remotion compatibility |
| 1B | `phase_1b_decompose.py` | Content mechanism decomposition (Gemini multimodal) |
| 1C | `phase_1c_edges.py` | Narrative edge mapping (Gemini text-only) |
| 2 | `phase_2_embed.py` | Multimodal embedding (1408-d vectors) |
| 3 | `phase_3_store.py` | Storage in Spanner + GCS tracking upload |
| 4 | `phase_4_auto_curate.py` | Auto-curator (full-graph sweep + Gemini scoring) |
| Render | (orchestrator) | Remotion render to 9:16 MP4 clips |

### Standalone Query-Based Retrieval

`phase_4_retrieve.py` is a standalone script (not part of the orchestrator) that finds a clip matching a natural language query via hybrid vector search + graph traversal:

```bash
# Edit USER_QUERY in the script, then:
python3 pipeline/phase_4_retrieve.py
```

## Optional: Local LoCoNet ASD Prototype

You can optionally inject a frame-level active-speaker timeline from a local ASD model (for example LoCoNet) without changing cloud services.

1. Place raw ASD output at:
   - `outputs/phase_1a_loconet_raw.json`
2. Run:
   ```bash
   python3 pipeline/phase_1a_asd.py
   ```
3. This writes:
   - `outputs/phase_1a_active_speaker_timeline.json`
4. Re-run render fetch + render:
   ```bash
   python3 -c "from pipeline.run_pipeline import setup_render_engine, run_fetch_tracking, run_remotion_render; setup_render_engine(); run_fetch_tracking(); run_remotion_render()"
   ```

`fetch_tracking.js` will automatically slice this timeline per clip and pass it to Remotion. If the file is missing, the pipeline falls back to existing speaker timelines.

Note: if `outputs/phase_1a_loconet_raw.json` is missing, `phase_1a_asd.py` now auto-generates a bootstrap raw file from STT diarization + face tracks, then builds `phase_1a_active_speaker_timeline.json`. This keeps ASD-enabled tracking running by default, but true LoCoNet quality requires replacing that raw file with real model predictions.

## Optional: True TalkNet ASD Runner

The repo now includes `pipeline/phase_1a_asd_infer.py`, which runs TalkNet and writes:

- `outputs/phase_1a_loconet_raw.json` (model-based frame detections)

Then `phase_1a_asd.py` converts that into:

- `outputs/phase_1a_active_speaker_timeline.json`

### Setup

1. Clone TalkNet:
   ```bash
   mkdir -p third_party
   git clone https://github.com/TaoRuijie/TalkNet-ASD third_party/TalkNet-ASD
   ```
2. Create/install TalkNet environment (their dependencies are separate from this repo):
   ```bash
   cd third_party/TalkNet-ASD
   pip install -r requirement.txt
   cd ../..
   ```
3. Point pipeline to that environment:
   ```bash
   export TALKNET_REPO_PATH=/absolute/path/to/third_party/TalkNet-ASD
   export TALKNET_PYTHON_BIN=/absolute/path/to/talknet-env/bin/python
   ```
4. Run model inference:
   ```bash
   python3 pipeline/phase_1a_asd_infer.py
   python3 pipeline/phase_1a_asd.py
   ```

### Notes

- TalkNet demo code is CUDA-first. If your machine has no supported NVIDIA GPU, run this phase on a GPU VM (or CPU-patched TalkNet fork).
- If `phase_1a_asd_infer.py` fails in the orchestrator, the pipeline continues with the bootstrap fallback automatically.

## Project Structure

```
Clypt-PreYC/
├── pipeline/                          # Python pipeline scripts
│   ├── run_pipeline.py                # Orchestrator (entry point)
│   ├── phase_1a_extract.py            # Deterministic extraction
│   ├── phase_1a_reconcile.py          # Speaker-to-face reconciliation
│   ├── phase_1b_decompose.py          # Content mechanism decomposition
│   ├── phase_1c_edges.py              # Narrative edge mapping
│   ├── phase_2_embed.py               # Multimodal embedding
│   ├── phase_3_store.py               # Spanner + GCS storage
│   ├── phase_4_auto_curate.py         # Auto-curator
│   └── phase_4_retrieve.py            # Query-based retrieval (standalone)
├── clypt-render-engine/               # Remotion rendering project
│   ├── src/
│   │   ├── Root.tsx                   # Composition registration
│   │   ├── ClyptViralShort.tsx        # Video component (spring camera + speaker tracking)
│   │   └── index.ts
│   ├── scripts/
│   │   └── fetch_tracking.js          # GCS tracking data fetcher
│   ├── public/                        # Video + tracking data (gitignored, symlinked at runtime)
│   └── out/                           # Rendered MP4 clips (gitignored)
├── outputs/                           # Intermediate JSON files (gitignored)
├── downloads/                         # Downloaded media (gitignored)
├── planning/                          # Architecture docs
│   └── Semantic_Graph_Architecture.md
├── spannerSchema.sql                  # Spanner DDL
├── requirements.txt                   # Python dependencies
└── .gitignore
```

## Updating GCP Resource Names

If you use different resource names than the defaults, update these constants across the pipeline scripts:

| Constant | Default | Files |
|---|---|---|
| `PROJECT_ID` | `clypt-preyc` | All pipeline scripts |
| `GCS_BUCKET` | `clypt-test-bucket` | `phase_1a_extract.py`, `phase_3_store.py` |
| `SPANNER_INSTANCE` | `clypt-preyc-db` | `phase_3_store.py`, `phase_4_auto_curate.py`, `phase_4_retrieve.py` |
| `SPANNER_DATABASE` | `clypt-db` | `phase_3_store.py`, `phase_4_auto_curate.py`, `phase_4_retrieve.py` |
| `GEMINI_MODEL` | `gemini-3.1-pro-preview` | `phase_1b_decompose.py`, `phase_1c_edges.py`, `phase_4_auto_curate.py`, `phase_4_retrieve.py` |

## Cost Considerations

The pipeline uses several billable GCP services. For a single video run:

- **Video Intelligence API** — 5 feature detections on the video
- **Speech-to-Text v2** — Chirp 3 transcription with diarization
- **Vertex AI** — Gemini 3.1 Pro calls (Phases 1B, 1C, 4) + multimodal embeddings (Phase 2)
- **Cloud Spanner** — 1 node instance (this is the most expensive resource at ~$0.90/hr for a single-node instance)
- **Cloud Storage** — minimal storage for video + tracking files

Spanner is billed continuously while the instance exists. Consider deleting the instance after testing:

```bash
gcloud spanner instances delete clypt-preyc-db
```
