# System Architecture

See also: [Planning Index](./README.md), [Product and Demo](./01-product-and-demo.md), [Agents and Clipping](./03-agents-and-clipping.md), [Data/Integrations](./04-data-integrations-and-reference.md)

---
## Architecture Overview

Source of truth: [Semantic_Graph_Architecture.md](./Semantic_Graph_Architecture.md)

The implemented system is a sequential pipeline:

```text
YouTube URL
  -> Phase 1A: Modal GPU Deterministic Extraction (webhook)
  -> FFmpeg Re-encode
  -> Phase 1B: Content Mechanism Decomposition
  -> Phase 1C: Narrative Edge Mapping
  -> Phase 2: Multimodal Embedding
  -> Phase 3: Storage & Graph Binding
  -> Phase 4: Auto-Curate / Retrieve
  -> Remotion Render
```

Phase 1A is now split across two execution environments:
- `pipeline/phase_1a_extract.py` in the main pipeline acts as a lightweight HTTP client.
- A Modal serverless GPU worker performs the actual extraction workloads.

---
## Modal Phase 1A Pipeline

The Modal microservice (Python/FastAPI on A10G/A100) runs these steps sequentially:
1. `yt-dlp` downloads and prepares the video inside the worker environment.
2. NVIDIA Canary-1B-v2 performs high-speed ASR with precise word-level timestamps and punctuation.
3. YOLOv12 + BoT-SORT runs dense, full-framerate detection/tracking and produces persistent face/person track IDs.
4. TalkNCE + LASER (with MediaPipe lip landmarks) fuses audio and visual signals, binding spoken words directly to BoT-SORT `track_id` values.
5. Worker returns `phase_1a_visual.json` and `phase_1a_audio.json` payloads to the calling pipeline.

This removes any need for post-hoc speaker reconciliation.

---
## Design Principles

1. **Math first, reasoning second**: deterministic timestamps and spatial trajectories are established before Gemini reasoning.
2. **SOTA extraction isolation**: all GPU-heavy multimodal grounding is isolated in a scalable Modal worker.
3. **Two-call graph generation**: Phase 1B (multimodal decomposition) + Phase 1C (text-only edge mapping) to control token budgets and output quality.
4. **Spanner multi-model usage**: relational + vector index + property graph in one ACID store.
5. **Late-fusion embeddings**: fused text + video vectors per semantic node.

---
## Runtime Components

| Component | Role |
|---|---|
| `pipeline/run_pipeline.py` | Orchestrates all phases and rendering handoff |
| `pipeline/phase_1a_extract.py` | Sends webhook request to Modal and writes returned ledgers |
| `modal` GPU service (FastAPI) | Runs NVIDIA Canary-1B-v2 + YOLOv12/BoT-SORT + TalkNCE/LASER extraction stack |
| `pipeline/phase_1b_decompose.py` | Gemini 3.1 Pro semantic node decomposition |
| `pipeline/phase_1c_edges.py` | Gemini 3.1 Pro narrative edge mapping |
| `pipeline/phase_2_embed.py` | Vertex multimodal embedding fusion |
| `pipeline/phase_3_store.py` | Spanner graph writes + GCS tracking payloads |
| `pipeline/phase_4_auto_curate.py` | Full-graph chapter scoring and clip selection |
| `pipeline/phase_4_retrieve.py` | Query-based hybrid retrieval (standalone) |
| `clypt-render-engine/` | Remotion 9:16 clip rendering with speaker-aware tracking |

---
## Render Path

1. Phase 4 writes Remotion payloads (`remotion_payloads_array.json` or `remotion_payload.json`).
2. `fetch_tracking.js` downloads per-node tracking JSON from GCS and merges frame tracks.
3. `Root.tsx` registers per-clip compositions.
4. `ClyptViralShort.tsx` applies direct coordinate transforms from BoT-SORT tracking (`transform: translate(x, y)`).

---
## Deployment Assumptions

- Phase 1A extraction service runs on Modal serverless GPUs (A10G/A100).
- Python pipeline still runs with ADC-authenticated access for GCS, Spanner, and Vertex.
- Storage: `gs://clypt-test-bucket`.
- Database: Spanner instance `clypt-preyc-db`, database `clypt-db`.
- Render engine is local/CI Node runtime using Remotion.
