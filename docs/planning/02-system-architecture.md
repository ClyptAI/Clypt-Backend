# System Architecture

See also: [Planning Index](./README.md), [Product and Demo](./01-product-and-demo.md), [Agents and Clipping](./03-agents-and-clipping.md), [Data/Integrations](./04-data-integrations-and-reference.md)

---
## Architecture Overview

Source of truth: [Semantic_Graph_Architecture.md](./Semantic_Graph_Architecture.md)

The implemented system is a sequential pipeline:

```text
YouTube URL
  -> Phase 1: DO GPU Deterministic Extraction
  -> FFmpeg Re-encode
  -> Phase 2A: Content Mechanism Decomposition
  -> Phase 2B: Narrative Edge Mapping
  -> Phase 3: Multimodal Embedding
  -> Phase 4: Storage & Graph Binding
  -> Phase 5: Auto-Curate / Retrieve
  -> Remotion Render
```

Phase 1 is split across two execution environments:
- `backend/pipeline/phase_1_do_pipeline.py` in the main pipeline acts as a lightweight HTTP client.
- A DO GPU worker performs the actual extraction workloads.

---
## DO Phase 1 Pipeline

The DO extraction service (Python/FastAPI on GPU) runs these steps sequentially:
1. `yt-dlp` downloads and prepares the video inside the worker environment.
2. NVIDIA Canary-1B-v2 performs high-speed ASR with precise word-level timestamps and punctuation.
3. YOLO11 + BoT-SORT runs dense, full-framerate detection/tracking and produces persistent face/person track IDs.
4. TalkNet fuses audio and visual signals, binding spoken words directly to BoT-SORT `track_id` values.
5. Worker returns `phase_1_visual.json` and `phase_1_audio.json` payloads to the calling pipeline.

This removes any need for post-hoc speaker reconciliation.

---
## Design Principles

1. **Math first, reasoning second**: deterministic timestamps and spatial trajectories are established before Gemini reasoning.
2. **SOTA extraction isolation**: all GPU-heavy multimodal grounding is isolated in a scalable DO worker.
3. **Two-call graph generation**: Phase 2A (multimodal decomposition) + Phase 2B (text-only edge mapping) to control token budgets and output quality.
4. **Spanner multi-model usage**: relational + vector index + property graph in one ACID store.
5. **Late-fusion embeddings**: fused text + video vectors per semantic node.

---
## Runtime Components

| Component | Role |
|---|---|
| `backend/pipeline/run_pipeline.py` | Orchestrates all phases and rendering handoff |
| `backend/pipeline/phase_1_do_pipeline.py` | Sends job request to the DO Phase 1 service and writes returned ledgers |
| `modal` GPU service (FastAPI) | Runs Parakeet + YOLO11/BoT-SORT + TalkNet extraction stack |
| `backend/pipeline/phase_2a_make_nodes.py` | Gemini 3.1 Pro semantic node decomposition |
| `backend/pipeline/phase_2b_draw_edges.py` | Gemini 3.1 Pro narrative edge mapping |
| `backend/pipeline/phase_3_multimodal_embeddings.py` | Vertex multimodal embedding fusion |
| `backend/pipeline/phase_4_store_graph.py` | Spanner graph writes + GCS tracking payloads |
| `backend/pipeline/phase_5_auto_curate.py` | Full-graph chapter scoring and clip selection |
| `backend/pipeline/phase_5_retrieve.py` | Query-based hybrid retrieval (standalone) |
| `remotion-render/` | Remotion 9:16 clip rendering with speaker-aware tracking |

---
## Render Path

1. Phase 5 writes Remotion payloads (`remotion_payloads_array.json` or `remotion_payload.json`).
2. `fetch_tracking.js` downloads per-node tracking JSON from GCS and merges frame tracks.
3. `Root.tsx` registers per-clip compositions.
4. `ClyptViralShort.tsx` applies direct coordinate transforms from BoT-SORT tracking (`transform: translate(x, y)`).

---
## Deployment Assumptions

- Phase 1 extraction service runs on DigitalOcean GPUs.
- Python pipeline still runs with ADC-authenticated access for GCS, Spanner, and Vertex.
- Storage: `gs://clypt-storage-v2`.
- Database: Spanner instance `clypt-spanner-v2`, database `clypt-graph-db-v2`.
- Render engine is local/CI Node runtime using Remotion.
