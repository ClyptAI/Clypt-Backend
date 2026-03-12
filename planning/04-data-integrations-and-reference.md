# Data, Integrations, and Reference

See also: [Planning Index](./README.md), [System Architecture](./02-system-architecture.md), [Agents and Clipping](./03-agents-and-clipping.md), [Semantic Graph Source](./Semantic_Graph_Architecture.md)

---
## Primary Integrations

| Integration | Usage |
|---|---|
| YouTube + `yt-dlp` | Source media acquisition in Phase 1 worker |
| Modal (FastAPI + serverless GPU) | Hosts deterministic extraction microservice |
| NVIDIA Canary-1B-v2 | High-speed transcription with word-level timestamps and punctuation |
| YOLO11 + BoT-SORT | Dense face/person tracking with persistent IDs |
| TalkNet + InsightFace | Active speaker binding from audio-visual synchrony |
| Gemini 3.1 Pro | Decomposition, edge mapping, clip scoring |
| Multimodal Embeddings (`multimodalembedding@001`) | Node and query vectorization |
| Cloud Spanner | Node/edge relational graph + vector index + property graph |
| Cloud Storage | Intermediate artifacts + per-node tracking payloads |
| Remotion | Final clip rendering |

---
## Core Data Artifacts

| File | Produced By | Consumed By |
|---|---|---|
| `phase_1_visual.json` | Phase 1 Modal worker | 2A, 4, Remotion tracking fetch |
| `phase_1_audio.json` | Phase 1 Modal worker | 2A, 4, 5 |
| `phase_2a_nodes.json` | Phase 2A | 2B, 3 |
| `phase_2b_narrative_edges.json` | Phase 2B | 4 |
| `phase_3_embeddings.json` | Phase 3 | 4 |
| `remotion_payloads_array.json` | Phase 5 Auto-Curate | Remotion |
| `remotion_payload.json` | Phase 5 Retrieve | Remotion standalone |

Artifact notes:
- `phase_1_visual.json` contains dense 60fps BoT-SORT arrays with persistent `track_id` assignments.
- `phase_1_audio.json` contains NVIDIA Canary-1B-v2 transcript words with direct `speaker_track_id` mapping.
- `phase_1a_speaker_map.json` is removed because speaker-to-track binding is native to Phase 1.

---
## Spanner Model

### `SemanticClipNode`
- `node_id`
- `start_time_ms`, `end_time_ms`
- transcript and mechanism fields
- `embedding` as `ARRAY<FLOAT32>(vector_length=>1408)`
- `spatial_tracking_uri`

### `NarrativeEdge`
- `edge_id`
- `from_node_id`, `to_node_id`
- edge label/classification/confidence

### Index / Graph
- `ClyptSemanticIndex` (ScaNN cosine vector index)
- `ClyptGraph` (property graph over nodes + edges)

---
## Infrastructure Summary

| Resource | Service | Purpose |
|---|---|---|
| Modal app endpoint | Modal | Phase 1 extraction webhook target |
| `clypt-v2` | GCP project | Core cloud project for non-extraction services |
| `clypt-storage-v2` | Cloud Storage | Uploads + tracking JSON |
| `clypt-spanner-v2` | Spanner instance | Database host |
| `clypt-graph-db-v2` | Spanner database | Graph + vectors |
| `gemini-3.1-pro-preview` | Vertex AI | LLM reasoning stages |
| `multimodalembedding@001` | Vertex AI | 1408-d embeddings |

---
## Challenge-Relevant References

- [Gemini Live API](https://ai.google.dev/gemini-api/docs/live)
- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [ADK Bidi-Streaming Dev Guide](https://google.github.io/adk-docs/streaming/dev-guide/part1/)
- [Live API Notebooks/Apps (Google Cloud)](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/gemini/multimodal-live-api)
- [Gemini API Docs](https://ai.google.dev/gemini-api/docs)
- [Challenge FAQ](https://geminiliveagentchallenge.devpost.com/details/faqs)

---
## Runtime Dependencies

Python (`requirements.txt`):
- `modal`
- `google-cloud-storage`
- `google-genai`
- `google-cloud-aiplatform`
- `google-cloud-spanner`
- `yt-dlp`

Node (`clypt-render-engine/`):
- Remotion render stack (`npm install` in render engine directory)
