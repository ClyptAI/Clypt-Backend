# Data, Integrations, and Reference

See also: [Planning Index](./README.md), [System Architecture](./02-system-architecture.md), [Agents and Clipping](./03-agents-and-clipping.md), [Semantic Graph Source](./Semantic_Graph_Architecture.md)

---
## Primary Integrations

| Integration | Usage |
|---|---|
| YouTube + `yt-dlp` | Source media acquisition in Phase 1A worker |
| Modal (FastAPI + serverless GPU) | Hosts deterministic extraction microservice on A10G/A100 |
| NVIDIA Canary-1B-v2 | High-speed transcription with word-level timestamps and punctuation |
| YOLOv12 + BoT-SORT | Dense 60fps face/person tracking with persistent IDs |
| TalkNCE + LASER + MediaPipe landmarks | Active speaker binding from audio-visual synchrony |
| Gemini 3.1 Pro | Decomposition, edge mapping, clip scoring |
| Multimodal Embeddings (`multimodalembedding@001`) | Node and query vectorization |
| Cloud Spanner | Node/edge relational graph + vector index + property graph |
| Cloud Storage | Intermediate artifacts + per-node tracking payloads |
| Remotion | Final clip rendering |

---
## Core Data Artifacts

| File | Produced By | Consumed By |
|---|---|---|
| `phase_1a_visual.json` | Phase 1A Modal worker | 1B, 3, Remotion tracking fetch |
| `phase_1a_audio.json` | Phase 1A Modal worker | 1B, 3, 4 |
| `phase_1b_nodes.json` | Phase 1B | 1C, 2 |
| `phase_1c_narrative_edges.json` | Phase 1C | 3 |
| `phase_2_embeddings.json` | Phase 2 | 3 |
| `remotion_payloads_array.json` | Phase 4 Auto-Curate | Remotion |
| `remotion_payload.json` | Phase 4 Retrieve | Remotion standalone |

Artifact notes:
- `phase_1a_visual.json` contains dense 60fps BoT-SORT arrays with persistent `track_id` assignments.
- `phase_1a_audio.json` contains NVIDIA Canary-1B-v2 transcript words with direct `speaker_track_id` mapping.
- `phase_1a_speaker_map.json` is removed because speaker-to-track binding is native to Phase 1A.

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
| Modal app endpoint | Modal | Phase 1A extraction webhook target |
| `clypt-preyc` | GCP project | Core cloud project for non-extraction services |
| `clypt-test-bucket` | Cloud Storage | Uploads + tracking JSON |
| `clypt-preyc-db` | Spanner instance | Database host |
| `clypt-db` | Spanner database | Graph + vectors |
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
