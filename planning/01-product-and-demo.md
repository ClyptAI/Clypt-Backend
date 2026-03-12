# Product and Demo

See also: [Planning Index](./README.md), [System Architecture](./02-system-architecture.md), [Agents and Clipping](./03-agents-and-clipping.md), [Data/Integrations](./04-data-integrations-and-reference.md)

---
## Semantic Graph Pipeline Strategy

**Event Context:** Gemini Live Agent Challenge (Devpost)
**Challenge:** [Gemini Live Agent Challenge: Redefining Interaction: From Static Chatbots to Immersive Experiences](https://geminiliveagentchallenge.devpost.com/)
**Format:** Online challenge submission + judging demo narrative

**Challenge-Relevant Tech:**
- **Gemini Live API** - challenge-mandated real-time multimodal interaction layer
- **Google ADK (Agent Development Kit)** - challenge-mandated agent framework option
- **Gemini multimodal + interleaved outputs** - required capability across tracks
- **Google Cloud deployment** - required for judging proof

**Current Implementation Scope (with new Phase 1):**
- End-to-end semantic graph pipeline from a YouTube URL to rendered 9:16 clips
- Deterministic grounding is now served by a Modal GPU microservice, not managed GCP extraction APIs
- Gemini continues to own semantic reasoning (Phase 2A/2B) and clip scoring (Phase 5)
- Spanner-backed graph + vector retrieval + Remotion rendering remain unchanged

---
## Product Capabilities

| Capability | What It Does | Backing Stages |
|---|---|---|
| **SOTA Deterministic Grounding** | Produces dense multimodal ledgers using Parakeet + YOLO11/BoT-SORT + TalkNet | Phase 1 (Modal worker) |
| **Semantic Graph Construction** | Builds semantic nodes and narrative edges with structured outputs | Phase 2A + 2B |
| **Graph-Driven Clip Generation** | Scores chapters, selects viral windows, renders clips with speaker-aware tracking | Phase 3 + 4 + 5 + Remotion |

---
## Demo Narrative

### How It Runs
1. User provides a YouTube URL.
2. `pipeline/phase_1_modal_pipeline.py` sends the URL to the Modal webhook and receives extraction JSON.
3. Pipeline continues through Phase 2A -> 2B -> 3 -> 4 -> 5.
4. Auto-curation outputs clip payloads.
5. Remotion renders 9:16 clips into `clypt-render-engine/out/`.

### What to Show in Demo
1. **Modal extraction output** proving deterministic grounding from SOTA models.
2. **`phase_1_visual.json`** with dense 60fps BoT-SORT track arrays and persistent `track_id` values.
3. **`phase_1_audio.json`** with NVIDIA Canary-1B-v2 word-level timestamps directly mapped to `speaker_track_id`.
4. **Semantic node and edge outputs** (`phase_2a_nodes.json`, `phase_2b_narrative_edges.json`) showing Gemini reasoning on top of grounded ledgers.
5. **Final rendered clips** where camera movement follows exact track coordinates.

### Submission Positioning
- Position this as a graph-first multimodal backend with a custom SOTA extraction layer.
- Emphasize that Live API and ADK are the real-time interaction layer to be added on top of this pipeline.
