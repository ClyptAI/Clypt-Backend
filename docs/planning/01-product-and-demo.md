# Product and Demo

See also: [Planning Index](./README.md), [System Architecture](./02-system-architecture.md), [Agents and Clipping](./03-agents-and-clipping.md), [Data/Integrations](./04-data-integrations-and-reference.md)

## Semantic Graph Pipeline Strategy

**Event Context:** Gemini Live Agent Challenge (Devpost)
**Challenge:** [Gemini Live Agent Challenge: Redefining Interaction: From Static Chatbots to Immersive Experiences](https://geminiliveagentchallenge.devpost.com/)
**Format:** online challenge submission + judging demo narrative

## Current Implementation Scope

- End-to-end pipeline from a source video URL to rendered 9:16 clips
- Phase 1 deterministic grounding runs on a DigitalOcean GPU extraction service
- Gemini owns semantic reasoning in Phases 2A, 2B, and 5
- Spanner-backed graph storage and GCS artifact persistence remain in place
- Remotion / QA render tooling consumes downstream payloads for final clip output

## Product Capabilities

| Capability | What It Does | Backing Stages |
|---|---|---|
| Deterministic multimodal grounding | Produces transcript, tracking, face, identity, and speaker-binding ledgers from the source video | Phase 1 |
| Semantic graph construction | Builds semantic nodes and narrative edges on top of grounded Phase 1 artifacts | Phases 2A + 2B |
| Graph-driven clip generation | Scores moments, selects clip windows, and renders shorts with speaker-aware framing | Phases 3 + 4 + 5 + render |

## Demo Narrative

### How it runs
1. User provides a video URL.
2. `backend/pipeline/phase_1_do_pipeline.py` submits the URL to the DO Phase 1 service.
3. The DO service returns a persisted manifest containing Phase 1 artifacts.
4. The local pipeline materializes compatibility ledgers for downstream phases.
5. Gemini phases build graph structure and clip candidates.
6. Renderers produce 9:16 output clips.

### What to show in the demo
1. **DO extraction output** proving deterministic grounding on the active stack.
2. **`phase_1_visual.json`** with tracking, person detections, face detections, and metrics.
3. **`phase_1_audio.json`** with Parakeet word timings and speaker bindings.
4. **Phase 2A / 2B outputs** showing Gemini reasoning on top of Phase 1 ledgers.
5. **Rendered clips** showing speaker-aware framing behavior.

## Submission Positioning

- Position Clypt as a graph-first multimodal backend with a custom extraction layer.
- Emphasize that the deterministic extraction stage is already separated from the reasoning layer.
- Treat Live API / ADK as future real-time interaction layers on top of the current backend, not as the present extraction runtime.
