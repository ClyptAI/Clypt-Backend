# Agents and Clipping

See also: [Planning Index](./README.md), [System Architecture](./02-system-architecture.md), [Data/Integrations](./04-data-integrations-and-reference.md)

## Agent Roles (Current)

The current system uses a hybrid runtime:
- **DO GPU extraction service** for deterministic Phase 1 grounding
- **Gemini** for semantic reasoning and clip scoring in later phases

| Role | Stage | Runtime | Output |
|---|---|---|---|
| Deterministic extraction worker | Phase 1 | DO GPU service | `phase_1_visual.json`, `phase_1_audio.json`, manifest artifacts |
| Content mechanism decomposition | Phase 2A | Gemini | `phase_2a_nodes.json` |
| Narrative edge mapping | Phase 2B | Gemini | `phase_2b_narrative_edges.json` |
| Clip scoring / auto-curation | Phase 5 | Gemini | ranked clip payloads |
| Retrieve clip scoring | Phase 5 retrieve | Gemini | query-specific clip payload |

## Phase 1 Extraction Mechanics

Inside the worker, the stack is currently:
1. Parakeet ASR
2. YOLOv26-seg + ByteTrack tracking
3. early face observations + identity features
4. global identity clustering
5. speaker binding (LR-ASD for `lrasd` / `auto` / `shared_analysis_proxy` resolutions; explicit `heuristic` mode; whole-job heuristic after LR-ASD `None` gated by `CLYPT_SPEAKER_BINDING_HEURISTIC_FALLBACK`)
6. final visual/audio ledger construction

Notes:
- LR-ASD is the primary audiovisual speaker-binding model path when the resolved mode is not `heuristic`.
- Heuristic binding remains available as an explicit mode and as an opt-in fallback when LR-ASD returns `None`.
- Face observations are reused across clustering, LR-ASD, and final ledgers.

## Clipping Logic

### Auto-curate mode
1. Load nodes + edges.
2. Build narrative chapters via graph connectivity.
3. Score windows for hook, payoff, pacing, and clip worthiness.
4. Emit render-ready payloads.

### Retrieve mode
1. Embed the query.
2. Find anchor nodes via vector search.
3. Expand through the local narrative neighborhood.
4. Score boundaries and emit focused clip payloads.

## Current Framing Policy Inputs

Phase 1 runtime controls currently encode:
- `single_person_plus_two_speaker`
- `shared_two_shot_or_explicit_split`

That means downstream renderers can distinguish between:
- single-person framing
- two-speaker shared shots
- two-speaker explicit split layouts

## Remotion / QA Construction

Current clip rendering consumes:
- clip timing from later phases
- Phase 1 tracking and speaker-binding outputs
- composition metadata and per-frame geometry

The QA renderer and Remotion-facing tooling are still evolving, but they both assume Phase 1 is the deterministic source for geometry and speaker-aware framing inputs.
