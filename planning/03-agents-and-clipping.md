# Agents and Clipping

See also: [Planning Index](./README.md), [System Architecture](./02-system-architecture.md), [Data/Integrations](./04-data-integrations-and-reference.md), [Semantic Graph Source](./Semantic_Graph_Architecture.md)

---
## Agent Roles (Implemented)

The current system uses a hybrid runtime:
- **Modal GPU microservice** for deterministic multimodal extraction in Phase 1A.
- **Gemini** for semantic reasoning and clip scoring in Phases 1B, 1C, and 4.

| Role | Stage | Runtime | Output |
|---|---|---|---|
| **Deterministic Extraction Worker** | Phase 1A | Modal (NVIDIA Canary-1B-v2 + YOLOv12/BoT-SORT + TalkNCE/LASER) | `phase_1a_visual.json`, `phase_1a_audio.json` |
| **Content Mechanism Decomposition** | Phase 1B | Gemini 3.1 Pro | `phase_1b_nodes.json` |
| **Narrative Edge Mapping** | Phase 1C | Gemini 3.1 Pro (text-only) | `phase_1c_narrative_edges.json` |
| **Auto-Curator ClipScoringAgent** | Phase 4 Auto-Curate | Gemini 3.1 Pro | ranked clip payloads |
| **Retrieve ClipScoringAgent** | Phase 4 Retrieve | Gemini 3.1 Pro | query-specific clip payload |

---
## Phase 1A Extraction Mechanics

`pipeline/phase_1a_extract.py` now performs three tasks only:
1. Build webhook request payload from input YouTube URL.
2. Call Modal endpoint and wait for extraction completion.
3. Persist returned visual/audio ledgers for downstream phases.

Inside Modal, the extraction stack runs in order:
1. `yt-dlp`
2. NVIDIA Canary-1B-v2 (word-level timestamps + punctuation)
3. YOLOv12 + BoT-SORT (dense 60fps persistent tracks)
4. TalkNCE + LASER + lip landmarks (active speaker to `track_id` binding)

No separate reconciliation phase exists after extraction.

---
## Clipping Logic

### Auto-Curate Mode
1. Load all nodes + edges from Spanner.
2. Build narrative chapters via graph connectivity.
3. Score each chapter with ClipScoringAgent using hook/payoff/pacing criteria.
4. Keep clips scoring 85+, or fallback to top 3.
5. Emit Remotion-ready payloads with `active_speaker_timeline` built from `speaker_track_id` bindings.

### Retrieve Mode
1. Embed query into 1408-d vector.
2. Find anchor node via ScaNN cosine search.
3. Expand with 1-hop graph neighbors.
4. Score the local sub-graph for optimal boundaries.
5. Emit single Remotion payload.

---
## Scoring Criteria (Current)

Clip scoring prompt evaluates:
- **Hook strength** in opening moments
- **Payoff quality** based on semantic mechanisms
- **Narrative coherence** from included nodes
- **Pacing and target duration** (short-form friendly)

---
## Remotion Clip Construction

The clip renderer uses:
- `clip_start_ms` / `clip_end_ms` from Phase 4 output
- `tracking_uris` merged into `merged_tracking.json`
- `active_speaker_timeline` for speaker-aware camera following
- direct per-frame `x/y` transforms from BoT-SORT trajectories in `ClyptViralShort.tsx`

The renderer no longer depends on additional smoothing logic because Phase 1A tracking data is already dense and smoothed.

---
## What Is Not Yet Reflected Here

- No Gemini Live API runtime loop is wired into this pipeline path yet.
- No ADK bidi-streaming session orchestration is currently part of the scripted phases.

These can be layered on top of the existing semantic graph backend.
