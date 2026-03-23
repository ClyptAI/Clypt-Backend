# Agents and Clipping

See also: [Planning Index](./README.md), [System Architecture](./02-system-architecture.md), [Data/Integrations](./04-data-integrations-and-reference.md), [Semantic Graph Source](./Semantic_Graph_Architecture.md)

---
## Agent Roles (Implemented)

The current system uses a hybrid runtime:
- **DO GPU microservice** for deterministic multimodal extraction in Phase 1.
- **Gemini** for semantic reasoning and clip scoring in Phases 2A, 2B, and 5.

| Role | Stage | Runtime | Output |
|---|---|---|---|
| **Deterministic Extraction Worker** | Phase 1 | DO GPU service (Parakeet + YOLO11/BoT-SORT + TalkNet) | `phase_1_visual.json`, `phase_1_audio.json` |
| **Content Mechanism Decomposition** | Phase 2A | Gemini 3.1 Pro | `phase_2a_nodes.json` |
| **Narrative Edge Mapping** | Phase 2B | Gemini 3.1 Pro (text-only) | `phase_2b_narrative_edges.json` |
| **Auto-Curator ClipScoringAgent** | Phase 5 Auto-Curate | Gemini 3.1 Pro | ranked clip payloads |
| **Retrieve ClipScoringAgent** | Phase 5 Retrieve | Gemini 3.1 Pro | query-specific clip payload |

---
## Phase 1 Extraction Mechanics

`backend/pipeline/phase_1_do_pipeline.py` now performs three tasks only:
1. Build webhook request payload from input YouTube URL.
2. Call the DO endpoint and wait for extraction completion.
3. Persist returned visual/audio ledgers for downstream phases.

Inside the DO extraction worker, the stack runs in order:
1. `yt-dlp`
2. NVIDIA Canary-1B-v2 (word-level timestamps + punctuation)
3. YOLO11 + BoT-SORT (dense persistent tracks)
4. TalkNet (active speaker to `track_id` binding)

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
- `clip_start_ms` / `clip_end_ms` from Phase 5 output
- `tracking_uris` merged into `merged_tracking.json`
- `active_speaker_timeline` for speaker-aware camera following
- direct per-frame `x/y` transforms from BoT-SORT trajectories in `ClyptViralShort.tsx`

The renderer no longer depends on additional smoothing logic because Phase 1 tracking data is already dense and smoothed.

---
## What Is Not Yet Reflected Here

- No Gemini Live API runtime loop is wired into this pipeline path yet.
- No ADK bidi-streaming session orchestration is currently part of the scripted phases.

These can be layered on top of the existing semantic graph backend.
