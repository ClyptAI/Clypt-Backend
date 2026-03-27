# Overlap-Aware Speaker Binding and Speed Pass Design

**Status:** Approved for implementation

## Goal
Improve Phase 1 speaker binding quality and throughput by making the analysis proxy the default, pruning weak LR-ASD candidates, reusing turn-level LR-ASD support, adding explicit overlap artifacts, and resolving overlap follow decisions in a post-Phase-1 Gemini pass while keeping overlays truthful about all active speakers.

## Current Context
The existing Phase 1 pipeline already has:
- shared analysis proxy support in [`/Users/rithvik/CascadeProjects/Clypt-V2/backend/do_phase1_worker.py`](/Users/rithvik/CascadeProjects/Clypt-V2/backend/do_phase1_worker.py)
- pyannote-backed `audio_speaker_turns`
- turn-to-local-track binding helpers
- local binding streams behind `CLYPT_EXPERIMENT_LOCAL_CLIP_BINDINGS`
- single-winner speaker candidate debug entries
- debug renderers that can show the chosen raw/follow speaker

The main remaining problems are:
- LR-ASD still spends work on too many candidate windows
- the word-level binding loop still recomputes local competition more often than necessary
- overlap turns are preserved in diarization but mostly collapsed before render/follow
- the renderer can only show a single chosen speaker instead of all active speakers
- CPU-side crop/resize/grayscale preprocessing still adds avoidable overhead before LR-ASD inference

## Non-Goals
- Replacing LR-ASD with an LLM
- Replacing pyannote or Parakeet
- Shipping GPU-native decode in the first correctness pass if it destabilizes the pipeline
- Making overlap adjudication part of the core Phase 1 binding loop

## Requirements

### 1. 1080p Analysis Proxy Default
Phase 1 analysis should default to a shared proxy with long edge 1920 unless explicitly disabled. Tracking, face analysis, diarization-aligned LR-ASD, and turn-level speaker binding should all operate on the analysis proxy. Final render should continue to use original source media.

### 2. Pre-LR-ASD Candidate Pruning
Before chunking tracks for LR-ASD, each local track must pass conservative eligibility gates. At minimum, gates should consider:
- minimum visible duration / frame count
- overlap with speech or diarized turns
- minimum median bounding-box area
- minimum track quality
- body-candidate hard rejects
- optional cap on maximum candidates per diarized turn or frame neighborhood

The pruning logic must favor false positives over false negatives; it should remove obvious junk work without excluding strong visible speakers.

### 3. Turn-Level LR-ASD Reuse
LR-ASD support should be accumulated per diarized turn so that the final word-level binding reuses turn-level ownership when visual evidence remains consistent. Word-level logic may still override a reused turn owner if there is a strong contrary local event. Ambiguous turns must remain explicit rather than being forced into a confident owner.

### 4. Explicit Overlap Artifact
Phase 1 outputs must persist overlap-aware active speaker spans in local-track space. Each span should contain:
- start/end time
- active audio speaker ids
- zero or more visible local track ids
- zero or more visible global track ids when available as metadata
- off-screen active speaker ids/count
- overlap flag
- confidence metadata and decision provenance

This artifact is intended for debug overlays and the post-Phase-1 overlap adjudicator.
The initial schema should be pinned to a single transcript artifact field:
- `active_speakers_local[]` with
  - `start_time_ms`
  - `end_time_ms`
  - `audio_speaker_ids[]`
  - `visible_local_track_ids[]`
  - `visible_track_ids[]`
  - `offscreen_audio_speaker_ids[]`
  - `overlap: bool`
  - `confidence: float | null`
  - `decision_source: str`

### 5. Post-Phase-1 Gemini Overlap Adjudication
Overlap follow decisions should run after Phase 1 and before rendering. For each overlap window, a Gemini pass should receive structured context including:
- transcript excerpt
- overlapping diarized speaker ids and timing
- active visible local tracks and candidate scores
- continuity context before/after overlap
- optional overlay/contact-sheet evidence hooks (not required in first pass)

It should return a structured decision with:
- `camera_target_local_track_id | null`
- `stay_wide: bool`
- rationale label / decision type
- optional confidence score

The official Gemini model code should be configurable, with default set to `gemini-3-flash-preview` as reflected in Google’s official Gemini API docs on March 27, 2026.

### 6. Overlay Truthfulness
Debug overlays must highlight all visible active speakers during overlap. The HUD must explicitly show when an active speaker is off-screen. Camera behavior may still use one follow target, but the overlay should never imply there is only one active speaker when the pipeline says otherwise.

### 7. GPU Preprocessing Pass
The LR-ASD preprocessing path should move the high-volume crop/resize/grayscale operations from CPU OpenCV into Torch-on-GPU where practical without changing model input semantics. This is a required implementation target for this pass.

### 8. GPU Decode Investigation
The codebase should gain a documented GPU decode path investigation with a safe integration boundary. If a stable GPU decode implementation cannot be completed in this pass without destabilizing the pipeline, the code and docs must capture the exact blocker and preserve the rest of the shipped improvements.

## Architecture

### Phase 1 Speed Path
1. Build or reuse the shared analysis proxy at long-edge 1920.
2. Run tracking/face analysis on the proxy.
3. Build local track indexes and track quality metrics.
4. Prune LR-ASD candidate tracks before chunk generation.
5. Run LR-ASD on the remaining eligible tracks.
6. Aggregate candidate support into diarized turn ownership.
7. Reuse turn-level ownership in word-level speaker assignment unless strong contrary visual evidence appears.

### Overlap Path
1. Use diarized turns to identify overlap windows.
2. Build `active_speakers_local` spans from turn bindings plus visible-candidate evidence.
3. Persist these spans in the Phase 1 artifacts.
4. Run a post-Phase-1 Gemini adjudicator over overlap windows only.
5. Persist overlap follow decisions separately from base speaker bindings.
6. Render overlays using all active visible speakers while camera follow uses the adjudicated single target or a wide fallback.

### GPU Path
1. Keep decode/frame access behavior unchanged initially unless a safe GPU path is proven.
2. Replace CPU-side crop/resize/grayscale with Torch GPU preprocessing in the LR-ASD preparation path.
3. Leave GPU decode behind an explicit integration seam so it can be adopted later without rewriting binding logic.

## Data Model Changes
Add overlap-oriented transcript/debug artifacts and optional post-pass outputs:
- `active_speakers_local`
- `overlap_follow_decisions`
- optional overlap adjudication metadata (`decision_model`, `decision_source`, `stay_wide`, `offscreen_active_speakers`)

Existing artifacts remain the base contract:
- `audio_speaker_turns`
- `speaker_candidate_debug`
- `speaker_bindings_local`
- `speaker_follow_bindings_local`
- `audio_speaker_local_track_map`
- `overlap_follow_decisions[]` with
  - `start_time_ms`
  - `end_time_ms`
  - `camera_target_local_track_id | null`
  - `camera_target_track_id | null`
  - `stay_wide: bool`
  - `visible_local_track_ids[]`
  - `offscreen_audio_speaker_ids[]`
  - `decision_model: str | null`
  - `decision_source: str`
  - `confidence: float | null`

## Failure Handling
- If pyannote is unavailable, overlap artifacts should be empty and speaker binding should continue without them.
- If Gemini overlap adjudication fails, rendering should fall back to deterministic overlap behavior using the explicit active-speaker spans.
- If Gemini and deterministic overlap logic disagree, the renderer should prefer Gemini only for camera targeting inside the overlap window; overlay truth should continue to come from `active_speakers_local`.
- If overlap windows have no visible candidates, mark off-screen activity clearly and avoid inventing an on-screen target.
- If GPU preprocessing is unavailable, fall back to the existing CPU path without changing outputs.

## Testing Strategy
- Unit tests for proxy-default selection and overrides
- Unit tests for LR-ASD eligibility pruning
- Unit tests for turn-level ownership reuse and override behavior
- Unit tests for overlap artifact generation, including off-screen speakers
- Unit tests for Gemini overlap request/response normalization and fallback behavior
- Renderer tests confirming multiple active visible speakers are highlighted together
- End-to-end validation run on known overlap-heavy videos

## Success Criteria
- Phase 1 defaults to a 1920-long-edge analysis proxy unless explicitly disabled.
- LR-ASD processes materially fewer candidate windows on representative runs.
- Turn-level reuse reduces redundant word-level re-decisions while preserving ambiguous handling.
- Overlap windows produce explicit active-speaker artifacts and off-screen indicators.
- Debug overlays highlight all active visible speakers during overlap.
- A Gemini post-pass can choose a single camera target for overlap windows without blocking the base Phase 1 artifacts.
- GPU preprocessing reduces CPU overhead in LR-ASD preparation without changing correctness.
