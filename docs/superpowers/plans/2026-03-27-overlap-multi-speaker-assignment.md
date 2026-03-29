# Overlap Multi-Speaker Assignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove LR-ASD turn-level candidate pruning, loosen diarization turn usage, and add true multi-speaker word assignment for overlap spans while keeping camera-follow arbitration as a separate concern.

**Architecture:** Speaker binding remains deterministic for truth-labeling. Pyannote diarization continues to provide speaker turns, but those turns become soft context rather than hard candidate-pruning boundaries. The Phase 1 audio contract gains overlap-aware word assignment data that can represent multiple active visible speakers plus off-screen speakers, while the existing overlap-follow/Gemini layer stays focused on choosing a single camera target when overlap exists.

**Tech Stack:** Python, pytest, pyannote diarization turns, LR-ASD speaker scoring, existing Phase 1 contract/storage pipeline, Gemini overlap follow post-pass.

## Status Update (2026-03-28)

- Status: Partially implemented, then superseded for the LR-ASD mainline branch.
- Landed pieces: turn-pruning removal, overlap artifacts, and camera-follow separation fed into `codex/overlap-speed-pass`.
- Not adopted on the promoted LR-ASD branch: the pyannote-driven multi-speaker truth path from this plan is not the branch being moved to `main` today; it remains a historical/superseded planning thread.

---

## File Map

**Modify:**
- `backend/do_phase1_worker.py`
  - Remove hard turn-level LR-ASD candidate pruning.
  - Loosen diarization turn handling around boundaries and overlap.
  - Build multi-speaker word assignment artifacts.
  - Preserve single-target follow generation as a separate step.
- `backend/pipeline/phase1_contract.py`
  - Extend Phase 1 audio contract with multi-speaker word assignment structures and any off-screen overlap metadata.
- `backend/overlap_follow.py`
  - Keep overlap follow focused on camera adjudication only; adapt input expectations if new overlap artifacts change shape.
- `backend/do_phase1_service/storage.py`
  - Ensure new Phase 1 audio fields persist cleanly.
- `backend/test-render/render_phase1_debug_video.py`
  - Render multiple active speakers at once and display off-screen overlap indicators.

**Test:**
- `tests/backend/do_phase1_service/test_extract.py`
  - LR-ASD no longer prunes candidates by turn top-K.
  - loosened turn matching around boundaries.
  - overlap words can receive multiple assigned local speakers.
  - off-screen overlap metadata persists correctly.
- `tests/backend/do_phase1_service/test_worker.py`
  - end-to-end contract persistence for the new overlap-aware audio fields.
- `tests/backend/test_render/test_render_phase1_debug_video.py`
  - overlay rendering highlights multiple active speakers and shows off-screen speaker state.

---

### Task 1: Remove Turn-Level LR-ASD Candidate Selection

**Files:**
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Write failing tests for disabled turn-level pruning**

```python
def test_run_lrasd_binding_scores_all_globally_eligible_candidates_when_turn_topk_disabled(...):
    ...
    assert set(worker._test_lrasd_scored_local_track_ids) == {"speaker", "listener"}
```

```python
def test_run_lrasd_binding_does_not_reduce_candidates_using_turn_subselection(...):
    ...
    assert metrics["with_scored_candidate"] >= baseline_expected
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/backend/do_phase1_service/test_extract.py -k "turn_topk_disabled or scores_all_globally_eligible"`
Expected: FAIL because worker still uses per-turn selected candidate subsets.

- [ ] **Step 3: Implement removal of turn-level candidate selection from LR-ASD scoring**

Implementation notes:
- Keep `_compute_lrasd_eligible_track_ids(...)` as the broad global gate.
- Do not use per-turn selected track IDs to reduce which LR-ASD windows are scored.
- Preserve debug data only if it remains useful for analysis, but it must no longer affect candidate availability.

- [ ] **Step 4: Re-run targeted tests**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/backend/do_phase1_service/test_extract.py -k "turn_topk_disabled or scores_all_globally_eligible"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/do_phase1_worker.py tests/backend/do_phase1_service/test_extract.py
git commit -m "fix: remove lrasd turn candidate pruning"
```

### Task 2: Loosen Diarization Turn Usage

**Files:**
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Write failing tests for softened turn boundaries**

```python
def test_word_near_turn_boundary_consults_adjacent_turn_context(...):
    ...
    assert word_debug["active_audio_speaker_ids"] == ["SPEAKER_00", "SPEAKER_01"]
```

```python
def test_tiny_same_speaker_gap_is_merged_for_binding_context(...):
    ...
    assert len(binding_turns) == 1
```

```python
def test_overlap_window_retains_multiple_active_turns(...):
    ...
    assert overlap_word["active_audio_speaker_ids"] == ["SPEAKER_00", "SPEAKER_01"]
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/backend/do_phase1_service/test_extract.py -k "boundary_consults_adjacent_turn or tiny_same_speaker_gap or multiple_active_turns"`
Expected: FAIL because current turn usage is too rigid.

- [ ] **Step 3: Implement softened turn handling**

Implementation notes:
- Add a helper that normalizes diarization turns for binding context:
  - merge tiny same-speaker gaps
  - pad turn boundaries by a small window
  - allow overlap windows to surface multiple active speaker IDs
- Use this normalized turn context for candidate lookup/debug, not as a hard pruning gate.

- [ ] **Step 4: Re-run targeted tests**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/backend/do_phase1_service/test_extract.py -k "boundary_consults_adjacent_turn or tiny_same_speaker_gap or multiple_active_turns"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/do_phase1_worker.py tests/backend/do_phase1_service/test_extract.py
git commit -m "feat: loosen diarization turn boundaries for binding"
```

### Task 3: Add Multi-Speaker Word Assignment Contract

**Files:**
- Modify: `backend/pipeline/phase1_contract.py`
- Modify: `backend/do_phase1_service/storage.py`
- Test: `tests/backend/do_phase1_service/test_worker.py`

- [ ] **Step 1: Write failing contract/storage tests**

```python
def test_phase1_audio_contract_accepts_multi_speaker_word_assignments():
    payload = {
        "word_active_speakers_local": [
            {
                "start_time_ms": 1000,
                "end_time_ms": 1200,
                "local_track_ids": ["track_a", "track_b"],
                "offscreen_speaker_ids": ["SPEAKER_02"],
                "overlap": True,
            }
        ]
    }
    ...
```

```python
def test_phase1_storage_persists_overlap_word_assignment_fields(...):
    ...
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/backend/do_phase1_service/test_worker.py -k "multi_speaker_word_assignments or overlap_word_assignment"`
Expected: FAIL because the contract doesn’t yet expose these fields.

- [ ] **Step 3: Extend the Phase 1 audio contract**

Implementation notes:
- Add a backwards-compatible field for overlap-aware speaker truth, for example:
  - `word_active_speakers_local`
- Each entry should support:
  - word/span timing
  - `local_track_ids`
  - `audio_speaker_ids`
  - `offscreen_speaker_ids`
  - `overlap`
  - optional `decision_source`/debug metadata if useful
- Keep existing single-speaker fields for compatibility.

- [ ] **Step 4: Re-run targeted tests**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/backend/do_phase1_service/test_worker.py -k "multi_speaker_word_assignments or overlap_word_assignment"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/phase1_contract.py backend/do_phase1_service/storage.py tests/backend/do_phase1_service/test_worker.py
git commit -m "feat: add overlap-aware speaker assignment contract"
```

### Task 4: Build Deterministic Multi-Speaker Word Assignment

**Files:**
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Write failing tests for overlap-aware assignment**

```python
def test_overlap_word_assigns_multiple_visible_speakers(...):
    ...
    assert overlap_entry["local_track_ids"] == ["speaker_a", "speaker_b"]
```

```python
def test_overlap_word_records_offscreen_audio_speaker_when_no_visible_match(...):
    ...
    assert overlap_entry["offscreen_speaker_ids"] == ["SPEAKER_02"]
```

```python
def test_non_overlap_word_still_emits_single_speaker_assignment(...):
    ...
    assert overlap_entry["local_track_ids"] == ["speaker_a"]
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/backend/do_phase1_service/test_extract.py -k "multiple_visible_speakers or offscreen_audio_speaker or non_overlap_word"`
Expected: FAIL because binding still collapses to a single winner/unknown contract.

- [ ] **Step 3: Implement deterministic overlap-aware assignment**

Implementation notes:
- For each word, find all normalized active audio speakers overlapping that word.
- For each active audio speaker, consult existing LR-ASD/local evidence and select the best visible local speaker if present.
- Aggregate visible speakers across the active audio speakers into a multi-speaker assignment set.
- If no visible match exists for an active audio speaker, record it in `offscreen_speaker_ids` rather than inventing a box.
- Preserve existing single-speaker fields for compatibility, but derive the overlap-aware structure in parallel.

- [ ] **Step 4: Re-run targeted tests**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/backend/do_phase1_service/test_extract.py -k "multiple_visible_speakers or offscreen_audio_speaker or non_overlap_word"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/do_phase1_worker.py tests/backend/do_phase1_service/test_extract.py
git commit -m "feat: assign multiple speakers during overlap"
```

### Task 5: Keep Camera Follow as a Separate Layer

**Files:**
- Modify: `backend/overlap_follow.py`
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Write failing tests for separation of concerns**

```python
def test_overlap_follow_decision_does_not_mutate_overlap_truth_assignments(...):
    ...
    assert word_active_speakers_local[0]["local_track_ids"] == ["speaker_a", "speaker_b"]
    assert follow_decision["camera_target_local_track_id"] == "speaker_a"
```

- [ ] **Step 2: Run targeted test to verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/backend/do_phase1_service/test_extract.py -k "does_not_mutate_overlap_truth_assignments"`
Expected: FAIL if current post-pass assumptions couple camera target and assignment truth too tightly.

- [ ] **Step 3: Adapt overlap follow to consume overlap truth but only emit follow decisions**

Implementation notes:
- Keep Gemini/deterministic overlap follow focused on a single camera target or stay-wide choice.
- Do not let the overlap follow layer erase or replace multi-speaker assignment truth.

- [ ] **Step 4: Re-run targeted test**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/backend/do_phase1_service/test_extract.py -k "does_not_mutate_overlap_truth_assignments"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/overlap_follow.py backend/do_phase1_worker.py tests/backend/do_phase1_service/test_extract.py
git commit -m "refactor: separate overlap truth from camera follow"
```

### Task 6: Update Debug Overlay Rendering

**Files:**
- Modify: `backend/test-render/render_phase1_debug_video.py`
- Test: `tests/backend/test_render/test_render_phase1_debug_video.py`

- [ ] **Step 1: Write failing render tests**

```python
def test_debug_overlay_highlights_multiple_active_visible_speakers(...):
    ...
```

```python
def test_debug_overlay_shows_offscreen_overlap_indicator(...):
    ...
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/backend/test_render/test_render_phase1_debug_video.py -k "multiple_active_visible_speakers or offscreen_overlap_indicator"`
Expected: FAIL because renderer still assumes a single active speaker highlight.

- [ ] **Step 3: Implement overlap-aware rendering**

Implementation notes:
- Highlight all visible active speakers for overlap spans.
- Add HUD text for off-screen active speakers.
- Keep single follow target visualization separate from active-speaker truth.

- [ ] **Step 4: Re-run targeted tests**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/backend/test_render/test_render_phase1_debug_video.py -k "multiple_active_visible_speakers or offscreen_overlap_indicator"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/test-render/render_phase1_debug_video.py tests/backend/test_render/test_render_phase1_debug_video.py
git commit -m "feat: render overlap-aware active speakers"
```

### Task 7: End-to-End Validation and Metrics Review

**Files:**
- Modify: `backend/do_phase1_worker.py` (only if metric output needs refinement)
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Add or update metric assertions/tests**

```python
def test_overlap_metrics_include_multi_speaker_word_counts(...):
    ...
```

- [ ] **Step 2: Run focused integration tests**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/backend/do_phase1_service/test_extract.py tests/backend/do_phase1_service/test_worker.py tests/backend/test_render/test_render_phase1_debug_video.py`
Expected: PASS.

- [ ] **Step 3: Run the full relevant suite**

Run: `PYTHONPATH=. .venv/bin/pytest -q tests/backend/do_phase1_service/test_extract.py tests/backend/do_phase1_service/test_worker.py tests/backend/test_render/test_render_phase1_debug_video.py tests/backend/pipeline/test_phase_1_pipeline_async.py`
Expected: PASS.

- [ ] **Step 4: Commit final validation/metric adjustments**

```bash
git add backend/do_phase1_worker.py tests/backend/do_phase1_service/test_extract.py tests/backend/do_phase1_service/test_worker.py tests/backend/test_render/test_render_phase1_debug_video.py tests/backend/pipeline/test_phase_1_pipeline_async.py
git commit -m "test: validate overlap-aware speaker assignment pipeline"
```

### Task 8: Deploy and Compare Against Baseline

**Files:**
- No code changes required unless issues are found.

- [ ] **Step 1: Deploy the branch to the GPU droplet**

Run the existing deploy workflow with:
- strict GPU decode on
- `CLYPT_LRASD_TOPK_PER_TURN=0`
- pyannote + Vertex/Gemini overlap follow enabled

- [ ] **Step 2: Run a fresh Phase 1 job on `2jW9lmlfiKQ`**

Capture:
- `with_scored_candidate`
- LR-ASD assignment ratio
- overlap-word multi-speaker counts
- final follow/overlay behavior

- [ ] **Step 3: Render the full debug overlay**

Verify:
- multiple active speakers highlight together
- off-screen speakers show in HUD
- follow target remains single and explicit

- [ ] **Step 4: Compare to the last bad run**

Record:
- assignment ratio delta
- whether overlap truth is now represented correctly
- whether single-target follow still looks reasonable
