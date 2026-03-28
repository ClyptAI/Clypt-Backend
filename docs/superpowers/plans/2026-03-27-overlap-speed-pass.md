# Overlap Speed Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a faster and more overlap-aware speaker binding/render pipeline by defaulting to the 1080p analysis proxy, pruning LR-ASD work, reusing turn-level ownership, persisting overlap artifacts, adding a Gemini overlap adjudication pass, and updating overlays to show all active speakers.

**Architecture:** Phase 1 remains the source of truth for tracking, diarization, LR-ASD scoring, and base local/global bindings. New overlap artifacts are persisted in Phase 1, then a post-Phase-1 Gemini pass resolves only overlap camera targeting. Renderers consume the richer artifact set while keeping fallback behavior deterministic.

**Tech Stack:** Python, pytest, decord, Torch/CUDA, pyannote, Gemini API via `google-genai`, ffmpeg-based renderers

## Status Update (2026-03-28)

- Status: Implemented and became the functional base branch for current LR-ASD work.
- Shipped outcomes: overlap-aware bindings/artifacts, local follow mode, overlap follow support, and renderer updates all landed from this plan family.
- Current promotion path: `codex/lrasd-prep-pool` extends this branch with GPU decode/prep-pool work plus follow/debug renderer rescue rules, and that is the branch being moved to `main`.

---

## File Map
- Modify: `/Users/rithvik/CascadeProjects/Clypt-V2/backend/do_phase1_worker.py`
- Modify: `/Users/rithvik/CascadeProjects/Clypt-V2/backend/pipeline/phase1_contract.py`
- Modify: `/Users/rithvik/CascadeProjects/Clypt-V2/backend/do_phase1_service/storage.py`
- Modify: `/Users/rithvik/CascadeProjects/Clypt-V2/backend/test-render/render_phase1_debug_video.py`
- Modify: `/Users/rithvik/CascadeProjects/Clypt-V2/backend/test-render/render_speaker_follow_clips.py`
- Create: `/Users/rithvik/CascadeProjects/Clypt-V2/backend/overlap_follow.py`
- Modify: `/Users/rithvik/CascadeProjects/Clypt-V2/requirements.txt`
- Create: `/Users/rithvik/CascadeProjects/Clypt-V2/docs/superpowers/notes/2026-03-27-gpu-decode-investigation.md`
- Modify: `/Users/rithvik/CascadeProjects/Clypt-V2/tests/backend/do_phase1_service/test_extract.py`
- Modify: `/Users/rithvik/CascadeProjects/Clypt-V2/tests/backend/pipeline/test_phase1_contract.py`
- Modify: `/Users/rithvik/CascadeProjects/Clypt-V2/tests/backend/do_phase1_service/test_storage.py`
- Modify: `/Users/rithvik/CascadeProjects/Clypt-V2/tests/backend/test_render/test_render_speaker_follow_clips.py`
- Create: `/Users/rithvik/CascadeProjects/Clypt-V2/tests/backend/test_render/test_render_phase1_debug_video.py`
- Create: `/Users/rithvik/CascadeProjects/Clypt-V2/tests/backend/test_overlap_follow.py`

### Task 1: Proxy Defaults and LR-ASD Candidate Pruning

**Files:**
- Modify: `/Users/rithvik/CascadeProjects/Clypt-V2/backend/do_phase1_worker.py`
- Test: `/Users/rithvik/CascadeProjects/Clypt-V2/tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Write failing tests for proxy defaults and LR-ASD eligibility pruning**
- [ ] **Step 2: Run the focused tests and verify they fail for the intended missing behavior**
- [ ] **Step 3: Implement 1920-long-edge proxy defaults and conservative LR-ASD eligibility gates**
- [ ] **Step 4: Re-run the focused tests until they pass**
- [ ] **Step 5: Commit with `feat: default phase1 to 1080p proxy and prune lrasd candidates`**

### Task 2: Turn-Level LR-ASD Reuse

**Files:**
- Modify: `/Users/rithvik/CascadeProjects/Clypt-V2/backend/do_phase1_worker.py`
- Test: `/Users/rithvik/CascadeProjects/Clypt-V2/tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Write failing tests for turn-level ownership reuse and strong-contrary override behavior**
- [ ] **Step 2: Run the focused tests and verify the failures reflect missing turn reuse**
- [ ] **Step 3: Implement turn-level LR-ASD support reuse in the word-level speaker binding path**
- [ ] **Step 4: Re-run the focused tests until they pass**
- [ ] **Step 5: Commit with `feat: reuse turn-level lrasd ownership in word binding`**

### Task 3: Overlap Artifacts and Contract Updates

**Files:**
- Modify: `/Users/rithvik/CascadeProjects/Clypt-V2/backend/do_phase1_worker.py`
- Modify: `/Users/rithvik/CascadeProjects/Clypt-V2/backend/pipeline/phase1_contract.py`
- Modify: `/Users/rithvik/CascadeProjects/Clypt-V2/backend/do_phase1_service/storage.py`
- Test: `/Users/rithvik/CascadeProjects/Clypt-V2/tests/backend/do_phase1_service/test_extract.py`
- Test: `/Users/rithvik/CascadeProjects/Clypt-V2/tests/backend/pipeline/test_phase1_contract.py`
- Test: `/Users/rithvik/CascadeProjects/Clypt-V2/tests/backend/do_phase1_service/test_storage.py`

- [ ] **Step 1: Write failing tests for `active_speakers_local`, off-screen indicators, and persisted overlap follow placeholders**
- [ ] **Step 2: Add failing degraded-mode tests for pyannote-unavailable overlap windows and no-visible-candidate overlap windows**
- [ ] **Step 3: Run the focused tests and verify schema/storage failures first**
- [ ] **Step 4: Implement the overlap artifact builders and manifest contract/storage updates**
- [ ] **Step 5: Re-run the focused tests until they pass**
- [ ] **Step 6: Commit with `feat: persist overlap-aware local speaker artifacts`**

### Task 4: Gemini Post-Phase-1 Overlap Adjudicator

**Files:**
- Create: `/Users/rithvik/CascadeProjects/Clypt-V2/backend/overlap_follow.py`
- Modify: `/Users/rithvik/CascadeProjects/Clypt-V2/backend/do_phase1_worker.py`
- Modify: `/Users/rithvik/CascadeProjects/Clypt-V2/requirements.txt`
- Test: `/Users/rithvik/CascadeProjects/Clypt-V2/tests/backend/test_overlap_follow.py`
- Test: `/Users/rithvik/CascadeProjects/Clypt-V2/tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Write failing tests for Gemini request normalization, structured response parsing, and deterministic fallback behavior**
- [ ] **Step 2: Run the focused tests and verify the failures show the post-pass is missing**
- [ ] **Step 3: Implement the overlap adjudication module with configurable default model `gemini-3-flash-preview`**
- [ ] **Step 4: Persist `overlap_follow_decisions` into `phase_1_audio` with deterministic fallback fields when Gemini declines or fails**
- [ ] **Step 5: Re-run the focused tests until they pass**
- [ ] **Step 6: Commit with `feat: add gemini overlap follow adjudication pass`**

### Task 5: Renderer and Overlay Updates

**Files:**
- Modify: `/Users/rithvik/CascadeProjects/Clypt-V2/backend/test-render/render_phase1_debug_video.py`
- Modify: `/Users/rithvik/CascadeProjects/Clypt-V2/backend/test-render/render_speaker_follow_clips.py`
- Test: `/Users/rithvik/CascadeProjects/Clypt-V2/tests/backend/test_render/test_render_speaker_follow_clips.py`
- Test: `/Users/rithvik/CascadeProjects/Clypt-V2/tests/backend/test_render/test_render_phase1_debug_video.py`

- [ ] **Step 1: Write failing tests proving multiple active visible speakers and off-screen active speakers are rendered explicitly**
- [ ] **Step 2: Run the focused renderer tests and verify the missing overlap behavior fails first**
- [ ] **Step 3: Implement overlay highlighting for all active visible speakers and HUD off-screen indicators, while preserving one camera target**
- [ ] **Step 4: Consume `overlap_follow_decisions` in `render_speaker_follow_clips.py` so overlap windows use the adjudicated target or `stay_wide` instead of the default follow binding**
- [ ] **Step 5: Re-run the focused tests until they pass**
- [ ] **Step 6: Commit with `feat: render overlap-aware active speaker overlays`**

### Task 6: GPU Preprocessing Improvements

**Files:**
- Modify: `/Users/rithvik/CascadeProjects/Clypt-V2/backend/do_phase1_worker.py`
- Test: `/Users/rithvik/CascadeProjects/Clypt-V2/tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Write failing tests for the new GPU preprocessing helper boundary and CPU fallback**
- [ ] **Step 2: Run the focused tests and verify the new helper contract is absent**
- [ ] **Step 3: Introduce a decode/preprocess boundary that lets LR-ASD swap frame sources without rewriting binding logic**
- [ ] **Step 4: Implement Torch-on-GPU crop/resize/grayscale preprocessing for LR-ASD with CPU fallback**
- [ ] **Step 5: Re-run the focused tests until they pass**
- [ ] **Step 6: Document decode-path blocker or safe GPU decode toggle status in `/Users/rithvik/CascadeProjects/Clypt-V2/docs/superpowers/notes/2026-03-27-gpu-decode-investigation.md`**
- [ ] **Step 7: Commit with `feat: move lrasd visual preprocessing onto gpu`**

### Task 7: Verification and Validation

**Files:**
- No new production files required
- Reuse existing test files and render scripts

- [ ] **Step 1: Run all focused backend tests for extract, contract, storage, overlap follow, and renderers**
- [ ] **Step 2: Run any broader affected suites that cover Phase 1 integration paths**
- [ ] **Step 3: Produce at least one fresh overlap-heavy Phase 1 run and render the full debug overlay**
- [ ] **Step 4: Confirm overlays show multiple active visible speakers and off-screen active speakers correctly**
- [ ] **Step 5: Commit any final verification-only adjustments with `test: validate overlap speed pass end to end`**
