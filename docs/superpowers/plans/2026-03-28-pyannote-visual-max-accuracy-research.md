# Pyannote Visual Max Accuracy Research Branch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a research-grade Phase 1 speaker binding path that treats pyannote as the audio truth layer and combines face identity, person ReID, pose, and selective active-speaker inference to maximize assignment accuracy and overlap correctness.

**Architecture:** Replace the current mostly word-first / LR-ASD-centric binding mindset with a span-first multimodal identity system. Pyannote supplies anonymous speaker activity and overlap structure; the visual system builds stable person identities from face embeddings, person ReID embeddings, pose support, and track continuity; a cross-modal mapping layer associates audio speakers to visual identities using clean spans only; active-speaker inference becomes a selective fallback for ambiguous spans rather than the default for every speaking region.

**Tech Stack:** pyannote diarization, Ultralytics tracking + pose, InsightFace face embeddings, Torchreid OSNet-AIN person embeddings, FAISS similarity search, existing Phase 1 worker/service stack, pytest.

---

## File / Module Plan

### New modules
- Create: `backend/speaker_binding/visual_identity.py`
  - Build stable visual identities from local tracks, face embeddings, person embeddings, pose/body support, and continuity constraints.
- Create: `backend/speaker_binding/audio_visual_mapping.py`
  - Learn soft `audio_speaker_id -> visual_identity_id` mappings from clean spans.
- Create: `backend/speaker_binding/assignment_engine.py`
  - Resolve each scheduled span using mapping priors first, selective active-speaker fallback second.
- Create: `backend/speaker_binding/visual_features.py`
  - Normalize access to face embeddings, person ReID embeddings, pose/body-derived quality, and caching.
- Create: `backend/speaker_binding/identity_store.py`
  - Typed data structures for visual identities, evidence weights, and mapping edges.
- Create: `backend/speaker_binding/evaluation.py`
  - Metrics for assignment coverage, overlap correctness, off-screen correctness, and mapping confidence.
- Create: `tests/backend/speaker_binding/test_visual_identity.py`
- Create: `tests/backend/speaker_binding/test_audio_visual_mapping.py`
- Create: `tests/backend/speaker_binding/test_assignment_engine.py`
- Create: `tests/backend/speaker_binding/test_evaluation.py`

### Existing modules to modify
- Modify: `backend/do_phase1_worker.py`
  - Insert the new max-accuracy visual identity + mapping path behind an experiment flag.
- Modify: `backend/speaker_binding/scheduler.py`
  - Extend scheduled spans to expose clean-span eligibility and mapping-friendly metadata.
- Modify: `backend/overlap_follow.py`
  - Consume richer overlap truth from the new assignment engine.
- Modify: `backend/do_phase1_service/test_extract.py`
  - Add end-to-end coverage for the new branch path.
- Modify: `backend/pipeline/phase1_contract.py`
  - Add optional fields for visual identity IDs, mapping confidence, and richer overlap assignments.
- Modify: `docs/superpowers/plans/2026-03-28-pyannote-visual-max-accuracy-research.md`
  - Keep updated if implementation discovers necessary plan corrections.

### Likely dependency / setup touchpoints
- Modify: `requirements-do-phase1.txt`
  - Add any new runtime deps only if not already present.
- Modify: `/etc/clypt-phase1/do-phase1.env` during deployment
  - Add experiment flags for the research path.

---

## Architecture Decisions

### 1. Pyannote owns speaker activity truth
- Pyannote is the authoritative source for:
  - speech vs non-speech
  - anonymous speaker turns
  - overlap windows
  - off-screen audio presence
- The visual system should never invent extra audio speakers or collapse overlap into a fake single winner.

### 2. Visual identity is separate from speaking activity
- We explicitly separate:
  - `who this visible person is`
  - `whether this visible person is speaking right now`
- Face identity is the highest-precision identity anchor.
- Person ReID and pose continuity preserve identity when faces are weak or missing.

### 3. Audio-to-visual mapping is learned from clean spans only
- Only accumulate mapping evidence from spans that satisfy all of:
  - one dominant pyannote speaker or a clean near-solo turn
  - one dominant visible identity or one clearly strongest candidate
  - strong visual continuity
  - good confidence from visual identity and active-speaker evidence
- Do not let noisy overlap windows teach the mapping table.

### 4. Active-speaker inference becomes selective fallback
- The max-accuracy branch should not run LR-ASD-style scoring everywhere by default.
- Use active-speaker inference only when:
  - multiple visible identities plausibly match the same audio speaker
  - rapid cuts or reaction shots make the mapping prior unsafe
  - the mapping table is weak or conflicting
  - overlap needs same-frame visible disambiguation

### 5. Multi-speaker overlap must be first-class
- A word/span can legitimately have:
  - multiple active visible speakers
  - one visible and one off-screen speaker
  - multiple off-screen speakers
- Preserve that truth in artifacts instead of collapsing to a single label.

---

## Experiment Flags

Add flags early so the branch is safe to iterate on without destabilizing the current baseline.

- `CLYPT_BINDING_STRATEGY=max_accuracy_mapping`
- `CLYPT_VISUAL_FACE_ID_ENABLE=1`
- `CLYPT_VISUAL_REID_ENABLE=1`
- `CLYPT_VISUAL_POSE_ENABLE=1`
- `CLYPT_AV_MAPPING_ENABLE=1`
- `CLYPT_ASD_FALLBACK_ENABLE=1`
- `CLYPT_OVERLAP_MULTI_ASSIGN_ENABLE=1`
- `CLYPT_BINDING_EVAL_DUMP=1`

The current production/default path must remain available until this branch beats it.

---

### Task 1: Add typed data structures for visual identities and audio-visual mappings

**Files:**
- Create: `backend/speaker_binding/identity_store.py`
- Test: `tests/backend/speaker_binding/test_visual_identity.py`

- [ ] **Step 1: Write failing tests for identity/mapping data normalization**
- [ ] **Step 2: Run the tests to confirm failure**
- [ ] **Step 3: Implement dataclasses / TypedDicts for visual identities, evidence edges, and mapping summaries**
- [ ] **Step 4: Add normalization helpers for confidence, track lists, and ordered IDs**
- [ ] **Step 5: Run the focused tests until they pass**
- [ ] **Step 6: Commit**

### Task 2: Build the visual identity graph from face + ReID + pose support

**Files:**
- Create: `backend/speaker_binding/visual_identity.py`
- Create: `backend/speaker_binding/visual_features.py`
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/speaker_binding/test_visual_identity.py`

- [ ] **Step 1: Write failing tests for merging local tracks into stable visual identities**
- [ ] **Step 2: Write failing tests for face-dominant, ReID-fallback, and pose-supported continuity cases**
- [ ] **Step 3: Run the focused tests to confirm failure**
- [ ] **Step 4: Implement normalized feature extraction hooks for:**
  - face embeddings already available in worker state
  - person ReID embeddings per local track span
  - pose/body quality and visibility continuity
- [ ] **Step 5: Implement identity clustering rules with same-frame exclusion and continuity constraints**
- [ ] **Step 6: Persist visual identity metadata into worker analysis context / debug artifacts**
- [ ] **Step 7: Run tests and fix edge cases**
- [ ] **Step 8: Commit**

### Task 3: Learn audio-speaker to visual-identity mappings from clean spans only

**Files:**
- Create: `backend/speaker_binding/audio_visual_mapping.py`
- Modify: `backend/speaker_binding/scheduler.py`
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/speaker_binding/test_audio_visual_mapping.py`

- [ ] **Step 1: Write failing tests for mapping accumulation on solo/clean spans**
- [ ] **Step 2: Write failing tests that noisy overlap spans do not pollute the mapping table**
- [ ] **Step 3: Run the focused tests to confirm failure**
- [ ] **Step 4: Extend scheduled spans with clean-span metadata and dominance hints**
- [ ] **Step 5: Implement weighted evidence aggregation:**
  - turn duration
  - visibility continuity
  - face anchor confidence
  - ReID support
  - optional active-speaker confidence if present
- [ ] **Step 6: Emit soft mappings with confidence and support diagnostics**
- [ ] **Step 7: Run tests and refine thresholds**
- [ ] **Step 8: Commit**

### Task 4: Implement the span-first assignment engine

**Files:**
- Create: `backend/speaker_binding/assignment_engine.py`
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/speaker_binding/test_assignment_engine.py`

- [ ] **Step 1: Write failing tests for clean single-speaker span assignment via mapping only**
- [ ] **Step 2: Write failing tests for ambiguous spans routing to fallback active-speaker inference**
- [ ] **Step 3: Write failing tests for off-screen speaker preservation**
- [ ] **Step 4: Run the focused tests to confirm failure**
- [ ] **Step 5: Implement decision policy:**
  - mapping-first for easy spans
  - fallback active-speaker scorer for ambiguous spans
  - preserve off-screen audio speakers explicitly
- [ ] **Step 6: Thread visual identity IDs, local track IDs, and global track IDs through results**
- [ ] **Step 7: Run tests and tighten decision logging**
- [ ] **Step 8: Commit**

### Task 5: Make overlap truly multi-speaker and mapping-aware

**Files:**
- Modify: `backend/speaker_binding/assignment_engine.py`
- Modify: `backend/speaker_binding/project_words.py`
- Modify: `backend/overlap_follow.py`
- Test: `tests/backend/speaker_binding/test_assignment_engine.py`

- [ ] **Step 1: Write failing tests for overlap spans assigning multiple visible speakers**
- [ ] **Step 2: Write failing tests for visible + off-screen overlap combinations**
- [ ] **Step 3: Run the tests to confirm failure**
- [ ] **Step 4: Update span projection so overlap words preserve the full active set**
- [ ] **Step 5: Make overlap follow consume the richer truth instead of re-deriving it**
- [ ] **Step 6: Keep backward-compatible dominant fields for legacy consumers where valid**
- [ ] **Step 7: Run tests and verify no false single-speaker collapse remains**
- [ ] **Step 8: Commit**

### Task 6: Add a stronger active-speaker fallback path only for hard spans

**Files:**
- Modify: `backend/do_phase1_worker.py`
- Modify: `backend/speaker_binding/assignment_engine.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Write failing tests that easy spans do not invoke fallback active-speaker scoring**
- [ ] **Step 2: Write failing tests that hard spans still invoke the fallback path**
- [ ] **Step 3: Run the focused tests to confirm failure**
- [ ] **Step 4: Define hard-span routing criteria:**
  - rapid cutbacks
  - conflicting mapping priors
  - multiple visible candidates
  - weak face/ReID confidence
- [ ] **Step 5: Reuse existing LR-ASD or a lighter ASD module behind a common fallback interface**
- [ ] **Step 6: Add metrics showing how often fallback was used and why**
- [ ] **Step 7: Run tests and verify routing behavior**
- [ ] **Step 8: Commit**

### Task 7: Instrument evaluation for the research branch

**Files:**
- Create: `backend/speaker_binding/evaluation.py`
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/speaker_binding/test_evaluation.py`

- [ ] **Step 1: Write failing tests for evaluation metric aggregation**
- [ ] **Step 2: Run the tests to confirm failure**
- [ ] **Step 3: Implement metrics for:**
  - words with >=1 assigned speaker
  - overlap words with multi-speaker assignment
  - off-screen overlap preservation
  - mapping confidence distribution
  - fallback invocation rate
  - assignment stability across adjacent words/spans
- [ ] **Step 4: Dump evaluation metrics into artifacts and logs when enabled**
- [ ] **Step 5: Run tests**
- [ ] **Step 6: Commit**

### Task 8: Integrate new dependencies and runtime guards safely

**Files:**
- Modify: `requirements-do-phase1.txt`
- Modify: `backend/do_phase1_service/extract.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Write failing tests for missing optional research dependencies producing clear errors**
- [ ] **Step 2: Run tests to confirm failure**
- [ ] **Step 3: Add dependency guards and actionable runtime error messages for:**
  - Torchreid / OSNet-AIN
  - pose model availability
  - optional FAISS usage
- [ ] **Step 4: Ensure the old path still runs when research flags are disabled**
- [ ] **Step 5: Run tests**
- [ ] **Step 6: Commit**

### Task 9: Run research-grade end-to-end validations on the droplet

**Files:**
- Modify: `scripts/do_phase1/run_remote_job.py`
- Modify: `scripts/do_phase1/run_remote_job.sh`
- Modify: `backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Add research-flag presets for this branch’s runs**
- [ ] **Step 2: Add log surfacing for:**
  - mapping table size
  - number of visual identities
  - fallback ASD invocation count
  - overlap assignment counts
- [ ] **Step 3: Run focused tests**
- [ ] **Step 4: Deploy to droplet in branch-isolated mode**
- [ ] **Step 5: Benchmark against current pyannote-scheduler branch on the same videos**
- [ ] **Step 6: Record assignment, overlap, off-screen, and runtime deltas**
- [ ] **Step 7: Commit**

---

## Validation Matrix

Run these at minimum during implementation:

- `PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q tests/backend/speaker_binding/test_visual_identity.py`
- `PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q tests/backend/speaker_binding/test_audio_visual_mapping.py`
- `PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q tests/backend/speaker_binding/test_assignment_engine.py`
- `PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q tests/backend/speaker_binding/test_evaluation.py`
- `PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q tests/backend/do_phase1_service/test_extract.py`

For droplet comparisons, record:
- assignment ratio
- with_scored_candidate ratio
- overlap multi-assignment count
- off-screen preserved count
- fallback ASD rate
- wall-clock speaker binding time
- total job time

---

## Success Criteria

The branch is a success when all of these are true:

- Assignment coverage improves materially over the current refactor branch.
- Overlap windows preserve multiple speakers instead of collapsing them.
- Off-screen speakers remain explicit instead of being forced onto visible listeners.
- Visual identity continuity is stronger under face flicker and profile views.
- Fallback active-speaker inference runs on a minority of spans, not the default path.
- Runtime is acceptable for a research branch even if slower than the speed-optimized branch.

## Non-Goals

- Do not land this directly into production defaults.
- Do not remove the existing LR-ASD path yet.
- Do not try to solve every model-training problem in this branch.
- Do not add LLM dependence to the core assignment truth layer.
