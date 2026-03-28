# Pyannote Visual Max Accuracy Research Branch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a research-grade Phase 1 speaker binding path that treats pyannote as the audio truth layer and combines face identity, high-accuracy person ReID, pose, and explicit visual speaking cues to maximize assignment accuracy and overlap correctness without relying on LR-ASD.

**Architecture:** Replace the current mostly word-first / LR-ASD-centric binding mindset with a span-first multimodal identity system. Pyannote supplies anonymous speaker activity and overlap structure; the visual system builds stable person identities from face embeddings, high-accuracy person ReID embeddings, pose support, face-landmark mouth-motion cues, and track continuity; a cross-modal mapping layer associates audio speakers to visual identities using clean spans only; hard spans are disambiguated by deterministic visual speaking evidence rather than LR-ASD. The branch is explicitly research-oriented and accuracy-first: it can be slower than the scheduler/cascade branch, but every expensive stage must be bounded, cached, and measurable.

**Tech Stack:** pyannote diarization, Ultralytics tracking + pose, InsightFace face embeddings, high-accuracy person ReID (BPBreID-class primary path), MediaPipe face landmarks for mouth-motion features, FAISS similarity search, existing Phase 1 worker/service stack, pytest.

---

## File / Module Plan

### New modules
- Create: `backend/speaker_binding/visual_identity.py`
  - Build stable visual identities from local tracks, face embeddings, person embeddings, pose/body support, and continuity constraints.
- Create: `backend/speaker_binding/audio_visual_mapping.py`
  - Learn soft `audio_speaker_id -> visual_identity_id` mappings from clean spans.
- Create: `backend/speaker_binding/assignment_engine.py`
  - Resolve each scheduled span using mapping priors first, deterministic visual-speaking disambiguation second.
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

## Exact Model / Runtime Strategy

### Visual tracking and person boxes
- Keep the existing Ultralytics person tracking path as the outer tracker.
- Default branch setting:
  - `CLYPT_TRACKER_BACKEND=botsort_reid`
- This branch should not replace the current tracker first. It should reuse:
  - current local track generation
  - current `track_to_dets` / `frame_to_dets` indexes
  - current clustering/identity surfaces where useful

### Face identity
- Reuse the existing InsightFace runtime already loaded in `load_model()`:
  - SCRFD detector: `det_10g.onnx`
  - ArcFace recognizer: `w600k_r50.onnx`
  - current cache root: `~/.insightface/models/buffalo_l`
- Do not introduce a second face stack.
- Treat face embeddings as the highest-precision identity anchor.

### Person ReID identity
- Use a **part-based / accuracy-first ReID path** as the primary body identity backbone.
- Exact first model target:
  - `BPBreID`-class integration as the primary research path
- Practical fallback if the droplet/runtime integration proves too heavy:
  - `osnet_ain_x1_0`
- Why this direction:
  - better occlusion robustness
  - stronger part-aware matching under profile / partial visibility
  - more aligned with the “accuracy over speed” goal for this branch
- Load once per worker process and cache on the worker instance.

### Pose support
- Add a dedicated Ultralytics pose model for visible-person structure.
- First model target:
  - `yolo11l-pose` if VRAM permits, otherwise `yolo11m-pose`
- Pose is not a standalone speaker classifier. It is a support signal for:
  - visibility quality
  - occlusion confidence
  - body-part continuity
  - candidate survival / tie-break support

### Face landmark speaking cues
- Add a face-landmark / mouth-motion path for hard-span disambiguation.
- First implementation target:
  - MediaPipe face landmarks / face mesh style mouth aperture + lip motion summaries
- Use this only on hard spans where mapping alone is not enough.
- This becomes the primary same-frame “is this visible person speaking right now?” cue instead of LR-ASD.

### Similarity search
- Use FAISS for nearest-neighbor lookups over:
  - person ReID embeddings
  - optional identity centroids
- Keep the first FAISS integration simple:
  - in-memory flat index for per-job work
  - no persistent ANN service

### Visual speaking disambiguation
- Do **not** introduce a new LLM-based speaker truth layer.
- Do **not** rely on LR-ASD in this branch.
- For hard spans, use deterministic visual-speaking evidence from:
  - mouth-motion / lip-aperture change
  - face visibility quality
  - short-window pose / head stability
  - continuity priors from neighboring spans
- If hard-span disambiguation is still weak, preserve ambiguity or off-screen truth rather than forcing a false single visible speaker.

### Overlap follow
- For this research branch, overlap truth is deterministic.
- The Gemini overlap-follow path must be treated as optional rendering/camera behavior only.
- Core assignment truth must remain LLM-free.

---

## Model Placement, Caching, and Lifecycle

### Worker process lifecycle
- All heavy models should be loaded in `ClyptWorker.load_model()` and reused for the lifetime of the worker child.
- Do not instantiate heavy models inside per-span or per-track loops.

### GPU vs CPU placement
- Keep on GPU:
  - Parakeet ASR
  - YOLO tracking model
  - InsightFace detector/recognizer as currently configured
  - person ReID model if stable on the droplet VRAM budget
- Keep flexible / test both:
  - pose model on GPU first; fall back to CPU if GPU pressure becomes unacceptable
  - face-landmark extraction on CPU first unless a GPU path is trivially available
- Keep on CPU:
  - FAISS index construction/search unless GPU FAISS is already trivially available
  - span scheduling
  - mapping aggregation
  - assignment logic

### Cache roots
- Face cache stays at:
  - `~/.insightface/models/buffalo_l`
- Add a dedicated cache root for research-branch visual models:
  - `/opt/clypt-phase1/models/research`
- Expected subdirectories:
  - `/opt/clypt-phase1/models/research/torchreid`
  - `/opt/clypt-phase1/models/research/ultralytics_pose`
- The branch must fail clearly if research flags are enabled but those assets are missing.

### Feature caching
- Cache per-job derived features in the Phase 1 job workspace, not globally:
  - person ReID embeddings per local track span
  - pose summaries per local track span
  - visual identity centroids
  - audio<->visual mapping table
- Keep cache artifacts small and typed so they can be included in debug outputs if needed.

### Bounded compute rules
- ReID and pose extraction must be span-aware and sampled, not frame-exhaustive.
- Default strategy:
  - sample representative frames per local track span
  - densify only on ambiguous spans
- Hard cap every expensive per-track/per-span routine and surface metrics when caps are hit.

---

## Exact Frame Sampling Policy For ReID / Pose

### Sampling unit
- The primary sampling unit is:
  - `(visual_identity_candidate, scheduled_span)`
- Do not sample over the full raw track first unless the span is missing enough evidence that a wider search is necessary.
- This keeps identity evidence aligned to the same audio decision unit used by pyannote scheduling.

### Pre-filter before any expensive extraction
- For each candidate local track within a scheduled span, build a lightweight frame list from existing `track_to_dets` entries.
- Drop frames immediately if any of these are true:
  - detection confidence below a configurable floor
  - tiny box area relative to frame
  - extreme truncation near image edge
  - frame lies inside a long occlusion gap for that track
- Mark, but do not immediately drop, profile / weak-face frames. Those are still useful for ReID/pose.

### Tier 1 sampling: cheap representative samples for every plausible candidate
- For each `(candidate, span)` pair, choose up to **6 representative frames**:
  - 1 near the span start
  - 1 near the span end
  - 1 near the temporal midpoint
  - up to 3 more spread across the span by quantiles
- Quantile targets:
  - `10%, 30%, 50%, 70%, 90%`
  - collapse duplicates if the span is short
- At each target point, choose the nearest valid frame with the best score from:
  - body box size
  - detector confidence
  - pose visibility if already available
  - face visibility bonus when present

### Tier 2 densification: only for ambiguous candidates/spans
- Densify only if the span is still unresolved after Tier 1, or if the mapping confidence is below threshold.
- Ambiguous conditions include:
  - multiple visible identities with similar mapping support
  - reaction-shot / cut-heavy spans
  - weak face anchor coverage
  - strong disagreement between face and ReID identity evidence
- Densification budget:
  - add up to **6 more frames** per `(candidate, span)`
  - max total **12 sampled frames**
- Densification should focus on:
  - frames nearest speaker-turn boundaries
  - frames immediately after cuts
  - frames with highest motion / mouth-region plausibility if available
  - frames where pose visibility is strongest

### Tier 3 fallback widening
- Only if a candidate still has no stable evidence for the current span:
  - widen to neighboring frames within the same local track
  - cap widening to a bounded time window around the span
- Initial widening rule:
  - up to `+/- 1200 ms` around the scheduled span
  - never exceed **16 total sampled frames** for that candidate across widened search
- This widening is for continuity recovery, not as the default path.

### ReID crop policy
- ReID should use the tracked person crop, not a full frame.
- Crop policy:
  - use the person box with a small context margin
  - keep aspect ratio stable
  - avoid overly aggressive padding that drags in background identity noise
- If pose is available, optionally bias the crop to keep torso + upper body centered.

### Pose extraction policy
- Pose does not need to run on every sampled ReID frame.
- Default:
  - run pose on the same Tier 1 sample set
  - reuse those pose summaries for Tier 1 scoring
- During densification:
  - run pose only on newly added frames
  - skip if the pose budget for that `(candidate, span)` is already exhausted
- Initial pose cap:
  - **8 pose frames** per `(candidate, span)`

### Face / ReID / pose interplay
- If a frame has a strong canonical face box:
  - it is highly valuable for face identity
  - but it does not remove the need for at least some ReID/pose support
- If a span has zero canonical face coverage:
  - ReID + pose become the primary visual identity evidence
  - but still stay within the bounded sample caps above
- If a frame is weak for face but strong for body visibility:
  - keep it for ReID and pose scoring

### Mouth-motion sampling on hard spans
- Only compute mouth-motion features for candidates in hard spans.
- Default mouth-motion sampling window:
  - center on the scheduled span
  - include a short pre/post context window
- Initial hard-span mouth-motion budget:
  - up to **12 face-landmark frames** per candidate/span
  - reuse already sampled face-visible frames when possible
- Prefer temporally contiguous mini-windows over isolated single frames for mouth-motion scoring.

### Per-track / per-span caps
- Hard caps for the first implementation:
  - Tier 1 ReID frames: `6`
  - Tier 2 added ReID frames: `6`
  - widened total ReID frames: `16`
  - pose frames: `8`
- Hard caps per local track across one job:
  - if the same track is sampled repeatedly across many spans, cache and reuse features
  - do not recompute the same frame crop embedding twice

### Caching keys
- Cache sampled visual features by:
  - `track_id`
  - `frame_idx`
  - crop type (`reid`, `pose`, optional `face`)
- Cache span summaries by:
  - `track_id`
  - `span_start_time_ms`
  - `span_end_time_ms`
  - sampling tier
- Reuse cached frame-level features whenever multiple spans touch the same track/window.

### Scoring summaries from sampled frames
- For each candidate span, compute:
  - ReID centroid
  - ReID dispersion / variance
  - pose visibility score
  - pose stability score
  - face-anchor count and mean face confidence
- Favor candidates whose sampled evidence is:
  - consistent across time
  - not just one lucky frame

### Failure / recovery rules
- If Tier 1 returns no usable frames:
  - mark the candidate as weak and skip to the next candidate unless the span is high priority
- If Tier 1 and Tier 2 both fail:
  - allow audio-only off-screen assignment or explicit ambiguity rather than unbounded widening
- Never keep widening indefinitely for a low-value candidate.

### Metrics to log
- For each run, log:
  - mean sampled ReID frames per candidate/span
  - mean sampled pose frames per candidate/span
  - mean sampled mouth-motion frames per hard candidate/span
  - percent of spans resolved at Tier 1
  - percent of spans needing densification
  - percent of candidates widened outside the span
  - cache hit rate for frame-level ReID and pose features

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

### 4. Visual speaking evidence is the hard-span disambiguator
- The max-accuracy branch should not run LR-ASD-style scoring.
- Use deterministic visual speaking evidence only when:
  - multiple visible identities plausibly match the same audio speaker
  - rapid cuts or reaction shots make the mapping prior unsafe
  - the mapping table is weak or conflicting
  - overlap needs same-frame visible disambiguation
- Hard-span disambiguation should be based on:
  - mouth-motion consistency over a short temporal window
  - face visibility / face-landmark confidence
  - pose / head stability
  - continuity from neighboring solved spans

### 5. Multi-speaker overlap must be first-class
- A word/span can legitimately have:
  - multiple active visible speakers
  - one visible and one off-screen speaker
  - multiple off-screen speakers
- Preserve that truth in artifacts instead of collapsing to a single label.

### 6. No LLM in the assignment truth path
- Gemini / overlap-follow can still exist for camera behavior experiments.
- It must not participate in deciding who the true active speakers are for words/spans on this branch.

### 7. Accuracy-first does not mean unbounded work
- Every new expensive stage needs:
  - sampling strategy
  - cache strategy
  - metrics
  - runtime guardrails
- We are optimizing for accuracy first, but not for runaway per-job cost.

### 8. Preserve uncertainty honestly
- If visual speaking evidence is weak and the mapping table is weak, keep:
  - multiple active speakers
  - or an off-screen assignment
  - or an explicitly ambiguous span
- Do not force a single visible winner just because a downstream consumer would prefer one.

---

## Experiment Flags

Add flags early so the branch is safe to iterate on without destabilizing the current baseline.

- `CLYPT_BINDING_STRATEGY=max_accuracy_mapping`
- `CLYPT_VISUAL_FACE_ID_ENABLE=1`
- `CLYPT_VISUAL_REID_ENABLE=1`
- `CLYPT_VISUAL_POSE_ENABLE=1`
- `CLYPT_VISUAL_MOUTH_MOTION_ENABLE=1`
- `CLYPT_AV_MAPPING_ENABLE=1`
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
- [ ] **Step 2: Write failing tests for face-dominant, part-ReID-supported, and pose-supported continuity cases**
- [ ] **Step 3: Run the focused tests to confirm failure**
- [ ] **Step 4: Implement normalized feature extraction hooks for:**
  - face embeddings already available in worker state
  - person ReID embeddings per local track span
  - pose/body quality and visibility continuity
- [ ] **Step 5: Add explicit feature sampling rules**
  - representative frames per span
  - max samples per local track
  - denser sampling only for ambiguous tracks/spans
- [ ] **Step 6: Implement identity clustering rules with same-frame exclusion and continuity constraints**
- [ ] **Step 7: Persist visual identity metadata into worker analysis context / debug artifacts**
- [ ] **Step 8: Run tests and fix edge cases**
- [ ] **Step 9: Commit**

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
  - optional mouth-motion support on hard confirmed spans when available
- [ ] **Step 6: Emit soft mappings with confidence and support diagnostics**
- [ ] **Step 7: Run tests and refine thresholds**
- [ ] **Step 8: Commit**

### Task 4: Implement the span-first assignment engine

**Files:**
- Create: `backend/speaker_binding/assignment_engine.py`
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/speaker_binding/test_assignment_engine.py`

- [ ] **Step 1: Write failing tests for clean single-speaker span assignment via mapping only**
- [ ] **Step 2: Write failing tests for ambiguous spans routing to visual-speaking disambiguation**
- [ ] **Step 3: Write failing tests for off-screen speaker preservation**
- [ ] **Step 4: Run the focused tests to confirm failure**
- [ ] **Step 5: Implement decision policy:**
  - mapping-first for easy spans
  - visual-speaking disambiguation for ambiguous spans
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

### Task 6: Add visual-speaking disambiguation for hard spans

**Files:**
- Modify: `backend/do_phase1_worker.py`
- Modify: `backend/speaker_binding/assignment_engine.py`
- Create: `backend/speaker_binding/mouth_motion.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Write failing tests that easy spans do not invoke hard-span mouth-motion / pose disambiguation**
- [ ] **Step 2: Write failing tests that hard spans do invoke visual-speaking disambiguation**
- [ ] **Step 3: Run the focused tests to confirm failure**
- [ ] **Step 4: Define hard-span routing criteria:**
  - rapid cutbacks
  - conflicting mapping priors
  - multiple visible candidates
  - weak face/ReID confidence
- [ ] **Step 5: Implement mouth-motion / face-landmark summaries and combine them with pose visibility**
- [ ] **Step 6: Add a strict routing guard so hard-span disambiguation is never called for clearly solved mapping-first spans**
- [ ] **Step 7: Add metrics showing how often hard-span visual disambiguation was used and why**
- [ ] **Step 8: Run tests and verify routing behavior**
- [ ] **Step 9: Commit**

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
  - hard-span visual disambiguation invocation rate
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
  - primary high-accuracy ReID backend
  - OSNet-AIN fallback ReID backend
  - pose model availability
  - optional FAISS usage
  - MediaPipe face landmark availability
- [ ] **Step 4: Add worker-level model caching and health logging for:**
  - ReID model load success/failure
  - pose model load success/failure
  - face landmark model load success/failure
  - cache roots and selected model names
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
  - hard-span visual disambiguation count
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
 - hard-span visual disambiguation rate
 - wall-clock speaker binding time
 - total job time

---

## Success Criteria

The branch is a success when all of these are true:

- Assignment coverage improves materially over the current refactor branch.
- Overlap windows preserve multiple speakers instead of collapsing them.
- Off-screen speakers remain explicit instead of being forced onto visible listeners.
- Visual identity continuity is stronger under face flicker and profile views.
 - Hard-span visual disambiguation runs on a minority of spans, not the default path.
- Runtime is acceptable for a research branch even if slower than the speed-optimized branch.

## Non-Goals

- Do not land this directly into production defaults.
- Do not delete the existing LR-ASD path from the codebase yet.
- Do not try to solve every model-training problem in this branch.
- Do not add LLM dependence to the core assignment truth layer.
