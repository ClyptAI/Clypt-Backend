# Pyannote Scheduler + Cascade LR-ASD Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current word-centric LR-ASD binding path with a span-first, pyannote-scheduled, cascade-based speaker-binding pipeline that runs LR-ASD only on hard spans, uses body support to survive face flicker, and parallelizes LR-ASD prep so we materially improve both wallclock and assignment quality.

**Architecture:** Build a dedicated `backend/speaker_binding/` package and move binding logic out of the worker monolith. Pyannote becomes the scheduler: it produces normalized single-speaker and overlap spans, plus soft context windows. A cascade binder resolves easy single-speaker spans cheaply, while LR-ASD runs only on ambiguous spans, overlaps, weak-visibility spans, and visual discontinuities. Binding becomes span-first and projects to words at the end, instead of letting every word fight independently.

**Tech Stack:** Python, existing Phase 1 worker, pyannote diarization 3.1, CUDA-enabled decord, Torch LR-ASD, current face/identity feature pipeline, existing body priors and track-quality metrics, pytest.

---

## Current Reality Snapshot

The plan below is based on the current committed code at `7002bf9` plus the live runtime env on `2026-03-27` from droplet `clypt-phase1-gpu-3`.

### Live env snapshot
- `CLYPT_SPEAKER_BINDING_MODE=lrasd`
- `CLYPT_TRACKING_MODE=auto`
- `CLYPT_ANALYSIS_PROXY_ENABLE=1`
- `CLYPT_ANALYSIS_PROXY_MAX_LONG_EDGE=1920`
- `CLYPT_SHARED_ANALYSIS_PROXY=1`
- `CLYPT_AUDIO_DIARIZATION_ENABLE=1`
- `CLYPT_AUDIO_DIARIZATION_MODEL=pyannote/speaker-diarization-3.1`
- `CLYPT_AUDIO_DIARIZATION_MIN_SEGMENT_MS=400`
- `CLYPT_EXPERIMENT_LOCAL_CLIP_BINDINGS=1`
- `CLYPT_OVERLAP_FOLLOW_ENABLE=1`
- `CLYPT_OVERLAP_FOLLOW_MODEL=gemini-3-flash-preview`
- `CLYPT_LRASD_GPU_PREPROCESS=1`
- `CLYPT_LRASD_GPU_DECODE=1`
- `CLYPT_LRASD_GPU_DECODE_STRICT=1`
- `CLYPT_LRASD_TOPK_PER_TURN=0`

### Important current code seams
- `backend/do_phase1_worker.py`
  - `_run_speaker_binding(...)` currently treats LR-ASD as a drop-in word binder with heuristic fallback.
  - `_run_lrasd_binding(...)` is still fundamentally word-centric even though it already aggregates some turn evidence.
  - `_bind_audio_turns_to_local_tracks(...)` already knows how to aggregate candidate evidence across diarized turns.
  - `_build_active_speakers_local(...)` already derives overlap-aware active speaker spans from audio-turn bindings.
  - `_prepare_lrasd_visual_batch(...)`, `_open_lrasd_video_reader(...)`, `_make_lrasd_frame_provider(...)`, `_lrasd_build_pending_subchunk(...)` already expose the prep/inference seams we need for batching and parallelism.
- `backend/pipeline/phase1_contract.py`
  - already has `audio_speaker_turns`, `speaker_candidate_debug`, `active_speakers_local`, and `overlap_follow_decisions`.
- `backend/overlap_follow.py`
  - already assumes overlap follow is a post-pass over `active_speakers_local`, which is the right layering.

### Key architectural problems to fix
1. LR-ASD still thinks in words too early, which duplicates work and makes candidate starvation easier.
2. Pyannote exists, but mostly as context for a word binder instead of as the scheduler that shapes the binding workload.
3. Downstream overlap and off-screen truth currently depend on `audio_turn_bindings` emitted from the LR-ASD path, which means easy-span skipping needs a replacement evidence path instead of simply bypassing LR-ASD.
4. “Easy” spans still pay too much of the expensive LR-ASD tax.
5. Current body support is mostly a scoring prior; it is not a first-class continuity/candidate-survival cue.
6. LR-ASD prep is more efficient than before, but it is not yet a proper producer/consumer pipeline around scheduled spans.
7. The binding code is concentrated in `backend/do_phase1_worker.py`, which makes performance changes riskier and harder to reason about.

## Performance-First Design Decisions

These are intentional departures from the current architecture because they should improve performance the most.

1. **Make binding span-first, not word-first.**
   - Words become a projection target after span decisions are made.
   - This removes repeated LR-ASD work across adjacent words in the same speech span.

2. **Make pyannote the scheduler, not just a context hint.**
   - Every LR-ASD decision starts from normalized diarized spans.
   - LR-ASD is skipped entirely outside scheduled speech spans.

3. **Introduce a real cascade.**
   - Easy single-speaker spans resolve cheaply.
   - LR-ASD is reserved for hard spans only.

4. **Treat body support as a first-class signal.**
   - Use existing body priors, track quality, box prominence, continuity, and identity features to keep good candidates alive when faces flicker.
   - Do not begin with a heavy pose-estimation model unless the cheap body-support layer fails.

5. **Refactor binding into its own package.**
   - The worker should orchestrate.
   - The scheduler, cascade, body support, and LR-ASD runner should live in dedicated modules.

## Non-Goals

- Do not replace LR-ASD with pyannote.
- Do not make the LLM part of assignment truth. LLM remains optional camera-follow arbitration only.
- Do not add a heavyweight pose model in the first pass if the same outcome can be achieved with existing body/track signals.
- Do not preserve the current worker-local architecture if it blocks the performance-first design.

## Proposed File Structure

### New package
- Create: `backend/speaker_binding/__init__.py`
  - public orchestration entrypoints.
- Create: `backend/speaker_binding/types.py`
  - light-weight dataclasses / typed dicts for spans, candidates, span decisions, prep jobs.
- Create: `backend/speaker_binding/scheduler.py`
  - normalize pyannote turns, merge tiny same-speaker gaps, pad boundaries, classify spans as `single`, `overlap`, `uncertain`, and surface soft context windows.
- Create: `backend/speaker_binding/easy_span_cascade.py`
  - cheap resolution path for easy single-speaker spans.
- Create: `backend/speaker_binding/body_support.py`
  - body continuity, prominence, candidate survival, flicker tolerance, tie-break scoring.
- Create: `backend/speaker_binding/discontinuity.py`
  - visual continuity-break detection so single-speaker spans with rapid cut-backs still route to LR-ASD.
- Create: `backend/speaker_binding/lrasd_runner.py`
  - scheduled-span LR-ASD execution, prep queues, batching, parallel prep, and turn/sub-turn result reuse.
- Create: `backend/speaker_binding/project_words.py`
  - project span-level assignments back onto words, including multi-speaker overlap words.
- Create: `backend/speaker_binding/metrics.py`
  - emit scheduler/cascade/LR-ASD counters for runtime visibility.

### Existing files to modify
- Modify: `backend/do_phase1_worker.py`
  - strip down binding orchestration and delegate to `backend/speaker_binding`.
- Modify: `backend/pipeline/phase1_contract.py`
  - extend transcript artifact contract for span-first outputs and multi-speaker word assignments.
- Modify: `backend/overlap_follow.py`
  - adapt overlap follow input to the new span-level assignment artifact, but keep it post-pass.
- Modify: `tests/backend/do_phase1_service/test_extract.py`
  - keep a thin integration layer for worker-level regression coverage.

### New tests
- Create: `tests/backend/speaker_binding/test_scheduler.py`
- Create: `tests/backend/speaker_binding/test_easy_span_cascade.py`
- Create: `tests/backend/speaker_binding/test_body_support.py`
- Create: `tests/backend/speaker_binding/test_discontinuity.py`
- Create: `tests/backend/speaker_binding/test_lrasd_runner.py`
- Create: `tests/backend/speaker_binding/test_project_words.py`
- Create: `tests/backend/speaker_binding/test_metrics.py`

## New Runtime Controls

Add explicit envs so the new behavior is measurable and tunable without code edits.

- `CLYPT_BINDING_SCHEDULER_ENABLE=1`
- `CLYPT_BINDING_SCHEDULER_TURN_PAD_MS=120`
- `CLYPT_BINDING_SCHEDULER_MERGE_GAP_MS=120`
- `CLYPT_BINDING_SCHEDULER_MIN_SINGLE_SPEAKER_MS=250`
- `CLYPT_BINDING_CASCADE_ENABLE=1`
- `CLYPT_BINDING_EASY_SPAN_MIN_DOMINANCE=0.80`
- `CLYPT_BINDING_EASY_SPAN_MIN_VISIBLE_RATIO=0.70`
- `CLYPT_BINDING_EASY_SPAN_MAX_COMPETITOR_RATIO=0.20`
- `CLYPT_BINDING_EASY_SPAN_MIN_CONTINUITY_SCORE=0.75`
- `CLYPT_BINDING_BODY_SUPPORT_ENABLE=1`
- `CLYPT_BINDING_BODY_SUPPORT_FACE_FLICKER_GAP_FRAMES=12`
- `CLYPT_BINDING_DISCONTINUITY_ENABLE=1`
- `CLYPT_BINDING_DISCONTINUITY_TRACK_JACCARD_THRESHOLD=0.35`
- `CLYPT_BINDING_DISCONTINUITY_HISTOGRAM_DELTA_THRESHOLD=0.18`
- `CLYPT_LRASD_PREP_WORKERS=4`
- `CLYPT_LRASD_PREP_QUEUE_SIZE=128`
- `CLYPT_LRASD_INFER_WORKERS=1` (raise only after benchmark proof)
- `CLYPT_LRASD_SPAN_REUSE_ENABLE=1`
- `CLYPT_LRASD_SPAN_REUSE_MIN_SUPPORT=0.75`

Keep existing envs in place:
- `CLYPT_ANALYSIS_PROXY_ENABLE`
- `CLYPT_SHARED_ANALYSIS_PROXY`
- `CLYPT_LRASD_GPU_PREPROCESS`
- `CLYPT_LRASD_GPU_DECODE`
- `CLYPT_LRASD_GPU_DECODE_STRICT`
- `CLYPT_AUDIO_DIARIZATION_*`

## Output Contract Changes

The refactor should promote span-first outputs to first-class artifacts.

### Add new transcript-level outputs
- `speaker_assignment_spans_local`
  - span-level decisions with `single`, `overlap`, `offscreen`, `easy_span`, `lrasd_span` source metadata.
- `speaker_assignment_spans_global`
  - projected global equivalent when available.
- `word_speaker_assignments`
  - explicit per-word structured objects containing:
    - `visible_local_track_ids`
    - `visible_track_ids`
    - `offscreen_audio_speaker_ids`
    - `decision_source`
    - `overlap`
  - this replaces the fragile parallel-array idea.

### Keep for backward compatibility
- `speaker_bindings`
- `speaker_bindings_local`
- `speaker_follow_bindings`
- `speaker_follow_bindings_local`
- `active_speakers_local`
- `overlap_follow_decisions`

Backward compatibility rule:
- legacy single-speaker fields still get populated where a single dominant visible speaker exists.
- legacy debug outputs (`speaker_candidate_debug`, `speaker_bindings_local`, `speaker_follow_bindings_local`) are dual-written until overlap follow, storage, and debug renderers are migrated.
- overlap-aware fields become the source of truth for debug overlays and future camera-follow work only after consumer migration is complete.

## Refactor Sequence

### Task 1: Carve speaker binding into its own package

**Files:**
- Create: `backend/speaker_binding/__init__.py`
- Create: `backend/speaker_binding/types.py`
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Write failing worker-level tests that assert binding calls a package entrypoint**

```python
def test_run_speaker_binding_delegates_to_binding_package(monkeypatch):
    ...
```

- [ ] **Step 2: Run targeted test to verify failure**

Run: `PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q tests/backend/do_phase1_service/test_extract.py -k delegate`
Expected: FAIL because no package entrypoint exists

- [ ] **Step 3: Create `backend/speaker_binding/types.py`**
  - add typed containers for:
    - diarized spans
    - scheduled spans
    - easy-span decisions
    - LR-ASD span jobs
    - span-level assignments

- [ ] **Step 4: Create `backend/speaker_binding/__init__.py` with a placeholder orchestration function**

```python
def bind_words_to_speakers(...):
    raise NotImplementedError
```

- [ ] **Step 5: Wire `backend/do_phase1_worker.py` to call the new entrypoint**
  - keep behavior unchanged for now
  - no scheduler/cascade logic yet

- [ ] **Step 6: Run targeted regression suite**

Run: `PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q tests/backend/do_phase1_service/test_extract.py -k "speaker_binding or lrasd"`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add backend/speaker_binding/__init__.py backend/speaker_binding/types.py backend/do_phase1_worker.py tests/backend/do_phase1_service/test_extract.py
git commit -m "refactor: extract speaker binding package entrypoint"
```

### Task 2: Make pyannote the scheduler

**Files:**
- Create: `backend/speaker_binding/scheduler.py`
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/speaker_binding/test_scheduler.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Write failing scheduler tests**
  - merges tiny same-speaker gaps
  - pads boundaries
  - emits `single` vs `overlap` scheduled spans
  - preserves adjacent overlap windows instead of collapsing to a single owner

- [ ] **Step 2: Implement scheduler primitives**
  - normalize diarization turns
  - merge same-speaker micro-gaps
  - pad boundaries
  - emit scheduled spans with:
    - `span_type`
    - `speaker_ids`
    - `exclusive`
    - `overlap`
    - `source_turn_ids`
    - `context_start/end`

- [ ] **Step 3: Add env-driven tuning hooks**
  - `CLYPT_BINDING_SCHEDULER_*`

- [ ] **Step 4: Thread scheduled spans into binding orchestration**
  - do not change final assignments yet
  - only swap the source of scheduling/context

- [ ] **Step 5: Run scheduler and integration tests**

Run: `PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q tests/backend/speaker_binding/test_scheduler.py tests/backend/do_phase1_service/test_extract.py -k "active_speakers_local or diarization or scheduler"`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/speaker_binding/scheduler.py backend/do_phase1_worker.py tests/backend/speaker_binding/test_scheduler.py tests/backend/do_phase1_service/test_extract.py
git commit -m "feat: schedule binding from normalized pyannote spans"
```

### Task 3: Add visual discontinuity detection so cut-back single-speaker scenes still use LR-ASD

**Files:**
- Create: `backend/speaker_binding/discontinuity.py`
- Modify: `backend/speaker_binding/scheduler.py`
- Test: `tests/backend/speaker_binding/test_discontinuity.py`

- [ ] **Step 1: Write failing discontinuity tests**
  - single-speaker diarized span with stable one-person visibility stays easy
  - same speaker but rapid shot/candidate discontinuity routes to hard span

- [ ] **Step 2: Implement lightweight discontinuity heuristics**
  - first pass:
    - local-track set Jaccard changes across the span
    - bbox prominence owner flips
    - optional frame-histogram delta on analysis proxy frames
  - mark spans with `requires_lrasd=True` when continuity breaks

- [ ] **Step 3: Thread discontinuity marks into scheduled span metadata**

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q tests/backend/speaker_binding/test_discontinuity.py tests/backend/speaker_binding/test_scheduler.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/speaker_binding/discontinuity.py backend/speaker_binding/scheduler.py tests/backend/speaker_binding/test_discontinuity.py tests/backend/speaker_binding/test_scheduler.py
git commit -m "feat: detect visual discontinuities in scheduled speech spans"
```

### Task 4: Promote body support to a first-class continuity layer

**Files:**
- Create: `backend/speaker_binding/body_support.py`
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/speaker_binding/test_body_support.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Write failing body-support tests**
  - stable body track survives face flicker
  - large/prominent stable body track beats tiny face-heavy fragment
  - existing identity features and track quality improve candidate survival

- [ ] **Step 2: Implement body support scoring**
  - inputs:
    - existing `body_prior`
    - track-quality metrics
    - area/prominence
    - visibility continuity
    - optional track identity feature presence
  - outputs:
    - continuity support score
    - candidate survival mask
    - tie-break bonus

- [ ] **Step 3: Replace direct face-coverage dependence in scheduling/cascade decisions with body-support outputs**
  - face coverage stays a bonus, not a hard gate

- [ ] **Step 4: Gate optional future pose work explicitly**
  - document that full pose estimation is out-of-scope for this first pass
  - keep a narrow seam for later optional pose/head-orientation cues if needed

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q tests/backend/speaker_binding/test_body_support.py tests/backend/do_phase1_service/test_extract.py -k "body or lrasd or candidate"`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/speaker_binding/body_support.py backend/do_phase1_worker.py tests/backend/speaker_binding/test_body_support.py tests/backend/do_phase1_service/test_extract.py
git commit -m "feat: add body continuity support for speaker binding"
```

### Task 5: Add the easy-span cascade

**Files:**
- Create: `backend/speaker_binding/easy_span_cascade.py`
- Modify: `backend/speaker_binding/scheduler.py`
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/speaker_binding/test_easy_span_cascade.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Write failing cascade tests**
  - easy solo span with dominant visible local track resolves without LR-ASD
  - overlap span never uses easy path
  - discontinuous cut-back single-speaker span still routes to LR-ASD
  - weak-visibility or handoff span routes to LR-ASD

- [ ] **Step 2: Implement cascade classifier**
  - easy span requires:
    - single active diarized speaker
    - no overlap
    - no discontinuity
    - dominant visible local track
    - strong body continuity
    - weak competitor ratio
  - otherwise mark span `hard`

- [ ] **Step 3: Implement cheap assignment for easy spans**
  - assign visible local/global track directly at span level
  - record `decision_source="easy_span"`
  - emit synthetic turn-binding evidence rows so downstream `active_speakers_local`, off-screen logic, and audio-speaker-to-local-track maps do not depend exclusively on LR-ASD spans

- [ ] **Step 4: Emit metrics**
  - scheduled spans total
  - easy spans resolved
  - hard spans routed to LR-ASD
  - overlap spans count

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q tests/backend/speaker_binding/test_easy_span_cascade.py tests/backend/do_phase1_service/test_extract.py -k "cascade or speaker_binding"`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/speaker_binding/easy_span_cascade.py backend/speaker_binding/scheduler.py backend/do_phase1_worker.py tests/backend/speaker_binding/test_easy_span_cascade.py tests/backend/do_phase1_service/test_extract.py
git commit -m "feat: add easy-span cascade before lrasd"
```

### Task 6: Rewrite LR-ASD execution to run on scheduled hard spans, not words

**Files:**
- Create: `backend/speaker_binding/lrasd_runner.py`
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/speaker_binding/test_lrasd_runner.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Write failing LR-ASD runner tests**
  - only scheduled hard spans generate LR-ASD jobs
  - solo easy spans generate zero LR-ASD jobs while still leaving valid turn-binding evidence behind
  - overlap span creates multi-speaker-capable LR-ASD jobs
  - span reuse avoids duplicate per-word work inside the same span/sub-span

- [ ] **Step 2: Implement span job creation**
  - for each scheduled hard span:
    - build candidate set from globally eligible visible tracks intersecting the span
    - use body-support survival mask
    - do not revive turn-top-K pruning

- [ ] **Step 3: Implement per-span / sub-span LR-ASD reuse**
  - score once for the span or sub-span
  - reuse that evidence for all words inside the span
  - split only when overlap or discontinuity demands it

- [ ] **Step 4: Keep strict GPU decode and GPU preprocess support intact**
  - move existing helpers from worker into runner module
  - preserve current strict error surfacing

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q tests/backend/speaker_binding/test_lrasd_runner.py tests/backend/do_phase1_service/test_extract.py -k "lrasd or speaker_binding"`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/speaker_binding/lrasd_runner.py backend/do_phase1_worker.py tests/backend/speaker_binding/test_lrasd_runner.py tests/backend/do_phase1_service/test_extract.py
git commit -m "refactor: run lrasd on scheduled hard spans"
```

### Task 7: Parallelize LR-ASD prep as a producer/consumer pipeline

**Files:**
- Modify: `backend/speaker_binding/lrasd_runner.py`
- Create: `backend/speaker_binding/metrics.py`
- Test: `tests/backend/speaker_binding/test_lrasd_runner.py`
- Test: `tests/backend/speaker_binding/test_metrics.py`

- [ ] **Step 1: Write failing concurrency tests around prep queue behavior**
  - multiple span jobs can prepare concurrently
  - inference consumes in bounded order
  - queue backpressure works
  - final scores stay deterministic

- [ ] **Step 2: Implement prep workers**
  - use `ThreadPoolExecutor` for frame/crop/audio prep across scheduled span jobs
  - keep decode/preprocess on current GPU path where available
  - push prepared jobs into a bounded queue

- [ ] **Step 3: Keep inference single-stream first**
  - default `CLYPT_LRASD_INFER_WORKERS=1`
  - preserve deterministic output ordering in aggregation
  - add a benchmark-only seam for `>1` workers, but do not enable it by default without proof on the target GPU

- [ ] **Step 4: Emit queue and throughput metrics**
  - prep queue depth
  - prep wallclock
  - infer wallclock
  - spans/sec
  - jobs skipped by easy cascade

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q tests/backend/speaker_binding/test_lrasd_runner.py tests/backend/speaker_binding/test_metrics.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/speaker_binding/lrasd_runner.py backend/speaker_binding/metrics.py tests/backend/speaker_binding/test_lrasd_runner.py tests/backend/speaker_binding/test_metrics.py
git commit -m "feat: parallelize lrasd prep pipeline"
```

### Task 8: Project span assignments back to words and support multi-speaker overlap words

**Files:**
- Create: `backend/speaker_binding/project_words.py`
- Modify: `backend/pipeline/phase1_contract.py`
- Modify: `backend/do_phase1_worker.py`
- Modify: `backend/overlap_follow.py`
- Test: `tests/backend/speaker_binding/test_project_words.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`
- Test: `tests/backend/pipeline/test_phase1_contract.py`

- [ ] **Step 1: Write failing projection/contract tests**
  - solo span maps one visible speaker to all words in range
  - overlap span maps multiple visible speakers to affected words
  - off-screen overlapping speaker is surfaced without inventing a box
  - legacy single-speaker fields still populate where applicable

- [ ] **Step 2: Implement span-to-word projection**
  - use span intersections, not per-word re-scoring
  - emit structured `word_speaker_assignments` objects
  - preserve legacy `speaker_local_track_id`/`speaker_track_id` when exactly one dominant visible speaker exists

- [ ] **Step 3: Extend transcript contract models**
  - add new structured per-word assignment model and validators
  - update `Phase1Word` or its enclosing transcript artifact so consumer ordering cannot drift

- [ ] **Step 4: Adapt overlap follow post-pass**
  - consume span-level overlap truth instead of reconstructing it from legacy fields
  - keep `speaker_candidate_debug` and legacy local bindings available until render/debug consumers migrate

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q tests/backend/speaker_binding/test_project_words.py tests/backend/do_phase1_service/test_extract.py tests/backend/pipeline/test_phase1_contract.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/speaker_binding/project_words.py backend/pipeline/phase1_contract.py backend/do_phase1_worker.py backend/overlap_follow.py tests/backend/speaker_binding/test_project_words.py tests/backend/do_phase1_service/test_extract.py tests/backend/pipeline/test_phase1_contract.py
git commit -m "feat: project span assignments to words with overlap support"
```

### Task 9: Remove dead paths and obsolete metrics from the old word-first architecture

**Files:**
- Modify: `backend/do_phase1_worker.py`
- Modify: `tests/backend/do_phase1_service/test_extract.py`
- Test: all touched suites

- [ ] **Step 1: Write failing regression tests for removed dead paths**
  - no turn-top-K pruning
  - no per-word LR-ASD job generation
  - no fallback gates depending on removed turn-subselection state

- [ ] **Step 2: Delete obsolete helpers and metrics**
  - remove dead turn-top-K selection code
  - remove only the truly dead word-first helpers
  - keep compatibility shims and dual-write outputs for `speaker_candidate_debug`, `speaker_bindings_local`, and `speaker_follow_bindings_local` until overlap follow, storage, and debug rendering have switched to span-level inputs

- [ ] **Step 3: Run focused full speaker-binding suites**

Run: `PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q tests/backend/speaker_binding tests/backend/do_phase1_service/test_extract.py tests/backend/pipeline/test_phase1_contract.py tests/backend/pipeline/test_phase_1_pipeline_async.py`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add backend/do_phase1_worker.py tests/backend/do_phase1_service/test_extract.py tests/backend/speaker_binding tests/backend/pipeline/test_phase1_contract.py tests/backend/pipeline/test_phase_1_pipeline_async.py
git commit -m "refactor: remove obsolete word-first speaker binding paths"
```

### Task 10: Benchmark and deploy safely

**Files:**
- Modify: `backend/do_phase1_worker.py` (only if metric names need final cleanup)
- Modify: runtime env on droplet after code lands
- Test: live droplet runs

- [ ] **Step 1: Add explicit runtime log lines**
  - scheduler summary
  - easy-span/hard-span counts
  - spans routed to LR-ASD
  - prep queue metrics
  - assignment coverage broken down by solo vs overlap spans

- [ ] **Step 2: Run local verification**

Run:
- `python3 -m py_compile backend/do_phase1_worker.py backend/speaker_binding/*.py`
- `PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q tests/backend/speaker_binding tests/backend/do_phase1_service/test_extract.py tests/backend/pipeline/test_phase1_contract.py tests/backend/pipeline/test_phase_1_pipeline_async.py`

Expected: PASS

- [ ] **Step 3: Deploy to droplet with current GPU env preserved**
  - keep strict GPU decode on
  - keep shared 1920 analysis proxy on
  - add new scheduler/cascade envs explicitly

- [ ] **Step 4: Run A/B validation on the same video**
  - compare old branch vs new branch on:
    - LR-ASD wallclock
    - `with_scored_candidate`
    - assignment ratio
    - overlap-visible speaker correctness
    - easy-span skip rate

- [ ] **Step 5: Commit any final metric-name cleanup only if required**

## Acceptance Criteria

1. LR-ASD no longer runs outside diarized scheduled speech spans.
2. Easy single-speaker spans skip LR-ASD entirely.
3. Single-speaker spans with strong visual discontinuity still route to LR-ASD.
4. Body support keeps good candidates alive through face flicker without making face presence a hard gate.
5. LR-ASD prep is parallelized with bounded producer/consumer queues.
6. Binding is span-first, and words are projected from span decisions.
7. Overlap words can carry multiple visible assigned speakers plus off-screen audio speakers.
8. The worker becomes orchestration glue, not the home for the whole speaker-binding algorithm.
9. On the reference video, the new path must show a measurable LR-ASD wallclock reduction without regressing overlap handling.

## Risk Register

- **Risk:** cheap cascade over-assigns in cut-heavy single-speaker scenes.
  - **Mitigation:** discontinuity gate forces LR-ASD on continuity breaks.
- **Risk:** span projection hides word-level edge timing.
  - **Mitigation:** allow sub-span splits where overlap/handoff/discontinuity occurs.
- **Risk:** body support helps the wrong person survive.
  - **Mitigation:** make body support a survival/tie-break cue, not a sole identity claim.
- **Risk:** refactor churn in the worker is too large.
  - **Mitigation:** extract aggressively into `backend/speaker_binding/` early so worker diffs shrink after Task 1.
- **Risk:** concurrency introduces nondeterministic assignments.
  - **Mitigation:** deterministic aggregation order and queue-bound tests before rollout.

## Suggested Execution Order

1. Package extraction
2. Scheduler
3. Discontinuity guard
4. Body support
5. Easy-span cascade
6. Span-first LR-ASD runner
7. Parallel prep
8. Word projection + overlap contract
9. Consumer migration + dead-path cleanup
10. Benchmark + deploy
