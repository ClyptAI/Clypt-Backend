# Local Track Clip Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an experiment path where speaker-follow clip rendering uses stitched local body tracks directly, minimizing dependence on polluted `Global_Person_*` remapping.

**Architecture:** Keep the existing clustering/global-ID pipeline for analytics and compatibility, but add a parallel local-track speaker-follow path. LR-ASD assignment should preserve local stitched track winners, emit local speaker bindings/follow bindings, and let the renderer follow those local track IDs directly under an experiment flag.

**Tech Stack:** Python, pytest, existing Phase 1 worker, existing test-render pipeline, LR-ASD, YOLO/BoT-SORT stitched person tracks.

## Status Update (2026-03-28)

- Status: Implemented.
- Shipped outcomes: `speaker_bindings_local`, `speaker_follow_bindings_local`, and local-track rendering/follow became part of the promoted LR-ASD path.
- Current branch lineage: the local-track clip mode work is present on `codex/lrasd-prep-pool` and is part of the mainline branch move requested today.

---

## File Structure

- Modify: `backend/do_phase1_worker.py`
  - Add local-track binding outputs alongside existing global outputs.
  - Preserve local stitched track IDs through LR-ASD scoring and follow-binding generation.
  - Add experiment flags to choose global-vs-local clip-binding exports.
- Modify: `backend/test-render/render_speaker_follow_clips.py`
  - Add a binding-selection path for local track bindings.
  - Follow local stitched body tracks directly when local mode is enabled.
- Modify: `tests/backend/do_phase1_service/test_extract.py`
  - Add worker tests for local binding export and remap minimization behavior.
- Modify: `tests/backend/test_render/test_render_speaker_follow_clips.py`
  - Add renderer tests proving local bindings drive camera target selection.
- Create: `docs/superpowers/plans/2026-03-25-local-track-clip-mode.md`
  - This plan document.

---

### Task 1: Define the experiment contract

**Files:**
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Write failing tests for local binding outputs**

Add tests that expect Phase 1 audio output to optionally include:
- `speaker_bindings_local`
- `speaker_follow_bindings_local`

And verify these bindings are keyed by stitched local track IDs rather than `Global_Person_*` IDs.

- [ ] **Step 2: Run the targeted worker tests to verify they fail**

Run:
```bash
cd /Users/rithvik/CascadeProjects/Clypt-V2/.worktrees/local-track-experiment
PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q tests/backend/do_phase1_service/test_extract.py -k local
```

Expected: FAIL because local binding fields do not exist yet.

- [ ] **Step 3: Add minimal config surface**

Add an experiment flag such as:
- `CLYPT_EXPERIMENT_LOCAL_CLIP_BINDINGS=1`

Define clear behavior:
- global bindings continue to exist
- local bindings are emitted in parallel when enabled
- no existing consumers break if the flag is off

- [ ] **Step 4: Run the targeted worker tests again**

Run the same command and verify the new contract is exercised.

- [ ] **Step 5: Commit**

```bash
git add backend/do_phase1_worker.py tests/backend/do_phase1_service/test_extract.py docs/superpowers/plans/2026-03-25-local-track-clip-mode.md
git commit -m "feat: add local clip binding contract"
```

### Task 2: Preserve stitched local winners through LR-ASD assignment

**Files:**
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Write a failing test for local winner preservation**

Create a regression test where:
- two distinct stitched local tracks later remap to the same polluted global
- the local output must still preserve them as separate local speaker winners

- [ ] **Step 2: Run the specific test to verify it fails**

Run:
```bash
cd /Users/rithvik/CascadeProjects/Clypt-V2/.worktrees/local-track-experiment
PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q tests/backend/do_phase1_service/test_extract.py::test_local_bindings_preserve_distinct_stitched_tracks
```

Expected: FAIL because remap still flattens the distinction.

- [ ] **Step 3: Implement local winner preservation**

Inside `backend/do_phase1_worker.py`:
- keep LR-ASD candidate competition in stitched local track space
- keep local winner IDs on words before any global remap
- build local binding segments from those local winners
- only derive global bindings afterward as a parallel projection

- [ ] **Step 4: Run the specific test to verify it passes**

Run the same test and confirm PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/do_phase1_worker.py tests/backend/do_phase1_service/test_extract.py
git commit -m "feat: preserve local LR-ASD winners for clip mode"
```

### Task 3: Build local follow bindings with the same smoothing policy

**Files:**
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Write a failing test for local follow smoothing**

Cover:
- short `A -> B -> A` local blips
- follow-rate limiting / segment cleanup
- output still staying in local track namespace

- [ ] **Step 2: Run the targeted smoothing test to verify it fails**

Run:
```bash
cd /Users/rithvik/CascadeProjects/Clypt-V2/.worktrees/local-track-experiment
PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q tests/backend/do_phase1_service/test_extract.py -k local_follow
```

Expected: FAIL until local follow binding generation is wired.

- [ ] **Step 3: Implement local follow binding generation**

Reuse the existing follow-binding builder where possible, but keep local track IDs intact.

- [ ] **Step 4: Run the targeted smoothing test to verify it passes**

Run the same command and confirm PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/do_phase1_worker.py tests/backend/do_phase1_service/test_extract.py
git commit -m "feat: add local follow bindings"
```

### Task 4: Teach the renderer to consume local clip bindings

**Files:**
- Modify: `backend/test-render/render_speaker_follow_clips.py`
- Test: `tests/backend/test_render/test_render_speaker_follow_clips.py`

- [ ] **Step 1: Write failing renderer tests**

Add tests proving that when local clip mode is enabled:
- renderer prefers `speaker_follow_bindings_local`
- falls back to `speaker_bindings_local`
- and follows local stitched body tracks directly

- [ ] **Step 2: Run the renderer tests to verify they fail**

Run:
```bash
cd /Users/rithvik/CascadeProjects/Clypt-V2/.worktrees/local-track-experiment
PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q tests/backend/test_render/test_render_speaker_follow_clips.py -k local
```

Expected: FAIL because renderer does not know about local binding fields yet.

- [ ] **Step 3: Implement local-binding selection in the renderer**

Add:
- local binding selection helper
- local track lookup path
- direct body-box follow using local stitched track IDs
- compatibility fallback to current global binding behavior when flag is off

- [ ] **Step 4: Run renderer tests to verify they pass**

Run the same command and confirm PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/test-render/render_speaker_follow_clips.py tests/backend/test_render/test_render_speaker_follow_clips.py
git commit -m "feat: render clips from local bindings"
```

### Task 5: Add remap-pollution instrumentation

**Files:**
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Write a failing test for remap-pollution diagnostics**

Expect metrics that flag cases where:
- distinct stitched local tracks
- repeatedly map to the same final global identity
- across strong solo-shot windows

- [ ] **Step 2: Run the metrics test to verify it fails**

Run:
```bash
cd /Users/rithvik/CascadeProjects/Clypt-V2/.worktrees/local-track-experiment
PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q tests/backend/do_phase1_service/test_extract.py -k remap_pollution
```

Expected: FAIL because the metrics do not exist yet.

- [ ] **Step 3: Implement diagnostics**

Add metrics such as:
- count of local stitched track IDs per final global speaker winner
- repeated solo-shot local tracks mapping to one global
- optional debug export for local-to-global speaker remap evidence

- [ ] **Step 4: Run the metrics test to verify it passes**

Run the same command and confirm PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/do_phase1_worker.py tests/backend/do_phase1_service/test_extract.py
git commit -m "feat: add local-to-global remap diagnostics"
```

### Task 6: Run full focused verification

**Files:**
- Modify: `backend/do_phase1_worker.py`
- Modify: `backend/test-render/render_speaker_follow_clips.py`
- Test: `tests/backend/do_phase1_service/test_extract.py`
- Test: `tests/backend/test_render/test_render_speaker_follow_clips.py`

- [ ] **Step 1: Run the focused backend/render suite**

Run:
```bash
cd /Users/rithvik/CascadeProjects/Clypt-V2/.worktrees/local-track-experiment
PYTHONPATH=. /Users/rithvik/CascadeProjects/Clypt-V2/.venv/bin/pytest -q \
  tests/backend/do_phase1_service/test_extract.py \
  tests/backend/test_render/test_render_speaker_follow_clips.py
```

Expected: PASS.

- [ ] **Step 2: Run smoke compile checks**

Run:
```bash
cd /Users/rithvik/CascadeProjects/Clypt-V2/.worktrees/local-track-experiment
python3 -m py_compile backend/do_phase1_worker.py backend/test-render/render_speaker_follow_clips.py
```

Expected: no output, exit 0.

- [ ] **Step 3: Record the experiment validation target**

Primary manual validation set after implementation:
- Divorce Lawyer / Andrew timestamps from `docs/planning/07-flagrant-divorce-lawyer-debug-analysis.md`
- especially `00:20-00:26`, `02:00-02:20`, `04:26-04:43`, `05:51-05:58`, `06:05-07:00`

- [ ] **Step 4: Commit final implementation state**

```bash
git add backend/do_phase1_worker.py backend/test-render/render_speaker_follow_clips.py tests/backend/do_phase1_service/test_extract.py tests/backend/test_render/test_render_speaker_follow_clips.py docs/superpowers/plans/2026-03-25-local-track-clip-mode.md
git commit -m "feat: add local-track clip mode experiment"
```
