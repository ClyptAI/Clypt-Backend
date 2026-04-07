# Phase 1 Parallel Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the V3.1 Phase 1 worker so media preparation fans out into a parent-coordinated parallel runtime with a visual GPU branch, an audio GPU branch, and a CPU YAMNet branch, while preserving the existing Phase 1 artifact contracts and keeping VibeVoice in its validated native subprocess environment.

**Architecture:** Introduce a small parent coordinator above the current Phase 1 sidecar flow. The coordinator prepares the workspace once, launches isolated branch workers as separate processes, waits for their result artifacts, and joins them into `Phase1SidecarOutputs`. The visual branch owns RF-DETR + ByteTrack, the audio branch owns `VibeVoice -> forced alignment -> emotion2vec+`, and the YAMNet branch runs independently on CPU in V1. VibeVoice remains a nested subprocess inside the audio branch because its validated runtime still requires a separate native venv.

**Tech Stack:** Python, subprocess-based process orchestration, SQLite job state, RF-DETR Large, ByteTrackTracker, VibeVoice native worker, NVIDIA NeMo Forced Aligner, funasr emotion2vec+, TensorFlow YAMNet, pytest.

---

## File Structure

### Existing files to modify

- `backend/phase1_runtime/extract.py`
  - Replace the current serial `run_phase1_sidecars(...)` flow with a parent-coordinator entrypoint plus result-join logic.
- `backend/phase1_runtime/factory.py`
  - Wire the new coordinator and branch worker factories into the default Phase 1 runtime assembly.
- `backend/phase1_runtime/models.py`
  - Add typed branch result models, branch status metadata, and coordinator-facing payload types.
- `backend/phase1_runtime/runner.py`
  - Call the new coordinator instead of the current serial sidecar helper.
- `backend/phase1_runtime/worker.py`
  - Preserve job-worker semantics while ensuring branch logs and child failures surface clearly.
- `backend/providers/config.py`
  - Add env/config knobs for branch parallelism, branch timeouts, and CPU YAMNet defaults.
- `docs/runtime/v3.1_runtime_guide.md`
  - Update the runtime architecture and env docs.
- `docs/deployment/v3.1_phase1_digitalocean.md`
  - Update DO runtime/deployment guidance for the parent + branch worker model.

### New files to create

- `backend/phase1_runtime/coordinator.py`
  - Parent Phase 1 coordinator: launch branches, watch processes, join outputs, handle failure/cancellation.
- `backend/phase1_runtime/branch_models.py`
  - Small JSON-serializable branch request/result/status contracts.
- `backend/phase1_runtime/branch_io.py`
  - Filesystem helpers for per-run branch directories, result files, status files, and logs.
- `backend/runtime/run_phase1_branch.py`
  - CLI entrypoint used by the coordinator to run a single branch process.
- `backend/phase1_runtime/branches/visual_branch.py`
  - Branch wrapper that invokes the existing visual extractor and writes `phase1_visual`.
- `backend/phase1_runtime/branches/audio_branch.py`
  - Branch wrapper that runs `VibeVoice -> forced alignment -> vibevoice_merge -> emotion2vec+`.
- `backend/phase1_runtime/branches/yamnet_branch.py`
  - Branch wrapper that runs YAMNet on CPU and writes the raw YAMNet payload.
- `tests/backend/phase1_runtime/test_branch_io.py`
  - Unit tests for branch filesystem contracts.
- `tests/backend/phase1_runtime/test_parallel_phase1_coordinator.py`
  - Coordinator and join-path tests with fake branch runners.
- `tests/backend/phase1_runtime/test_audio_branch.py`
  - Audio branch orchestration tests, including preservation of the native VibeVoice subprocess provider path.
- `tests/backend/phase1_runtime/test_phase1_branch_cli.py`
  - CLI/entrypoint tests for branch selection and output writing.

### Existing files to inspect during implementation

- `backend/runtime/vibevoice_native_worker.py`
  - Keep unchanged unless branch orchestration exposes a real bug; this remains the validated nested subprocess path.
- `backend/providers/vibevoice.py`
  - Preserve native-subprocess behavior; do not inline native VibeVoice into the main worker env.
- `backend/providers/forced_aligner.py`
  - Confirm GPU/CPU expectations for the audio branch.
- `backend/providers/emotion2vec.py`
  - Confirm branch-local sequencing and logging.
- `backend/providers/yamnet.py`
  - Ensure CPU branch behavior is explicit and does not silently use GPU in V1.
- `backend/phase1_runtime/visual.py`
  - Reuse as the visual branch engine rather than re-implementing extraction logic.

## Implementation Notes

- Keep the current Phase 1 artifact contracts stable:
  - `phase1_visual`
  - `diarization_payload`
  - `emotion2vec_payload`
  - `yamnet_payload`
- Do not remove the nested VibeVoice native subprocess. The audio branch should orchestrate around it, not absorb it.
- Use **process-level parallelism**, not threads and not manual cross-library CUDA stream management.
- V1 shared-GPU rule:
  - exactly **two** GPU-heavy branches may overlap on a single H100/H200:
    - `visual`
    - `audio`
  - `yamnet` stays on CPU in V1
  - do not add a third GPU branch
  - do not attempt manual CUDA-stream arbitration across libraries
  - the coordinator should enforce this by construction rather than relying on operator discipline
- Use the filesystem as the inter-process contract:
  - branch request JSON
  - branch result JSON
  - branch status JSON
  - branch log file
- The parent coordinator owns branch lifecycle and final join logic.
- In V1, YAMNet must run on CPU even if GPU is available.
- If one branch hard-fails, the coordinator should terminate sibling branches and fail the Phase 1 job.
- Keep the existing Phase 1 job-store contract stable unless the new branch model requires additional status fields.

## Task 1: Add branch runtime contracts and filesystem helpers

**Files:**
- Create: `backend/phase1_runtime/branch_models.py`
- Create: `backend/phase1_runtime/branch_io.py`
- Modify: `backend/phase1_runtime/models.py`
- Test: `tests/backend/phase1_runtime/test_branch_io.py`

- [ ] **Step 1: Write the failing tests for branch IO and models**

```python
from backend.phase1_runtime.branch_io import BranchPaths, build_branch_paths
from backend.phase1_runtime.branch_models import BranchKind


def test_build_branch_paths_creates_stable_layout(tmp_path):
    paths = build_branch_paths(run_root=tmp_path, branch=BranchKind.VISUAL)
    assert paths.request_path.name == "request.json"
    assert paths.result_path.name == "result.json"
    assert paths.status_path.name == "status.json"
    assert paths.log_path.name == "branch.log"


def test_branch_status_payload_is_json_round_trip():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/backend/phase1_runtime/test_branch_io.py -q`

Expected: FAIL because the branch modules do not exist yet.

- [ ] **Step 3: Implement minimal branch contracts**

Create:
- `BranchKind` enum with `visual`, `audio`, `yamnet`
- `BranchRequest`
- `BranchResultEnvelope`
- `BranchStatus`
- `BranchPaths`
- filesystem helpers to create branch directories and read/write request/result/status JSON

Keep the models focused and JSON-serializable.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/backend/phase1_runtime/test_branch_io.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/phase1_runtime/branch_models.py backend/phase1_runtime/branch_io.py backend/phase1_runtime/models.py tests/backend/phase1_runtime/test_branch_io.py
git commit -m "feat: add phase1 branch runtime contracts"
```

## Task 2: Add branch entrypoints for visual, audio, and YAMNet

**Files:**
- Create: `backend/phase1_runtime/branches/visual_branch.py`
- Create: `backend/phase1_runtime/branches/audio_branch.py`
- Create: `backend/phase1_runtime/branches/yamnet_branch.py`
- Create: `backend/runtime/run_phase1_branch.py`
- Test: `tests/backend/phase1_runtime/test_audio_branch.py`
- Test: `tests/backend/phase1_runtime/test_phase1_branch_cli.py`

- [ ] **Step 1: Write the failing tests for the branch wrappers**

```python
def test_audio_branch_runs_vibevoice_then_alignment_then_emotion(tmp_path, fake_audio_deps):
    result = run_audio_branch(...)
    assert result["diarization_payload"]["turns"]
    assert result["emotion2vec_payload"]["segments"]


def test_branch_cli_dispatches_visual_branch(tmp_path, monkeypatch):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/backend/phase1_runtime/test_audio_branch.py tests/backend/phase1_runtime/test_phase1_branch_cli.py -q`

Expected: FAIL because the branch modules/CLI do not exist yet.

- [ ] **Step 3: Implement the branch wrappers**

Implement:
- `visual_branch.run_visual_branch(...)`
  - invoke the existing visual extractor
  - write `phase1_visual`
- `audio_branch.run_audio_branch(...)`
  - call `vibevoice_provider.run(...)`
  - build preliminary turns
  - call `forced_aligner.run(...)`
  - call `merge_vibevoice_outputs(...)`
  - call `emotion_provider.run(...)`
  - return `diarization_payload` + `emotion2vec_payload`
- `yamnet_branch.run_yamnet_branch(...)`
  - call `yamnet_provider.run(...)`

Implement `backend/runtime/run_phase1_branch.py` to:
- read a branch request JSON
- instantiate the needed branch dependencies
- run exactly one branch
- write result/status JSON
- exit non-zero on branch failure

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/backend/phase1_runtime/test_audio_branch.py tests/backend/phase1_runtime/test_phase1_branch_cli.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/phase1_runtime/branches backend/runtime/run_phase1_branch.py tests/backend/phase1_runtime/test_audio_branch.py tests/backend/phase1_runtime/test_phase1_branch_cli.py
git commit -m "feat: add phase1 branch worker entrypoints"
```

## Task 3: Build the parent coordinator and branch lifecycle management

**Files:**
- Create: `backend/phase1_runtime/coordinator.py`
- Modify: `backend/phase1_runtime/extract.py`
- Modify: `backend/phase1_runtime/models.py`
- Test: `tests/backend/phase1_runtime/test_parallel_phase1_coordinator.py`

- [ ] **Step 1: Write the failing coordinator tests**

```python
def test_coordinator_launches_visual_audio_and_yamnet_branches(tmp_path, fake_branch_runner):
    outputs = run_parallel_phase1_sidecars(...)
    assert outputs.phase1_visual["tracks"]
    assert outputs.diarization_payload["turns"]
    assert outputs.yamnet_payload["events"] == []


def test_coordinator_kills_siblings_when_one_branch_fails(tmp_path, fake_branch_runner):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/backend/phase1_runtime/test_parallel_phase1_coordinator.py -q`

Expected: FAIL because the coordinator does not exist yet.

- [ ] **Step 3: Implement the parent coordinator**

Implement in `coordinator.py`:
- branch request creation
- branch directory initialization
- `subprocess.Popen(...)` launch for:
  - visual branch
  - audio branch
  - yamnet branch
- enforce the V1 scheduling policy:
  - launch `visual` and `audio` immediately
  - launch `yamnet` as a CPU branch with no CUDA env exposure
- per-branch timeout handling
- polling loop for exit codes
- sibling termination on hard failure
- result JSON loading and validation
- final join into `Phase1SidecarOutputs`

Then refactor `extract.py`:
- replace the current serial `run_phase1_sidecars(...)`
- expose `run_parallel_phase1_sidecars(...)`
- keep a small join layer there if needed, but move orchestration into `coordinator.py`

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/backend/phase1_runtime/test_parallel_phase1_coordinator.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/phase1_runtime/coordinator.py backend/phase1_runtime/extract.py backend/phase1_runtime/models.py tests/backend/phase1_runtime/test_parallel_phase1_coordinator.py
git commit -m "feat: add phase1 parallel coordinator"
```

## Task 4: Wire the new coordinator into the default Phase 1 runtime

**Files:**
- Modify: `backend/phase1_runtime/factory.py`
- Modify: `backend/phase1_runtime/runner.py`
- Modify: `backend/phase1_runtime/worker.py`
- Modify: `backend/providers/config.py`
- Test: `tests/backend/phase1_runtime/test_parallel_phase1_coordinator.py`
- Test: `tests/backend/providers/test_provider_config_and_clients.py`

- [ ] **Step 1: Write or extend the failing wiring/config tests**

```python
def test_default_phase1_job_runner_uses_parallel_sidecar_runtime(...):
    runner = build_default_phase1_job_runner(...)
    assert runner.run_phase1_sidecars.__name__ == "run_parallel_phase1_sidecars"


def test_phase1_runtime_settings_include_branch_timeouts(...):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/backend/phase1_runtime/test_parallel_phase1_coordinator.py tests/backend/providers/test_provider_config_and_clients.py -q`

Expected: FAIL because the runtime/config wiring is not present yet.

- [ ] **Step 3: Implement the wiring**

Update:
- `factory.py`
  - build the coordinator-aware sidecar runner
  - force YAMNet branch device to CPU for V1
- `runner.py`
  - invoke the new parallel Phase 1 sidecar path
- `worker.py`
  - preserve current job-store behavior while ensuring branch failure surfaces with useful logs
- `config.py`
  - add:
    - branch timeout settings
    - branch poll interval
    - explicit `phase1_parallel_enabled`
    - explicit `phase1_parallel_gpu_branch_limit=2`
    - explicit `yamnet_branch_device=cpu`

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/backend/phase1_runtime/test_parallel_phase1_coordinator.py tests/backend/providers/test_provider_config_and_clients.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/phase1_runtime/factory.py backend/phase1_runtime/runner.py backend/phase1_runtime/worker.py backend/providers/config.py tests/backend/phase1_runtime/test_parallel_phase1_coordinator.py tests/backend/providers/test_provider_config_and_clients.py
git commit -m "feat: wire phase1 parallel runtime into worker stack"
```

## Task 5: Update compatibility tests that encode the old serial runtime

**Files:**
- Modify: `tests/backend/phase1_runtime/test_runner.py`
- Modify: `tests/backend/providers/test_storage_and_phase1_runtime.py`
- Test: `tests/backend/phase1_runtime/test_runner.py`
- Test: `tests/backend/providers/test_storage_and_phase1_runtime.py`

- [ ] **Step 1: Write or update the failing compatibility assertions**

Replace serial-specific expectations with assertions that match the new architecture:
- the runner invokes the coordinator-backed sidecar path
- YAMNet is configured for CPU in V1
- `phase1_audio`, `phase1_visual`, `diarization_payload`, `emotion2vec_payload`, and `yamnet_payload` still appear in the final outputs

- [ ] **Step 2: Run tests to verify they fail against the old assumptions**

Run: `./.venv/bin/python -m pytest tests/backend/phase1_runtime/test_runner.py tests/backend/providers/test_storage_and_phase1_runtime.py -q`

Expected: FAIL because the tests still encode the serial runtime shape.

- [ ] **Step 3: Implement the compatibility test updates**

Update the tests so they validate:
- coordinator-backed execution instead of direct serial sidecar execution
- branch-aware runtime settings
- unchanged Phase 1 output contracts

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/backend/phase1_runtime/test_runner.py tests/backend/providers/test_storage_and_phase1_runtime.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/backend/phase1_runtime/test_runner.py tests/backend/providers/test_storage_and_phase1_runtime.py
git commit -m "test: update phase1 runtime tests for parallel coordinator"
```

## Task 6: Harden failure semantics, logging, and branch observability

**Files:**
- Modify: `backend/phase1_runtime/coordinator.py`
- Modify: `backend/runtime/run_phase1_branch.py`
- Modify: `backend/phase1_runtime/worker.py`
- Test: `tests/backend/phase1_runtime/test_parallel_phase1_coordinator.py`

- [ ] **Step 1: Add failing tests for branch failure visibility**

```python
def test_coordinator_persists_branch_status_and_log_paths_on_failure(...):
    ...


def test_branch_cli_writes_status_json_before_exiting_nonzero(...):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/backend/phase1_runtime/test_parallel_phase1_coordinator.py tests/backend/phase1_runtime/test_phase1_branch_cli.py -q`

Expected: FAIL because the observability semantics are incomplete.

- [ ] **Step 3: Implement logging and failure details**

Ensure:
- every branch writes:
  - request JSON
  - status JSON
  - result JSON on success
  - branch log file
- coordinator writes a top-level branch summary
- child stderr/stdout is captured into per-branch log files
- sibling termination is logged with branch names and pids
- worker failure payload includes the failing branch and branch log path

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/backend/phase1_runtime/test_parallel_phase1_coordinator.py tests/backend/phase1_runtime/test_phase1_branch_cli.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/phase1_runtime/coordinator.py backend/runtime/run_phase1_branch.py backend/phase1_runtime/worker.py tests/backend/phase1_runtime/test_parallel_phase1_coordinator.py tests/backend/phase1_runtime/test_phase1_branch_cli.py
git commit -m "feat: add branch logging and failure reporting"
```

## Task 7: Update docs for the new parent/branch Phase 1 runtime

**Files:**
- Modify: `docs/runtime/v3.1_runtime_guide.md`
- Modify: `docs/deployment/v3.1_phase1_digitalocean.md`
- Modify: `README.md`

- [ ] **Step 1: Write the doc changes**

Update the docs to reflect:
- parent coordinator + branch model
- visual/audio parallelism
- YAMNet CPU branch in V1
- why VibeVoice remains a nested subprocess
- new env/config knobs
- new log and workspace structure
- DO operational notes for branch logs and branch failures

- [ ] **Step 2: Verify docs match the implemented runtime**

Read:
- `backend/phase1_runtime/coordinator.py`
- `backend/runtime/run_phase1_branch.py`
- `backend/phase1_runtime/branches/audio_branch.py`

Expected: docs and code describe the same flow.

- [ ] **Step 3: Commit**

```bash
git add docs/runtime/v3.1_runtime_guide.md docs/deployment/v3.1_phase1_digitalocean.md README.md
git commit -m "docs: describe phase1 parallel coordinator runtime"
```

## Task 8: Run the focused verification suite

**Files:**
- Test only

- [ ] **Step 1: Run the Phase 1 runtime tests**

Run:

```bash
./.venv/bin/python -m pytest \
  tests/backend/phase1_runtime/test_branch_io.py \
  tests/backend/phase1_runtime/test_audio_branch.py \
  tests/backend/phase1_runtime/test_phase1_branch_cli.py \
  tests/backend/phase1_runtime/test_parallel_phase1_coordinator.py \
  tests/backend/phase1_runtime/test_runner.py \
  tests/backend/providers/test_storage_and_phase1_runtime.py -q
```

Expected: PASS

- [ ] **Step 2: Run the broader backend regression slice**

Run:

```bash
./.venv/bin/python -m pytest \
  tests/backend/phase1_runtime \
  tests/backend/providers \
  tests/backend/pipeline -q
```

Expected: PASS

- [ ] **Step 3: Run compile verification**

Run:

```bash
python3 -m py_compile $(find backend -type f -name '*.py' | sort)
```

Expected: no output, exit code 0

- [ ] **Step 4: Commit final verification adjustments if needed**

```bash
git add -A
git commit -m "test: verify phase1 parallel runtime"
```

## Design Constraints To Preserve

- Do not inline native VibeVoice into the audio worker’s main environment unless we intentionally take on a new env-unification project.
- Do not use Python threads for branch orchestration.
- Do not use manual cross-library CUDA streams as the top-level orchestration mechanism.
- Do not move YAMNet back onto GPU in this plan.
- Do not change downstream Phase 1 artifact schemas.
- Do not fold RF-DETR and tracking back into one opaque library call.

## Follow-Up Work Explicitly Out Of Scope

- Running YAMNet concurrently on GPU
- CUDA MPS experiments
- Full branch-level GPU telemetry/profiling beyond basic timings/logging
- Collapsing the VibeVoice native subprocess into a single audio-branch environment
- DO smoke runs on this plan document alone
