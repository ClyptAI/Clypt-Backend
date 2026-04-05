# V3.1 Backend Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the refactored Phase 1-4 backend into a real runnable V3.1 system with live providers, a rebuilt Phase 1 worker/service, and restored V3.1 deployment/runtime docs.

**Architecture:** Keep the current refactored pipeline under `backend/pipeline` as the canonical semantic core. Add provider adapters, runtime entrypoints, and a new V3.1 Phase 1 worker/service around that core, reusing operational patterns from `main` without restoring legacy LR-ASD-era architecture.

**Tech Stack:** Python, Pydantic, FastAPI, httpx, Vertex AI / Gemini, pyannote cloud API, emotion2vec+, TensorFlow/YAMNet, ffmpeg/ffprobe, DigitalOcean worker/runtime.

---

## File Map

### Existing files to extend

- Modify: `backend/pipeline/config.py`
- Modify: `backend/pipeline/contracts.py`
- Modify: `backend/pipeline/orchestrator.py`
- Modify: `backend/pipeline/semantics/prompts.py`
- Modify: `backend/pipeline/semantics/node_embeddings.py`
- Modify: `backend/pipeline/candidates/query_embeddings.py`
- Modify: `backend/pipeline/timeline/emotion_events.py`
- Modify: `backend/pipeline/timeline/audio_events.py`
- Modify: `backend/pipeline/timeline/tracklets.py`
- Modify: `requirements.txt`
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `docs/planning/v3.1_backend_completion_spec.md`

### New provider/runtime files

- Create: `backend/providers/__init__.py`
- Create: `backend/providers/config.py`
- Create: `backend/providers/pyannote.py`
- Create: `backend/providers/gemini.py`
- Create: `backend/providers/embeddings.py`
- Create: `backend/providers/emotion2vec.py`
- Create: `backend/providers/yamnet.py`
- Create: `backend/providers/youtube.py`
- Create: `backend/runtime/__init__.py`
- Create: `backend/runtime/run_phase1.py`
- Create: `backend/runtime/run_phase14.py`
- Create: `backend/runtime/phase1_job_models.py`
- Create: `backend/runtime/phase1_job_store.py`

### New pipeline runtime helpers

- Create: `backend/pipeline/semantics/runtime.py`
- Create: `backend/pipeline/graph/prompts.py`
- Create: `backend/pipeline/graph/runtime.py`
- Create: `backend/pipeline/candidates/prompts.py`
- Create: `backend/pipeline/candidates/runtime.py`

### New Phase 1 worker/service files

- Create: `backend/phase1_runtime/__init__.py`
- Create: `backend/phase1_runtime/app.py`
- Create: `backend/phase1_runtime/worker.py`
- Create: `backend/phase1_runtime/extract.py`
- Create: `backend/phase1_runtime/jobs.py`
- Create: `backend/phase1_runtime/models.py`
- Create: `backend/phase1_runtime/state_store.py`
- Create: `backend/phase1_runtime/storage.py`
- Create: `backend/phase1_runtime/visual.py`

### New deployment/docs files

- Create: `requirements-phase1-worker.txt`
- Create: `scripts/do_phase1/bootstrap_gpu_droplet.sh`
- Create: `scripts/do_phase1/deploy_phase1_service.sh`
- Create: `scripts/do_phase1/run_remote_job.py`
- Create: `scripts/do_phase1/run_phase1_only_remote.sh`
- Create: `scripts/do_phase1/run_worker_service.sh`
- Create: `scripts/do_phase1/systemd/clypt-phase1-api.service`
- Create: `scripts/do_phase1/systemd/clypt-phase1-worker.service`
- Create: `docs/deployment/do-phase1-v31-digitalocean.md`
- Create: `docs/runtime/phase1_runtime.md`

### New tests

- Create: `tests/backend/providers/test_pyannote.py`
- Create: `tests/backend/providers/test_embeddings.py`
- Create: `tests/backend/providers/test_gemini.py`
- Create: `tests/backend/providers/test_emotion2vec.py`
- Create: `tests/backend/providers/test_yamnet.py`
- Create: `tests/backend/runtime/test_run_phase1.py`
- Create: `tests/backend/runtime/test_run_phase14.py`
- Create: `tests/backend/phase1_runtime/test_models.py`
- Create: `tests/backend/phase1_runtime/test_state_store.py`
- Create: `tests/backend/phase1_runtime/test_extract.py`
- Create: `tests/backend/phase1_runtime/test_worker.py`
- Modify: `tests/backend/pipeline/test_orchestrator_phase14.py`

## Task 1: Provider Configuration And Env Discovery

**Files:**
- Create: `backend/providers/config.py`
- Modify: `.env.example`
- Modify: `requirements.txt`
- Test: `tests/backend/providers/test_pyannote.py`

- [ ] **Step 1: Write failing config/env tests**

Add tests for:
- loading pyannote API key from env
- loading Vertex project/location/bucket from env or gcloud defaults
- failing clearly when required provider env is missing

- [ ] **Step 2: Run provider-config tests to verify failure**

Run: `python -m pytest tests/backend/providers/test_pyannote.py -q`
Expected: FAIL because provider config module does not exist yet.

- [ ] **Step 3: Implement provider config discovery**

Build:
- env-backed pyannote config
- env-or-gcloud-backed Vertex config
- shared bucket/project resolution helpers

Use:
- `PYANNOTE_API_KEY`
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`
- `GCS_BUCKET`

Allow fallback discovery via `gcloud config get-value project` when env is absent.

- [ ] **Step 4: Add the minimal runtime deps**

Update `requirements.txt` with the live runtime surface needed now:
- `httpx`
- `fastapi`
- `uvicorn`
- `google-cloud-aiplatform` or Vertex-compatible SDK dependency
- any other minimal provider/runtime packages needed for the foundation

- [ ] **Step 5: Re-run provider-config tests**

Run: `python -m pytest tests/backend/providers/test_pyannote.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/providers/config.py .env.example requirements.txt tests/backend/providers/test_pyannote.py
git commit -m "feat: add provider config discovery for v3.1"
```

## Task 2: Pyannote Cloud Client

**Files:**
- Create: `backend/providers/pyannote.py`
- Modify: `backend/pipeline/timeline/pyannote_merge.py`
- Test: `tests/backend/providers/test_pyannote.py`

- [ ] **Step 1: Extend tests for pyannote request/response normalization**

Cover:
- `/v1/diarize` request shaping
- optional `/v1/identify` request shaping
- normalization of raw payloads into the existing merge layer contract
- cloud-only behavior

- [ ] **Step 2: Run pyannote tests to verify failure**

Run: `python -m pytest tests/backend/providers/test_pyannote.py -q`
Expected: FAIL because the client is not implemented yet.

- [ ] **Step 3: Implement pyannote cloud client**

Build:
- authenticated request helper
- `run_diarize(...)`
- `run_identify(...)`
- optional parallel call orchestration helper
- raw payload persistence helper hooks

Do not add local HF fallback in this task.

- [ ] **Step 4: Keep the current merge layer unchanged as canonical normalization**

Use [pyannote_merge.py](/Users/rithvik/Clypt-V3/backend/pipeline/timeline/pyannote_merge.py) as the canonical normalization layer.
Only modify it if raw cloud payload normalization needs small compatibility additions.

- [ ] **Step 5: Re-run pyannote tests**

Run: `python -m pytest tests/backend/providers/test_pyannote.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/providers/pyannote.py backend/pipeline/timeline/pyannote_merge.py tests/backend/providers/test_pyannote.py
git commit -m "feat: add pyannote cloud client for v3.1"
```

## Task 3: Vertex Gemini + Embeddings Runtime

**Files:**
- Create: `backend/providers/gemini.py`
- Create: `backend/providers/embeddings.py`
- Modify: `backend/pipeline/_embedding_utils.py`
- Modify: `backend/pipeline/semantics/node_embeddings.py`
- Modify: `backend/pipeline/candidates/query_embeddings.py`
- Test: `tests/backend/providers/test_gemini.py`
- Test: `tests/backend/providers/test_embeddings.py`

- [ ] **Step 1: Write failing Gemini and embeddings tests**

Cover:
- structured JSON generation calls
- embeddings calls for node text and prompt text
- deterministic local fallback for tests

- [ ] **Step 2: Run provider tests to verify failure**

Run: `python -m pytest tests/backend/providers/test_gemini.py tests/backend/providers/test_embeddings.py -q`
Expected: FAIL

- [ ] **Step 3: Implement Vertex Gemini client**

Build:
- model execution wrapper
- structured-output helper
- retry/timeout handling
- request metadata logging

- [ ] **Step 4: Implement embeddings provider**

Build:
- provider-backed text embeddings
- deterministic fallback mode for tests

- [ ] **Step 5: Refactor pipeline embedding callers**

Make:
- `embed_semantic_nodes(...)`
- `embed_prompt_texts(...)`

accept an injected embeddings provider/runtime instead of always using the local hash helper.

Keep the current deterministic helper as a fallback/test path.

- [ ] **Step 6: Re-run provider tests**

Run: `python -m pytest tests/backend/providers/test_gemini.py tests/backend/providers/test_embeddings.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add backend/providers/gemini.py backend/providers/embeddings.py backend/pipeline/_embedding_utils.py backend/pipeline/semantics/node_embeddings.py backend/pipeline/candidates/query_embeddings.py tests/backend/providers/test_gemini.py tests/backend/providers/test_embeddings.py
git commit -m "feat: add Vertex Gemini and embeddings providers"
```

## Task 4: emotion2vec+ And YAMNet Providers

**Files:**
- Create: `backend/providers/emotion2vec.py`
- Create: `backend/providers/yamnet.py`
- Modify: `backend/pipeline/timeline/emotion_events.py`
- Modify: `backend/pipeline/timeline/audio_events.py`
- Test: `tests/backend/providers/test_emotion2vec.py`
- Test: `tests/backend/providers/test_yamnet.py`

- [ ] **Step 1: Write failing provider tests**

Cover:
- turn-level emotion2vec+ output normalization
- YAMNet raw detections to merged-event conversion
- provider output shape compatibility with the current timeline builders

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/backend/providers/test_emotion2vec.py tests/backend/providers/test_yamnet.py -q`
Expected: FAIL

- [ ] **Step 3: Implement emotion2vec+ provider**

Build:
- model loader
- turn-level inference API
- output normalization to the current Phase 1 sidecar shape

- [ ] **Step 4: Implement YAMNet provider**

Build:
- audio loading/resampling flow
- inference entrypoint
- raw detections
- merged curated event spans

- [ ] **Step 5: Keep the timeline builders as artifact shapers**

Do not move model execution into:
- `emotion_events.py`
- `audio_events.py`

These should remain artifact shapers over normalized provider output.

- [ ] **Step 6: Re-run tests**

Run: `python -m pytest tests/backend/providers/test_emotion2vec.py tests/backend/providers/test_yamnet.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add backend/providers/emotion2vec.py backend/providers/yamnet.py backend/pipeline/timeline/emotion_events.py backend/pipeline/timeline/audio_events.py tests/backend/providers/test_emotion2vec.py tests/backend/providers/test_yamnet.py
git commit -m "feat: add emotion2vec and YAMNet providers"
```

## Task 5: Media Acquisition And Visual Phase 1 Foundation

**Files:**
- Create: `backend/providers/youtube.py`
- Create: `backend/phase1_runtime/visual.py`
- Test: `tests/backend/phase1_runtime/test_extract.py`

- [ ] **Step 1: Write failing media/runtime tests**

Cover:
- source URL media acquisition
- local workspace preparation
- normalized source video/audio paths
- visual payload shape generation

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/backend/phase1_runtime/test_extract.py -q`
Expected: FAIL

- [ ] **Step 3: Adapt the useful media helpers from `main`**

Use `main:backend/pipeline/phase_1_do_pipeline.py` as the operational reference for:
- media download
- ffprobe helpers
- ffmpeg normalization

Copy/adapt only what is needed for V3.1 source preparation.

- [ ] **Step 4: Implement the minimal visual Phase 1 foundation**

Build a V3.1 visual extraction path that produces:
- `video_metadata`
- `shot_changes`
- `tracks`

Do not restore speaker-binding outputs.

- [ ] **Step 5: Re-run tests**

Run: `python -m pytest tests/backend/phase1_runtime/test_extract.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/providers/youtube.py backend/phase1_runtime/visual.py tests/backend/phase1_runtime/test_extract.py
git commit -m "feat: add media acquisition and visual phase1 foundation"
```

## Task 6: Phase 2 Runtime Execution

**Files:**
- Create: `backend/pipeline/semantics/runtime.py`
- Modify: `backend/pipeline/semantics/prompts.py`
- Modify: `backend/pipeline/orchestrator.py`
- Test: `tests/backend/pipeline/test_semantics_phase2.py`
- Test: `tests/backend/pipeline/test_orchestrator_phase14.py`

- [ ] **Step 1: Extend tests for real runtime orchestration**

Cover:
- building merge/classify prompts
- invoking provider-backed Gemini runtime
- invoking boundary reconciliation runtime
- preserving current validation behavior

- [ ] **Step 2: Run Phase 2 tests to verify failure**

Run: `python -m pytest tests/backend/pipeline/test_semantics_phase2.py tests/backend/pipeline/test_orchestrator_phase14.py -q`
Expected: FAIL

- [ ] **Step 3: Implement Phase 2 runtime execution layer**

Build:
- neighborhood request batching
- prompt construction
- provider calls
- response adaptation through existing functions

- [ ] **Step 4: Refactor orchestrator Phase 2 to support live provider mode**

Support both:
- injected response mode for tests
- live provider mode for runtime

- [ ] **Step 5: Re-run Phase 2 tests**

Run: `python -m pytest tests/backend/pipeline/test_semantics_phase2.py tests/backend/pipeline/test_orchestrator_phase14.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/pipeline/semantics/runtime.py backend/pipeline/semantics/prompts.py backend/pipeline/orchestrator.py tests/backend/pipeline/test_semantics_phase2.py tests/backend/pipeline/test_orchestrator_phase14.py
git commit -m "feat: add live phase2 Gemini runtime"
```

## Task 7: Phase 3 Runtime Execution

**Files:**
- Create: `backend/pipeline/graph/prompts.py`
- Create: `backend/pipeline/graph/runtime.py`
- Modify: `backend/pipeline/orchestrator.py`
- Test: `tests/backend/pipeline/test_semantic_edges_phase3.py`

- [ ] **Step 1: Extend Phase 3 tests for provider-backed execution**

Cover:
- local semantic edge batch building
- long-range pair adjudication prompts
- live runtime execution using mocked Gemini provider

- [ ] **Step 2: Run Phase 3 tests to verify failure**

Run: `python -m pytest tests/backend/pipeline/test_semantic_edges_phase3.py -q`
Expected: FAIL

- [ ] **Step 3: Implement graph prompt builders**

Build:
- local semantic edge prompt builder
- long-range pair adjudication prompt builder

- [ ] **Step 4: Implement graph runtime execution**

Build:
- local batch construction
- Gemini execution
- long-range adjudication execution
- adaptation through existing edge validators

- [ ] **Step 5: Re-run Phase 3 tests**

Run: `python -m pytest tests/backend/pipeline/test_semantic_edges_phase3.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/pipeline/graph/prompts.py backend/pipeline/graph/runtime.py backend/pipeline/orchestrator.py tests/backend/pipeline/test_semantic_edges_phase3.py
git commit -m "feat: add live phase3 graph runtime"
```

## Task 8: Phase 4 Runtime Execution And Budget Enforcement

**Files:**
- Create: `backend/pipeline/candidates/prompts.py`
- Create: `backend/pipeline/candidates/runtime.py`
- Modify: `backend/pipeline/config.py`
- Modify: `backend/pipeline/orchestrator.py`
- Test: `tests/backend/pipeline/test_candidate_review_phase4.py`
- Test: `tests/backend/pipeline/test_candidates_phase4.py`

- [ ] **Step 1: Extend Phase 4 tests**

Cover:
- subgraph review prompt building
- pooled review prompt building
- budget enforcement
- fail-hard behavior on invalid pooled review

- [ ] **Step 2: Run Phase 4 tests to verify failure**

Run: `python -m pytest tests/backend/pipeline/test_candidate_review_phase4.py tests/backend/pipeline/test_candidates_phase4.py -q`
Expected: FAIL

- [ ] **Step 3: Implement candidate prompt builders**

Build:
- subgraph review prompt builder
- pooled review prompt builder

- [ ] **Step 4: Implement candidate runtime execution**

Build:
- subgraph review calls
- pooled final review call
- debug artifact recording

- [ ] **Step 5: Enforce budget config**

Actually enforce:
- max total prompts
- max subgraphs per run
- max final review calls

- [ ] **Step 6: Re-run Phase 4 tests**

Run: `python -m pytest tests/backend/pipeline/test_candidate_review_phase4.py tests/backend/pipeline/test_candidates_phase4.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add backend/pipeline/candidates/prompts.py backend/pipeline/candidates/runtime.py backend/pipeline/config.py backend/pipeline/orchestrator.py tests/backend/pipeline/test_candidate_review_phase4.py tests/backend/pipeline/test_candidates_phase4.py
git commit -m "feat: add live phase4 runtime and budget enforcement"
```

## Task 9: Rebuild The Phase 1 Worker / Service

**Files:**
- Create: `backend/phase1_runtime/app.py`
- Create: `backend/phase1_runtime/worker.py`
- Create: `backend/phase1_runtime/extract.py`
- Create: `backend/phase1_runtime/jobs.py`
- Create: `backend/phase1_runtime/models.py`
- Create: `backend/phase1_runtime/state_store.py`
- Create: `backend/phase1_runtime/storage.py`
- Create: `backend/runtime/phase1_job_models.py`
- Create: `backend/runtime/phase1_job_store.py`
- Test: `tests/backend/phase1_runtime/test_models.py`
- Test: `tests/backend/phase1_runtime/test_state_store.py`
- Test: `tests/backend/phase1_runtime/test_worker.py`

- [ ] **Step 1: Write failing Phase 1 runtime tests**

Cover:
- job creation and status persistence
- worker execution lifecycle
- log path handling
- manifest/result persistence

- [ ] **Step 2: Run runtime tests to verify failure**

Run: `python -m pytest tests/backend/phase1_runtime/test_models.py tests/backend/phase1_runtime/test_state_store.py tests/backend/phase1_runtime/test_worker.py -q`
Expected: FAIL

- [ ] **Step 3: Adapt the operational architecture from `main`**

Use `main` as the reference for:
- FastAPI service layout
- SQLite job store pattern
- job polling model
- log streaming model

Do not carry over legacy extraction internals.

- [ ] **Step 4: Implement the new worker execution flow**

Worker should:
- prepare media
- kick off pyannote cloud in parallel
- run local visual extraction
- run emotion2vec+ on the worker
- run YAMNet on the worker
- join outputs
- call the current Phase 1 artifact builders
- optionally continue into full Phase 1-4 orchestration

- [ ] **Step 5: Re-run Phase 1 runtime tests**

Run: `python -m pytest tests/backend/phase1_runtime/test_models.py tests/backend/phase1_runtime/test_state_store.py tests/backend/phase1_runtime/test_worker.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/phase1_runtime backend/runtime/phase1_job_models.py backend/runtime/phase1_job_store.py tests/backend/phase1_runtime
git commit -m "feat: rebuild v3.1 phase1 worker and service"
```

## Task 10: Local Entrypoints

**Files:**
- Create: `backend/runtime/run_phase1.py`
- Create: `backend/runtime/run_phase14.py`
- Test: `tests/backend/runtime/test_run_phase1.py`
- Test: `tests/backend/runtime/test_run_phase14.py`

- [ ] **Step 1: Write failing runner tests**

Cover:
- Phase 1 local command path
- Phase 1-4 local command path
- source URL execution
- artifact root output

- [ ] **Step 2: Run runner tests to verify failure**

Run: `python -m pytest tests/backend/runtime/test_run_phase1.py tests/backend/runtime/test_run_phase14.py -q`
Expected: FAIL

- [ ] **Step 3: Implement local runners**

Build:
- a Phase 1-only CLI/module
- a Phase 1-4 CLI/module

Both should use the rebuilt runtime and orchestrator rather than reimplementing logic.

- [ ] **Step 4: Re-run runner tests**

Run: `python -m pytest tests/backend/runtime/test_run_phase1.py tests/backend/runtime/test_run_phase14.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/runtime/run_phase1.py backend/runtime/run_phase14.py tests/backend/runtime/test_run_phase1.py tests/backend/runtime/test_run_phase14.py
git commit -m "feat: add v3.1 local runtime entrypoints"
```

## Task 11: Worker Requirements And DO Scripts

**Files:**
- Create: `requirements-phase1-worker.txt`
- Create: `scripts/do_phase1/bootstrap_gpu_droplet.sh`
- Create: `scripts/do_phase1/deploy_phase1_service.sh`
- Create: `scripts/do_phase1/run_remote_job.py`
- Create: `scripts/do_phase1/run_phase1_only_remote.sh`
- Create: `scripts/do_phase1/run_worker_service.sh`
- Create: `scripts/do_phase1/systemd/clypt-phase1-api.service`
- Create: `scripts/do_phase1/systemd/clypt-phase1-worker.service`
- Test: lightweight smoke validation via shell syntax and import checks

- [ ] **Step 1: Draft worker dependency file from `main`**

Start from `main:requirements-do-phase1.txt`, then:
- remove LR-ASD-only deps
- keep what is truly needed for V3.1 worker runtime
- add emotion2vec+/YAMNet/runtime deps

- [ ] **Step 2: Recreate DO scripts in V3.1 form**

Use `main` operational flow as reference for:
- droplet bootstrap
- service deploy
- remote job submission
- worker service startup

Point them at the new V3.1 runtime paths.

- [ ] **Step 3: Validate script syntax and imports**

Run:
- `bash -n scripts/do_phase1/bootstrap_gpu_droplet.sh`
- `bash -n scripts/do_phase1/deploy_phase1_service.sh`
- `bash -n scripts/do_phase1/run_phase1_only_remote.sh`
- `python3 -m py_compile backend/phase1_runtime/*.py backend/runtime/*.py`

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add requirements-phase1-worker.txt scripts/do_phase1 backend/phase1_runtime backend/runtime
git commit -m "feat: restore v3.1 worker requirements and do scripts"
```

## Task 12: Documentation Pass

**Files:**
- Create: `docs/deployment/do-phase1-v31-digitalocean.md`
- Create: `docs/runtime/phase1_runtime.md`
- Modify: `docs/planning/v3.1_backend_completion_spec.md`
- Modify: `README.md`

- [ ] **Step 1: Write the V3.1 DO deployment doc**

Use `main:docs/deployment/do-phase1-digitalocean.md` as the operational reference.

Required content:
- project/resource naming
- `atl1` primary, `nyc` fallback
- image/size details
- bootstrap/deploy flow
- env/secret setup
- systemd services
- remote log watching

Do not create the droplet yet.

- [ ] **Step 2: Write a V3.1 runtime doc**

Document:
- local run commands
- worker/service layout
- provider dependencies
- artifact flow

- [ ] **Step 3: Update top-level docs**

Make README reflect:
- the new runtime entrypoints
- the provider/runtime architecture
- the fact that DO deployment exists again in V3.1 form

- [ ] **Step 4: Commit**

```bash
git add docs/deployment/do-phase1-v31-digitalocean.md docs/runtime/phase1_runtime.md docs/planning/v3.1_backend_completion_spec.md README.md
git commit -m "docs: add v3.1 runtime and deployment guides"
```

## Task 13: End-to-End Verification

**Files:**
- Modify as needed based on failures from earlier tasks
- Test: full backend suite

- [ ] **Step 1: Run focused provider/runtime tests**

Run:
`python -m pytest tests/backend/providers tests/backend/phase1_runtime tests/backend/runtime -q`

Expected: PASS

- [ ] **Step 2: Run full pipeline tests**

Run:
`python -m pytest tests/backend/pipeline -q`

Expected: PASS

- [ ] **Step 3: Run combined suite**

Run:
`python -m pytest tests/backend -q`

Expected: PASS

- [ ] **Step 4: Compile-check the runtime tree**

Run:
`python3 -m py_compile backend/providers/*.py backend/phase1_runtime/*.py backend/runtime/*.py backend/pipeline/*.py backend/pipeline/timeline/*.py backend/pipeline/semantics/*.py backend/pipeline/graph/*.py backend/pipeline/candidates/*.py`

Expected: PASS

- [ ] **Step 5: Commit any final fixes**

```bash
git add .
git commit -m "test: verify v3.1 backend completion stack"
```
