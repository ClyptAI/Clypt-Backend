# Balanced Hybrid Hackathon Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Modal with a DigitalOcean GPU Droplet-backed asynchronous Phase 1 extraction service, keep the existing DeepMind/GCP graph pipeline intact, and create the support artifacts needed for a Lovable-built frontend handoff.

**Architecture:** Introduce a versioned Phase 1 extraction contract and a DO-hosted extraction service that owns long-running job execution. Refactor the current pipeline so Phase 1 becomes an async job submission and resume flow, while Phases 2-5 and Remotion continue to operate on the existing graph/retrieval path. Frontend implementation stays teammate-owned in Lovable, with this repo gaining prompt, token, and integration guidance rather than a parallel UI rewrite.

**Tech Stack:** Python 3.11, FastAPI-style extraction service, DigitalOcean GPU Droplets, Google Vertex AI / Gemini, Cloud Spanner, GCS, Next.js, Lovable, pytest

---

## Planned File Map

### Backend contract and pipeline

- Create: `backend/pipeline/phase1_contract.py`
  Purpose: Versioned manifest types, job states, artifact references, retry/failure payloads, and validation helpers.
- Create: `tests/backend/pipeline/test_phase1_contract.py`
  Purpose: Contract validation tests and backward-compatibility expectations.
- Create: `backend/pipeline/do_phase1_client.py`
  Purpose: Client for submitting, polling, resuming, and downloading manifest results from the DO extraction service.
- Modify: `backend/pipeline/phase_1_modal_pipeline.py`
  Purpose: Replace Modal orchestration with async DO job orchestration while preserving downstream semantics.
- Modify: `backend/pipeline/run_pipeline.py`
  Purpose: Update end-to-end pipeline entrypoint to use the new Phase 1 job model.

### DigitalOcean extraction service

- Modify: `requirements.txt`
  Purpose: Add FastAPI/httpx/uvicorn or equivalent dependencies needed by the DO service and tests.
- Create: `backend/do_phase1_service/__init__.py`
  Purpose: Package marker.
- Create: `backend/do_phase1_service/app.py`
  Purpose: HTTP entrypoint for create-job, get-job, and result endpoints.
- Create: `backend/do_phase1_service/models.py`
  Purpose: Service request/response models mapped to the contract.
- Create: `backend/do_phase1_service/jobs.py`
  Purpose: Long-running job lifecycle, persistence hooks, and retry/resume handling.
- Create: `backend/do_phase1_service/extract.py`
  Purpose: Actual Phase 1 extraction runner ported from the current Modal worker path.
- Create: `backend/do_phase1_service/worker.py`
  Purpose: Dedicated worker-loop entrypoint that dequeues and executes extraction jobs outside the request path.
- Create: `backend/do_phase1_service/state_store.py`
  Purpose: Durable job state persistence backed by SQLite on the droplet.
- Create: `backend/do_phase1_service/storage.py`
  Purpose: Artifact writing and manifest upload helpers.
- Create: `tests/backend/do_phase1_service/test_jobs.py`
  Purpose: Job state lifecycle and resumability tests.
- Create: `tests/backend/do_phase1_service/test_app.py`
  Purpose: API behavior tests for create/poll/result flows.
- Create: `tests/backend/do_phase1_service/test_extract.py`
  Purpose: Extraction runner contract tests against the Phase 1 workload.
- Create: `tests/backend/do_phase1_service/test_state_store.py`
  Purpose: Persistence and restart/recovery tests.
- Create: `tests/backend/do_phase1_service/test_storage.py`
  Purpose: Durable artifact upload and manifest URI tests.
- Create: `tests/backend/do_phase1_service/test_worker.py`
  Purpose: Queue transition tests for `queued -> running -> succeeded/failed`.

### Docs and frontend support

- Create: `docs/frontend/lovable-handoff.md`
  Purpose: Frontend integration contract, import expectations, backend endpoints, and teammate handoff notes.
- Create: `docs/frontend/lovable-prompt-pack.md`
  Purpose: Prompting guidance for using Lovable on Cortex and related flows.
- Create: `docs/frontend/brand-tokens.md`
  Purpose: Brand tokens, design principles, and UI constraints for Lovable-generated work.
- Modify: `README.md`
  Purpose: Update stack story, runtime instructions, and infrastructure notes.

## Task 1: Lock the Phase 1 Extraction Contract

**Files:**
- Create: `backend/pipeline/phase1_contract.py`
- Test: `tests/backend/pipeline/test_phase1_contract.py`

- [ ] **Step 1: Write the failing contract tests**

```python
from backend.pipeline.phase1_contract import Phase1Manifest, JobState


def test_manifest_requires_contract_version_and_job_state():
    manifest = Phase1Manifest.model_validate(
        {
            "contract_version": "v1",
            "job_id": "job_123",
            "status": JobState.SUCCEEDED,
            "source_video": {"source_url": "https://youtube.com/watch?v=x"},
            "artifacts": {
                "transcript": {"uri": "gs://bucket/transcript.json"},
                "visual_tracking": {"uri": "gs://bucket/tracking.json"},
            },
            "metadata": {"runtime": {"provider": "digitalocean"}, "timings": {}},
        }
    )
    assert manifest.contract_version == "v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/backend/pipeline/test_phase1_contract.py -v`
Expected: FAIL because `phase1_contract.py` does not exist yet.

- [ ] **Step 3: Implement the minimal contract**

```python
class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Phase1Manifest(BaseModel):
    contract_version: str
    job_id: str
    status: JobState
```

- [ ] **Step 4: Add the missing required fields**

Add:
- source video reference
- canonical video GCS URI for downstream phases, preserving `gs://clypt-storage-v2/phase_1/video.mp4` or its configured equivalent
- transcript artifact
- visual tracking artifact
- optional events artifact
- runtime/timing metadata
- optional quality metrics
- retry/failure metadata placeholders
- compatibility guarantees for legacy downstream payload needs:
  - transcript payload includes `words`
  - transcript payload includes `speaker_bindings`
  - visual payload includes `tracks`
  - visual payload includes detection blocks required by existing consumers

- [ ] **Step 5: Re-run the tests**

Run: `pytest tests/backend/pipeline/test_phase1_contract.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/pipeline/phase1_contract.py tests/backend/pipeline/test_phase1_contract.py
git commit -m "feat: add phase 1 extraction contract"
```

## Task 2: Build the DigitalOcean Phase 1 Service Skeleton

**Files:**
- Modify: `requirements.txt`
- Create: `backend/do_phase1_service/app.py`
- Create: `backend/do_phase1_service/models.py`
- Create: `backend/do_phase1_service/jobs.py`
- Create: `backend/do_phase1_service/extract.py`
- Create: `backend/do_phase1_service/worker.py`
- Create: `backend/do_phase1_service/state_store.py`
- Create: `backend/do_phase1_service/storage.py`
- Create: `tests/backend/do_phase1_service/test_app.py`
- Create: `tests/backend/do_phase1_service/test_jobs.py`
- Create: `tests/backend/do_phase1_service/test_extract.py`
- Create: `tests/backend/do_phase1_service/test_state_store.py`
- Create: `tests/backend/do_phase1_service/test_storage.py`
- Create: `tests/backend/do_phase1_service/test_worker.py`

- [ ] **Step 1: Write failing job lifecycle tests**

```python
def test_create_job_returns_queued_manifest():
    payload = {"source_url": "https://youtube.com/watch?v=x"}
    job = create_job(payload)
    assert job.status == "queued"
```

- [ ] **Step 2: Write failing persistence/restart tests**

```python
def test_job_state_survives_process_restart(tmp_path):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    store.save_job(job_id="job_123", status="running")

    reloaded = SQLiteJobStore(tmp_path / "jobs.db")
    assert reloaded.get_job("job_123").status == "running"
```

- [ ] **Step 3: Write the failing extraction-runner tests**

```python
def test_extract_job_produces_manifest_and_artifacts(tmp_path):
    result = run_extraction_job(source_video=tmp_path / "video.mp4", output_dir=tmp_path)
    assert result.status == "succeeded"
    assert result.artifacts.transcript.uri
    assert result.artifacts.visual_tracking.uri
```

- [ ] **Step 4: Write the failing durable-storage tests**

```python
def test_manifest_uses_durable_storage_uris(tmp_path):
    manifest = persist_phase1_outputs(...)
    assert manifest.artifacts.transcript.uri.startswith("gs://")
    assert manifest.artifacts.visual_tracking.uri.startswith("gs://")
```

- [ ] **Step 5: Run the job, persistence, extraction, and storage tests**

Run: `pytest tests/backend/do_phase1_service/test_jobs.py tests/backend/do_phase1_service/test_state_store.py tests/backend/do_phase1_service/test_extract.py tests/backend/do_phase1_service/test_storage.py -v`
Expected: FAIL because service modules do not exist yet.

- [ ] **Step 6: Implement the durable state store**

Decision:
- Use SQLite on the droplet's persistent disk as the default job-state backing store.
- Persist job status, retries, timestamps, manifest location, and failure payload.

- [ ] **Step 6A: Add the service/runtime dependencies**

Update `requirements.txt` to include the minimum stack needed for this service and its tests:
- `fastapi`
- `uvicorn`
- `httpx`
- any minimal validation/runtime helpers actually chosen for implementation

- [ ] **Step 7: Port the actual extraction workload**

Requirements:
- Reuse logic from `backend/modal_worker.py` by extracting shared code or moving it into `backend/do_phase1_service/extract.py`
- The job runner must actually perform ASR, tracking, face identity stabilization, and speaker binding
- The job runner must emit the Phase 1 manifest and artifact set defined by `phase1_contract.py`
- The job runner must upload transcript and tracking artifacts to durable storage before returning a manifest
- During the hackathon, durable storage should default to GCS so downstream phases keep receiving stable artifact URIs
- The job runner must ensure the canonical source video is uploaded to `gs://clypt-storage-v2/phase_1/video.mp4` or the configured equivalent before downstream phases begin
- Avoid duplicating the entire worker blindly if a shared extraction core can be introduced cleanly

- [ ] **Step 8: Choose and implement the execution model**

Execution model decision:
- `POST /jobs` only enqueues work in SQLite and returns immediately
- `backend/do_phase1_service/worker.py` is the dedicated worker-loop entrypoint on the droplet
- the worker process reads queued jobs from SQLite, marks them `running`, executes extraction, and writes final state back to SQLite
- on service restart, the worker loop scans for `queued` and recoverable `running` jobs and resumes them
- SQLite remains the source of truth for queued/running/succeeded/failed transitions

- [ ] **Step 8A: Add the failing worker-loop test**

```python
def test_worker_promotes_job_through_lifecycle(sqlite_store):
    enqueue_job(sqlite_store, job_id="job_123")
    run_worker_once(sqlite_store)
    assert sqlite_store.get_job("job_123").status in {"running", "succeeded"}
```

- [ ] **Step 9: Implement durable artifact persistence**

Storage requirements:
- Upload transcript and visual-tracking artifacts to GCS before marking the job successful
- Keep local droplet disk as temporary working storage only
- Ensure the manifest never points at ephemeral local paths

- [ ] **Step 10: Implement the minimal job state machine**

Start with:
- `create_job`
- `get_job`
- `mark_running`
- `mark_succeeded`
- `mark_failed`

- [ ] **Step 11: Write failing API tests**

```python
def test_post_jobs_returns_202(client):
    response = client.post("/jobs", json={"source_url": "https://youtube.com/watch?v=x"})
    assert response.status_code == 202
```

- [ ] **Step 12: Implement the HTTP surface**

Endpoints:
- `GET /healthz`
- `POST /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/result`

Behavior:
- Health check verifies SQLite connectivity and configured state/artifact paths
- Create returns job handle
- Poll returns status
- Result returns manifest only on success

- [ ] **Step 13: Re-run service tests**

Run: `pytest tests/backend/do_phase1_service/test_jobs.py tests/backend/do_phase1_service/test_state_store.py tests/backend/do_phase1_service/test_extract.py tests/backend/do_phase1_service/test_storage.py tests/backend/do_phase1_service/test_worker.py tests/backend/do_phase1_service/test_app.py -v`
Expected: PASS

- [ ] **Step 14: Commit**

```bash
git add requirements.txt backend/do_phase1_service tests/backend/do_phase1_service
git commit -m "feat: add digitalocean phase 1 service skeleton"
```

## Task 3: Move Phase 1 Pipeline Orchestration to Async DO Jobs

**Files:**
- Create: `backend/pipeline/do_phase1_client.py`
- Modify: `backend/pipeline/phase_1_modal_pipeline.py`
- Modify: `backend/pipeline/run_pipeline.py`
- Test: `tests/backend/pipeline/test_do_phase1_client.py`
- Test: `tests/backend/pipeline/test_phase_1_pipeline_async.py`

- [ ] **Step 1: Write failing client tests**

```python
def test_submit_job_returns_job_handle(httpx_mock):
    httpx_mock.add_response(method="POST", url="http://do-service/jobs", json={"job_id": "job_123", "status": "queued"})
    client = DOPhase1Client("http://do-service")
    result = client.submit_job(source_url="https://youtube.com/watch?v=x")
    assert result.job_id == "job_123"
```

- [ ] **Step 2: Run the client tests**

Run: `pytest tests/backend/pipeline/test_do_phase1_client.py -v`
Expected: FAIL because the client does not exist yet.

- [ ] **Step 3: Implement the DO client**

Methods:
- `submit_job`
- `poll_job`
- `wait_for_completion`
- `fetch_result`

- [ ] **Step 4: Write failing pipeline orchestration tests**

```python
def test_phase_1_pipeline_waits_for_manifest(monkeypatch):
    manifest = main("https://youtube.com/watch?v=x")
    assert manifest["status"] == "succeeded"
```

- [ ] **Step 5: Refactor the pipeline to use the new client**

Requirements:
- No Modal SDK usage in the active path
- Async polling/resume support
- Manifest-driven artifact consumption
- Explicit timeout/resume logging

- [ ] **Step 6: Update the full pipeline entrypoint**

Update `backend/pipeline/run_pipeline.py` so full-pipeline execution assumes async Phase 1 and consumes the returned manifest.

- [ ] **Step 7: Re-run orchestration tests**

Run: `pytest tests/backend/pipeline/test_do_phase1_client.py tests/backend/pipeline/test_phase_1_pipeline_async.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add backend/pipeline/do_phase1_client.py backend/pipeline/phase_1_modal_pipeline.py backend/pipeline/run_pipeline.py tests/backend/pipeline/test_do_phase1_client.py tests/backend/pipeline/test_phase_1_pipeline_async.py
git commit -m "refactor: move phase 1 orchestration to digitalocean jobs"
```

## Task 4: Preserve Downstream Compatibility Through a Manifest Adapter

**Files:**
- Modify: `backend/pipeline/phase_1_modal_pipeline.py`
- Modify: `backend/pipeline/phase_2a_make_nodes.py`
- Modify: `backend/pipeline/phase_4_store_graph.py`
- Test: `tests/backend/pipeline/test_phase1_manifest_adapter.py`

- [ ] **Step 1: Write failing adapter tests**

```python
def test_manifest_adapter_returns_expected_phase_1_paths(tmp_path):
    manifest = {...}
    adapted = adapt_phase1_manifest(manifest)
    assert "phase_1_visual.json" in adapted.visual_uri
```

- [ ] **Step 2: Run the adapter tests**

Run: `pytest tests/backend/pipeline/test_phase1_manifest_adapter.py -v`
Expected: FAIL because no adapter exists.

- [ ] **Step 3: Implement the adapter**

Responsibilities:
- Translate manifest references into the inputs expected by downstream stages
- Keep downstream code changes minimal
- Centralize any legacy-to-new contract bridging
- Materialize or preserve the legacy artifacts the repo already expects:
  - `backend/outputs/phase_1_audio.json`
  - `backend/outputs/phase_1_visual.json`
  - `words`
  - `speaker_bindings`
  - `tracks`
  - canonical GCS video URI

- [ ] **Step 4: Update downstream consumers only where necessary**

Limit changes to places that currently assume Modal-era artifact semantics.

- [ ] **Step 5: Re-run targeted tests**

Run: `pytest tests/backend/pipeline/test_phase1_manifest_adapter.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/pipeline/phase_1_modal_pipeline.py backend/pipeline/phase_2a_make_nodes.py backend/pipeline/phase_4_store_graph.py tests/backend/pipeline/test_phase1_manifest_adapter.py
git commit -m "feat: add phase 1 manifest adapter"
```

## Task 5: Document the Lovable Frontend Handoff and Brand System

**Files:**
- Create: `docs/frontend/lovable-handoff.md`
- Create: `docs/frontend/lovable-prompt-pack.md`
- Create: `docs/frontend/brand-tokens.md`
- Create: `docs/frontend/lovable-delivery-checklist.md`
- Modify: `clypt-skills-doc.md`

- [ ] **Step 1: Write the handoff doc**

Must include:
- what the frontend owns
- what the backend contract guarantees
- import expectations when Lovable output is brought back into this repo
- API/data dependencies for Cortex-facing flows

- [ ] **Step 2: Write the prompt pack**

Include:
- product framing
- Cortex UX goals
- constraints from the existing spec
- what Lovable should not change

- [ ] **Step 3: Write brand tokens**

Include:
- typography direction
- color tokens
- spacing principles
- graph/editor visual constraints
- anti-slop rules

- [ ] **Step 4: Cross-link the docs**

Update `clypt-skills-doc.md` or another existing reference point so these frontend support docs are discoverable.

- [ ] **Step 5: Write the Lovable delivery checklist**

Define the accepted handoff formats:
- exported frontend files staged at `cortex-ui/.lovable-import/`
- or a teammate branch named `lovable-export/<date-or-topic>`

Include:
- required files
- expected app surfaces
- import ownership
- what to do if delivery is delayed

- [ ] **Step 6: Commit the handoff docs**

```bash
git add docs/frontend/lovable-handoff.md docs/frontend/lovable-prompt-pack.md docs/frontend/brand-tokens.md docs/frontend/lovable-delivery-checklist.md clypt-skills-doc.md
git commit -m "docs: add lovable handoff, prompts, and brand guides"
```

## Task 6: Import and Validate the Lovable Frontend

**Files:**
- Modify: `cortex-ui/app/page.tsx`
- Modify: `cortex-ui/app/graph/page.tsx`
- Modify: `cortex-ui/app/layout.tsx`
- Modify: `cortex-ui/app/globals.css`

- [ ] **Step 1: Confirm the Lovable artifact exists**

Accept either:
- files staged at `cortex-ui/.lovable-import/`
- or a teammate branch named `lovable-export/<date-or-topic>`

If neither exists, stop this task and leave it pending while backend tasks continue.

- [ ] **Step 2: Import the Lovable output into `cortex-ui`**

Requirements:
- Replace or merge the current Cortex-facing surfaces with the teammate-produced Lovable output
- Keep the imported UI wired to the documented backend contract and routes
- Prefer modifying `cortex-ui/app/page.tsx`, `cortex-ui/app/graph/page.tsx`, `cortex-ui/app/layout.tsx`, and `cortex-ui/app/globals.css` unless the delivered UI requires a better-scoped split

- [ ] **Step 3: Verify the imported frontend actually builds**

Run: `cd cortex-ui && npm run build`
Expected: PASS

- [ ] **Step 4: Run one lightweight runtime smoke check**

Example:
- start the frontend locally
- verify the imported Cortex shell loads
- verify one documented backend route such as `/api/pipeline/status` or `/api/graph` is still called successfully

- [ ] **Step 5: Commit**

```bash
git add cortex-ui/app/page.tsx cortex-ui/app/graph/page.tsx cortex-ui/app/layout.tsx cortex-ui/app/globals.css
git commit -m "feat: import lovable frontend output"
```

## Task 7: Add Droplet Deployment and Runtime Artifacts

**Files:**
- Create: `backend/do_phase1_service/Dockerfile`
- Create: `ops/digitalocean/do-phase1.service`
- Create: `ops/digitalocean/do-phase1-worker.service`
- Create: `ops/digitalocean/deploy-phase1.sh`
- Create: `ops/digitalocean/healthcheck.sh`
- Create: `ops/digitalocean/cloud-init.yaml`
- Create: `ops/digitalocean/mount-state-volume.sh`
- Create: `tests/backend/smoke/test_do_phase1_deployment_contract.py`

- [ ] **Step 1: Write the failing deployment smoke test**

```python
def test_do_phase1_service_exposes_health_and_jobs_endpoints():
    assert service_has_endpoint("/healthz")
    assert service_has_endpoint("/jobs")
```

- [ ] **Step 2: Run the deployment smoke test**

Run: `pytest tests/backend/smoke/test_do_phase1_deployment_contract.py -v`
Expected: FAIL because deployment artifacts do not exist yet.

- [ ] **Step 3: Add the container/runtime artifacts**

Must include:
- GPU-capable container startup
- environment variable loading
- health endpoint expectation
- artifact/runtime path configuration
- explicit SQLite path such as `/mnt/clypt-state/jobs.db`
- explicit artifact root such as `/mnt/clypt-artifacts`

- [ ] **Step 4: Add the droplet runbook script**

Include:
- image/bootstrap assumptions
- API service install or restart command
- worker service install or restart command
- health check verification
- state volume mount command
- persistent directory creation

- [ ] **Step 5: Add the machine bootstrap and mount scripts**

Must define:
- how the persistent volume is attached or mounted
- where SQLite lives
- where artifacts live
- what env vars the systemd unit reads

- [ ] **Step 6: Re-run the deployment smoke test**

Run: `pytest tests/backend/smoke/test_do_phase1_deployment_contract.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add backend/do_phase1_service/Dockerfile ops/digitalocean/do-phase1.service ops/digitalocean/do-phase1-worker.service ops/digitalocean/deploy-phase1.sh ops/digitalocean/healthcheck.sh ops/digitalocean/cloud-init.yaml ops/digitalocean/mount-state-volume.sh tests/backend/smoke/test_do_phase1_deployment_contract.py
git commit -m "ops: add digitalocean droplet deployment artifacts"
```

## Task 8: Update Runbooks and Verify the New Story

**Files:**
- Modify: `README.md`
- Modify: `docs/modal_worker.md`
- Create: `tests/backend/smoke/test_balanced_hybrid_path.py`

- [ ] **Step 1: Write the smoke test**

```python
def test_balanced_hybrid_pipeline_routes_phase1_through_do_client():
    result = run_phase1_path(...)
    assert result.provider == "digitalocean"
    assert result.job_mode == "async"
    assert result.canonical_video_uri.startswith("gs://")
```

- [ ] **Step 2: Run the smoke test**

Run: `pytest tests/backend/smoke/test_balanced_hybrid_path.py -v`
Expected: FAIL because the config contract is not wired yet.

- [ ] **Step 3: Update the docs**

Document:
- DigitalOcean for Phase 1
- GCP for downstream graph/reasoning
- long-running async job model
- frontend/Lovable ownership boundaries

- [ ] **Step 4: Make the smoke test pass**

Add the minimum config plumbing and routing checks needed for the documented architecture to be true.

- [ ] **Step 5: Add one downstream-consumer smoke check**

Add a narrow end-to-end check proving the async Phase 1 flow produces artifacts consumable by one existing downstream stage, preferably `backend/pipeline/phase_2a_make_nodes.py`.

- [ ] **Step 6: Run targeted verification**

Run: `pytest tests/backend/smoke/test_balanced_hybrid_path.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add README.md docs/modal_worker.md tests/backend/smoke/test_balanced_hybrid_path.py
git commit -m "docs: update runbooks for balanced hybrid stack"
```

## Verification Checklist

- `pytest tests/backend/pipeline/test_phase1_contract.py -v`
- `pytest tests/backend/do_phase1_service/test_jobs.py tests/backend/do_phase1_service/test_app.py -v`
- `pytest tests/backend/pipeline/test_do_phase1_client.py tests/backend/pipeline/test_phase_1_pipeline_async.py -v`
- `pytest tests/backend/pipeline/test_phase1_manifest_adapter.py -v`
- `pytest tests/backend/smoke/test_balanced_hybrid_path.py -v`

## Notes for Execution

- Treat all Phase 1 video jobs as asynchronous from the first commit.
- Prefer keeping GCS as the durable artifact source of truth during the hackathon unless a later plan explicitly changes that.
- Do not start a frontend rewrite inside this repo until Lovable output is ready to import.
- When frontend import begins, use `docs/frontend/lovable-handoff.md`, `docs/frontend/lovable-prompt-pack.md`, and `docs/frontend/brand-tokens.md` as the coordination source of truth.
