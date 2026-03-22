from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from backend.do_phase1_service.jobs import create_job, get_job
from backend.do_phase1_service.models import JobCreatePayload
from backend.do_phase1_service.state_store import SQLiteJobStore


DEFAULT_STATE_ROOT = Path(os.getenv("DO_PHASE1_STATE_ROOT", "/var/lib/clypt/do_phase1_service"))
DEFAULT_DB_PATH = Path(os.getenv("DO_PHASE1_DB_PATH", str(DEFAULT_STATE_ROOT / "jobs.db")))
DEFAULT_OUTPUT_ROOT = Path(os.getenv("DO_PHASE1_OUTPUT_ROOT", str(DEFAULT_STATE_ROOT / "workdir")))


def create_app(*, store: SQLiteJobStore | None = None, output_root: str | Path | None = None) -> FastAPI:
    output_root = Path(output_root or DEFAULT_OUTPUT_ROOT)

    app = FastAPI(title="Clypt DO Phase 1 Service")
    app.state.store = store
    app.state.db_path = DEFAULT_DB_PATH
    app.state.output_root = output_root

    def _store() -> SQLiteJobStore:
        if app.state.store is None:
            app.state.store = SQLiteJobStore(app.state.db_path)
        return app.state.store

    def _output_root() -> Path:
        app.state.output_root.mkdir(parents=True, exist_ok=True)
        return app.state.output_root

    @app.get("/healthz")
    def healthz() -> dict:
        live_store = _store()
        live_output_root = _output_root()
        return {
            "status": "ok",
            "sqlite": live_store.healthcheck(),
            "db_path": str(live_store.db_path),
            "output_root": str(live_output_root),
        }

    @app.post("/jobs", status_code=202)
    def post_jobs(payload: JobCreatePayload) -> dict:
        job = create_job(_store(), payload)
        return job.model_dump(mode="json")

    @app.get("/jobs/{job_id}")
    def get_job_status(job_id: str) -> dict:
        job = get_job(_store(), job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job.model_dump(mode="json")

    @app.get("/jobs/{job_id}/result")
    def get_job_result(job_id: str) -> dict:
        job = get_job(_store(), job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status != "succeeded" or job.manifest is None:
            raise HTTPException(status_code=409, detail="job result not ready")
        return job.manifest

    return app


app = create_app()
