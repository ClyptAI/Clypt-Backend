from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from backend.do_phase1_service.jobs import create_job, get_job
from backend.do_phase1_service.models import JobCreatePayload
from backend.do_phase1_service.state_store import SQLiteJobStore


DEFAULT_DB_PATH = Path(os.getenv("DO_PHASE1_DB_PATH", "backend/do_phase1_service/jobs.db"))
DEFAULT_OUTPUT_ROOT = Path(os.getenv("DO_PHASE1_OUTPUT_ROOT", "backend/do_phase1_service/workdir"))


def create_app(*, store: SQLiteJobStore | None = None, output_root: str | Path | None = None) -> FastAPI:
    store = store or SQLiteJobStore(DEFAULT_DB_PATH)
    output_root = Path(output_root or DEFAULT_OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Clypt DO Phase 1 Service")
    app.state.store = store
    app.state.output_root = output_root

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "sqlite": store.healthcheck(),
            "db_path": str(store.db_path),
            "output_root": str(output_root),
        }

    @app.post("/jobs", status_code=202)
    def post_jobs(payload: JobCreatePayload) -> dict:
        job = create_job(store, payload)
        return job.model_dump(mode="json")

    @app.get("/jobs/{job_id}")
    def get_job_status(job_id: str) -> dict:
        job = get_job(store, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job.model_dump(mode="json")

    @app.get("/jobs/{job_id}/result")
    def get_job_result(job_id: str) -> dict:
        job = get_job(store, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status != "succeeded" or job.manifest is None:
            raise HTTPException(status_code=409, detail="job result not ready")
        return job.manifest

    return app


app = create_app()
