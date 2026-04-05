from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from .jobs import create_job, get_job
from .models import Phase1JobCreatePayload
from .state_store import SQLiteJobStore


DEFAULT_STATE_ROOT = Path("backend/outputs/v3_1_phase1_service")
DEFAULT_LOGS_ROOT = DEFAULT_STATE_ROOT / "logs"


def create_app(*, store: SQLiteJobStore | None = None, logs_root: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="Clypt V3.1 Phase 1 Service")
    app.state.store = store
    app.state.logs_root = Path(logs_root or DEFAULT_LOGS_ROOT)

    def _store() -> SQLiteJobStore:
        if app.state.store is None:
            app.state.store = SQLiteJobStore(DEFAULT_STATE_ROOT / "jobs.db")
        return app.state.store

    def _logs_root() -> Path:
        app.state.logs_root.mkdir(parents=True, exist_ok=True)
        return app.state.logs_root

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "sqlite": _store().healthcheck(),
        }

    @app.post("/jobs", status_code=202)
    def post_jobs(payload: Phase1JobCreatePayload) -> dict:
        return create_job(_store(), payload).model_dump(mode="json")

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
        if job.status != "succeeded" or job.result is None:
            raise HTTPException(status_code=409, detail="job result not ready")
        return job.result

    @app.get("/jobs/{job_id}/logs")
    def get_job_logs(job_id: str, tail_lines: int = 200) -> dict:
        job = get_job(_store(), job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        log_path = Path(job.log_path) if job.log_path else (_logs_root() / f"{job_id}.log")
        if not log_path.exists():
            return {"job_id": job_id, "log_path": str(log_path), "lines": []}
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail_lines = max(1, min(2000, int(tail_lines)))
        return {
            "job_id": job_id,
            "log_path": str(log_path),
            "lines": lines[-tail_lines:],
        }

    return app


app = create_app()


__all__ = ["app", "create_app"]
