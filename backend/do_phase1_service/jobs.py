from __future__ import annotations

from datetime import UTC, datetime

from backend.do_phase1_service.models import JobCreatePayload, JobRecord
from backend.do_phase1_service.state_store import SQLiteJobStore


def create_job(store: SQLiteJobStore, payload: JobCreatePayload) -> JobRecord:
    now = datetime.now(UTC)
    record = JobRecord(source_url=payload.source_url, status="queued", created_at=now, updated_at=now)
    return store.save_job(
        job_id=record.job_id,
        source_url=record.source_url,
        status=record.status,
        retries=record.retries,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def enqueue_job(store: SQLiteJobStore, *, job_id: str, payload: dict) -> JobRecord:
    now = datetime.now(UTC)
    return store.save_job(
        job_id=job_id,
        source_url=str(payload["source_url"]),
        status="queued",
        retries=0,
        created_at=now,
        updated_at=now,
    )


def get_job(store: SQLiteJobStore, job_id: str) -> JobRecord | None:
    return store.get_job(job_id)


def mark_running(store: SQLiteJobStore, job_id: str) -> JobRecord:
    job = _require_job(store, job_id)
    now = datetime.now(UTC)
    return store.save_job(
        job_id=job.job_id,
        source_url=job.source_url,
        status="running",
        retries=job.retries + 1,
        manifest=job.manifest,
        manifest_uri=job.manifest_uri,
        failure=job.failure,
        created_at=job.created_at,
        updated_at=now,
        started_at=job.started_at or now,
        completed_at=None,
    )


def mark_succeeded(
    store: SQLiteJobStore,
    job_id: str,
    *,
    manifest: dict,
    manifest_uri: str,
) -> JobRecord:
    job = _require_job(store, job_id)
    now = datetime.now(UTC)
    return store.save_job(
        job_id=job.job_id,
        source_url=job.source_url,
        status="succeeded",
        retries=job.retries,
        manifest=manifest,
        manifest_uri=manifest_uri,
        failure=None,
        created_at=job.created_at,
        updated_at=now,
        started_at=job.started_at,
        completed_at=now,
    )


def mark_failed(
    store: SQLiteJobStore,
    job_id: str,
    *,
    error_type: str,
    error_message: str,
    failed_step: str | None = None,
) -> JobRecord:
    job = _require_job(store, job_id)
    now = datetime.now(UTC)
    failure = {
        "error_type": error_type,
        "error_message": error_message,
        "failed_step": failed_step,
    }
    return store.save_job(
        job_id=job.job_id,
        source_url=job.source_url,
        status="failed",
        retries=job.retries,
        manifest=job.manifest,
        manifest_uri=job.manifest_uri,
        failure=failure,
        created_at=job.created_at,
        updated_at=now,
        started_at=job.started_at,
        completed_at=now,
    )


def _require_job(store: SQLiteJobStore, job_id: str) -> JobRecord:
    job = store.get_job(job_id)
    if job is None:
        raise KeyError(f"unknown job_id: {job_id}")
    return job
