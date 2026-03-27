from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from backend.do_phase1_service.models import JobCreatePayload, JobRecord
from backend.do_phase1_service.state_store import SQLiteJobStore


def create_job(store: SQLiteJobStore, payload: JobCreatePayload) -> JobRecord:
    now = datetime.now(UTC)
    record = JobRecord(
        source_url=payload.source_url,
        runtime_controls=payload.runtime_controls,
        status="queued",
        created_at=now,
        updated_at=now,
    )
    return store.save_job(
        job_id=record.job_id,
        source_url=record.source_url,
        runtime_controls=record.runtime_controls,
        status=record.status,
        retries=record.retries,
        claim_token=None,
        current_step="queued",
        progress_message="Queued for DigitalOcean Phase 1 worker",
        progress_pct=0.0,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def enqueue_job(store: SQLiteJobStore, *, job_id: str, payload: dict) -> JobRecord:
    now = datetime.now(UTC)
    return store.save_job(
        job_id=job_id,
        source_url=str(payload["source_url"]),
        runtime_controls=dict(payload.get("runtime_controls") or {}) or None,
        status="queued",
        retries=0,
        claim_token=None,
        current_step="queued",
        progress_message="Queued for DigitalOcean Phase 1 worker",
        progress_pct=0.0,
        created_at=now,
        updated_at=now,
    )


def get_job(store: SQLiteJobStore, job_id: str) -> JobRecord | None:
    return store.get_job(job_id)


def mark_running(store: SQLiteJobStore, job_id: str) -> JobRecord:
    job = _require_job(store, job_id)
    if job.status != "queued":
        raise ValueError(f"cannot mark job {job_id} running from status {job.status}")
    now = datetime.now(UTC)
    return store.save_job(
        job_id=job.job_id,
        source_url=job.source_url,
        runtime_controls=job.runtime_controls,
        status="running",
        retries=job.retries + 1,
        claim_token=job.claim_token or uuid4().hex,
        manifest=job.manifest,
        manifest_uri=job.manifest_uri,
        failure=job.failure,
        current_step="starting",
        progress_message="Worker claimed job",
        progress_pct=0.01,
        log_path=job.log_path,
        created_at=job.created_at,
        updated_at=now,
        started_at=job.started_at or now,
        completed_at=None,
    )


def mark_succeeded(
    store: SQLiteJobStore,
    job_id: str,
    *,
    claim_token: str,
    manifest: dict,
    manifest_uri: str,
) -> JobRecord | None:
    return store.complete_job(
        job_id=job_id,
        claim_token=claim_token,
        manifest=manifest,
        manifest_uri=manifest_uri,
    )


def mark_failed(
    store: SQLiteJobStore,
    job_id: str,
    *,
    claim_token: str,
    error_type: str,
    error_message: str,
    failed_step: str | None = None,
) -> JobRecord | None:
    return store.fail_job(
        job_id=job_id,
        claim_token=claim_token,
        error_type=error_type,
        error_message=error_message,
        failed_step=failed_step,
    )


def _require_job(store: SQLiteJobStore, job_id: str) -> JobRecord:
    job = store.get_job(job_id)
    if job is None:
        raise KeyError(f"unknown job_id: {job_id}")
    return job
UTC = timezone.utc
