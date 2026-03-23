from pathlib import Path

import pytest

from backend.do_phase1_service.jobs import create_job, enqueue_job, mark_failed, mark_running, mark_succeeded
from backend.do_phase1_service.models import JobCreatePayload
from backend.do_phase1_service.state_store import SQLiteJobStore


def test_create_job_returns_queued_manifest(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    payload = JobCreatePayload(
        source_url="https://youtube.com/watch?v=x",
        runtime_controls={"speaker_binding_mode": "lrasd"},
    )

    job = create_job(store, payload)

    assert job.status == "queued"
    assert job.manifest is None
    assert job.retries == 0
    assert job.runtime_controls == {"speaker_binding_mode": "lrasd"}


def test_job_lifecycle_updates_status_and_manifest(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    payload = JobCreatePayload(source_url="https://youtube.com/watch?v=x")
    created = create_job(store, payload)

    running = mark_running(store, created.job_id)
    assert running.status == "running"
    assert running.claim_token is not None

    manifest = {"job_id": created.job_id, "status": "succeeded"}
    succeeded = mark_succeeded(
        store,
        created.job_id,
        claim_token=running.claim_token,
        manifest=manifest,
        manifest_uri="gs://bucket/job.json",
    )
    assert succeeded.status == "succeeded"
    assert succeeded.manifest == manifest
    assert succeeded.manifest_uri == "gs://bucket/job.json"

    created2 = create_job(store, payload)
    rerun = mark_running(store, created2.job_id)
    failed = mark_failed(
        store,
        created2.job_id,
        claim_token=rerun.claim_token,
        error_type="RuntimeError",
        error_message="boom",
    )
    assert failed.status == "failed"
    assert failed.failure["error_type"] == "RuntimeError"

    with pytest.raises(ValueError, match="cannot mark job .* running from status succeeded"):
        mark_running(store, created.job_id)


def test_enqueue_job_persists_specific_job_id(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.db")

    job = enqueue_job(store, job_id="job_123", payload={"source_url": "https://youtube.com/watch?v=x"})

    assert job.job_id == "job_123"
    assert store.get_job("job_123").status == "queued"
