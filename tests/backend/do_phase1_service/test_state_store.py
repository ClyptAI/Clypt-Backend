from pathlib import Path
from datetime import datetime, timedelta, timezone

UTC = timezone.utc

from backend.do_phase1_service.state_store import SQLiteJobStore


def test_job_state_survives_process_restart(tmp_path: Path):
    db_path = tmp_path / "jobs.db"
    store = SQLiteJobStore(db_path)
    store.save_job(job_id="job_123", source_url="https://youtube.com/watch?v=x", status="running")

    reloaded = SQLiteJobStore(db_path)

    assert reloaded.get_job("job_123").status == "running"


def test_recoverable_jobs_include_only_queued_work(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    store.save_job(job_id="queued_job", source_url="https://youtube.com/watch?v=q", status="queued")
    store.save_job(job_id="running_job", source_url="https://youtube.com/watch?v=r", status="running")
    store.save_job(job_id="done_job", source_url="https://youtube.com/watch?v=s", status="succeeded")

    recoverable_ids = {job.job_id for job in store.list_recoverable_jobs()}

    assert recoverable_ids == {"queued_job"}


def test_claim_next_job_is_atomic_for_queued_work(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    store.save_job(job_id="job_123", source_url="https://youtube.com/watch?v=x", status="queued")

    first_claim = store.claim_next_job(stale_after_seconds=1800)
    second_claim = store.claim_next_job(stale_after_seconds=1800)

    assert first_claim is not None
    assert first_claim.job_id == "job_123"
    assert first_claim.status == "running"
    assert first_claim.claim_token
    assert second_claim is None


def test_claim_next_job_does_not_reclaim_running_job(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    store.save_job(job_id="job_123", source_url="https://youtube.com/watch?v=x", status="running", retries=1)

    claim = store.claim_next_job(stale_after_seconds=60)

    assert claim is None


def test_active_heartbeat_prevents_running_job_reclaim(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    started_at = datetime.now(UTC) - timedelta(minutes=10)
    old_updated_at = datetime.now(UTC) - timedelta(minutes=5)
    store.save_job(
        job_id="job_123",
        source_url="https://youtube.com/watch?v=x",
        status="running",
        retries=1,
        claim_token="active-token",
        started_at=started_at,
        updated_at=old_updated_at,
    )

    heartbeat = store.heartbeat_job("job_123", "active-token")
    claim = store.claim_next_job(stale_after_seconds=60)

    assert heartbeat is not None
    assert claim is None


def test_stale_running_job_is_reclaimable(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    old_updated_at = datetime.now(UTC) - timedelta(hours=2)
    started_at = datetime.now(UTC) - timedelta(hours=2)
    store.save_job(
        job_id="job_123",
        source_url="https://youtube.com/watch?v=x",
        status="running",
        retries=1,
        started_at=started_at,
        updated_at=old_updated_at,
    )

    claim = store.claim_next_job(stale_after_seconds=60)

    assert claim is not None
    assert claim.job_id == "job_123"
    assert claim.status == "running"
    assert claim.retries == 2
    assert claim.claim_token


def test_stale_worker_cannot_complete_after_reclaim(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    old_updated_at = datetime.now(UTC) - timedelta(hours=2)
    first = store.save_job(
        job_id="job_123",
        source_url="https://youtube.com/watch?v=x",
        status="running",
        retries=1,
        claim_token="old-token",
        started_at=old_updated_at,
        updated_at=old_updated_at,
    )

    reclaimed = store.claim_next_job(stale_after_seconds=60)
    assert reclaimed is not None
    assert reclaimed.claim_token != first.claim_token

    stale_complete = store.complete_job(
        job_id="job_123",
        claim_token="old-token",
        manifest={"job_id": "job_123", "status": "succeeded"},
        manifest_uri="gs://bucket/old.json",
    )
    assert stale_complete is None

    fresh_complete = store.complete_job(
        job_id="job_123",
        claim_token=reclaimed.claim_token,
        manifest={"job_id": "job_123", "status": "succeeded"},
        manifest_uri="gs://bucket/new.json",
    )
    assert fresh_complete is not None
    assert fresh_complete.manifest_uri == "gs://bucket/new.json"
