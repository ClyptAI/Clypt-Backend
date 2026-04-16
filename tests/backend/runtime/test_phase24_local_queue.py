from __future__ import annotations

import time
from pathlib import Path

import pytest


def _minimal_payload(run_id: str = "run-a") -> dict:
    return {
        "run_id": run_id,
        "source_url": "https://example.com/v",
        "phase1_outputs": {
            "phase1_audio": {
                "source_audio": "https://example.com/v",
                "video_gcs_uri": "gs://b/v.mp4",
                "local_video_path": "/tmp/x.mp4",
            },
            "diarization_payload": {"words": [], "turns": []},
            "phase1_visual": {"video_metadata": {"fps": 30.0}, "shot_changes": [], "tracks": []},
            "emotion2vec_payload": {"segments": []},
            "yamnet_payload": {"events": []},
        },
    }


def test_phase24_local_queue_enqueue_is_idempotent_by_run_id(tmp_path: Path) -> None:
    from backend.runtime.phase24_local_queue import Phase24LocalQueue

    db = tmp_path / "q.sqlite"
    queue = Phase24LocalQueue(db)
    p = _minimal_payload()
    j1 = queue.enqueue("run-a", p)
    j2 = queue.enqueue("run-a", {**p, "extra": 1})
    assert j1 == j2
    row = queue.get_job(j1)
    assert row is not None
    assert row["run_id"] == "run-a"
    assert "extra" not in row["payload_json"]


def test_phase24_local_queue_claim_mark_succeeded(tmp_path: Path) -> None:
    from backend.runtime.phase24_local_queue import Phase24LocalQueue

    queue = Phase24LocalQueue(tmp_path / "q.sqlite")
    payload = _minimal_payload("run-b")
    job_id = queue.enqueue("run-b", payload)
    row = queue.claim_next("worker-1", lease_timeout_s=60)
    assert row is not None
    assert row["job_id"] == job_id
    assert row["attempt_count"] == 1
    assert row["payload"]["run_id"] == "run-b"
    queue.mark_succeeded(job_id)
    assert queue.get_job(job_id)["status"] == "succeeded"


def test_phase24_local_queue_mark_failed_retry_then_complete(tmp_path: Path) -> None:
    from backend.runtime.phase24_local_queue import Phase24LocalQueue

    queue = Phase24LocalQueue(tmp_path / "q.sqlite")
    job_id = queue.enqueue("run-c", _minimal_payload("run-c"))
    r1 = queue.claim_next("w", 60)
    assert r1 is not None
    queue.mark_failed(job_id, error="boom", retry=True, retry_delay_s=0.0)
    assert queue.get_job(job_id)["status"] == "queued"
    r2 = queue.claim_next("w", 60)
    assert r2 is not None
    assert r2["attempt_count"] == 2
    queue.mark_succeeded(job_id)
    assert queue.get_job(job_id)["status"] == "succeeded"


def test_phase24_local_queue_reclaims_expired_lease(tmp_path: Path) -> None:
    from backend.runtime.phase24_local_queue import Phase24LocalQueue

    queue = Phase24LocalQueue(tmp_path / "q.sqlite")
    job_id = queue.enqueue("run-d", _minimal_payload("run-d"))
    r1 = queue.claim_next("w1", lease_timeout_s=1)
    assert r1 is not None
    time.sleep(1.25)
    r2 = queue.claim_next("w2", lease_timeout_s=1)
    assert r2 is not None
    assert r2["job_id"] == job_id
    assert r2["attempt_count"] == 2


def test_phase24_local_queue_can_disable_expired_lease_reclaim(tmp_path: Path) -> None:
    from backend.runtime.phase24_local_queue import Phase24LocalQueue

    queue = Phase24LocalQueue(tmp_path / "q.sqlite")
    job_id = queue.enqueue("run-disable-reclaim", _minimal_payload("run-disable-reclaim"))
    r1 = queue.claim_next("w1", lease_timeout_s=1)
    assert r1 is not None
    time.sleep(1.25)
    r2 = queue.claim_next("w2", lease_timeout_s=1, reclaim_expired_leases=False)
    assert r2 is None
    row = queue.get_job(job_id)
    assert row is not None
    assert row["status"] == "running"


def test_phase24_local_queue_enforces_global_max_inflight(tmp_path: Path) -> None:
    from backend.runtime.phase24_local_queue import Phase24LocalQueue

    queue = Phase24LocalQueue(tmp_path / "q.sqlite")
    first = queue.enqueue("run-e1", _minimal_payload("run-e1"))
    second = queue.enqueue("run-e2", _minimal_payload("run-e2"))
    r1 = queue.claim_next("w1", lease_timeout_s=60, max_inflight=1)
    assert r1 is not None and r1["job_id"] == first
    r2 = queue.claim_next("w2", lease_timeout_s=60, max_inflight=1)
    assert r2 is None
    queue.mark_succeeded(first)
    r3 = queue.claim_next("w2", lease_timeout_s=60, max_inflight=1)
    assert r3 is not None and r3["job_id"] == second


def test_load_provider_settings_includes_phase24_local_queue_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "clypt-v3")
    monkeypatch.setenv("GCS_BUCKET", "bucket-a")
    monkeypatch.setenv("VIBEVOICE_VLLM_BASE_URL", "http://127.0.0.1:8000")

    settings = load_provider_settings()

    assert settings.phase24_local_queue.queue_backend == "local_sqlite"
    assert settings.phase24_local_queue.poll_interval_ms == 500
    assert "phase24_local_queue.sqlite" in str(settings.phase24_local_queue.path)
    assert settings.phase24_local_queue.reclaim_expired_leases is False
    assert settings.phase24_local_queue.fail_fast_on_stale_running is True
