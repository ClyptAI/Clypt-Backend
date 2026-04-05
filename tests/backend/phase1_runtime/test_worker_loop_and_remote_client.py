from __future__ import annotations

from pathlib import Path


def test_phase1_worker_run_forever_stops_after_idle_loops(tmp_path):
    from backend.phase1_runtime.jobs import create_job
    from backend.phase1_runtime.models import Phase1JobCreatePayload
    from backend.phase1_runtime.state_store import SQLiteJobStore
    from backend.phase1_runtime.worker import Phase1Worker

    store = SQLiteJobStore(tmp_path / "jobs.db")
    create_job(store, Phase1JobCreatePayload(source_url="https://youtube.com/watch?v=test"))
    processed: list[str] = []

    def fake_runner(*, job_id: str, source_url: str | None, source_path: str | None, runtime_controls: dict | None):
        processed.append(job_id)
        return {"ok": True}

    worker = Phase1Worker(store=store, run_job=fake_runner, logs_root=tmp_path / "logs")
    worker.run_forever(poll_interval_s=0.0, stop_after_idle_loops=1)

    assert len(processed) == 1


def test_remote_job_client_builds_submit_and_logs_urls():
    from scripts.do_phase1.run_remote_job import Phase1RemoteClient

    client = Phase1RemoteClient(base_url="http://127.0.0.1:8080")

    assert client.jobs_url == "http://127.0.0.1:8080/jobs"
    assert client.job_url(job_id="job_123") == "http://127.0.0.1:8080/jobs/job_123"
    assert client.logs_url(job_id="job_123", tail_lines=50) == "http://127.0.0.1:8080/jobs/job_123/logs?tail_lines=50"
