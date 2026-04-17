from __future__ import annotations

import logging
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
    from scripts.do_phase1_visual.run_remote_job import Phase1RemoteClient

    client = Phase1RemoteClient(base_url="http://127.0.0.1:8080")

    assert client.jobs_url == "http://127.0.0.1:8080/jobs"
    assert client.job_url(job_id="job_123") == "http://127.0.0.1:8080/jobs/job_123"
    assert client.logs_url(job_id="job_123", tail_lines=50) == "http://127.0.0.1:8080/jobs/job_123/logs?tail_lines=50"


def test_run_phase1_worker_configures_info_logging(monkeypatch, tmp_path):
    from backend.runtime import run_phase1_worker

    captured: dict[str, object] = {}

    def fake_basic_config(**kwargs):
        captured.update(kwargs)

    class _FakeWorker:
        def __init__(self, *, store, run_job, logs_root):
            self.store = store
            self.run_job = run_job
            self.logs_root = logs_root

        def run_forever(self, *, poll_interval_s: float):
            captured["poll_interval_s"] = poll_interval_s

    class _FakeRunner:
        def __init__(self):
            self.run_job = lambda **_: {"ok": True}

    monkeypatch.setattr(run_phase1_worker.logging, "basicConfig", fake_basic_config)
    monkeypatch.setattr(run_phase1_worker, "SQLiteJobStore", lambda path: ("store", path))
    monkeypatch.setattr(run_phase1_worker, "build_default_phase1_job_runner", lambda: _FakeRunner())
    monkeypatch.setattr(run_phase1_worker, "Phase1Worker", _FakeWorker)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_phase1_worker.py",
            "--db-path",
            str(tmp_path / "jobs.db"),
            "--logs-root",
            str(tmp_path / "logs"),
            "--poll-interval-s",
            "0.5",
        ],
    )

    exit_code = run_phase1_worker.main()

    assert exit_code == 0
    assert captured["level"] == logging.INFO
    assert "format" in captured
    assert captured["poll_interval_s"] == 0.5
