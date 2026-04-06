from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def test_phase1_worker_processes_one_job_to_success(tmp_path):
    from backend.phase1_runtime.jobs import create_job
    from backend.phase1_runtime.models import Phase1JobCreatePayload, Phase1SidecarOutputs
    from backend.phase1_runtime.state_store import SQLiteJobStore
    from backend.phase1_runtime.worker import Phase1Worker

    store = SQLiteJobStore(tmp_path / "jobs.db")
    job = create_job(
        store,
        Phase1JobCreatePayload(source_url="https://youtube.com/watch?v=test"),
    )

    processed: list[str] = []

    def fake_runner(*, job_id: str, source_url: str | None, source_path: str | None, runtime_controls: dict | None):
        processed.append(job_id)
        return {
            "phase1": Phase1SidecarOutputs(
                phase1_audio={"source_audio": source_url, "video_gcs_uri": "gs://bucket/source.mp4"},
                diarization_payload={"turns": [], "words": []},
                phase1_visual={"video_metadata": {"fps": 30.0}, "shot_changes": [], "tracks": []},
                emotion2vec_payload={"segments": []},
                yamnet_payload={"events": []},
            ),
            "summary": {"artifact_paths": {"canonical_timeline": "tmp/timeline.json"}},
        }

    worker = Phase1Worker(store=store, run_job=fake_runner)
    result = worker.run_next_job_once()

    assert result is True
    assert processed == [job.job_id]
    updated = store.get_job(job.job_id)
    assert updated is not None
    assert updated.status == "succeeded"
    assert updated.result is not None
    assert updated.result["summary"]["artifact_paths"]["canonical_timeline"] == "tmp/timeline.json"


def test_phase1_service_app_creates_jobs_and_returns_status(tmp_path):
    from backend.phase1_runtime.app import create_app
    from backend.phase1_runtime.state_store import SQLiteJobStore

    store = SQLiteJobStore(tmp_path / "jobs.db")
    logs_root = tmp_path / "logs"
    client = TestClient(create_app(store=store, logs_root=logs_root))

    create_response = client.post(
        "/jobs",
        json={"source_url": "https://youtube.com/watch?v=test"},
    )
    assert create_response.status_code == 202
    job = create_response.json()
    assert job["status"] == "queued"

    status_response = client.get(f"/jobs/{job['job_id']}")
    assert status_response.status_code == 200
    assert status_response.json()["job_id"] == job["job_id"]

    result_response = client.get(f"/jobs/{job['job_id']}/result")
    assert result_response.status_code == 409


def test_phase1_service_app_returns_job_logs(tmp_path):
    from backend.phase1_runtime.app import create_app
    from backend.phase1_runtime.state_store import SQLiteJobStore

    store = SQLiteJobStore(tmp_path / "jobs.db")
    logs_root = tmp_path / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    log_path = logs_root / "job_123.log"
    log_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
    store.save_job(
        job_id="job_123",
        source_url="https://youtube.com/watch?v=test",
        source_path=None,
        runtime_controls=None,
        status="running",
        log_path=str(log_path),
        current_step="running",
        progress_message="running",
        progress_pct=0.5,
    )

    client = TestClient(create_app(store=store, logs_root=logs_root))
    response = client.get("/jobs/job_123/logs?tail_lines=2")

    assert response.status_code == 200
    assert response.json()["lines"] == ["line2", "line3"]


def test_phase1_worker_writes_job_log_file(tmp_path):
    from backend.phase1_runtime.jobs import create_job
    from backend.phase1_runtime.models import Phase1JobCreatePayload
    from backend.phase1_runtime.state_store import SQLiteJobStore
    from backend.phase1_runtime.worker import Phase1Worker

    store = SQLiteJobStore(tmp_path / "jobs.db")
    logs_root = tmp_path / "logs"
    job = create_job(
        store,
        Phase1JobCreatePayload(source_url="https://youtube.com/watch?v=test"),
    )

    def fake_runner(*, job_id: str, source_url: str | None, source_path: str | None, runtime_controls: dict | None):
        print(f"processing {job_id}")
        return {"ok": True}

    worker = Phase1Worker(store=store, run_job=fake_runner, logs_root=logs_root)
    worker.run_next_job_once()

    updated = store.get_job(job.job_id)
    assert updated is not None
    assert updated.log_path is not None
    assert Path(updated.log_path).read_text(encoding="utf-8").strip() == f"processing {job.job_id}"
