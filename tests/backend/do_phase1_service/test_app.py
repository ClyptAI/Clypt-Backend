from pathlib import Path
import sqlite3

from fastapi.testclient import TestClient

from backend.do_phase1_service.app import create_app
from backend.do_phase1_service.jobs import enqueue_job
from backend.do_phase1_service.state_store import SQLiteJobStore
from datetime import datetime, timezone

UTC = timezone.utc


def test_post_jobs_returns_202(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    client = TestClient(create_app(store=store, output_root=tmp_path))

    response = client.post("/jobs", json={"source_url": "https://youtube.com/watch?v=x"})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["job_id"]


def test_healthz_reports_ok(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    client = TestClient(create_app(store=store, output_root=tmp_path))

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_dashboard_page_renders(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    client = TestClient(create_app(store=store, output_root=tmp_path))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "Clypt Phase 1 Monitor" in response.text
    assert "/dashboard/api/jobs" in response.text
    assert "Wrap lines" in response.text
    assert "Tail lines" in response.text
    assert "Copy logs" in response.text


def test_get_jobs_returns_job_status(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    enqueue_job(store, job_id="job_123", payload={"source_url": "https://youtube.com/watch?v=x"})
    client = TestClient(create_app(store=store, output_root=tmp_path))

    response = client.get("/jobs/job_123")

    assert response.status_code == 200
    assert response.json()["job_id"] == "job_123"
    assert response.json()["status"] == "queued"


def test_get_job_result_returns_409_until_success(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    enqueue_job(store, job_id="job_123", payload={"source_url": "https://youtube.com/watch?v=x"})
    client = TestClient(create_app(store=store, output_root=tmp_path))

    response = client.get("/jobs/job_123/result")

    assert response.status_code == 409


def test_get_job_result_returns_manifest_on_success(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    manifest = {
        "contract_version": "v2",
        "job_id": "job_123",
        "status": "succeeded",
        "source_video": {"source_url": "https://youtube.com/watch?v=x"},
        "canonical_video_gcs_uri": "gs://bucket/phase_1/video.mp4",
        "artifacts": {
            "transcript": {
                "uri": "gs://bucket/transcript.json",
                "source_audio": "https://youtube.com/watch?v=x",
                "video_gcs_uri": "gs://bucket/phase_1/video.mp4",
                "words": [],
                "speaker_bindings": [],
            },
            "visual_tracking": {
                "uri": "gs://bucket/visual.json",
                "source_video": "https://youtube.com/watch?v=x",
                "video_gcs_uri": "gs://bucket/phase_1/video.mp4",
                "schema_version": "2.0.0",
                "task_type": "person_tracking",
                "coordinate_space": "absolute_original_frame_xyxy",
                "geometry_type": "aabb",
                "class_taxonomy": {"0": "person"},
                "tracking_metrics": {"schema_pass_rate": 1.0},
                "tracks": [],
                "face_detections": [],
                "person_detections": [],
                "label_detections": [],
                "object_tracking": [],
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "video_metadata": {"width": 1920, "height": 1080, "fps": 30.0, "duration_ms": 1000},
            },
            "events": None,
        },
        "metadata": {
            "runtime": {"provider": "digitalocean", "worker_id": "worker-1", "region": None},
            "timings": {"ingest_ms": 1, "processing_ms": 1, "upload_ms": 1},
            "quality_metrics": {"schema_pass_rate": 1.0, "transcript_coverage": 1.0, "tracking_confidence": 1.0},
            "retry": None,
            "failure": None,
        },
    }
    store.save_job(
        job_id="job_123",
        source_url="https://youtube.com/watch?v=x",
        status="succeeded",
        manifest=manifest,
        manifest_uri="gs://bucket/manifest.json",
    )
    client = TestClient(create_app(store=store, output_root=tmp_path))

    response = client.get("/jobs/job_123/result")

    assert response.status_code == 200
    assert response.json()["job_id"] == "job_123"
    assert response.json()["status"] == "succeeded"


def test_get_job_logs_returns_tail(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    log_path = tmp_path / "logs" / "job_123.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("line-1\nline-2\nline-3\n", encoding="utf-8")
    store.save_job(
        job_id="job_123",
        source_url="https://youtube.com/watch?v=x",
        status="running",
        current_step="asr_tracking",
        progress_message="Running ASR and tracking",
        progress_pct=0.2,
        log_path=str(log_path),
    )
    client = TestClient(create_app(store=store, output_root=tmp_path))

    response = client.get("/jobs/job_123/logs?tail_lines=2")

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job_123",
        "log_path": str(log_path),
        "lines": ["line-2", "line-3"],
    }


def test_dashboard_jobs_lists_recent_jobs(tmp_path: Path):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    store.save_job(
        job_id="job_1",
        source_url="https://youtube.com/watch?v=1",
        status="running",
        current_step="tracking",
        progress_message="Running visual tracking",
        progress_pct=0.3,
    )
    store.save_job(
        job_id="job_2",
        source_url="https://youtube.com/watch?v=2",
        status="queued",
        current_step="queued",
        progress_message="Queued",
        progress_pct=0.0,
    )
    client = TestClient(create_app(store=store, output_root=tmp_path))

    response = client.get("/dashboard/api/jobs?limit=5")

    assert response.status_code == 200
    body = response.json()
    assert [job["job_id"] for job in body["jobs"]] == ["job_2", "job_1"]


def test_dashboard_jobs_tolerates_malformed_legacy_failure_json(tmp_path: Path):
    db_path = tmp_path / "jobs.db"
    store = SQLiteJobStore(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO jobs (
                job_id, source_url, runtime_controls_json, status, retries, claim_token, manifest_json,
                manifest_uri, failure_json, current_step, progress_message, progress_pct, log_path,
                created_at, updated_at, started_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job_bad",
                "https://youtube.com/watch?v=bad",
                None,
                "failed",
                0,
                None,
                None,
                None,
                "Internal Server Error",
                "speaker_binding",
                "failed",
                0.9,
                None,
                datetime.now(UTC).isoformat(),
                datetime.now(UTC).isoformat(),
                None,
                None,
            ),
        )
    client = TestClient(create_app(store=store, output_root=tmp_path))

    response = client.get("/dashboard/api/jobs?limit=5")

    assert response.status_code == 200
    body = response.json()
    assert body["jobs"][0]["job_id"] == "job_bad"
    assert body["jobs"][0]["failure"] == {
        "_parse_error": "invalid_json",
        "_field": "failure_json",
        "_raw": "Internal Server Error",
    }


def test_dashboard_proxy_jobs_uses_remote_base(tmp_path: Path, monkeypatch):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    client = TestClient(create_app(store=store, output_root=tmp_path))

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"jobs": [{"job_id": "job_remote"}]}

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            self.base_url = kwargs["base_url"]

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, path, params=None):
            assert str(self.base_url) == "http://remote.example"
            assert path == "/dashboard/api/jobs"
            assert params == {"limit": 10}
            return DummyResponse()

    monkeypatch.setenv("DO_PHASE1_DASHBOARD_REMOTE_BASE_URL", "http://remote.example")
    monkeypatch.setattr("backend.do_phase1_service.app.httpx.AsyncClient", DummyAsyncClient)

    response = client.get("/dashboard/api/jobs?remote=1&limit=10")

    assert response.status_code == 200
    assert response.json() == {"jobs": [{"job_id": "job_remote"}]}
