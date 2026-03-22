from pathlib import Path

from fastapi.testclient import TestClient

from backend.do_phase1_service.app import create_app
from backend.do_phase1_service.jobs import enqueue_job
from backend.do_phase1_service.state_store import SQLiteJobStore


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
