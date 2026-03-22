from pathlib import Path

from backend.do_phase1_service.jobs import enqueue_job
from backend.do_phase1_service.state_store import SQLiteJobStore
from backend.do_phase1_service.worker import run_worker_loop, run_worker_once


class DummyResult:
    status = "succeeded"
    manifest_uri = "gs://bucket/manifests/job_123.json"

    def model_dump(self, mode: str = "python"):
        return {
            "contract_version": "v1",
            "job_id": "job_123",
            "status": "succeeded",
            "source_video": {"source_url": "https://youtube.com/watch?v=x"},
            "canonical_video_gcs_uri": "gs://bucket/phase_1/jobs/job_123/source_video.mp4",
            "artifacts": {
                "transcript": {
                    "uri": "gs://bucket/transcript.json",
                    "source_audio": "https://youtube.com/watch?v=x",
                    "video_gcs_uri": "gs://bucket/phase_1/jobs/job_123/source_video.mp4",
                    "words": [],
                    "speaker_bindings": [],
                },
                "visual_tracking": {
                    "uri": "gs://bucket/visual.json",
                    "source_video": "https://youtube.com/watch?v=x",
                    "video_gcs_uri": "gs://bucket/phase_1/jobs/job_123/source_video.mp4",
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
            "manifest_uri": self.manifest_uri,
        }


def test_worker_promotes_job_through_lifecycle(tmp_path: Path, monkeypatch):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    enqueue_job(store, job_id="job_123", payload={"source_url": "https://youtube.com/watch?v=x"})
    observed_statuses = []

    def fake_run_extraction_job(**kwargs):
        observed_statuses.append(store.get_job(kwargs["job_id"]).status)
        return DummyResult()

    monkeypatch.setattr("backend.do_phase1_service.worker.run_extraction_job", fake_run_extraction_job)

    run_worker_once(store, output_root=tmp_path)

    assert observed_statuses == ["running"]
    job = store.get_job("job_123")
    assert job is not None
    assert job.status == "succeeded"
    assert job.manifest_uri == "gs://bucket/manifests/job_123.json"


def test_worker_marks_failed_job_and_continues_processing(tmp_path: Path, monkeypatch):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    enqueue_job(store, job_id="job_fail", payload={"source_url": "https://youtube.com/watch?v=fail"})
    enqueue_job(store, job_id="job_next", payload={"source_url": "https://youtube.com/watch?v=next"})
    calls = []

    def fake_run_extraction_job(**kwargs):
        calls.append(kwargs["job_id"])
        if kwargs["job_id"] == "job_fail":
            raise RuntimeError("boom")
        return DummyResult()

    monkeypatch.setattr("backend.do_phase1_service.worker.run_extraction_job", fake_run_extraction_job)

    assert run_worker_once(store, output_root=tmp_path) is True

    failed_job = store.get_job("job_fail")
    assert failed_job is not None
    assert failed_job.status == "failed"
    assert failed_job.failure == {
        "error_type": "RuntimeError",
        "error_message": "boom",
        "failed_step": "extraction",
    }

    assert run_worker_once(store, output_root=tmp_path) is True

    next_job = store.get_job("job_next")
    assert next_job is not None
    assert next_job.status == "succeeded"
    assert calls == ["job_fail", "job_next"]


def test_worker_loop_continues_after_failure(tmp_path: Path, monkeypatch):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    enqueue_job(store, job_id="job_fail", payload={"source_url": "https://youtube.com/watch?v=fail"})
    enqueue_job(store, job_id="job_next", payload={"source_url": "https://youtube.com/watch?v=next"})
    seen = []

    def fake_run_extraction_job(**kwargs):
        seen.append(kwargs["job_id"])
        if kwargs["job_id"] == "job_fail":
            raise RuntimeError("boom")
        return DummyResult()

    def stop_after_idle(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr("backend.do_phase1_service.worker.run_extraction_job", fake_run_extraction_job)
    monkeypatch.setattr("backend.do_phase1_service.worker.time.sleep", stop_after_idle)

    try:
        run_worker_loop(store=store, output_root=tmp_path, poll_interval_seconds=0)
    except KeyboardInterrupt:
        pass

    assert seen == ["job_fail", "job_next"]
    assert store.get_job("job_fail").status == "failed"
    assert store.get_job("job_next").status == "succeeded"


def test_worker_contains_control_plane_failure_and_processes_later_job(tmp_path: Path, monkeypatch):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    enqueue_job(store, job_id="job_fail", payload={"source_url": "https://youtube.com/watch?v=fail"})
    enqueue_job(store, job_id="job_next", payload={"source_url": "https://youtube.com/watch?v=next"})
    seen = []
    original_mark_succeeded = __import__("backend.do_phase1_service.worker", fromlist=["mark_succeeded"]).mark_succeeded

    def fake_run_extraction_job(**kwargs):
        seen.append(kwargs["job_id"])
        return DummyResult()

    def fake_mark_succeeded(store, job_id, *, claim_token, manifest, manifest_uri):
        if job_id == "job_fail":
            raise RuntimeError("persist boom")
        return original_mark_succeeded(
            store,
            job_id,
            claim_token=claim_token,
            manifest=manifest,
            manifest_uri=manifest_uri,
        )

    monkeypatch.setattr("backend.do_phase1_service.worker.run_extraction_job", fake_run_extraction_job)
    monkeypatch.setattr("backend.do_phase1_service.worker.mark_succeeded", fake_mark_succeeded)

    assert run_worker_once(store, output_root=tmp_path) is True
    failed_job = store.get_job("job_fail")
    assert failed_job is not None
    assert failed_job.status == "failed"
    assert failed_job.failure == {
        "error_type": "RuntimeError",
        "error_message": "persist boom",
        "failed_step": "persist_success",
    }

    assert run_worker_once(store, output_root=tmp_path) is True
    assert store.get_job("job_next").status == "succeeded"
    assert seen == ["job_fail", "job_next"]


def test_worker_loop_backs_off_on_unexpected_exception(tmp_path: Path, monkeypatch):
    store = SQLiteJobStore(tmp_path / "jobs.db")
    calls = []
    sleeps = []

    def fake_run_worker_once(*args, **kwargs):
        calls.append("run")
        if len(calls) == 1:
            raise RuntimeError("claim boom")
        return False

    def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 2:
            raise KeyboardInterrupt

    monkeypatch.setattr("backend.do_phase1_service.worker.run_worker_once", fake_run_worker_once)
    monkeypatch.setattr("backend.do_phase1_service.worker.time.sleep", fake_sleep)

    try:
        run_worker_loop(store=store, output_root=tmp_path, poll_interval_seconds=0.5)
    except KeyboardInterrupt:
        pass

    assert calls == ["run", "run"]
    assert sleeps == [2.0, 0.5]
