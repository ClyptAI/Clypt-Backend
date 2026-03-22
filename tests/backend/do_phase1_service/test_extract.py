from pathlib import Path
import threading
import time

from backend.do_phase1_service.extract import host_extraction_lock, run_extraction_job


class FakeStorage:
    def __init__(self):
        self.uploaded = []

    def upload_file(self, source_path, object_name: str) -> str:
        self.uploaded.append((str(source_path), object_name))
        return f"gs://bucket/{object_name}"

    def upload_bytes(self, data: bytes, object_name: str) -> str:
        self.uploaded.append((len(data), object_name))
        return f"gs://bucket/{object_name}"


def test_extract_job_produces_manifest_and_artifacts(tmp_path: Path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")

    monkeypatch.setattr(
        "backend.do_phase1_service.extract.download_media",
        lambda url: (str(video_path), str(audio_path)),
    )
    monkeypatch.setattr(
        "backend.do_phase1_service.extract.execute_local_extraction",
        lambda video_path, audio_path, youtube_url: {
            "status": "success",
            "phase_1_visual": {
                "source_video": youtube_url,
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
            "phase_1_audio": {
                "source_audio": youtube_url,
                "words": [],
                "speaker_bindings": [],
            },
        },
    )
    monkeypatch.setattr("backend.do_phase1_service.extract.enrich_visual_ledger_for_downstream", lambda phase_1_visual, phase_1_audio, video_path: phase_1_visual)
    monkeypatch.setattr("backend.do_phase1_service.extract.validate_phase_handoff", lambda visual_ledger, audio_ledger: None)

    result = run_extraction_job(
        source_url="https://youtube.com/watch?v=x",
        job_id="job_123",
        output_dir=tmp_path,
        storage=FakeStorage(),
        host_lock_path=tmp_path / "extract.lock",
    )

    assert result.status == "succeeded"
    assert result.artifacts.transcript.uri
    assert result.artifacts.visual_tracking.uri
    assert result.canonical_video_gcs_uri == "gs://bucket/phase_1/jobs/job_123/source_video.mp4"
    assert result.metadata.retry is None


def test_host_lock_serializes_second_extraction_attempt(tmp_path: Path):
    lock_path = tmp_path / "extract.lock"
    events = []

    def second_waiter():
        events.append("second-start")
        with host_extraction_lock(lock_path):
            events.append("second-acquired")

    with host_extraction_lock(lock_path):
        worker = threading.Thread(target=second_waiter)
        worker.start()
        time.sleep(0.1)
        events.append("first-held")
        assert events == ["second-start", "first-held"]
    worker.join(timeout=1.0)

    assert events == ["second-start", "first-held", "second-acquired"]


def test_host_lock_covers_download_and_waits_before_second_download(tmp_path: Path, monkeypatch):
    lock_path = tmp_path / "extract.lock"
    order = []
    download_allowed = threading.Event()
    uploaded_bytes = {}
    enriched_bytes = {}

    def fake_download_media(url):
        order.append(f"download:{url}")
        if url.endswith("first"):
            download_allowed.wait(timeout=2.0)
        video_path = tmp_path / f"{url.rsplit('=', 1)[-1]}.mp4"
        audio_path = tmp_path / f"{url.rsplit('=', 1)[-1]}.wav"
        video_path.write_bytes(url.encode("utf-8"))
        audio_path.write_bytes(b"audio")
        return str(video_path), str(audio_path)

    def fake_execute_local_extraction(**kwargs):
        order.append(f"extract:{kwargs['youtube_url']}")
        return {
            "status": "success",
            "phase_1_visual": {
                "source_video": kwargs["youtube_url"],
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
            "phase_1_audio": {
                "source_audio": kwargs["youtube_url"],
                "words": [],
                "speaker_bindings": [],
            },
        }

    monkeypatch.setattr("backend.do_phase1_service.extract.download_media", fake_download_media)
    monkeypatch.setattr("backend.do_phase1_service.extract.execute_local_extraction", fake_execute_local_extraction)
    class InspectingStorage(FakeStorage):
        def upload_file(self, source_path, object_name: str) -> str:
            uploaded_bytes[object_name] = Path(source_path).read_bytes()
            return super().upload_file(source_path, object_name)

    def fake_enrich_visual_ledger_for_downstream(phase_1_visual, phase_1_audio, video_path):
        enriched_bytes[str(video_path)] = Path(video_path).read_bytes()
        return phase_1_visual

    monkeypatch.setattr("backend.do_phase1_service.extract.enrich_visual_ledger_for_downstream", fake_enrich_visual_ledger_for_downstream)
    monkeypatch.setattr("backend.do_phase1_service.extract.validate_phase_handoff", lambda visual_ledger, audio_ledger: None)

    first_done = []

    def run_first():
        result = run_extraction_job(
            source_url="https://youtube.com/watch?v=first",
            job_id="job_first",
            output_dir=tmp_path,
            storage=InspectingStorage(),
            host_lock_path=lock_path,
        )
        first_done.append(result.job_id)

    worker = threading.Thread(target=run_first)
    worker.start()
    time.sleep(0.1)

    second = threading.Thread(
        target=lambda: run_extraction_job(
            source_url="https://youtube.com/watch?v=second",
            job_id="job_second",
            output_dir=tmp_path,
            storage=InspectingStorage(),
            host_lock_path=lock_path,
        )
    )
    second.start()
    time.sleep(0.1)
    assert order == ["download:https://youtube.com/watch?v=first"]

    download_allowed.set()
    worker.join(timeout=2.0)
    second.join(timeout=2.0)

    assert first_done == ["job_first"]
    assert order == [
        "download:https://youtube.com/watch?v=first",
        "extract:https://youtube.com/watch?v=first",
        "download:https://youtube.com/watch?v=second",
        "extract:https://youtube.com/watch?v=second",
    ]
    assert uploaded_bytes["phase_1/jobs/job_first/source_video.mp4"] == b"https://youtube.com/watch?v=first"
    assert uploaded_bytes["phase_1/jobs/job_second/source_video.mp4"] == b"https://youtube.com/watch?v=second"
    assert enriched_bytes[str(tmp_path / "first.mp4")] == b"https://youtube.com/watch?v=first"
    assert enriched_bytes[str(tmp_path / "second.mp4")] == b"https://youtube.com/watch?v=second"
