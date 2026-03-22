from pathlib import Path

import pytest

from backend.do_phase1_service.extract import HostExtractionBusyError, host_extraction_lock, run_extraction_job


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


def test_host_lock_rejects_second_extraction_attempt(tmp_path: Path):
    lock_path = tmp_path / "extract.lock"

    with host_extraction_lock(lock_path):
        with pytest.raises(HostExtractionBusyError, match="already running"):
            with host_extraction_lock(lock_path):
                pass
