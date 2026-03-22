from pathlib import Path

from backend.do_phase1_service.storage import LocalGCSStorage, persist_phase1_outputs


def test_manifest_uses_durable_storage_uris(tmp_path: Path):
    storage = LocalGCSStorage(bucket="test-bucket", root_dir=tmp_path / "gcs")
    canonical_video_uri = storage.upload_bytes(b"video", object_name="phase_1/jobs/job_123/source_video.mp4")

    manifest = persist_phase1_outputs(
        storage=storage,
        output_dir=tmp_path,
        job_id="job_123",
        source_url="https://youtube.com/watch?v=x",
        canonical_video_uri=canonical_video_uri,
        phase_1_audio={
            "source_audio": "https://youtube.com/watch?v=x",
            "words": [],
            "speaker_bindings": [],
        },
        phase_1_visual={
            "source_video": "https://youtube.com/watch?v=x",
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
    )

    assert manifest.artifacts.transcript.uri.startswith("gs://")
    assert manifest.artifacts.visual_tracking.uri.startswith("gs://")
    assert manifest.canonical_video_gcs_uri == canonical_video_uri
    assert manifest.canonical_video_gcs_uri.endswith("/phase_1/jobs/job_123/source_video.mp4")
    assert manifest.metadata.retry is None
