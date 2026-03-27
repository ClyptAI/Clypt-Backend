from pathlib import Path

import pytest
from pydantic import ValidationError

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
    assert manifest.artifacts.transcript.audio_speaker_turns == []
    assert manifest.canonical_video_gcs_uri == canonical_video_uri
    assert manifest.canonical_video_gcs_uri.endswith("/phase_1/jobs/job_123/source_video.mp4")
    assert manifest.metadata.retry is None


def test_manifest_persists_local_clip_experiment_fields(tmp_path: Path):
    storage = LocalGCSStorage(bucket="test-bucket", root_dir=tmp_path / "gcs")
    canonical_video_uri = storage.upload_bytes(b"video", object_name="phase_1/jobs/job_456/source_video.mp4")

    manifest = persist_phase1_outputs(
        storage=storage,
        output_dir=tmp_path,
        job_id="job_456",
        source_url="https://youtube.com/watch?v=local",
        canonical_video_uri=canonical_video_uri,
        phase_1_audio={
            "source_audio": "https://youtube.com/watch?v=local",
            "words": [
                {
                    "word": "hello",
                    "start_time_ms": 0,
                    "end_time_ms": 100,
                    "speaker_track_id": "Global_Person_0",
                    "speaker_tag": "Global_Person_0",
                    "speaker_local_track_id": "track_1",
                    "speaker_local_tag": "track_1",
                }
            ],
            "audio_speaker_turns": [
                {
                    "speaker_id": "SPEAKER_00",
                    "start_time_ms": 0,
                    "end_time_ms": 1500,
                    "exclusive": True,
                    "overlap": False,
                    "confidence": 0.91,
                }
            ],
            "speaker_bindings": [
                {"track_id": "Global_Person_0", "start_time_ms": 0, "end_time_ms": 100, "word_count": 1}
            ],
            "speaker_bindings_local": [
                {"track_id": "track_1", "start_time_ms": 0, "end_time_ms": 100, "word_count": 1}
            ],
            "speaker_follow_bindings_local": [
                {"track_id": "track_1", "start_time_ms": 0, "end_time_ms": 100, "word_count": 1}
            ],
            "audio_speaker_local_track_map": [
                {
                    "speaker_id": "SPEAKER_00",
                    "local_track_id": "track_1",
                    "support_segments": 2,
                    "support_ms": 1800,
                    "confidence": 0.86,
                }
            ],
            "speaker_candidate_debug": [
                {
                    "word": "hello",
                    "start_time_ms": 0,
                    "end_time_ms": 100,
                    "active_audio_speaker_id": "SPEAKER_00",
                    "active_audio_local_track_id": "track_1",
                    "chosen_track_id": "Global_Person_0",
                    "chosen_local_track_id": "track_1",
                    "decision_source": "audio_boosted_visual",
                    "ambiguous": False,
                    "top_1_top_2_margin": 0.081,
                    "candidates": [
                        {
                            "local_track_id": "track_1",
                            "track_id": "Global_Person_0",
                            "blended_score": 0.301,
                            "asd_probability": 0.18,
                            "body_prior": 0.56,
                            "detection_confidence": 0.99,
                        },
                        {
                            "local_track_id": "track_9",
                            "track_id": "Global_Person_1",
                            "blended_score": 0.22,
                            "asd_probability": 0.16,
                            "body_prior": 0.44,
                            "detection_confidence": 0.88,
                        },
                    ],
                }
            ],
            "active_speakers_local": [
                {
                    "start_time_ms": 0,
                    "end_time_ms": 100,
                    "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
                    "visible_local_track_ids": ["track_1"],
                    "visible_track_ids": ["Global_Person_0"],
                    "offscreen_audio_speaker_ids": ["SPEAKER_01"],
                    "overlap": True,
                    "confidence": 0.73,
                    "decision_source": "turn_binding",
                }
            ],
            "overlap_follow_decisions": [
                {
                    "start_time_ms": 0,
                    "end_time_ms": 100,
                    "camera_target_local_track_id": "track_1",
                    "camera_target_track_id": "Global_Person_0",
                    "stay_wide": False,
                    "visible_local_track_ids": ["track_1"],
                    "offscreen_audio_speaker_ids": ["SPEAKER_01"],
                    "decision_model": "gemini-3-flash-preview",
                    "decision_source": "gemini",
                    "confidence": 0.81,
                }
            ],
        },
        phase_1_visual={
            "source_video": "https://youtube.com/watch?v=local",
            "schema_version": "2.0.0",
            "task_type": "person_tracking",
            "coordinate_space": "absolute_original_frame_xyxy",
            "geometry_type": "aabb",
            "class_taxonomy": {"0": "person"},
            "tracking_metrics": {"schema_pass_rate": 1.0},
            "tracks": [],
            "tracks_local": [
                {
                    "frame_idx": 0,
                    "local_frame_idx": 0,
                    "chunk_idx": 0,
                    "track_id": "track_1",
                    "local_track_id": 1,
                    "class_id": 0,
                    "label": "person",
                    "confidence": 0.99,
                    "x1": 100.0,
                    "y1": 120.0,
                    "x2": 220.0,
                    "y2": 340.0,
                    "x_center": 160.0,
                    "y_center": 230.0,
                    "width": 120.0,
                    "height": 220.0,
                    "source": "do_phase1",
                    "geometry_type": "aabb",
                    "bbox_norm_xywh": {
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.2,
                        "height": 0.3,
                    },
                }
            ],
            "face_detections": [],
            "person_detections": [],
            "label_detections": [],
            "object_tracking": [],
            "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
            "video_metadata": {"width": 1920, "height": 1080, "fps": 30.0, "duration_ms": 1000},
        },
    )

    assert manifest.artifacts.transcript.words[0].speaker_local_track_id == "track_1"
    assert manifest.artifacts.transcript.audio_speaker_turns[0].speaker_id == "SPEAKER_00"
    assert manifest.artifacts.transcript.audio_speaker_turns[0].confidence == 0.91
    assert manifest.artifacts.transcript.speaker_bindings_local[0].track_id == "track_1"
    assert manifest.artifacts.transcript.speaker_follow_bindings_local[0].track_id == "track_1"
    assert manifest.artifacts.transcript.audio_speaker_local_track_map[0].speaker_id == "SPEAKER_00"
    assert manifest.artifacts.transcript.audio_speaker_local_track_map[0].local_track_id == "track_1"
    assert manifest.artifacts.transcript.audio_speaker_local_track_map[0].support_ms == 1800
    assert manifest.artifacts.transcript.speaker_candidate_debug[0].decision_source == "audio_boosted_visual"
    assert manifest.artifacts.transcript.speaker_candidate_debug[0].candidates[0].local_track_id == "track_1"
    assert manifest.artifacts.transcript.active_speakers_local[0].offscreen_audio_speaker_ids == ["SPEAKER_01"]
    assert manifest.artifacts.transcript.overlap_follow_decisions[0].decision_source == "gemini"
    assert manifest.artifacts.visual_tracking.tracks_local[0].track_id == "track_1"


def test_manifest_persistence_rejects_unknown_audio_turn_fields(tmp_path: Path):
    storage = LocalGCSStorage(bucket="test-bucket", root_dir=tmp_path / "gcs")
    canonical_video_uri = storage.upload_bytes(b"video", object_name="phase_1/jobs/job_789/source_video.mp4")

    with pytest.raises(ValidationError, match="unexpected_field"):
        persist_phase1_outputs(
            storage=storage,
            output_dir=tmp_path,
            job_id="job_789",
            source_url="https://youtube.com/watch?v=bad",
            canonical_video_uri=canonical_video_uri,
            phase_1_audio={
                "source_audio": "https://youtube.com/watch?v=bad",
                "words": [],
                "speaker_bindings": [],
                "audio_speaker_turns": [
                    {
                        "speaker_id": "SPEAKER_00",
                        "start_time_ms": 0,
                        "end_time_ms": 1000,
                        "unexpected_field": True,
                    }
                ],
            },
            phase_1_visual={
                "source_video": "https://youtube.com/watch?v=bad",
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


def test_manifest_persistence_rejects_unknown_speaker_candidate_debug_fields(tmp_path: Path):
    storage = LocalGCSStorage(bucket="test-bucket", root_dir=tmp_path / "gcs")
    canonical_video_uri = storage.upload_bytes(b"video", object_name="phase_1/jobs/job_790/source_video.mp4")

    with pytest.raises(ValidationError, match="unexpected_field"):
        persist_phase1_outputs(
            storage=storage,
            output_dir=tmp_path,
            job_id="job_790",
            source_url="https://youtube.com/watch?v=bad-debug",
            canonical_video_uri=canonical_video_uri,
            phase_1_audio={
                "source_audio": "https://youtube.com/watch?v=bad-debug",
                "words": [],
                "speaker_bindings": [],
                "speaker_candidate_debug": [
                    {
                        "word": "hello",
                        "start_time_ms": 0,
                        "end_time_ms": 100,
                        "decision_source": "visual",
                        "ambiguous": False,
                        "candidates": [],
                        "unexpected_field": True,
                    }
                ],
            },
            phase_1_visual={
                "source_video": "https://youtube.com/watch?v=bad-debug",
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


def test_manifest_persistence_rejects_unknown_overlap_follow_decision_fields(tmp_path: Path):
    storage = LocalGCSStorage(bucket="test-bucket", root_dir=tmp_path / "gcs")
    canonical_video_uri = storage.upload_bytes(b"video", object_name="phase_1/jobs/job_791/source_video.mp4")

    with pytest.raises(ValidationError, match="unexpected_field"):
        persist_phase1_outputs(
            storage=storage,
            output_dir=tmp_path,
            job_id="job_791",
            source_url="https://youtube.com/watch?v=bad-overlap",
            canonical_video_uri=canonical_video_uri,
            phase_1_audio={
                "source_audio": "https://youtube.com/watch?v=bad-overlap",
                "words": [],
                "speaker_bindings": [],
                "overlap_follow_decisions": [
                    {
                        "start_time_ms": 0,
                        "end_time_ms": 100,
                        "camera_target_local_track_id": None,
                        "camera_target_track_id": None,
                        "stay_wide": True,
                        "visible_local_track_ids": [],
                        "offscreen_audio_speaker_ids": ["SPEAKER_02"],
                        "decision_model": None,
                        "decision_source": "deterministic",
                        "confidence": None,
                        "unexpected_field": True,
                    }
                ],
            },
            phase_1_visual={
                "source_video": "https://youtube.com/watch?v=bad-overlap",
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
