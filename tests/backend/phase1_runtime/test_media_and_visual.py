from __future__ import annotations

from pathlib import Path

import pytest


def test_prepare_workspace_media_copies_local_video_and_extracts_audio(tmp_path: Path):
    from backend.phase1_runtime.media import prepare_workspace_media
    from backend.phase1_runtime.models import Phase1Workspace

    calls: list[str] = []
    source_video = tmp_path / "input.mp4"
    source_video.write_text("video", encoding="utf-8")

    def fake_audio_extractor(*, video_path: Path, audio_path: Path) -> None:
        calls.append(f"audio:{video_path.name}")
        audio_path.write_text("audio", encoding="utf-8")

    workspace = Phase1Workspace.create(root=tmp_path, run_id="run_001")
    prepared = prepare_workspace_media(
        source_path=str(source_video),
        workspace=workspace,
        audio_extractor=fake_audio_extractor,
    )

    assert prepared.video_path == workspace.video_path
    assert prepared.audio_path == workspace.audio_path
    assert workspace.video_path.read_text(encoding="utf-8") == "video"
    assert workspace.audio_path.read_text(encoding="utf-8") == "audio"
    assert calls == ["audio:source_video.mp4"]


def test_simple_visual_extractor_emits_single_shot_payload(tmp_path: Path):
    from backend.phase1_runtime.models import Phase1Workspace
    from backend.phase1_runtime.visual import V31VisualExtractor

    workspace = Phase1Workspace.create(root=tmp_path, run_id="run_001")
    workspace.video_path.write_text("video", encoding="utf-8")

    extractor = V31VisualExtractor(
        metadata_probe=lambda video_path: {
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "duration_ms": 12000,
        },
        shot_detector=lambda video_path, duration_ms: [],
        tracker_runner=lambda video_path: [],
    )
    payload = extractor.extract(video_path=workspace.video_path, workspace=workspace)

    assert payload["video_metadata"] == {
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "duration_ms": 12000,
    }
    assert payload["shot_changes"] == [{"start_time_ms": 0, "end_time_ms": 12000}]
    assert payload["tracks"] == []
    assert payload["person_detections"] == []
    assert payload["face_detections"] == []
    assert payload["visual_identities"] == []
    assert payload["mask_stability_signals"] == []
    assert payload["tracking_metrics"]["tracker_backend"] == "rfdetr_small_bytetrack"
    assert payload["tracking_metrics"]["input_track_rows"] == 0
    assert payload["tracking_metrics"]["emitted_track_rows"] == 0
    assert payload["tracking_metrics"]["emitted_person_detection_segments"] == 0
    assert payload["tracking_metrics"]["shot_count"] == 1


def test_visual_extractor_detects_shots_splits_tracks_and_builds_person_detections(tmp_path: Path):
    from backend.phase1_runtime.models import Phase1Workspace
    from backend.phase1_runtime.visual import V31VisualExtractor

    workspace = Phase1Workspace.create(root=tmp_path, run_id="run_002")
    workspace.video_path.write_text("video", encoding="utf-8")

    extractor = V31VisualExtractor(
        metadata_probe=lambda video_path: {
            "width": 1280,
            "height": 720,
            "fps": 25.0,
            "duration_ms": 4000,
        },
        shot_detector=lambda video_path, duration_ms: [500, 3000],
        tracker_runner=lambda video_path: [
            {"frame_idx": 12, "track_id": "1", "x1": 10.0, "y1": 20.0, "x2": 50.0, "y2": 80.0, "confidence": 0.8},
            {"frame_idx": 13, "track_id": "1", "x1": 12.0, "y1": 22.0, "x2": 52.0, "y2": 82.0, "confidence": 0.85},
            {"frame_idx": 60, "track_id": 2, "x1": 100.0, "y1": 120.0, "x2": 180.0, "y2": 260.0, "confidence": 0.7},
        ],
    )

    payload = extractor.extract(video_path=workspace.video_path, workspace=workspace)

    assert payload["shot_changes"] == [
        {"start_time_ms": 0, "end_time_ms": 500},
        {"start_time_ms": 500, "end_time_ms": 3000},
        {"start_time_ms": 3000, "end_time_ms": 4000},
    ]
    assert len(payload["tracks"]) == 3
    assert payload["tracks"][0]["track_id"].startswith("track_")
    assert payload["tracks"][0]["track_id"] != payload["tracks"][1]["track_id"]
    assert payload["tracks"][0]["local_track_id"] != payload["tracks"][1]["local_track_id"]
    assert payload["tracks"][0]["frame_idx"] == 12
    assert payload["tracks"][0]["chunk_idx"] == 0
    assert payload["tracks"][0]["label"] == "person"
    assert payload["tracks"][0]["geometry_type"] == "aabb"
    assert payload["tracks"][0]["bbox_norm_xywh"] == {
        "x_center": pytest.approx((30.0 / 1280.0), abs=1e-6),
        "y_center": pytest.approx((50.0 / 720.0), abs=1e-6),
        "width": pytest.approx((40.0 / 1280.0), abs=1e-6),
        "height": pytest.approx((60.0 / 720.0), abs=1e-6),
    }
    assert len(payload["person_detections"]) == 3
    assert payload["person_detections"][0]["track_id"] == payload["tracks"][0]["track_id"]
    assert payload["person_detections"][0]["segment_start_ms"] == 480
    assert payload["person_detections"][0]["segment_end_ms"] == 480
    assert payload["person_detections"][0]["timestamped_objects"][0]["bounding_box"] == {
        "left": pytest.approx(10.0 / 1280.0, abs=1e-6),
        "top": pytest.approx(20.0 / 720.0, abs=1e-6),
        "right": pytest.approx(50.0 / 1280.0, abs=1e-6),
        "bottom": pytest.approx(80.0 / 720.0, abs=1e-6),
    }
    assert payload["tracking_metrics"]["tracker_backend"] == "rfdetr_small_bytetrack"
    assert payload["tracking_metrics"]["input_track_rows"] == 3
    assert payload["tracking_metrics"]["emitted_track_rows"] == 3
    assert payload["tracking_metrics"]["emitted_person_detection_segments"] == 3
    assert payload["tracking_metrics"]["shot_count"] == 3
    assert payload["tracking_metrics"]["camera_cut_gating_enabled"] is True
    assert payload["tracking_metrics"]["camera_cut_split_source_tracks"] == 1
    assert payload["tracking_metrics"]["camera_cut_split_emitted_segments"] == 2


def test_visual_extractor_fails_hard_when_tracker_not_available(tmp_path: Path):
    from backend.phase1_runtime.models import Phase1Workspace
    from backend.phase1_runtime.visual import V31VisualExtractor

    workspace = Phase1Workspace.create(root=tmp_path, run_id="run_003")
    workspace.video_path.write_text("video", encoding="utf-8")

    extractor = V31VisualExtractor(
        metadata_probe=lambda video_path: {
            "width": 1280,
            "height": 720,
            "fps": 25.0,
            "duration_ms": 4000,
        },
        shot_detector=lambda video_path, duration_ms: [],
        tracker_runner=lambda video_path: (_ for _ in ()).throw(RuntimeError("tracker missing")),
    )

    with pytest.raises(RuntimeError, match="tracker missing"):
        extractor.extract(video_path=workspace.video_path, workspace=workspace)
