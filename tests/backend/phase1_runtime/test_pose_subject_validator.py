from __future__ import annotations

import pytest
import numpy as np


def test_pose_quality_requires_head_and_upper_body_anchor_but_not_duration():
    from backend.phase1_runtime.pose_subject_validator import (
        PoseSampleEvidence,
        evaluate_tracklet_pose_quality,
    )

    report = evaluate_tracklet_pose_quality(
        track_id="track_1",
        rfdetr_confidences=[0.91],
        samples=[
            PoseSampleEvidence(
                frame_idx=42,
                has_head_evidence=True,
                has_upper_body_anchor=True,
            )
        ],
        min_rfdetr_confidence=0.85,
        min_head_evidence_ratio=0.40,
        min_upper_body_anchor_ratio=0.25,
    )

    assert report["auto_follow_eligible"] is True
    assert report["subject_quality"]["sampled_frames"] == 1
    assert report["subject_quality"]["head_evidence_ratio"] == pytest.approx(1.0)
    assert report["subject_quality"]["upper_body_anchor_ratio"] == pytest.approx(1.0)
    assert report["subject_quality"]["median_rfdetr_confidence"] == pytest.approx(0.91)


def test_pose_quality_rejects_headless_high_confidence_body_fragment():
    from backend.phase1_runtime.pose_subject_validator import (
        PoseSampleEvidence,
        evaluate_tracklet_pose_quality,
    )

    report = evaluate_tracklet_pose_quality(
        track_id="track_arm",
        rfdetr_confidences=[0.99, 0.98, 0.97],
        samples=[
            PoseSampleEvidence(
                frame_idx=10,
                has_head_evidence=False,
                has_upper_body_anchor=False,
            ),
            PoseSampleEvidence(
                frame_idx=22,
                has_head_evidence=False,
                has_upper_body_anchor=True,
            ),
        ],
        min_rfdetr_confidence=0.85,
        min_head_evidence_ratio=0.40,
        min_upper_body_anchor_ratio=0.25,
    )

    assert report["auto_follow_eligible"] is False
    assert report["subject_quality"]["median_rfdetr_confidence"] == pytest.approx(0.98)
    assert report["subject_quality"]["head_evidence_ratio"] == pytest.approx(0.0)
    assert report["subject_quality"]["upper_body_anchor_ratio"] == pytest.approx(0.5)


def test_pose_quality_requires_sustained_upper_body_anchor():
    from backend.phase1_runtime.pose_subject_validator import (
        PoseSampleEvidence,
        evaluate_tracklet_pose_quality,
    )

    report = evaluate_tracklet_pose_quality(
        track_id="track_partial",
        rfdetr_confidences=[0.98, 0.97, 0.96, 0.95],
        samples=[
            PoseSampleEvidence(
                frame_idx=1,
                has_head_evidence=True,
                has_upper_body_anchor=True,
            ),
            PoseSampleEvidence(
                frame_idx=2,
                has_head_evidence=True,
                has_upper_body_anchor=False,
            ),
            PoseSampleEvidence(
                frame_idx=3,
                has_head_evidence=True,
                has_upper_body_anchor=False,
            ),
            PoseSampleEvidence(
                frame_idx=4,
                has_head_evidence=True,
                has_upper_body_anchor=False,
            ),
        ],
        min_rfdetr_confidence=0.85,
        min_head_evidence_ratio=0.40,
        min_upper_body_anchor_ratio=0.50,
    )

    assert report["auto_follow_eligible"] is False
    assert report["subject_quality"]["head_evidence_ratio"] == pytest.approx(1.0)
    assert report["subject_quality"]["upper_body_anchor_ratio"] == pytest.approx(0.25)


def test_pose_evidence_maps_crop_keypoints_to_source_coordinates():
    from types import SimpleNamespace

    from backend.phase1_runtime.pose_subject_validator import _pose_evidence_from_result

    data = np.zeros((1, 17, 3), dtype=np.float32)
    data[0, 0] = [10.0, 20.0, 0.95]
    data[0, 5] = [8.0, 60.0, 0.90]
    data[0, 6] = [32.0, 60.0, 0.90]
    result = SimpleNamespace(keypoints=SimpleNamespace(data=data))

    evidence = _pose_evidence_from_result(
        result,
        keypoint_confidence=0.35,
        offset_xy=(100, 200),
    )

    assert evidence.has_head_evidence is True
    assert evidence.has_upper_body_anchor is True
    assert evidence.head_center_xy == pytest.approx((110.0, 220.0))
    assert evidence.shoulder_center_xy == pytest.approx((120.0, 260.0))
    assert evidence.upper_torso_anchor_xy == pytest.approx((116.5, 246.0))


def test_pose_evidence_from_result_rejects_multi_person_pose_batches():
    from types import SimpleNamespace

    from backend.phase1_runtime.pose_subject_validator import _pose_evidence_from_result

    data = np.zeros((2, 17, 3), dtype=np.float32)
    data[0, 0] = [10.0, 20.0, 0.95]
    data[1, 0] = [60.0, 20.0, 0.95]
    result = SimpleNamespace(keypoints=SimpleNamespace(data=data))

    with pytest.raises(RuntimeError, match="requires exactly one pose instance"):
        _pose_evidence_from_result(
            result,
            keypoint_confidence=0.35,
            offset_xy=(0, 0),
        )


def test_yolo_pose_validator_preserves_source_pose_anchor_points(monkeypatch):
    from types import SimpleNamespace

    from backend.phase1_runtime import pose_subject_validator as module

    class FakeModel:
        def predict(self, crop_batch, imgsz, conf, verbose):  # noqa: ARG002
            data = np.zeros((1, 17, 3), dtype=np.float32)
            data[0, 0] = [10.0, 20.0, 0.95]
            data[0, 5] = [8.0, 60.0, 0.90]
            data[0, 6] = [32.0, 60.0, 0.90]
            return [SimpleNamespace(keypoints=SimpleNamespace(data=data)) for _ in crop_batch]

    decoded = SimpleNamespace(
        frame_idx=7,
        rgb=np.zeros((240, 320, 3), dtype=np.uint8),
    )

    monkeypatch.setattr(module, "decode_video_frames", lambda **kwargs: iter([decoded]))

    config = SimpleNamespace(
        pose_max_samples_per_tracklet=1,
        pose_imgsz=256,
        pose_confidence=0.25,
        pose_keypoint_confidence=0.35,
        pose_batch_size=4,
        pose_crop_padding_ratio=0.0,
        frame_decode_backend="cpu",
        gpu_decode_backend="none",
        pose_min_rfdetr_confidence=0.85,
        pose_min_head_evidence_ratio=0.40,
        pose_min_upper_body_anchor_ratio=0.25,
    )
    validator = module.YoloPoseSubjectValidator(config=config)
    monkeypatch.setattr(validator, "_load_model", lambda: FakeModel())

    reports = validator(
        video_path=SimpleNamespace(),
        tracks=[
            {
                "track_id": "track_1",
                "frame_idx": 7,
                "x1": 100.0,
                "y1": 50.0,
                "x2": 180.0,
                "y2": 170.0,
                "confidence": 0.99,
            }
        ],
        metadata={"fps": 30.0},
        config=config,
    )

    anchors = reports["track_1"]["subject_quality"]["pose_anchor_points"]
    assert anchors == [
        {
            "frame_idx": 7,
            "head_center_xy": pytest.approx([110.0, 70.0]),
            "shoulder_center_xy": pytest.approx([120.0, 110.0]),
            "upper_torso_anchor_xy": pytest.approx([116.5, 96.0]),
        }
    ]


def test_yolo_pose_validator_uses_track_mask_to_select_single_pose_instance(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from backend.phase1_runtime import pose_subject_validator as module
    from backend.phase1_runtime.masks import MaskArtifactWriter

    class FakeModel:
        def predict(self, crop_batch, imgsz, conf, verbose):  # noqa: ARG002
            data = np.zeros((2, 17, 3), dtype=np.float32)
            data[0, 0] = [12.0, 20.0, 0.95]
            data[0, 5] = [10.0, 60.0, 0.90]
            data[0, 6] = [30.0, 60.0, 0.90]
            data[1, 0] = [65.0, 22.0, 0.95]
            data[1, 5] = [55.0, 62.0, 0.90]
            data[1, 6] = [75.0, 62.0, 0.90]
            return [SimpleNamespace(keypoints=SimpleNamespace(data=data)) for _ in crop_batch]

    decoded = SimpleNamespace(
        frame_idx=7,
        rgb=np.zeros((240, 320, 3), dtype=np.uint8),
    )

    monkeypatch.setattr(module, "decode_video_frames", lambda **kwargs: iter([decoded]))

    video_path = tmp_path / "source_video.mp4"
    video_path.write_text("video", encoding="utf-8")
    mask_writer = MaskArtifactWriter(
        artifact_path=tmp_path / "visual_masks_lowres_v1.npz",
    )
    mask_ref = mask_writer.add(
        frame_idx=7,
        detection_id="raw_7_0",
        mask=np.array(
            [
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
            ],
            dtype=np.uint8,
        ),
        bbox_xyxy=[100.0, 50.0, 180.0, 170.0],
        source_size=(240, 320),
    )
    artifact = mask_writer.finalize()
    assert artifact is not None

    config = SimpleNamespace(
        pose_max_samples_per_tracklet=1,
        pose_imgsz=256,
        pose_confidence=0.25,
        pose_keypoint_confidence=0.35,
        pose_batch_size=4,
        pose_crop_padding_ratio=0.0,
        frame_decode_backend="cpu",
        gpu_decode_backend="none",
        pose_min_rfdetr_confidence=0.85,
        pose_min_head_evidence_ratio=0.40,
        pose_min_upper_body_anchor_ratio=0.25,
    )
    validator = module.YoloPoseSubjectValidator(config=config)
    monkeypatch.setattr(validator, "_load_model", lambda: FakeModel())

    reports = validator(
        video_path=video_path,
        tracks=[
            {
                "track_id": "track_1",
                "frame_idx": 7,
                "x1": 100.0,
                "y1": 50.0,
                "x2": 180.0,
                "y2": 170.0,
                "confidence": 0.99,
                "mask_ref": mask_ref,
            }
        ],
        metadata={"fps": 30.0},
        config=config,
    )

    anchors = reports["track_1"]["subject_quality"]["pose_anchor_points"]
    assert anchors == [
        {
            "frame_idx": 7,
            "head_center_xy": pytest.approx([165.0, 72.0]),
            "shoulder_center_xy": pytest.approx([165.0, 112.0]),
            "upper_torso_anchor_xy": pytest.approx([165.0, 98.0]),
        }
    ]
