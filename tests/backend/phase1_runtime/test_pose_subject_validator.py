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
