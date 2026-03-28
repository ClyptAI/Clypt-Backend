from __future__ import annotations

from backend.speaker_binding.visual_signals import (
    combine_visual_candidate_signals,
    sample_span_frame_indices,
    summarize_mouth_landmark_signal,
    summarize_pose_signal,
)


def test_sample_span_frame_indices_spreads_samples_across_span() -> None:
    assert sample_span_frame_indices([9, 0, 3, 1, 7, 5, 2, 8, 4, 6], max_samples=6) == [0, 1, 3, 5, 7, 9]


def test_summarize_mouth_landmark_signal_reports_visibility_motion_and_blendshape_support() -> None:
    summary = summarize_mouth_landmark_signal(
        [
            {
                "frame_idx": 10,
                "mouth_open_ratio": 0.9,
                "mouth_wide_ratio": 0.8,
                "blendshape_jaw_open": 0.9,
                "blendshape_mouth_open": 0.8,
                "face_detected": True,
            },
            {
                "frame_idx": 12,
                "mouth_open_ratio": 0.7,
                "mouth_wide_ratio": 0.6,
                "blendshape_jaw_open": 0.8,
                "blendshape_mouth_open": 0.7,
                "face_detected": True,
            },
            {
                "frame_idx": 11,
                "mouth_open_ratio": 0.3,
                "mouth_wide_ratio": 0.2,
                "face_detected": False,
            },
        ]
    )

    assert summary["usable_face_frame_count"] == 2
    assert summary["mouth_motion_score"] == 0.635
    assert summary["face_visibility_score"] == 0.667
    assert summary["blendshape_support_score"] == 0.667


def test_summarize_pose_signal_reports_visibility_and_stability() -> None:
    summary = summarize_pose_signal(
        [
            {
                "frame_idx": 10,
                "upper_body_visibility": 0.9,
                "head_visibility": 0.8,
                "frontal_support": 0.7,
                "torso_center_x": 0.5,
                "torso_center_y": 0.4,
                "shoulder_span": 0.2,
            },
            {
                "frame_idx": 11,
                "upper_body_visibility": 0.7,
                "head_visibility": 0.6,
                "frontal_support": 0.9,
                "torso_center_x": 0.51,
                "torso_center_y": 0.405,
                "shoulder_span": 0.198,
            },
            {
                "frame_idx": 12,
                "upper_body_visibility": 0.0,
                "head_visibility": 0.0,
                "frontal_support": 0.0,
            },
        ]
    )

    assert summary["usable_pose_frame_count"] == 2
    assert summary["pose_visibility_score"] == 0.775
    assert summary["pose_stability_score"] == 0.947


def test_combine_visual_candidate_signals_merges_scores_for_ranking() -> None:
    candidate = combine_visual_candidate_signals(
        visual_identity_id=" Global_Person_4 ",
        mouth_summary={
            "mouth_motion_score": 0.635,
            "face_visibility_score": 0.667,
            "blendshape_support_score": 0.667,
            "usable_face_frame_count": 2,
        },
        pose_summary={
            "pose_visibility_score": 0.775,
            "pose_stability_score": 0.947,
            "usable_pose_frame_count": 2,
        },
        mapping_summary={
            "mapping_confidence": 0.9,
        },
        local_track_id=" track-7 ",
    )

    assert candidate == {
        "visual_identity_id": "Global_Person_4",
        "local_track_id": "track-7",
        "mouth_motion_score": 0.635,
        "face_visibility_score": 0.667,
        "blendshape_support_score": 0.667,
        "usable_face_frame_count": 2,
        "pose_visibility_score": 0.775,
        "pose_stability_score": 0.947,
        "usable_pose_frame_count": 2,
        "mapping_confidence": 0.9,
        "composite_score": 0.728,
    }
