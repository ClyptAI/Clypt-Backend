from __future__ import annotations

from backend.speaker_binding.mouth_motion import choose_visual_speaking_candidate


def test_choose_visual_speaking_candidate_returns_clear_winner() -> None:
    result = choose_visual_speaking_candidate(
        [
            {
                "visual_identity_id": "Global_Person_0",
                "mouth_motion_score": 0.9,
                "pose_visibility_score": 0.8,
                "face_visibility_score": 0.85,
                "mapping_confidence": 0.7,
            },
            {
                "visual_identity_id": "Global_Person_1",
                "mouth_motion_score": 0.4,
                "pose_visibility_score": 0.8,
                "face_visibility_score": 0.7,
                "mapping_confidence": 0.7,
            },
        ]
    )

    assert result["winner_visual_identity_id"] == "Global_Person_0"
    assert result["unresolved"] is False
    assert [item["visual_identity_id"] for item in result["ranked_candidates"]] == [
        "Global_Person_0",
        "Global_Person_1",
    ]


def test_choose_visual_speaking_candidate_marks_close_scores_ambiguous() -> None:
    result = choose_visual_speaking_candidate(
        [
            {
                "visual_identity_id": "Global_Person_0",
                "mouth_motion_score": 0.7,
                "pose_visibility_score": 0.7,
                "face_visibility_score": 0.7,
                "mapping_confidence": 0.6,
            },
            {
                "visual_identity_id": "Global_Person_1",
                "mouth_motion_score": 0.69,
                "pose_visibility_score": 0.7,
                "face_visibility_score": 0.7,
                "mapping_confidence": 0.6,
            },
        ],
        winning_margin=0.03,
    )

    assert result["winner_visual_identity_id"] is None
    assert result["unresolved"] is True
    assert result["reason"] == "ambiguous_visual_signals"


def test_choose_visual_speaking_candidate_breaks_ties_deterministically() -> None:
    result = choose_visual_speaking_candidate(
        [
            {
                "visual_identity_id": "Global_Person_2",
                "mouth_motion_score": 0.5,
                "pose_visibility_score": 0.5,
                "face_visibility_score": 0.5,
                "mapping_confidence": 0.5,
            },
            {
                "visual_identity_id": "Global_Person_1",
                "mouth_motion_score": 0.5,
                "pose_visibility_score": 0.5,
                "face_visibility_score": 0.5,
                "mapping_confidence": 0.5,
            },
        ],
        winning_margin=0.0,
    )

    assert result["winner_visual_identity_id"] == "Global_Person_1"
    assert [item["visual_identity_id"] for item in result["ranked_candidates"]] == [
        "Global_Person_1",
        "Global_Person_2",
    ]
