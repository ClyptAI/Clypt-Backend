from backend.speaker_binding.discontinuity import classify_visual_discontinuity


def test_classify_visual_discontinuity_keeps_stable_single_person_span_easy():
    result = classify_visual_discontinuity(
        [
            {
                "time_ms": 0,
                "local_track_ids": ["track-a"],
                "prominent_track_id": "track-a",
            },
            {
                "time_ms": 320,
                "local_track_ids": ["track-a"],
                "prominent_track_id": "track-a",
            },
        ]
    )

    assert result == {
        "requires_lrasd": False,
        "discontinuity_reasons": [],
    }


def test_classify_visual_discontinuity_marks_cut_back_candidate_flip_as_hard():
    result = classify_visual_discontinuity(
        [
            {
                "time_ms": 0,
                "local_track_ids": ["track-a", "track-b"],
                "prominent_track_id": "track-a",
            },
            {
                "time_ms": 320,
                "local_track_ids": ["track-c", "track-d"],
                "prominent_track_id": "track-c",
            },
        ]
    )

    assert result["requires_lrasd"] is True
    assert "track_set_jaccard_drop" in result["discontinuity_reasons"]
    assert "prominent_track_flip" in result["discontinuity_reasons"]
