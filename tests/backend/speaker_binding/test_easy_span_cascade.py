from backend.speaker_binding.easy_span_cascade import classify_easy_span


def test_classify_easy_span_accepts_dominant_single_speaker_span():
    decision = classify_easy_span(
        {
            "span_id": "scheduled-0",
            "span_type": "single",
            "speaker_ids": ["SPEAKER_00"],
            "overlap": False,
            "requires_lrasd": False,
        },
        [
            {
                "local_track_id": "local-1",
                "candidate_survives": True,
                "speech_overlap_ratio": 0.94,
                "continuity_support_score": 0.88,
                "rank_score": 0.91,
            },
            {
                "local_track_id": "local-2",
                "candidate_survives": True,
                "speech_overlap_ratio": 0.18,
                "continuity_support_score": 0.44,
                "rank_score": 0.23,
            },
        ],
    )

    assert decision["decision"] == "easy"
    assert decision["local_track_id"] == "local-1"
    assert decision["decision_source"] == "easy_span"


def test_classify_easy_span_rejects_overlap_spans():
    decision = classify_easy_span(
        {
            "span_id": "scheduled-1",
            "span_type": "overlap",
            "speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
            "overlap": True,
        },
        [
            {
                "local_track_id": "local-1",
                "candidate_survives": True,
                "speech_overlap_ratio": 0.98,
                "continuity_support_score": 0.90,
                "rank_score": 0.95,
            }
        ],
    )

    assert decision["decision"] == "hard"
    assert decision["reason"] == "overlap_span"


def test_classify_easy_span_rejects_discontinuous_single_speaker_span():
    decision = classify_easy_span(
        {
            "span_id": "scheduled-2",
            "span_type": "single",
            "speaker_ids": ["SPEAKER_00"],
            "overlap": False,
            "requires_lrasd": True,
            "discontinuity_reasons": ["prominent_track_flip"],
        },
        [
            {
                "local_track_id": "local-1",
                "candidate_survives": True,
                "speech_overlap_ratio": 0.96,
                "continuity_support_score": 0.89,
                "rank_score": 0.93,
            }
        ],
    )

    assert decision["decision"] == "hard"
    assert decision["reason"] == "visual_discontinuity"


def test_classify_easy_span_rejects_weak_visibility_or_handoff_span():
    decision = classify_easy_span(
        {
            "span_id": "scheduled-3",
            "span_type": "single",
            "speaker_ids": ["SPEAKER_00"],
            "overlap": False,
        },
        [
            {
                "local_track_id": "local-1",
                "candidate_survives": True,
                "speech_overlap_ratio": 0.56,
                "continuity_support_score": 0.69,
                "rank_score": 0.64,
            },
            {
                "local_track_id": "local-2",
                "candidate_survives": True,
                "speech_overlap_ratio": 0.46,
                "continuity_support_score": 0.62,
                "rank_score": 0.53,
            },
        ],
    )

    assert decision["decision"] == "hard"
    assert decision["reason"] == "weak_visibility_or_handoff"
