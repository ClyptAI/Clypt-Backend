from backend.speaker_binding.easy_span_cascade import build_easy_span_binding_rows, classify_easy_span


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


def test_classify_easy_span_rejects_winner_without_local_track_id():
    decision = classify_easy_span(
        {
            "span_id": "scheduled-4",
            "span_type": "single",
            "speaker_ids": ["SPEAKER_00"],
            "overlap": False,
        },
        [
            {
                "local_track_id": "",
                "candidate_survives": True,
                "speech_overlap_ratio": 0.98,
                "continuity_support_score": 0.92,
                "rank_score": 0.95,
            },
            {
                "local_track_id": "local-2",
                "candidate_survives": True,
                "speech_overlap_ratio": 0.15,
                "continuity_support_score": 0.40,
                "rank_score": 0.21,
            },
        ],
    )

    assert decision["decision"] == "hard"
    assert decision["reason"] == "missing_local_track"


def test_build_easy_span_binding_rows_uses_source_turn_time_ranges():
    rows = build_easy_span_binding_rows(
        {
            "span_id": "scheduled-0",
            "speaker_ids": ["SPEAKER_00"],
            "context_start_time_ms": 0,
            "context_end_time_ms": 200,
            "source_turn_ids": ["turn-0", "turn-1"],
        },
        {
            "decision": "easy",
            "speaker_id": "SPEAKER_00",
            "local_track_id": "local-1",
            "winning_score": 0.92,
            "winning_margin": 0.81,
            "support_ratio": 0.95,
            "max_visible_candidates": 1,
        },
        global_track_id="Global_Person_0",
        source_turn_ranges={
            "turn-0": {"start_time_ms": 0, "end_time_ms": 100},
            "turn-1": {"start_time_ms": 105, "end_time_ms": 200},
        },
    )

    assert rows == [
        {
            "speaker_id": "SPEAKER_00",
            "source_turn_id": "turn-0",
            "span_id": "scheduled-0",
            "start_time_ms": 0,
            "end_time_ms": 100,
            "local_track_id": "local-1",
            "ambiguous": False,
            "decision_source": "easy_span",
            "winning_score": 0.92,
            "winning_margin": 0.81,
            "support_ratio": 0.95,
            "max_visible_candidates": 1,
            "track_id": "Global_Person_0",
        },
        {
            "speaker_id": "SPEAKER_00",
            "source_turn_id": "turn-1",
            "span_id": "scheduled-0",
            "start_time_ms": 105,
            "end_time_ms": 200,
            "local_track_id": "local-1",
            "ambiguous": False,
            "decision_source": "easy_span",
            "winning_score": 0.92,
            "winning_margin": 0.81,
            "support_ratio": 0.95,
            "max_visible_candidates": 1,
            "track_id": "Global_Person_0",
        },
    ]
