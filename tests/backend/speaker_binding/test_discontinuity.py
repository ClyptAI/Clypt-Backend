from backend.speaker_binding.discontinuity import classify_visual_discontinuity
from backend.speaker_binding.scheduler import schedule_diarized_spans


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


def test_schedule_diarized_spans_threads_discontinuity_metadata_for_single_speaker_spans():
    spans = schedule_diarized_spans(
        [
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 400,
                "visual_samples": [
                    {
                        "time_ms": 0,
                        "local_track_ids": ["track-a", "track-b"],
                        "prominent_track_id": "track-a",
                    },
                    {
                        "time_ms": 200,
                        "local_track_ids": ["track-c", "track-d"],
                        "prominent_track_id": "track-c",
                    },
                ],
            },
        ],
        same_speaker_gap_ms=0,
        boundary_pad_ms=0,
    )

    assert spans == [
        {
            "span_id": "scheduled-0",
            "span_type": "single",
            "speaker_ids": ["SPEAKER_00"],
            "exclusive": True,
            "overlap": False,
            "start_time_ms": 0,
            "end_time_ms": 400,
            "context_start_time_ms": 0,
            "context_end_time_ms": 400,
            "source_turn_ids": ["turn-0"],
            "requires_lrasd": True,
            "discontinuity_reasons": [
                "track_set_jaccard_drop",
                "prominent_track_flip",
            ],
        }
    ]


def test_schedule_diarized_spans_skips_discontinuity_metadata_for_overlap_spans():
    spans = schedule_diarized_spans(
        [
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 400,
                "visual_samples": [
                    {
                        "time_ms": 0,
                        "local_track_ids": ["track-a", "track-b"],
                        "prominent_track_id": "track-a",
                    },
                    {
                        "time_ms": 200,
                        "local_track_ids": ["track-c", "track-d"],
                        "prominent_track_id": "track-c",
                    },
                ],
            },
            {
                "speaker_id": "SPEAKER_01",
                "start_time_ms": 0,
                "end_time_ms": 400,
            },
        ],
        same_speaker_gap_ms=0,
        boundary_pad_ms=0,
    )

    assert spans == [
        {
            "span_id": "scheduled-0",
            "span_type": "overlap",
            "speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
            "exclusive": False,
            "overlap": True,
            "start_time_ms": 0,
            "end_time_ms": 400,
            "context_start_time_ms": 0,
            "context_end_time_ms": 400,
            "source_turn_ids": ["turn-0", "turn-1"],
        }
    ]
