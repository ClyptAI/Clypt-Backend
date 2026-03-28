from backend.speaker_binding.lrasd_runner import build_lrasd_span_jobs


def test_build_lrasd_span_jobs_includes_only_hard_spans_and_surviving_candidates():
    jobs, debug_rows = build_lrasd_span_jobs(
        scheduled_hard_spans=[
            {
                "span_id": "scheduled-1",
                "span_type": "single",
                "speaker_ids": ["SPEAKER_00"],
                "speaker_id": "SPEAKER_00",
                "overlap": False,
                "context_start_time_ms": 500,
                "context_end_time_ms": 1000,
                "source_turn_ids": ["turn-1"],
            }
        ],
        words=[
            {"text": "hard", "start_time_ms": 520, "end_time_ms": 640},
            {"text": "span", "start_time_ms": 660, "end_time_ms": 880},
        ],
        rank_candidates_fn=lambda span: [
            {
                "local_track_id": "survivor",
                "candidate_survives": True,
                "rank_score": 0.9,
            },
            {
                "local_track_id": "rejected",
                "candidate_survives": False,
                "rank_score": 0.8,
            },
        ],
    )

    assert jobs == [
        {
            "job_id": "scheduled-1:subspan-0",
            "span_id": "scheduled-1",
            "span_type": "single",
            "speaker_ids": ["SPEAKER_00"],
            "source_turn_ids": ["turn-1"],
            "overlap": False,
            "context_start_time_ms": 500,
            "context_end_time_ms": 1000,
            "start_time_ms": 520,
            "end_time_ms": 880,
            "selected_local_track_ids": ["survivor"],
            "candidate_rows": [
                {
                    "local_track_id": "survivor",
                    "candidate_survives": True,
                    "rank_score": 0.9,
                }
            ],
            "word_indices": [0, 1],
        }
    ]
    assert debug_rows == [
        {
            "span_id": "scheduled-1",
            "job_count": 1,
            "selected_local_track_ids": ["survivor"],
        }
    ]


def test_build_lrasd_span_jobs_preserves_overlap_multi_speaker_context():
    jobs, _debug_rows = build_lrasd_span_jobs(
        scheduled_hard_spans=[
            {
                "span_id": "scheduled-2",
                "span_type": "overlap",
                "speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
                "overlap": True,
                "context_start_time_ms": 1000,
                "context_end_time_ms": 1600,
                "source_turn_ids": ["turn-2", "turn-3"],
            }
        ],
        words=[
            {"text": "both", "start_time_ms": 1040, "end_time_ms": 1200},
        ],
        rank_candidates_fn=lambda span: [
            {"local_track_id": "left", "candidate_survives": True, "rank_score": 0.81},
            {"local_track_id": "right", "candidate_survives": True, "rank_score": 0.79},
        ],
    )

    assert jobs[0]["span_type"] == "overlap"
    assert jobs[0]["speaker_ids"] == ["SPEAKER_00", "SPEAKER_01"]
    assert jobs[0]["overlap"] is True
    assert jobs[0]["selected_local_track_ids"] == ["left", "right"]
    assert jobs[0]["context_start_time_ms"] == 1000
    assert jobs[0]["context_end_time_ms"] == 1600


def test_build_lrasd_span_jobs_reuses_contiguous_words_within_same_span():
    jobs, _debug_rows = build_lrasd_span_jobs(
        scheduled_hard_spans=[
            {
                "span_id": "scheduled-3",
                "span_type": "single",
                "speaker_ids": ["SPEAKER_02"],
                "speaker_id": "SPEAKER_02",
                "overlap": False,
                "context_start_time_ms": 0,
                "context_end_time_ms": 1200,
                "source_turn_ids": ["turn-4"],
            }
        ],
        words=[
            {"text": "one", "start_time_ms": 0, "end_time_ms": 120},
            {"text": "two", "start_time_ms": 200, "end_time_ms": 320},
            {"text": "three", "start_time_ms": 900, "end_time_ms": 1100},
        ],
        rank_candidates_fn=lambda span: [
            {"local_track_id": "speaker", "candidate_survives": True, "rank_score": 0.88},
        ],
    )

    assert [job["job_id"] for job in jobs] == [
        "scheduled-3:subspan-0",
        "scheduled-3:subspan-1",
    ]
    assert jobs[0]["start_time_ms"] == 0
    assert jobs[0]["end_time_ms"] == 320
    assert jobs[0]["word_indices"] == [0, 1]
    assert jobs[1]["start_time_ms"] == 900
    assert jobs[1]["end_time_ms"] == 1100
    assert jobs[1]["word_indices"] == [2]
