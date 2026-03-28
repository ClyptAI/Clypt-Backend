import pytest

from backend.speaker_binding.scheduler import schedule_diarized_spans


def test_schedule_diarized_spans_merges_tiny_same_speaker_gaps():
    spans = schedule_diarized_spans(
        [
            {"speaker_id": "SPEAKER_00", "start_time_ms": 0, "end_time_ms": 100},
            {"speaker_id": "SPEAKER_00", "start_time_ms": 108, "end_time_ms": 220},
        ],
        same_speaker_gap_ms=10,
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
            "end_time_ms": 220,
            "context_start_time_ms": 0,
            "context_end_time_ms": 220,
            "source_turn_ids": ["turn-0", "turn-1"],
        }
    ]


def test_schedule_diarized_spans_pads_boundaries_without_losing_context():
    spans = schedule_diarized_spans(
        [
            {"speaker_id": "SPEAKER_01", "start_time_ms": 100, "end_time_ms": 200},
        ],
        same_speaker_gap_ms=0,
        boundary_pad_ms=25,
    )

    assert spans == [
        {
            "span_id": "scheduled-0",
            "span_type": "single",
            "speaker_ids": ["SPEAKER_01"],
            "exclusive": True,
            "overlap": False,
            "start_time_ms": 75,
            "end_time_ms": 225,
            "context_start_time_ms": 100,
            "context_end_time_ms": 200,
            "source_turn_ids": ["turn-0"],
        }
    ]


def test_schedule_diarized_spans_emits_single_and_overlap_windows():
    spans = schedule_diarized_spans(
        [
            {"speaker_id": "SPEAKER_00", "start_time_ms": 0, "end_time_ms": 100},
            {"speaker_id": "SPEAKER_01", "start_time_ms": 50, "end_time_ms": 150},
        ],
        same_speaker_gap_ms=0,
        boundary_pad_ms=0,
    )

    assert [
        (span["span_type"], span["speaker_ids"], span["context_start_time_ms"], span["context_end_time_ms"])
        for span in spans
    ] == [
        ("single", ["SPEAKER_00"], 0, 50),
        ("overlap", ["SPEAKER_00", "SPEAKER_01"], 50, 100),
        ("single", ["SPEAKER_01"], 100, 150),
    ]
    assert spans[0]["exclusive"] is True
    assert spans[1]["exclusive"] is False
    assert spans[1]["overlap"] is True
    assert spans[2]["exclusive"] is True



def test_schedule_diarized_spans_preserves_adjacent_overlap_windows():
    spans = schedule_diarized_spans(
        [
            {"speaker_id": "SPEAKER_00", "start_time_ms": 0, "end_time_ms": 120},
            {"speaker_id": "SPEAKER_01", "start_time_ms": 20, "end_time_ms": 60},
            {"speaker_id": "SPEAKER_02", "start_time_ms": 60, "end_time_ms": 100},
        ],
        same_speaker_gap_ms=0,
        boundary_pad_ms=0,
    )

    overlap_spans = [span for span in spans if span["span_type"] == "overlap"]

    assert overlap_spans == [
        {
            "span_id": "scheduled-1",
            "span_type": "overlap",
            "speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
            "exclusive": False,
            "overlap": True,
            "start_time_ms": 20,
            "end_time_ms": 60,
            "context_start_time_ms": 20,
            "context_end_time_ms": 60,
            "source_turn_ids": ["turn-0", "turn-1"],
        },
        {
            "span_id": "scheduled-2",
            "span_type": "overlap",
            "speaker_ids": ["SPEAKER_00", "SPEAKER_02"],
            "exclusive": False,
            "overlap": True,
            "start_time_ms": 60,
            "end_time_ms": 100,
            "context_start_time_ms": 60,
            "context_end_time_ms": 100,
            "source_turn_ids": ["turn-0", "turn-2"],
        },
    ]
