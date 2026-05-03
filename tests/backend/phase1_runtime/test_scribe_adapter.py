from __future__ import annotations

import pytest

from backend.phase1_runtime.scribe_adapter import (
    ScribeAdapterError,
    adapt_scribe_response,
)


def test_adapt_scribe_response_maps_words_turns_events_and_empty_emotion_payload() -> None:
    result = adapt_scribe_response(
        {
            "text": "Hello world. Again",
            "words": [
                {
                    "type": "word",
                    "text": "Hello",
                    "start": 0.0,
                    "end": 0.3,
                    "speaker_id": "speaker_0",
                    "logprob": -0.1,
                },
                {
                    "type": "word",
                    "text": "world",
                    "start": 0.35,
                    "end": 0.7,
                    "speaker_id": "speaker_0",
                },
                {
                    "type": "audio_event",
                    "text": "(laughter)",
                    "start": 0.72,
                    "end": 1.0,
                    "confidence": 0.82,
                    "future": "kept",
                },
                {
                    "type": "word",
                    "text": "Again",
                    "start": 2.2,
                    "end": 2.6,
                    "speaker_id": "speaker_1",
                },
            ],
        }
    )

    assert [word["word_id"] for word in result.diarization_payload.words] == [
        "w_000001",
        "w_000002",
        "w_000003",
    ]
    assert result.diarization_payload.words[0]["speaker_id"] == "SPEAKER_0"
    assert result.diarization_payload.words[2]["speaker_id"] == "SPEAKER_1"
    assert result.diarization_payload.words[0]["scribe"]["logprob"] == -0.1
    assert [turn["turn_id"] for turn in result.diarization_payload.turns] == [
        "t_000001",
        "t_000002",
    ]
    assert result.diarization_payload.turns[0]["transcript_text"] == "Hello world"
    assert result.diarization_payload.turns[0]["word_ids"] == ["w_000001", "w_000002"]
    assert result.diarization_payload.turns[1]["speaker_id"] == "SPEAKER_1"
    assert result.yamnet_payload.events == [
        {
            "event_label": "laughter",
            "start_ms": 720,
            "end_ms": 1000,
            "confidence": 0.82,
            "source": "scribe_v2",
            "scribe": {
                "type": "audio_event",
                "text": "(laughter)",
                "start": 0.72,
                "end": 1.0,
                "confidence": 0.82,
                "future": "kept",
            },
        }
    ]
    assert result.emotion2vec_payload.segments == []


def test_adapt_scribe_response_splits_turns_on_gap_even_with_same_speaker() -> None:
    result = adapt_scribe_response(
        {
            "words": [
                {
                    "type": "word",
                    "text": "First",
                    "start": 0.0,
                    "end": 0.2,
                    "speaker_id": "speaker_0",
                },
                {
                    "type": "word",
                    "text": "second",
                    "start": 2.0,
                    "end": 2.2,
                    "speaker_id": "speaker_0",
                },
            ]
        },
        turn_gap_ms=1200,
    )

    assert len(result.diarization_payload.turns) == 2
    assert result.diarization_payload.turns[0]["word_ids"] == ["w_000001"]
    assert result.diarization_payload.turns[1]["word_ids"] == ["w_000002"]


def test_adapt_scribe_response_sanitizes_unknown_speaker_labels_deterministically() -> None:
    result = adapt_scribe_response(
        {
            "words": [
                {
                    "type": "word",
                    "text": "Hi",
                    "start": 0.0,
                    "end": 0.2,
                    "speaker_id": "host alpha",
                },
                {
                    "type": "word",
                    "text": "there",
                    "start": 0.25,
                    "end": 0.5,
                    "speaker_id": "host alpha",
                },
            ]
        }
    )

    assert {word["speaker_id"] for word in result.diarization_payload.words} == {
        "SPEAKER_HOST_ALPHA"
    }


def test_adapt_scribe_response_omits_ambiguous_untimed_tags() -> None:
    result = adapt_scribe_response(
        {
            "words": [
                {"type": "audio_event", "text": "(music)"},
                {
                    "type": "word",
                    "text": "Hello",
                    "start": 0.0,
                    "end": 0.2,
                    "speaker_id": "speaker_0",
                },
            ]
        }
    )

    assert result.yamnet_payload.events == []


def test_adapt_scribe_response_repairs_zero_duration_word_tokens() -> None:
    result = adapt_scribe_response(
        {
            "words": [
                {
                    "type": "word",
                    "text": "Wait",
                    "start": 5.49,
                    "end": 5.49,
                    "speaker_id": "speaker_0",
                },
                {
                    "type": "word",
                    "text": "what",
                    "start": 5.49,
                    "end": 5.49,
                    "speaker_id": "speaker_0",
                },
                {
                    "type": "word",
                    "text": "happened",
                    "start": 5.5,
                    "end": 5.8,
                    "speaker_id": "speaker_0",
                },
            ]
        }
    )

    words = result.diarization_payload.words
    assert words[0]["start_ms"] == 5490
    assert words[0]["end_ms"] == 5491
    assert words[0]["scribe_timing_repaired"] is True
    assert words[1]["start_ms"] == 5491
    assert words[1]["end_ms"] == 5492
    assert words[1]["scribe_timing_repaired"] is True
    assert words[2]["start_ms"] == 5500
    assert words[2]["end_ms"] == 5800
    assert "scribe_timing_repaired" not in words[2]
    assert result.diarization_payload.turns[0]["transcript_text"] == "Wait what happened"


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({}, "words list"),
        ({"words": [{"type": "word", "text": "x", "end": 0.2, "speaker_id": "speaker_0"}]}, "start"),
        ({"words": [{"type": "word", "text": "x", "start": 0.0, "end": 0.2}]}, "speaker_id"),
        ({"words": [{"type": "audio_event", "text": "(music)", "start": 0.0, "end": 0.2}]}, "no timed word"),
    ],
)
def test_adapt_scribe_response_fails_on_caption_critical_missing_word_data(
    raw: dict,
    match: str,
) -> None:
    with pytest.raises(ScribeAdapterError, match=match):
        adapt_scribe_response(raw)
