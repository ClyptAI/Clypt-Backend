from __future__ import annotations

import pytest

from backend.pipeline.contracts import (
    AudioEvent,
    AudioEventTimeline,
    CanonicalTimeline,
    CanonicalTurn,
    SpeechEmotionEvent,
    SpeechEmotionTimeline,
    TranscriptWord,
)
from backend.pipeline.semantics.merge_and_classify import merge_and_classify_neighborhood
from backend.pipeline.semantics.responses import SemanticsMergeAndClassifyBatchResponse
from backend.pipeline.semantics.turn_neighborhoods import build_turn_neighborhoods


def _canonical_timeline() -> CanonicalTimeline:
    return CanonicalTimeline(
        words=[
            TranscriptWord(word_id="w_000001", text="I", start_ms=0, end_ms=100, speaker_id="S1"),
            TranscriptWord(word_id="w_000002", text="thought", start_ms=100, end_ms=300, speaker_id="S1"),
            TranscriptWord(word_id="w_000003", text="this", start_ms=300, end_ms=450, speaker_id="S1"),
            TranscriptWord(word_id="w_000004", text="would", start_ms=450, end_ms=600, speaker_id="S1"),
            TranscriptWord(word_id="w_000005", text="fail", start_ms=600, end_ms=800, speaker_id="S1"),
            TranscriptWord(word_id="w_000006", text="yeah", start_ms=900, end_ms=1000, speaker_id="S2"),
            TranscriptWord(word_id="w_000007", text="but", start_ms=1100, end_ms=1250, speaker_id="S1"),
            TranscriptWord(word_id="w_000008", text="it", start_ms=1250, end_ms=1350, speaker_id="S1"),
            TranscriptWord(word_id="w_000009", text="worked", start_ms=1350, end_ms=1600, speaker_id="S1"),
            TranscriptWord(word_id="w_000010", text="wow", start_ms=1700, end_ms=1800, speaker_id="S2"),
        ],
        turns=[
            CanonicalTurn(
                turn_id="t_000001",
                speaker_id="S1",
                start_ms=0,
                end_ms=800,
                word_ids=["w_000001", "w_000002", "w_000003", "w_000004", "w_000005"],
                transcript_text="I thought this would fail",
            ),
            CanonicalTurn(
                turn_id="t_000002",
                speaker_id="S2",
                start_ms=900,
                end_ms=1000,
                word_ids=["w_000006"],
                transcript_text="yeah",
            ),
            CanonicalTurn(
                turn_id="t_000003",
                speaker_id="S1",
                start_ms=1100,
                end_ms=1600,
                word_ids=["w_000007", "w_000008", "w_000009"],
                transcript_text="but it worked",
            ),
            CanonicalTurn(
                turn_id="t_000004",
                speaker_id="S2",
                start_ms=1700,
                end_ms=1800,
                word_ids=["w_000010"],
                transcript_text="wow",
            ),
        ],
    )


def test_build_turn_neighborhoods_creates_overlapping_target_blocks_with_halos():
    neighborhoods = build_turn_neighborhoods(
        canonical_timeline=_canonical_timeline(),
        speech_emotion_timeline=SpeechEmotionTimeline(
            events=[
                SpeechEmotionEvent(
                    turn_id="t_000001",
                    primary_emotion_label="fearful",
                    primary_emotion_score=0.72,
                    per_class_scores={"fearful": 0.72, "neutral": 0.28},
                ),
                SpeechEmotionEvent(
                    turn_id="t_000004",
                    primary_emotion_label="surprised",
                    primary_emotion_score=0.91,
                    per_class_scores={"surprised": 0.91, "happy": 0.09},
                ),
            ]
        ),
        audio_event_timeline=AudioEventTimeline(
            events=[
                AudioEvent(event_label="Laughter", start_ms=850, end_ms=1750, confidence=0.88),
            ]
        ),
        target_turn_count=2,
        halo_turn_count=1,
    )

    assert len(neighborhoods) == 2

    first = neighborhoods[0]
    assert first["target_turn_ids"] == ["t_000001", "t_000002"]
    assert "left_halo_turn_ids" not in first
    assert "right_halo_turn_ids" not in first
    assert [turn["turn_id"] for turn in first["turns"]] == ["t_000001", "t_000002", "t_000003"]
    assert first["turns"][0]["role"] == "target"
    assert first["turns"][2]["role"] == "halo"
    assert "word_ids" not in first["turns"][0]
    assert first["turns"][0]["emotion_labels"] == ["fearful"]
    assert first["turns"][1]["audio_events"] == ["Laughter"]

    second = neighborhoods[1]
    assert second["target_turn_ids"] == ["t_000003", "t_000004"]
    assert "left_halo_turn_ids" not in second
    assert "right_halo_turn_ids" not in second
    assert [turn["turn_id"] for turn in second["turns"]] == ["t_000002", "t_000003", "t_000004"]
    assert second["turns"][2]["emotion_labels"] == ["surprised"]


def test_merge_and_classify_neighborhood_builds_nodes_from_gemini_partition():
    neighborhood = build_turn_neighborhoods(
        canonical_timeline=_canonical_timeline(),
        speech_emotion_timeline=SpeechEmotionTimeline(
            events=[
                SpeechEmotionEvent(
                    turn_id="t_000001",
                    primary_emotion_label="fearful",
                    primary_emotion_score=0.72,
                    per_class_scores={"fearful": 0.72, "neutral": 0.28},
                ),
                SpeechEmotionEvent(
                    turn_id="t_000004",
                    primary_emotion_label="surprised",
                    primary_emotion_score=0.91,
                    per_class_scores={"surprised": 0.91, "happy": 0.09},
                ),
            ]
        ),
        audio_event_timeline=AudioEventTimeline(
            events=[
                AudioEvent(event_label="Laughter", start_ms=850, end_ms=1750, confidence=0.88),
            ]
        ),
        target_turn_count=2,
        halo_turn_count=1,
    )[0]

    llm_response = SemanticsMergeAndClassifyBatchResponse.model_validate({
        "merged_nodes": [
            {
                "source_turn_ids": ["t_000001", "t_000002"],
                "node_type": "setup_payoff",
                "node_flags": ["backchannel_dense", "high_resonance_candidate"],
                "summary": "Expectation of failure gets immediate listener buy-in.",
            }
        ]
    })

    nodes = merge_and_classify_neighborhood(
        neighborhood_payload=neighborhood,
        llm_response=llm_response,
        turn_word_ids_by_turn_id={
            turn.turn_id: list(turn.word_ids)
            for turn in _canonical_timeline().turns
        },
    )

    assert len(nodes) == 1
    node = nodes[0]
    assert node.start_ms == 0
    assert node.end_ms == 1000
    assert node.source_turn_ids == ["t_000001", "t_000002"]
    assert node.word_ids == ["w_000001", "w_000002", "w_000003", "w_000004", "w_000005", "w_000006"]
    assert node.transcript_text == "I thought this would fail yeah"
    assert node.evidence.emotion_labels == ["fearful"]
    assert node.evidence.audio_events == ["Laughter"]
    assert node.node_type == "setup_payoff"
    assert node.node_flags == ["backchannel_dense", "high_resonance_candidate"]


def test_merge_and_classify_neighborhood_rejects_incomplete_target_partition():
    neighborhood = build_turn_neighborhoods(
        canonical_timeline=_canonical_timeline(),
        target_turn_count=2,
        halo_turn_count=1,
    )[0]

    with pytest.raises(ValueError, match="partition"):
        merge_and_classify_neighborhood(
            neighborhood_payload=neighborhood,
            llm_response=SemanticsMergeAndClassifyBatchResponse.model_validate({
                "merged_nodes": [
                    {
                        "source_turn_ids": ["t_000001"],
                        "node_type": "claim",
                        "node_flags": [],
                        "summary": "Incomplete",
                    }
                ]
            }),
        )
