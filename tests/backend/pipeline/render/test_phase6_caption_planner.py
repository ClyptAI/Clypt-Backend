from __future__ import annotations

import pytest

from backend.pipeline.contracts import CanonicalTimeline, CanonicalTurn, ClipCandidate, TranscriptWord


def _timeline() -> CanonicalTimeline:
    return CanonicalTimeline(
        words=[
            TranscriptWord(word_id="w1", text="This", start_ms=1000, end_ms=1120, speaker_id="SPEAKER_0"),
            TranscriptWord(word_id="w2", text="is", start_ms=1120, end_ms=1210, speaker_id="SPEAKER_0"),
            TranscriptWord(word_id="w3", text="the", start_ms=1210, end_ms=1310, speaker_id="SPEAKER_0"),
            TranscriptWord(word_id="w4", text="hook.", start_ms=1310, end_ms=1490, speaker_id="SPEAKER_0"),
            TranscriptWord(word_id="w5", text="Wait", start_ms=1900, end_ms=2030, speaker_id="SPEAKER_1"),
            TranscriptWord(word_id="w6", text="for", start_ms=2030, end_ms=2140, speaker_id="SPEAKER_1"),
            TranscriptWord(word_id="w7", text="it", start_ms=2140, end_ms=2260, speaker_id="SPEAKER_1"),
        ],
        turns=[
            CanonicalTurn(
                turn_id="t1",
                speaker_id="SPEAKER_0",
                start_ms=1000,
                end_ms=1490,
                word_ids=["w1", "w2", "w3", "w4"],
                transcript_text="This is the hook.",
            ),
            CanonicalTurn(
                turn_id="t2",
                speaker_id="SPEAKER_1",
                start_ms=1900,
                end_ms=2260,
                word_ids=["w5", "w6", "w7"],
                transcript_text="Wait for it",
            ),
        ],
    )


def test_plan_caption_artifacts_splits_on_speaker_boundaries_and_uses_clip_local_word_timings() -> None:
    from backend.pipeline.render.phase6 import plan_caption_artifacts

    caption_plan = plan_caption_artifacts(
        run_id="run_phase6",
        canonical_timeline=_timeline(),
        candidates=[
            ClipCandidate(
                clip_id="clip_001",
                node_ids=["node_1"],
                start_ms=1000,
                end_ms=2300,
                score=8.7,
                rationale="Strong short hook.",
            )
        ],
        preset_id="karaoke_focus",
    )

    clip = caption_plan.clips[0]
    assert clip.clip_id == "clip_001"
    assert clip.clip_start_ms == 1000
    assert clip.clip_end_ms == 2300
    assert clip.preset_id == "karaoke_focus"
    assert [segment.text for segment in clip.segments] == ["This is the hook.", "Wait for it"]
    assert clip.segments[0].speaker_ids == ["SPEAKER_0"]
    assert clip.segments[1].speaker_ids == ["SPEAKER_1"]
    assert clip.segments[0].word_ids == ["w1", "w2", "w3", "w4"]
    assert clip.segments[0].start_ms == 0
    assert clip.segments[1].start_ms >= 850
    assert clip.segments[0].highlight_mode == "word_highlight"
    assert [item.word_id for item in clip.segments[0].active_word_timings] == ["w1", "w2", "w3", "w4"]
    assert clip.segments[0].active_word_timings[0].start_ms == 0
    assert clip.segments[0].active_word_timings[-1].end_ms <= clip.segments[0].end_ms


def test_plan_caption_artifacts_clips_segment_and_word_timings_to_clip_boundaries() -> None:
    from backend.pipeline.render.phase6 import plan_caption_artifacts

    caption_plan = plan_caption_artifacts(
        run_id="run_phase6",
        canonical_timeline=_timeline(),
        candidates=[
            ClipCandidate(
                clip_id="clip_boundary",
                node_ids=["node_1"],
                start_ms=1050,
                end_ms=1400,
                score=8.4,
                rationale="Boundary-clipped hook.",
            )
        ],
        preset_id="karaoke_focus",
    )

    segment = caption_plan.clips[0].segments[0]
    assert segment.text == "This is the hook."
    assert segment.start_ms == 0
    assert segment.end_ms == 350
    assert [timing.model_dump(mode="json") for timing in segment.active_word_timings] == [
        {"word_id": "w1", "start_ms": 0, "end_ms": 70, "text": "This"},
        {"word_id": "w2", "start_ms": 70, "end_ms": 160, "text": "is"},
        {"word_id": "w3", "start_ms": 160, "end_ms": 260, "text": "the"},
        {"word_id": "w4", "start_ms": 260, "end_ms": 350, "text": "hook."},
    ]


def test_plan_caption_artifacts_fails_when_turn_word_ids_are_missing_from_canonical_words() -> None:
    from backend.pipeline.render.phase6 import plan_caption_artifacts

    broken_timeline = CanonicalTimeline(
        words=[
            TranscriptWord(word_id="w1", text="This", start_ms=1000, end_ms=1120, speaker_id="SPEAKER_0"),
        ],
        turns=[
            CanonicalTurn(
                turn_id="t1",
                speaker_id="SPEAKER_0",
                start_ms=1000,
                end_ms=1490,
                word_ids=["w1", "w_missing"],
                transcript_text="This hook",
            )
        ],
    )

    with pytest.raises(ValueError, match="missing canonical words"):
        plan_caption_artifacts(
            run_id="run_phase6",
            canonical_timeline=broken_timeline,
            candidates=[
                ClipCandidate(
                    clip_id="clip_missing_word",
                    node_ids=["node_1"],
                    start_ms=1000,
                    end_ms=1490,
                    score=7.9,
                    rationale="Broken canonical payload.",
                )
            ],
            preset_id="karaoke_focus",
        )


def test_plan_caption_artifacts_fails_when_clip_interval_has_no_canonical_words() -> None:
    from backend.pipeline.render.phase6 import plan_caption_artifacts

    with pytest.raises(ValueError, match="cannot be resolved against canonical timeline"):
        plan_caption_artifacts(
            run_id="run_phase6",
            canonical_timeline=_timeline(),
            candidates=[
                ClipCandidate(
                    clip_id="clip_empty",
                    node_ids=["node_1"],
                    start_ms=2301,
                    end_ms=2600,
                    score=7.5,
                    rationale="No transcript coverage.",
                )
            ],
            preset_id="karaoke_focus",
        )
