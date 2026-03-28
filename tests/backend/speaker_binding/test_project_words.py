from __future__ import annotations

from backend.speaker_binding.project_words import project_words


def test_project_words_sets_legacy_single_speaker_fields_when_valid() -> None:
    words = [{"word": "hello", "start_time_ms": 100, "end_time_ms": 200}]
    projected = project_words(
        words=words,
        span_assignments=[
            {
                "start_time_ms": 0,
                "end_time_ms": 500,
                "assigned_visual_identity_ids": ("Global_Person_0",),
                "offscreen_audio_speaker_ids": (),
                "decision_source": "mapping",
                "require_hard_disambiguation": False,
            }
        ],
    )

    assert projected[0]["speaker_track_id"] == "Global_Person_0"
    assert projected[0]["speaker_track_ids"] == ["Global_Person_0"]
    assert projected[0]["offscreen_audio_speaker_ids"] == []


def test_project_words_preserves_multi_speaker_overlap_without_forcing_legacy_single_winner() -> None:
    words = [{"word": "hello", "start_time_ms": 100, "end_time_ms": 200}]
    projected = project_words(
        words=words,
        span_assignments=[
            {
                "start_time_ms": 0,
                "end_time_ms": 500,
                "assigned_visual_identity_ids": ("Global_Person_0", "Global_Person_1"),
                "offscreen_audio_speaker_ids": ("SPEAKER_02",),
                "decision_source": "overlap_mapping",
                "require_hard_disambiguation": False,
            }
        ],
    )

    assert projected[0]["speaker_track_id"] is None
    assert projected[0]["speaker_track_ids"] == ["Global_Person_0", "Global_Person_1"]
    assert projected[0]["offscreen_audio_speaker_ids"] == ["SPEAKER_02"]


def test_project_words_marks_offscreen_only_span_without_visible_assignment() -> None:
    words = [{"word": "hello", "start_time_ms": 100, "end_time_ms": 200}]
    projected = project_words(
        words=words,
        span_assignments=[
            {
                "start_time_ms": 0,
                "end_time_ms": 500,
                "assigned_visual_identity_ids": (),
                "offscreen_audio_speaker_ids": ("SPEAKER_03",),
                "decision_source": "mapping_offscreen",
                "require_hard_disambiguation": False,
            }
        ],
    )

    assert projected[0]["speaker_track_id"] is None
    assert projected[0]["speaker_track_ids"] == []
    assert projected[0]["offscreen_audio_speaker_ids"] == ["SPEAKER_03"]


def test_project_words_does_not_force_legacy_single_winner_when_overlap_has_offscreen_speaker() -> None:
    words = [{"word": "hello", "start_time_ms": 100, "end_time_ms": 200}]
    projected = project_words(
        words=words,
        span_assignments=[
            {
                "start_time_ms": 0,
                "end_time_ms": 500,
                "assigned_visual_identity_ids": ("Global_Person_0",),
                "offscreen_audio_speaker_ids": ("SPEAKER_04",),
                "decision_source": "overlap_mapping",
                "require_hard_disambiguation": False,
            }
        ],
    )

    assert projected[0]["speaker_track_id"] is None
    assert projected[0]["speaker_track_ids"] == ["Global_Person_0"]
    assert projected[0]["offscreen_audio_speaker_ids"] == ["SPEAKER_04"]
