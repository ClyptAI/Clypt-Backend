from __future__ import annotations

from backend.speaker_binding.audio_visual_mapping import learn_audio_visual_mappings


def test_learn_audio_visual_mappings_aggregates_clean_span_support() -> None:
    mappings = learn_audio_visual_mappings(
        [
            {
                "audio_speaker_id": "speaker-1",
                "visual_identity_id": "visual-2",
                "confidence": 0.9,
                "support_track_ids": ["Global_Person_2"],
                "overlap": False,
                "offscreen": False,
            },
            {
                "audio_speaker_id": "speaker-1",
                "visual_identity_id": "visual-2",
                "confidence": 0.6,
                "support_track_ids": ["Global_Person_2", "Global_Person_2"],
                "overlap": False,
                "offscreen": False,
            },
            {
                "audio_speaker_id": "speaker-1",
                "visual_identity_id": "visual-4",
                "confidence": 0.2,
                "support_track_ids": ["Global_Person_4"],
                "overlap": False,
                "offscreen": False,
            },
        ]
    )

    assert len(mappings) == 1
    mapping = mappings[0]
    assert mapping.audio_speaker_id == "speaker-1"
    assert mapping.matched_visual_identity_id == "visual-2"
    assert mapping.candidate_visual_identity_ids == ("visual-2", "visual-4")
    assert mapping.supporting_track_ids == ("Global_Person_2", "Global_Person_4")
    assert mapping.ambiguous is False
    assert mapping.mapping_strategy == "clean-span-aggregation"
    assert mapping.confidence == 0.882
    assert [edge.visual_identity_id for edge in mapping.evidence_edges] == [
        "visual-2",
        "visual-2",
        "visual-4",
    ]


def test_learn_audio_visual_mappings_ignores_overlap_and_offscreen_teaching_spans() -> None:
    mappings = learn_audio_visual_mappings(
        [
            {
                "audio_speaker_id": "speaker-1",
                "visual_identity_id": "visual-2",
                "confidence": 0.9,
                "support_track_ids": ["Global_Person_2"],
                "overlap": True,
                "offscreen": False,
            },
            {
                "audio_speaker_id": "speaker-1",
                "visual_identity_id": "visual-3",
                "confidence": 0.7,
                "support_track_ids": [],
                "overlap": False,
                "offscreen": True,
            },
        ]
    )

    assert mappings == []


def test_learn_audio_visual_mappings_marks_close_candidates_ambiguous() -> None:
    mappings = learn_audio_visual_mappings(
        [
            {
                "audio_speaker_id": "speaker-3",
                "visual_identity_id": "visual-1",
                "confidence": 0.55,
                "support_track_ids": ["Global_Person_1"],
                "overlap": False,
                "offscreen": False,
            },
            {
                "audio_speaker_id": "speaker-3",
                "visual_identity_id": "visual-2",
                "confidence": 0.5,
                "support_track_ids": ["Global_Person_2"],
                "overlap": False,
                "offscreen": False,
            },
        ]
    )

    assert len(mappings) == 1
    mapping = mappings[0]
    assert mapping.audio_speaker_id == "speaker-3"
    assert mapping.matched_visual_identity_id == "visual-1"
    assert mapping.candidate_visual_identity_ids == ("visual-1", "visual-2")
    assert mapping.ambiguous is True
    assert mapping.confidence == 0.524
