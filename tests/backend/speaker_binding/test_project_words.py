from backend.speaker_binding.project_words import (
    build_speaker_assignment_spans,
    project_span_assignments_to_words,
)


def test_project_span_assignments_maps_solo_visible_speaker_to_all_covered_words():
    words = [
        {"text": "hello", "start_time_ms": 0, "end_time_ms": 100},
        {"text": "world", "start_time_ms": 120, "end_time_ms": 240},
    ]

    assignments = project_span_assignments_to_words(
        words=words,
        speaker_assignment_spans_local=[
            {
                "start_time_ms": 0,
                "end_time_ms": 300,
                "audio_speaker_ids": ["SPEAKER_00"],
                "visible_local_track_ids": ["local-1"],
                "visible_track_ids": ["Global_Person_0"],
                "offscreen_audio_speaker_ids": [],
                "overlap": False,
                "confidence": 0.91,
                "decision_source": "easy_span",
            }
        ],
    )

    assert assignments == [
        {
            "start_time_ms": 0,
            "end_time_ms": 100,
            "audio_speaker_ids": ["SPEAKER_00"],
            "visible_local_track_ids": ["local-1"],
            "visible_track_ids": ["Global_Person_0"],
            "offscreen_audio_speaker_ids": [],
            "dominant_visible_local_track_id": "local-1",
            "dominant_visible_track_id": "Global_Person_0",
            "decision_source": "easy_span",
            "overlap": False,
        },
        {
            "start_time_ms": 120,
            "end_time_ms": 240,
            "audio_speaker_ids": ["SPEAKER_00"],
            "visible_local_track_ids": ["local-1"],
            "visible_track_ids": ["Global_Person_0"],
            "offscreen_audio_speaker_ids": [],
            "dominant_visible_local_track_id": "local-1",
            "dominant_visible_track_id": "Global_Person_0",
            "decision_source": "easy_span",
            "overlap": False,
        },
    ]
    assert words[0]["speaker_local_track_id"] == "local-1"
    assert words[0]["speaker_track_id"] == "Global_Person_0"
    assert words[1]["speaker_local_track_id"] == "local-1"
    assert words[1]["speaker_track_id"] == "Global_Person_0"


def test_project_span_assignments_preserves_multi_visible_overlap_without_single_legacy_track():
    words = [
        {"text": "shared", "start_time_ms": 500, "end_time_ms": 700},
    ]

    assignments = project_span_assignments_to_words(
        words=words,
        speaker_assignment_spans_local=[
            {
                "start_time_ms": 400,
                "end_time_ms": 900,
                "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
                "visible_local_track_ids": ["local-1", "local-2"],
                "visible_track_ids": ["Global_Person_0", "Global_Person_1"],
                "offscreen_audio_speaker_ids": [],
                "overlap": True,
                "confidence": 0.74,
                "decision_source": "lrasd_span",
            }
        ],
    )

    assert assignments == [
        {
            "start_time_ms": 500,
            "end_time_ms": 700,
            "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
            "visible_local_track_ids": ["local-1", "local-2"],
            "visible_track_ids": ["Global_Person_0", "Global_Person_1"],
            "offscreen_audio_speaker_ids": [],
            "dominant_visible_local_track_id": None,
            "dominant_visible_track_id": None,
            "decision_source": "lrasd_span",
            "overlap": True,
        }
    ]
    assert words[0]["speaker_local_track_id"] is None
    assert words[0]["speaker_track_id"] is None


def test_project_span_assignments_surfaces_offscreen_overlap_without_inventing_visible_box():
    words = [
        {"text": "reply", "start_time_ms": 300, "end_time_ms": 500},
    ]

    assignments = project_span_assignments_to_words(
        words=words,
        speaker_assignment_spans_local=[
            {
                "start_time_ms": 200,
                "end_time_ms": 600,
                "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
                "visible_local_track_ids": ["local-1"],
                "visible_track_ids": ["Global_Person_0"],
                "offscreen_audio_speaker_ids": ["SPEAKER_01"],
                "overlap": True,
                "confidence": 0.68,
                "decision_source": "turn_binding",
            }
        ],
    )

    assert assignments == [
        {
            "start_time_ms": 300,
            "end_time_ms": 500,
            "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
            "visible_local_track_ids": ["local-1"],
            "visible_track_ids": ["Global_Person_0"],
            "offscreen_audio_speaker_ids": ["SPEAKER_01"],
            "dominant_visible_local_track_id": "local-1",
            "dominant_visible_track_id": "Global_Person_0",
            "decision_source": "turn_binding",
            "overlap": True,
        }
    ]
    assert words[0]["speaker_local_track_id"] == "local-1"
    assert words[0]["speaker_track_id"] == "Global_Person_0"


def test_build_speaker_assignment_spans_projects_global_visibility_without_dropping_offscreen():
    local_spans, global_spans = build_speaker_assignment_spans(
        active_speakers_local=[
            {
                "start_time_ms": 100,
                "end_time_ms": 300,
                "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
                "visible_local_track_ids": ["local-1"],
                "visible_track_ids": [],
                "offscreen_audio_speaker_ids": ["SPEAKER_01"],
                "overlap": True,
                "confidence": 0.63,
                "decision_source": "audio_only",
            }
        ],
        local_to_global_track_id={"local-1": "Global_Person_0"},
    )

    assert local_spans[0]["visible_local_track_ids"] == ["local-1"]
    assert global_spans == [
        {
            "start_time_ms": 100,
            "end_time_ms": 300,
            "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
            "visible_local_track_ids": ["local-1"],
            "visible_track_ids": ["Global_Person_0"],
            "offscreen_audio_speaker_ids": ["SPEAKER_01"],
            "overlap": True,
            "confidence": 0.63,
            "decision_source": "audio_only",
        }
    ]


def test_project_span_assignments_keeps_dominant_handoff_side_without_false_overlap_union():
    words = [
        {
            "text": "handoff",
            "start_time_ms": 90,
            "end_time_ms": 140,
            "speaker_track_id": "stale-global",
            "speaker_tag": "stale-global",
            "speaker_local_track_id": "stale-local",
            "speaker_local_tag": "stale-local",
        }
    ]

    assignments = project_span_assignments_to_words(
        words=words,
        speaker_assignment_spans_local=[
            {
                "start_time_ms": 0,
                "end_time_ms": 120,
                "audio_speaker_ids": ["SPEAKER_00"],
                "visible_local_track_ids": ["local-1"],
                "visible_track_ids": ["Global_Person_0"],
                "offscreen_audio_speaker_ids": [],
                "overlap": False,
                "confidence": 0.91,
                "decision_source": "turn_binding",
            },
            {
                "start_time_ms": 120,
                "end_time_ms": 220,
                "audio_speaker_ids": ["SPEAKER_01"],
                "visible_local_track_ids": ["local-2"],
                "visible_track_ids": ["Global_Person_1"],
                "offscreen_audio_speaker_ids": [],
                "overlap": False,
                "confidence": 0.89,
                "decision_source": "turn_binding",
            },
        ],
    )

    assert assignments == [
        {
            "start_time_ms": 90,
            "end_time_ms": 140,
            "audio_speaker_ids": ["SPEAKER_00"],
            "visible_local_track_ids": ["local-1"],
            "visible_track_ids": ["Global_Person_0"],
            "offscreen_audio_speaker_ids": [],
            "dominant_visible_local_track_id": "local-1",
            "dominant_visible_track_id": "Global_Person_0",
            "decision_source": "turn_binding",
            "overlap": False,
        }
    ]
    assert words[0]["speaker_local_track_id"] == "local-1"
    assert words[0]["speaker_track_id"] == "Global_Person_0"


def test_project_span_assignments_keeps_existing_global_assignment_when_single_local_has_no_projected_global():
    words = [
        {
            "text": "solo",
            "start_time_ms": 0,
            "end_time_ms": 100,
            "speaker_track_id": "Global_Person_0",
            "speaker_tag": "Global_Person_0",
            "speaker_local_track_id": None,
            "speaker_local_tag": "unknown",
        }
    ]

    assignments = project_span_assignments_to_words(
        words=words,
        speaker_assignment_spans_local=[
            {
                "start_time_ms": 0,
                "end_time_ms": 100,
                "audio_speaker_ids": ["SPEAKER_00"],
                "visible_local_track_ids": ["local-1"],
                "visible_track_ids": [],
                "offscreen_audio_speaker_ids": [],
                "overlap": False,
                "confidence": 0.93,
                "decision_source": "turn_binding",
            }
        ],
    )

    assert assignments == [
        {
            "start_time_ms": 0,
            "end_time_ms": 100,
            "audio_speaker_ids": ["SPEAKER_00"],
            "visible_local_track_ids": ["local-1"],
            "visible_track_ids": ["Global_Person_0"],
            "offscreen_audio_speaker_ids": [],
            "dominant_visible_local_track_id": "local-1",
            "dominant_visible_track_id": "Global_Person_0",
            "decision_source": "turn_binding",
            "overlap": False,
        }
    ]
    assert words[0]["speaker_local_track_id"] == "local-1"
    assert words[0]["speaker_track_id"] == "Global_Person_0"
