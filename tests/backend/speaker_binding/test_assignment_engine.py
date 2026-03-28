from __future__ import annotations

from backend.speaker_binding.assignment_engine import resolve_span_assignments


def test_resolve_span_assignments_uses_mapping_for_clean_visible_single_speaker_span() -> None:
    assignments = resolve_span_assignments(
        spans=[
            {
                "start_time_ms": 0,
                "end_time_ms": 800,
                "audio_speaker_ids": ["speaker-0"],
                "visible_track_ids": ["Global_Person_0"],
                "offscreen_audio_speaker_ids": [],
                "overlap": False,
            }
        ],
        mapping_summaries=[
            {
                "audio_speaker_id": "speaker-0",
                "matched_visual_identity_id": "Global_Person_0",
                "confidence": 0.92,
                "candidate_visual_identity_ids": ["Global_Person_0"],
                "ambiguous": False,
            }
        ],
    )

    assert assignments == [
        {
            "start_time_ms": 0,
            "end_time_ms": 800,
            "audio_speaker_ids": ("speaker-0",),
            "assigned_visual_identity_ids": ("Global_Person_0",),
            "dominant_visual_identity_id": "Global_Person_0",
            "offscreen_audio_speaker_ids": (),
            "unresolved_audio_speaker_ids": (),
            "require_hard_disambiguation": False,
            "decision_source": "mapping",
        }
    ]


def test_resolve_span_assignments_preserves_offscreen_when_mapped_identity_is_not_visible() -> None:
    assignments = resolve_span_assignments(
        spans=[
            {
                "start_time_ms": 100,
                "end_time_ms": 500,
                "audio_speaker_ids": ["speaker-1"],
                "visible_track_ids": ["Global_Person_8"],
                "offscreen_audio_speaker_ids": [],
                "overlap": False,
            }
        ],
        mapping_summaries=[
            {
                "audio_speaker_id": "speaker-1",
                "matched_visual_identity_id": "Global_Person_2",
                "confidence": 0.88,
                "candidate_visual_identity_ids": ["Global_Person_2"],
                "ambiguous": False,
            }
        ],
    )

    assert assignments == [
        {
            "start_time_ms": 100,
            "end_time_ms": 500,
            "audio_speaker_ids": ("speaker-1",),
            "assigned_visual_identity_ids": (),
            "dominant_visual_identity_id": None,
            "offscreen_audio_speaker_ids": ("speaker-1",),
            "unresolved_audio_speaker_ids": (),
            "require_hard_disambiguation": False,
            "decision_source": "mapping_offscreen",
        }
    ]


def test_resolve_span_assignments_routes_overlap_or_ambiguous_cases_to_hard_disambiguation() -> None:
    assignments = resolve_span_assignments(
        spans=[
            {
                "start_time_ms": 0,
                "end_time_ms": 600,
                "audio_speaker_ids": ["speaker-2", "speaker-3"],
                "visible_track_ids": ["Global_Person_2", "Global_Person_3"],
                "offscreen_audio_speaker_ids": [],
                "overlap": True,
            },
            {
                "start_time_ms": 700,
                "end_time_ms": 1200,
                "audio_speaker_ids": ["speaker-4"],
                "visible_track_ids": ["Global_Person_4", "Global_Person_5"],
                "offscreen_audio_speaker_ids": [],
                "overlap": False,
            },
        ],
        mapping_summaries=[
            {
                "audio_speaker_id": "speaker-4",
                "matched_visual_identity_id": "Global_Person_4",
                "confidence": 0.51,
                "candidate_visual_identity_ids": ["Global_Person_4", "Global_Person_5"],
                "ambiguous": True,
            }
        ],
    )

    assert assignments == [
        {
            "start_time_ms": 0,
            "end_time_ms": 600,
            "audio_speaker_ids": ("speaker-2", "speaker-3"),
            "assigned_visual_identity_ids": (),
            "dominant_visual_identity_id": None,
            "offscreen_audio_speaker_ids": (),
            "unresolved_audio_speaker_ids": ("speaker-2", "speaker-3"),
            "require_hard_disambiguation": True,
            "decision_source": "overlap",
        },
        {
            "start_time_ms": 700,
            "end_time_ms": 1200,
            "audio_speaker_ids": ("speaker-4",),
            "assigned_visual_identity_ids": (),
            "dominant_visual_identity_id": None,
            "offscreen_audio_speaker_ids": (),
            "unresolved_audio_speaker_ids": ("speaker-4",),
            "require_hard_disambiguation": True,
            "decision_source": "ambiguous_mapping",
        },
    ]
