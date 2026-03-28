from __future__ import annotations

from backend.speaker_binding.identity_store import (
    AudioVisualMappingSummary,
    VisualIdentity,
    VisualIdentityEvidenceEdge,
    normalize_confidence,
    normalize_ordered_unique_ids,
    normalize_track_ids,
)


def test_normalize_confidence_clamps_to_unit_interval() -> None:
    assert normalize_confidence(None) == 0.0
    assert normalize_confidence(-0.2) == 0.0
    assert normalize_confidence(0.25) == 0.25
    assert normalize_confidence(1.4) == 1.0


def test_normalize_ordered_unique_ids_preserves_first_occurrence_and_skips_blanks() -> None:
    assert normalize_ordered_unique_ids(["  track-a  ", "track-b", "track-a", "", "track-c", "track-b"]) == (
        "track-a",
        "track-b",
        "track-c",
    )


def test_normalize_track_ids_matches_ordered_unique_ids() -> None:
    assert normalize_track_ids(["track-3", "track-1", "track-3", "track-2"]) == (
        "track-3",
        "track-1",
        "track-2",
    )


def test_visual_identity_normalizes_track_lists_and_confidence() -> None:
    identity = VisualIdentity(
        identity_id="  visual-7  ",
        confidence=1.25,
        track_ids=[" track-2 ", "track-1", "track-2", None, "track-3"],
        face_track_ids=["face-9", "face-9", " face-10 "],
        person_track_ids=("person-5", "person-4", "person-5"),
    )

    assert identity.identity_id == "visual-7"
    assert identity.confidence == 1.0
    assert identity.track_ids == ("track-2", "track-1", "track-3")
    assert identity.face_track_ids == ("face-9", "face-10")
    assert identity.person_track_ids == ("person-5", "person-4")


def test_evidence_edge_normalizes_ids_and_confidence() -> None:
    edge = VisualIdentityEvidenceEdge(
        audio_speaker_id="  speaker-1 ",
        visual_identity_id=" visual-3 ",
        confidence=-0.5,
        support_track_ids=["track-8", "track-8", "track-9"],
        evidence_kind="clean_span",
    )

    assert edge.audio_speaker_id == "speaker-1"
    assert edge.visual_identity_id == "visual-3"
    assert edge.confidence == 0.0
    assert edge.support_track_ids == ("track-8", "track-9")


def test_mapping_summary_deduplicates_candidate_ids_and_keeps_edge_order() -> None:
    edge_a = VisualIdentityEvidenceEdge(
        audio_speaker_id="speaker-1",
        visual_identity_id="visual-2",
        confidence=0.8,
        support_track_ids=["track-1"],
    )
    edge_b = VisualIdentityEvidenceEdge(
        audio_speaker_id="speaker-1",
        visual_identity_id="visual-2",
        confidence=0.6,
        support_track_ids=["track-2", "track-2"],
    )

    summary = AudioVisualMappingSummary(
        audio_speaker_id="speaker-1",
        matched_visual_identity_id=" visual-2 ",
        candidate_visual_identity_ids=["visual-2", "visual-3", "visual-2"],
        confidence=1.1,
        evidence_edges=[edge_a, edge_b],
        supporting_track_ids=["track-1", "track-2", "track-1"],
        mapping_strategy="clean-span",
    )

    assert summary.audio_speaker_id == "speaker-1"
    assert summary.matched_visual_identity_id == "visual-2"
    assert summary.candidate_visual_identity_ids == ("visual-2", "visual-3")
    assert summary.confidence == 1.0
    assert summary.supporting_track_ids == ("track-1", "track-2")
    assert summary.evidence_edges == (edge_a, edge_b)
