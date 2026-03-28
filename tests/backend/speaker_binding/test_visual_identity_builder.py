from __future__ import annotations

from backend.speaker_binding.visual_identity import build_visual_identities


def test_build_visual_identities_merges_clustered_track_evidence_deterministically() -> None:
    tracks = [
        {"track_id": " Global_Person_2 ", "confidence": 0.2, "label": "person"},
        {"track_id": "Global_Person_1", "confidence": 0.7, "label": "person"},
        {"track_id": "Global_Person_2", "confidence": 0.6, "label": "person"},
        {"track_id": "   ", "confidence": 0.9, "label": "person"},
    ]
    track_identity_features = {
        "Global_Person_1": {
            "confidence": 0.9,
            "embedding": [0.1, 0.2, 0.3],
            "embedding_source": "face",
        },
        "Global_Person_2": {
            "confidence": 0.5,
            "embedding": [0.4, 0.5, 0.6],
            "embedding_source": "face",
            "face_observations": [{"frame_idx": 4, "confidence": 0.4}],
        },
    }
    face_track_features = {
        "face-b": {
            "embedding": [0.7, 0.8, 0.9],
            "embedding_count": 2,
            "associated_track_counts": {"Global_Person_2": 3, "Global_Person_1": 1},
            "dominant_track_id": " Global_Person_2 ",
        },
        "face-a": {
            "embedding": [1.0, 1.1, 1.2],
            "associated_track_counts": {"Global_Person_2": 1},
            "dominant_track_id": "Global_Person_2",
        },
    }

    identities = build_visual_identities(
        tracks=tracks,
        track_identity_features=track_identity_features,
        face_track_features=face_track_features,
    )

    assert [identity.identity_id for identity in identities] == ["Global_Person_1", "Global_Person_2"]

    first, second = identities
    assert first.track_ids == ("Global_Person_1",)
    assert first.person_track_ids == ("Global_Person_1",)
    assert first.face_track_ids == ()
    assert first.confidence == 0.9
    assert first.metadata == {
        "track_count": 1,
        "person_track_count": 1,
        "face_track_count": 0,
        "source_counts": {
            "tracks": 1,
            "track_identity_features": 1,
            "face_track_features": 0,
        },
        "sources": ("track_identity_features", "tracks"),
    }

    assert second.track_ids == ("Global_Person_2",)
    assert second.person_track_ids == ("Global_Person_2",)
    assert second.face_track_ids == ("face-a", "face-b")
    assert second.confidence == 0.6
    assert second.metadata == {
        "track_count": 2,
        "person_track_count": 2,
        "face_track_count": 2,
        "source_counts": {
            "tracks": 2,
            "track_identity_features": 1,
            "face_track_features": 2,
        },
        "sources": ("face_track_features", "track_identity_features", "tracks"),
    }


def test_build_visual_identities_creates_identities_for_track_only_inputs() -> None:
    identities = build_visual_identities(
        tracks=[
            {"track_id": "track-2", "confidence": 0.25, "label": "person"},
            {"track_id": "track-1", "confidence": 0.75, "label": "person"},
        ],
        track_identity_features=None,
        face_track_features=None,
    )

    assert [identity.identity_id for identity in identities] == ["track-1", "track-2"]
    assert identities[0].track_ids == ("track-1",)
    assert identities[0].person_track_ids == ("track-1",)
    assert identities[0].face_track_ids == ()
    assert identities[0].confidence == 0.75
    assert identities[0].metadata["source_counts"] == {
        "tracks": 1,
        "track_identity_features": 0,
        "face_track_features": 0,
    }
