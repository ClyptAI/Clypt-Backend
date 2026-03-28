from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .identity_store import VisualIdentity, normalize_confidence, normalize_track_ids


def _normalized_track_id(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def _iter_face_identity_ids(feature: dict) -> Iterable[str]:
    dominant_track_id = _normalized_track_id(feature.get("dominant_track_id"))
    if dominant_track_id:
        yield dominant_track_id
        return

    associated_track_counts = dict(feature.get("associated_track_counts", {}) or {})
    if not associated_track_counts:
        return

    sorted_candidates = sorted(
        (
            (_normalized_track_id(track_id), int(count))
            for track_id, count in associated_track_counts.items()
            if _normalized_track_id(track_id) and int(count) > 0
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if sorted_candidates:
        yield sorted_candidates[0][0]


def build_visual_identities(
    *,
    tracks: list[dict],
    track_identity_features: dict[str, dict] | None,
    face_track_features: dict[str, dict] | None,
) -> list[VisualIdentity]:
    grouped_track_rows: dict[str, list[dict]] = defaultdict(list)
    grouped_face_track_ids: dict[str, list[str]] = defaultdict(list)
    grouped_source_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "tracks": 0,
            "track_identity_features": 0,
            "face_track_features": 0,
        }
    )
    grouped_confidences: dict[str, list[float]] = defaultdict(list)

    for track in tracks or []:
        track_id = _normalized_track_id(track.get("track_id"))
        if not track_id:
            continue
        grouped_track_rows[track_id].append(dict(track))
        grouped_source_counts[track_id]["tracks"] += 1
        grouped_confidences[track_id].append(normalize_confidence(track.get("confidence")))

    for track_id, feature in sorted((track_identity_features or {}).items()):
        normalized_track_id = _normalized_track_id(track_id)
        if not normalized_track_id:
            continue
        grouped_source_counts[normalized_track_id]["track_identity_features"] += 1
        grouped_confidences[normalized_track_id].append(normalize_confidence(feature.get("confidence")))

    for face_track_id, feature in sorted((face_track_features or {}).items()):
        normalized_face_track_id = _normalized_track_id(face_track_id)
        if not normalized_face_track_id:
            continue
        for identity_id in _iter_face_identity_ids(feature):
            grouped_face_track_ids[identity_id].append(normalized_face_track_id)
            grouped_source_counts[identity_id]["face_track_features"] += 1

    all_identity_ids = sorted(
        {
            *grouped_track_rows.keys(),
            *(_normalized_track_id(track_id) for track_id in (track_identity_features or {}).keys()),
            *grouped_face_track_ids.keys(),
        }
        - {""}
    )

    identities: list[VisualIdentity] = []
    for identity_id in all_identity_ids:
        track_rows = grouped_track_rows.get(identity_id, [])
        face_track_ids = normalize_track_ids(grouped_face_track_ids.get(identity_id))
        confidences = grouped_confidences.get(identity_id, [])
        source_counts = grouped_source_counts[identity_id]
        sources = tuple(sorted(source for source, count in source_counts.items() if count > 0))

        identities.append(
            VisualIdentity(
                identity_id=identity_id,
                confidence=max(confidences) if confidences else 0.0,
                track_ids=(identity_id,),
                person_track_ids=(identity_id,),
                face_track_ids=face_track_ids,
                metadata={
                    "track_count": len(track_rows),
                    "person_track_count": len(track_rows),
                    "face_track_count": len(face_track_ids),
                    "source_counts": dict(source_counts),
                    "sources": sources,
                },
            )
        )

    return identities
