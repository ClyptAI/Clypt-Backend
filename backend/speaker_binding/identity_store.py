from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


def normalize_confidence(value: float | int | None, *, default: float = 0.0) -> float:
    if value is None:
        value = default
    confidence = float(value)
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def normalize_ordered_unique_ids(values: Iterable[str | None] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def normalize_track_ids(values: Iterable[str | None] | None) -> tuple[str, ...]:
    return normalize_ordered_unique_ids(values)


@dataclass(frozen=True, slots=True)
class VisualIdentity:
    identity_id: str
    confidence: float = 0.0
    track_ids: tuple[str, ...] = field(default_factory=tuple)
    face_track_ids: tuple[str, ...] = field(default_factory=tuple)
    person_track_ids: tuple[str, ...] = field(default_factory=tuple)
    evidence_edge_ids: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity_id", self.identity_id.strip())
        object.__setattr__(self, "confidence", normalize_confidence(self.confidence))
        object.__setattr__(self, "track_ids", normalize_track_ids(self.track_ids))
        object.__setattr__(self, "face_track_ids", normalize_track_ids(self.face_track_ids))
        object.__setattr__(self, "person_track_ids", normalize_track_ids(self.person_track_ids))
        object.__setattr__(self, "evidence_edge_ids", normalize_ordered_unique_ids(self.evidence_edge_ids))


@dataclass(frozen=True, slots=True)
class VisualIdentityEvidenceEdge:
    audio_speaker_id: str
    visual_identity_id: str
    confidence: float = 0.0
    support_track_ids: tuple[str, ...] = field(default_factory=tuple)
    evidence_kind: str = "clean_span"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "audio_speaker_id", self.audio_speaker_id.strip())
        object.__setattr__(self, "visual_identity_id", self.visual_identity_id.strip())
        object.__setattr__(self, "confidence", normalize_confidence(self.confidence))
        object.__setattr__(self, "support_track_ids", normalize_track_ids(self.support_track_ids))
        object.__setattr__(self, "evidence_kind", self.evidence_kind.strip())


@dataclass(frozen=True, slots=True)
class AudioVisualMappingSummary:
    audio_speaker_id: str
    matched_visual_identity_id: str | None = None
    confidence: float = 0.0
    candidate_visual_identity_ids: tuple[str, ...] = field(default_factory=tuple)
    evidence_edges: tuple[VisualIdentityEvidenceEdge, ...] = field(default_factory=tuple)
    supporting_track_ids: tuple[str, ...] = field(default_factory=tuple)
    mapping_strategy: str = "clean-span"
    ambiguous: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "audio_speaker_id", self.audio_speaker_id.strip())
        if self.matched_visual_identity_id is not None:
            object.__setattr__(self, "matched_visual_identity_id", self.matched_visual_identity_id.strip())
        object.__setattr__(self, "confidence", normalize_confidence(self.confidence))
        object.__setattr__(
            self,
            "candidate_visual_identity_ids",
            normalize_ordered_unique_ids(self.candidate_visual_identity_ids),
        )
        object.__setattr__(self, "supporting_track_ids", normalize_track_ids(self.supporting_track_ids))
        object.__setattr__(self, "mapping_strategy", self.mapping_strategy.strip())
        object.__setattr__(self, "evidence_edges", tuple(self.evidence_edges))
