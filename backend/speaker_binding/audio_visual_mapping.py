from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .identity_store import (
    AudioVisualMappingSummary,
    VisualIdentityEvidenceEdge,
    normalize_confidence,
    normalize_ordered_unique_ids,
    normalize_track_ids,
)


def _normalized_id(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


@dataclass(frozen=True, slots=True)
class AudioVisualMappingEvidence:
    audio_speaker_id: str
    visual_identity_id: str
    confidence: float = 0.0
    overlap: bool = False
    offscreen: bool = False
    support_track_ids: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "audio_speaker_id", self.audio_speaker_id.strip())
        object.__setattr__(self, "visual_identity_id", self.visual_identity_id.strip())
        object.__setattr__(self, "confidence", normalize_confidence(self.confidence))
        object.__setattr__(self, "overlap", bool(self.overlap))
        object.__setattr__(self, "offscreen", bool(self.offscreen))
        object.__setattr__(self, "support_track_ids", normalize_track_ids(self.support_track_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))


def _coerce_evidence_item(item: AudioVisualMappingEvidence | Mapping[str, Any]) -> AudioVisualMappingEvidence:
    if isinstance(item, AudioVisualMappingEvidence):
        return item
    return AudioVisualMappingEvidence(
        audio_speaker_id=_normalized_id(item.get("audio_speaker_id")),
        visual_identity_id=_normalized_id(item.get("visual_identity_id")),
        confidence=item.get("confidence", 0.0),
        overlap=bool(item.get("overlap", False)),
        offscreen=bool(item.get("offscreen", False)),
        support_track_ids=tuple(item.get("support_track_ids") or ()),
        metadata=dict(item.get("metadata") or {}),
    )


def _round_score(value: float) -> float:
    return round(value, 3)


def build_audio_visual_mapping_summaries(
    evidence_items: Iterable[AudioVisualMappingEvidence | Mapping[str, Any]],
    *,
    ambiguity_margin: float = 0.05,
    ambiguous_margin: float | None = None,
) -> list[AudioVisualMappingSummary]:
    if ambiguous_margin is not None:
        ambiguity_margin = ambiguous_margin

    evidence_by_pair: dict[tuple[str, str], list[AudioVisualMappingEvidence]] = defaultdict(list)
    evidence_by_speaker: dict[str, list[AudioVisualMappingEvidence]] = defaultdict(list)
    ignored_count = 0

    for raw_item in evidence_items or []:
        item = _coerce_evidence_item(raw_item)
        audio_speaker_id = _normalized_id(item.audio_speaker_id)
        visual_identity_id = _normalized_id(item.visual_identity_id)
        if not audio_speaker_id or not visual_identity_id or item.overlap or item.offscreen:
            ignored_count += 1
            continue

        evidence_by_pair[(audio_speaker_id, visual_identity_id)].append(item)
        evidence_by_speaker[audio_speaker_id].append(item)

    summaries: list[AudioVisualMappingSummary] = []
    for audio_speaker_id in sorted(evidence_by_speaker):
        speaker_items = evidence_by_speaker[audio_speaker_id]
        pair_rows: list[tuple[str, int, float, float, list[AudioVisualMappingEvidence]]] = []

        for visual_identity_id in sorted({item.visual_identity_id for item in speaker_items}):
            pair_items = evidence_by_pair[(audio_speaker_id, visual_identity_id)]
            support_count = len(pair_items)
            confidence_sum = sum(item.confidence for item in pair_items)
            average_confidence = confidence_sum / support_count if support_count else 0.0
            pair_rows.append(
                (
                    visual_identity_id,
                    support_count,
                    _round_score(confidence_sum),
                    _round_score(average_confidence),
                    pair_items,
                )
            )

        pair_rows.sort(key=lambda row: (-row[2], -row[1], row[0]))
        if not pair_rows:
            continue

        top_visual_identity_id, _, top_score, _, _ = pair_rows[0]
        second_score = pair_rows[1][2] if len(pair_rows) > 1 else 0.0
        score_margin = _round_score(top_score - second_score)
        ambiguous = len(pair_rows) > 1 and score_margin <= ambiguity_margin
        candidate_visual_identity_ids = tuple(row[0] for row in pair_rows)

        evidence_edges = tuple(
            sorted(
                (
                    VisualIdentityEvidenceEdge(
                        audio_speaker_id=audio_speaker_id,
                        visual_identity_id=item.visual_identity_id,
                        confidence=item.confidence,
                        support_track_ids=item.support_track_ids,
                        evidence_kind="clean_span",
                        metadata=dict(item.metadata),
                    )
                    for item in speaker_items
                ),
                key=lambda edge: (edge.visual_identity_id, -edge.confidence, edge.support_track_ids),
            )
        )
        supporting_track_ids = normalize_track_ids(
            track_id for edge in evidence_edges for track_id in edge.support_track_ids
        )

        summary_metadata = {
            "candidate_stats": tuple(
                {
                    "visual_identity_id": visual_identity_id,
                    "support_count": support_count,
                    "confidence_sum": confidence_sum,
                    "average_confidence": average_confidence,
                }
                for visual_identity_id, support_count, confidence_sum, average_confidence, _ in pair_rows
            ),
            "clean_evidence_count": len(speaker_items),
            "ignored_evidence_count": ignored_count,
            "top_score_margin": score_margin,
        }

        total_clean_confidence = sum(row[2] for row in pair_rows)
        mapping_confidence = _round_score(top_score / total_clean_confidence) if total_clean_confidence > 0 else 0.0

        summaries.append(
            AudioVisualMappingSummary(
                audio_speaker_id=audio_speaker_id,
                matched_visual_identity_id=top_visual_identity_id,
                confidence=mapping_confidence,
                candidate_visual_identity_ids=candidate_visual_identity_ids,
                evidence_edges=evidence_edges,
                supporting_track_ids=supporting_track_ids,
                mapping_strategy="clean-span-aggregation",
                ambiguous=ambiguous,
                metadata=summary_metadata,
            )
        )

    return summaries


def learn_audio_visual_mappings(
    records: Iterable[AudioVisualMappingEvidence | Mapping[str, Any]],
    *,
    ambiguity_margin: float = 0.05,
    ambiguous_margin: float | None = None,
) -> list[AudioVisualMappingSummary]:
    return build_audio_visual_mapping_summaries(
        records,
        ambiguity_margin=ambiguity_margin,
        ambiguous_margin=ambiguous_margin,
    )
