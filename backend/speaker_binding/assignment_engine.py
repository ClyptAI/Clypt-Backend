from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .identity_store import normalize_confidence, normalize_ordered_unique_ids


def _normalized_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalized_ids(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    return normalize_ordered_unique_ids(_normalized_id(value) for value in values)


def _coerce_mapping_summaries(values: Any) -> tuple[dict[str, Any], ...]:
    if not values:
        return ()

    summaries: list[dict[str, Any]] = []
    for raw_summary in values:
        summary = dict(raw_summary or {})
        summary["audio_speaker_id"] = _normalized_id(summary.get("audio_speaker_id"))
        summary["matched_visual_identity_id"] = _normalized_id(summary.get("matched_visual_identity_id"))
        summary["confidence"] = normalize_confidence(summary.get("confidence", 0.0))
        summary["ambiguous"] = bool(summary.get("ambiguous", False))
        summary["candidate_visual_identity_ids"] = _normalized_ids(summary.get("candidate_visual_identity_ids"))
        summaries.append(summary)
    return tuple(summaries)


def _pick_best_mapping(
    *,
    audio_speaker_ids: tuple[str, ...],
    mapping_summaries: tuple[dict[str, Any], ...],
) -> dict[str, Any] | None:
    if not audio_speaker_ids or not mapping_summaries:
        return None

    candidates = [
        summary
        for summary in mapping_summaries
        if summary["audio_speaker_id"] in set(audio_speaker_ids) and summary["matched_visual_identity_id"]
    ]
    if not candidates:
        return None

    candidates.sort(
        key=lambda summary: (
            -float(summary["confidence"]),
            summary["audio_speaker_id"],
            summary["matched_visual_identity_id"],
        )
    )
    return candidates[0]


def _span_sort_key(span: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        int(span.get("start_time_ms", 0) or 0),
        int(span.get("end_time_ms", 0) or 0),
        _normalized_id(span.get("span_id")),
    )


def _base_result(
    *,
    span: Mapping[str, Any],
    audio_speaker_ids: tuple[str, ...],
    visible_track_ids: tuple[str, ...],
    offscreen_audio_speaker_ids: tuple[str, ...],
    unresolved_audio_speaker_ids: tuple[str, ...],
    assigned_visual_identity_ids: tuple[str, ...],
    dominant_visual_identity_id: str | None,
    require_hard_disambiguation: bool,
    decision_source: str,
) -> dict[str, Any]:
    return {
        "start_time_ms": int(span.get("start_time_ms", 0) or 0),
        "end_time_ms": int(span.get("end_time_ms", 0) or 0),
        "audio_speaker_ids": audio_speaker_ids,
        "assigned_visual_identity_ids": assigned_visual_identity_ids,
        "dominant_visual_identity_id": dominant_visual_identity_id,
        "offscreen_audio_speaker_ids": offscreen_audio_speaker_ids,
        "unresolved_audio_speaker_ids": unresolved_audio_speaker_ids,
        "require_hard_disambiguation": require_hard_disambiguation,
        "decision_source": decision_source,
    }


def resolve_span_assignments(
    spans: Iterable[Mapping[str, Any]] | None,
    *,
    mapping_summaries: Iterable[Mapping[str, Any]] | None = None,
    strong_visible_mapping_threshold: float = 0.8,
) -> list[dict[str, Any]]:
    span_rows = sorted([dict(span) for span in (spans or [])], key=_span_sort_key)
    global_mapping_summaries = _coerce_mapping_summaries(mapping_summaries)

    assignments: list[dict[str, Any]] = []
    for span in span_rows:
        audio_speaker_ids = _normalized_ids(span.get("audio_speaker_ids"))
        visible_track_ids = _normalized_ids(span.get("visible_track_ids"))
        offscreen_audio_speaker_ids = _normalized_ids(span.get("offscreen_audio_speaker_ids"))
        overlap = bool(span.get("overlap", False))
        span_mapping_summaries = _coerce_mapping_summaries(span.get("mapping_summaries"))
        active_mapping_summaries = span_mapping_summaries or global_mapping_summaries
        best_mapping = _pick_best_mapping(
            audio_speaker_ids=audio_speaker_ids,
            mapping_summaries=active_mapping_summaries,
        )

        if overlap or len(audio_speaker_ids) != 1:
            assignments.append(
                _base_result(
                    span=span,
                    audio_speaker_ids=audio_speaker_ids,
                    visible_track_ids=visible_track_ids,
                    offscreen_audio_speaker_ids=offscreen_audio_speaker_ids,
                    unresolved_audio_speaker_ids=audio_speaker_ids,
                    assigned_visual_identity_ids=(),
                    dominant_visual_identity_id=None,
                    require_hard_disambiguation=True,
                    decision_source="overlap",
                )
            )
            continue

        if best_mapping is None or best_mapping["ambiguous"]:
            assignments.append(
                _base_result(
                    span=span,
                    audio_speaker_ids=audio_speaker_ids,
                    visible_track_ids=visible_track_ids,
                    offscreen_audio_speaker_ids=offscreen_audio_speaker_ids,
                    unresolved_audio_speaker_ids=audio_speaker_ids,
                    assigned_visual_identity_ids=(),
                    dominant_visual_identity_id=None,
                    require_hard_disambiguation=True,
                    decision_source="ambiguous_mapping",
                )
            )
            continue

        matched_visual_identity_id = _normalized_id(best_mapping["matched_visual_identity_id"])
        if not matched_visual_identity_id:
            assignments.append(
                _base_result(
                    span=span,
                    audio_speaker_ids=audio_speaker_ids,
                    visible_track_ids=visible_track_ids,
                    offscreen_audio_speaker_ids=offscreen_audio_speaker_ids,
                    unresolved_audio_speaker_ids=audio_speaker_ids,
                    assigned_visual_identity_ids=(),
                    dominant_visual_identity_id=None,
                    require_hard_disambiguation=True,
                    decision_source="ambiguous_mapping",
                )
            )
            continue

        if matched_visual_identity_id in visible_track_ids and best_mapping["confidence"] >= strong_visible_mapping_threshold:
            assignments.append(
                _base_result(
                    span=span,
                    audio_speaker_ids=audio_speaker_ids,
                    visible_track_ids=visible_track_ids,
                    offscreen_audio_speaker_ids=offscreen_audio_speaker_ids,
                    unresolved_audio_speaker_ids=(),
                    assigned_visual_identity_ids=(matched_visual_identity_id,),
                    dominant_visual_identity_id=matched_visual_identity_id,
                    require_hard_disambiguation=False,
                    decision_source="mapping",
                )
            )
            continue

        if matched_visual_identity_id not in visible_track_ids:
            assignments.append(
                _base_result(
                    span=span,
                    audio_speaker_ids=audio_speaker_ids,
                    visible_track_ids=visible_track_ids,
                    offscreen_audio_speaker_ids=offscreen_audio_speaker_ids or audio_speaker_ids,
                    unresolved_audio_speaker_ids=(),
                    assigned_visual_identity_ids=(),
                    dominant_visual_identity_id=None,
                    require_hard_disambiguation=False,
                    decision_source="mapping_offscreen",
                )
            )
            continue

        assignments.append(
            _base_result(
                span=span,
                audio_speaker_ids=audio_speaker_ids,
                visible_track_ids=visible_track_ids,
                offscreen_audio_speaker_ids=offscreen_audio_speaker_ids,
                unresolved_audio_speaker_ids=audio_speaker_ids,
                assigned_visual_identity_ids=(),
                dominant_visual_identity_id=None,
                require_hard_disambiguation=True,
                decision_source="ambiguous_mapping",
            )
        )

    return assignments


def assign_span_bindings(
    span_records: Iterable[Mapping[str, Any]] | None,
    *,
    mapping_summaries: Iterable[Mapping[str, Any]] | None = None,
    strong_visible_mapping_threshold: float = 0.8,
) -> list[dict[str, Any]]:
    return resolve_span_assignments(
        span_records,
        mapping_summaries=mapping_summaries,
        strong_visible_mapping_threshold=strong_visible_mapping_threshold,
    )

