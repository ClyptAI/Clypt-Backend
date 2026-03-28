"""Cheap visual discontinuity heuristics for scheduled speaker spans."""

from __future__ import annotations

from collections.abc import Iterable


def _normalize_track_set(sample: dict) -> set[str]:
    return {
        str(track_id)
        for track_id in (sample.get("local_track_ids") or [])
        if str(track_id or "")
    }


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def classify_visual_discontinuity(samples: Iterable[dict] | None) -> dict[str, bool | list[str]]:
    ordered_samples = [sample for sample in (samples or []) if isinstance(sample, dict)]
    if len(ordered_samples) < 2:
        return {"requires_lrasd": False, "discontinuity_reasons": []}

    reasons: list[str] = []
    for previous_sample, current_sample in zip(ordered_samples, ordered_samples[1:]):
        previous_tracks = _normalize_track_set(previous_sample)
        current_tracks = _normalize_track_set(current_sample)
        if _jaccard_similarity(previous_tracks, current_tracks) < 0.5:
            reasons.append("track_set_jaccard_drop")

        previous_owner = str(previous_sample.get("prominent_track_id") or "")
        current_owner = str(current_sample.get("prominent_track_id") or "")
        if previous_owner and current_owner and previous_owner != current_owner:
            reasons.append("prominent_track_flip")

    deduped_reasons = list(dict.fromkeys(reasons))
    return {
        "requires_lrasd": bool(deduped_reasons),
        "discontinuity_reasons": deduped_reasons,
    }


__all__ = ["classify_visual_discontinuity"]
