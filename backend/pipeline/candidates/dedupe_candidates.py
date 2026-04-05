from __future__ import annotations

from ..contracts import ClipCandidate

NODE_OVERLAP_THRESHOLD = 0.70
SPAN_IOU_THRESHOLD = 0.70


def _node_overlap_ratio(left: ClipCandidate, right: ClipCandidate) -> float:
    left_ids = set(left.node_ids)
    right_ids = set(right.node_ids)
    return len(left_ids & right_ids) / max(1, len(left_ids | right_ids))


def _span_iou(left: ClipCandidate, right: ClipCandidate) -> float:
    intersection = max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))
    union = max(left.end_ms, right.end_ms) - min(left.start_ms, right.start_ms)
    if union <= 0:
        return 0.0
    return intersection / union


def _rationale_strength(candidate: ClipCandidate) -> int:
    text = candidate.rationale.lower()
    keywords = (
        "complete",
        "self-contained",
        "self contained",
        "payoff",
        "reveal",
        "standalone",
        "hook",
    )
    return sum(1 for keyword in keywords if keyword in text)


def _sort_key(candidate: ClipCandidate) -> tuple[float, int, int, int, str]:
    duration_ms = candidate.end_ms - candidate.start_ms
    stable_id = candidate.clip_id or "|".join(candidate.node_ids)
    return (
        -candidate.score,
        -_rationale_strength(candidate),
        duration_ms,
        candidate.start_ms,
        stable_id,
    )


def _is_near_duplicate(left: ClipCandidate, right: ClipCandidate) -> bool:
    return (
        _node_overlap_ratio(left, right) > NODE_OVERLAP_THRESHOLD
        or _span_iou(left, right) > SPAN_IOU_THRESHOLD
    )


def dedupe_clip_candidates(*, candidates: list[ClipCandidate]) -> list[ClipCandidate]:
    """Deduplicate near-identical clip candidates before pooled review."""
    ordered_candidates = sorted(candidates, key=_sort_key)
    kept: list[ClipCandidate] = []
    for candidate in ordered_candidates:
        if any(_is_near_duplicate(candidate, existing) for existing in kept):
            continue
        kept.append(candidate)
    return kept
