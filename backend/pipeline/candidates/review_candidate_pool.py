from __future__ import annotations

from ..contracts import ClipCandidate, PooledCandidateReviewResponse, RankedCandidateDecision


def _candidate_temp_id_map(candidates: list[ClipCandidate]) -> dict[str, ClipCandidate]:
    mapping: dict[str, ClipCandidate] = {}
    for idx, candidate in enumerate(candidates, start=1):
        candidate_temp_id = candidate.clip_id or f"cand_tmp_{idx:03d}"
        if candidate_temp_id in mapping:
            raise ValueError(f"duplicate pooled candidate id: {candidate_temp_id}")
        mapping[candidate_temp_id] = candidate
    return mapping


def review_candidate_pool(*, candidates: list[ClipCandidate], gemini_response: dict | None = None) -> PooledCandidateReviewResponse:
    """Validate or adapt the final pooled Gemini candidate-review response."""
    if gemini_response is None:
        raise ValueError("pooled review call failed: gemini_response is required")

    candidate_map = _candidate_temp_id_map(candidates)
    candidate_ids = set(candidate_map)
    parsed = PooledCandidateReviewResponse.model_validate(gemini_response)

    kept_ids: list[str] = []
    ranked: list[RankedCandidateDecision] = []
    for decision in parsed.ranked_candidates:
        if decision.candidate_temp_id not in candidate_ids:
            raise ValueError(f"unknown candidate_temp_id: {decision.candidate_temp_id}")
        if not decision.keep:
            raise ValueError("ranked_candidates entries must have keep=true")
        kept_ids.append(decision.candidate_temp_id)
        ranked.append(decision)

    dropped_ids = list(parsed.dropped_candidate_temp_ids)
    if any(candidate_id not in candidate_ids for candidate_id in dropped_ids):
        unknown = [candidate_id for candidate_id in dropped_ids if candidate_id not in candidate_ids][0]
        raise ValueError(f"unknown dropped candidate_temp_id: {unknown}")

    kept_id_set = set(kept_ids)
    dropped_id_set = set(dropped_ids)
    if len(kept_id_set) != len(kept_ids):
        raise ValueError("candidate_temp_id values must be unique among kept candidates")
    if kept_id_set & dropped_id_set:
        raise ValueError("dropped candidates must not also appear as kept")

    accounted_ids = kept_id_set | dropped_id_set
    if accounted_ids != candidate_ids:
        raise ValueError("pooled review must account for every candidate exactly once")

    pool_ranks = [decision.pool_rank for decision in ranked]
    if any(rank is None for rank in pool_ranks):
        raise ValueError("pool_rank is required for every kept candidate")
    sorted_ranks = sorted(rank for rank in pool_ranks if rank is not None)
    expected_ranks = list(range(1, len(ranked) + 1))
    if sorted_ranks != expected_ranks:
        raise ValueError("pool_rank values must form a contiguous rank order starting at 1")

    ranked.sort(key=lambda decision: (decision.pool_rank or 0, decision.candidate_temp_id))
    dropped_ids.sort(key=lambda candidate_id: list(candidate_map).index(candidate_id))

    return PooledCandidateReviewResponse(
        ranked_candidates=ranked,
        dropped_candidate_temp_ids=dropped_ids,
    )
