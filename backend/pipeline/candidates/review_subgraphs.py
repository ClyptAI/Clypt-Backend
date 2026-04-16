from __future__ import annotations

from pydantic import ValidationError

from ..contracts import ClipCandidate, LocalSubgraph, SubgraphReviewResponse


def _invalid_subgraph_response(*, subgraph: LocalSubgraph, reason: str) -> SubgraphReviewResponse:
    return SubgraphReviewResponse(
        subgraph_id=subgraph.subgraph_id,
        seed_node_id=subgraph.seed_node_id,
        reject_all=True,
        reject_reason=f"invalid_structured_output: {reason}",
        candidates=[],
    )


def review_local_subgraph(*, subgraph: LocalSubgraph, llm_response: dict | None = None) -> SubgraphReviewResponse:
    """Validate or adapt one Qwen subgraph-review response."""
    if llm_response is None:
        raise ValueError("llm_response is required")

    node_order = [node.node_id for node in subgraph.nodes]
    node_id_set = set(node_order)
    node_by_id = {node.node_id: node for node in subgraph.nodes}

    try:
        if llm_response.get("subgraph_id") != subgraph.subgraph_id:
            raise ValueError("subgraph_id must match the reviewed subgraph")
        if llm_response.get("seed_node_id") != subgraph.seed_node_id:
            raise ValueError("seed_node_id must match the reviewed subgraph")

        reject_all = bool(llm_response.get("reject_all"))
        reject_reason = str(llm_response.get("reject_reason") or "")
        raw_candidates = list(llm_response.get("candidates") or [])

        if len(raw_candidates) > 3:
            raise ValueError("subgraph review may return at most 3 candidates")
        if reject_all and raw_candidates:
            raise ValueError("reject_all=true requires candidates=[]")
        if not reject_all and not raw_candidates:
            raise ValueError("reject_all=false requires at least one candidate")
        if reject_all:
            if not reject_reason.strip():
                raise ValueError("reject_all=true requires a non-empty reject_reason")
            return SubgraphReviewResponse(
                subgraph_id=subgraph.subgraph_id,
                seed_node_id=subgraph.seed_node_id,
                reject_all=True,
                reject_reason=reject_reason,
                candidates=[],
            )

        candidates: list[ClipCandidate] = []
        for idx, raw_candidate in enumerate(raw_candidates, start=1):
            candidate_node_ids = list(raw_candidate.get("node_ids") or [])
            if not candidate_node_ids:
                raise ValueError("candidate node_ids are required")
            if any(node_id not in node_id_set for node_id in candidate_node_ids):
                raise ValueError("candidate references unknown node_id")

            positions = [node_order.index(node_id) for node_id in candidate_node_ids]
            span_positions = list(range(min(positions), max(positions) + 1))
            if positions != span_positions:
                raise ValueError("candidate node_ids must form one contiguous chronological span")

            expected_span_node_ids = node_order[min(positions) : max(positions) + 1]
            if candidate_node_ids != expected_span_node_ids:
                raise ValueError("candidate node_ids must preserve chronological order")

            first_node = node_by_id[candidate_node_ids[0]]
            last_node = node_by_id[candidate_node_ids[-1]]
            if raw_candidate.get("start_ms") != first_node.start_ms:
                raise ValueError("candidate start_ms must match the chosen node span exactly")
            if raw_candidate.get("end_ms") != last_node.end_ms:
                raise ValueError("candidate end_ms must match the chosen node span exactly")

            candidates.append(
                ClipCandidate(
                    clip_id=f"{subgraph.subgraph_id}_cand_{idx:02d}",
                    node_ids=candidate_node_ids,
                    start_ms=first_node.start_ms,
                    end_ms=last_node.end_ms,
                    score=float(raw_candidate["score"]),
                    rationale=str(raw_candidate.get("rationale") or "").strip(),
                    source_prompt_ids=list(subgraph.source_prompt_ids),
                    seed_node_id=subgraph.seed_node_id,
                    subgraph_id=subgraph.subgraph_id,
                )
            )

        return SubgraphReviewResponse(
            subgraph_id=subgraph.subgraph_id,
            seed_node_id=subgraph.seed_node_id,
            reject_all=False,
            reject_reason=reject_reason,
            candidates=candidates,
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        return _invalid_subgraph_response(subgraph=subgraph, reason=str(exc))
