from __future__ import annotations

from dataclasses import dataclass

from ..config import Phase4SubgraphConfig
from ..contracts import LocalSubgraph, LocalSubgraphNode, LocalSubgraphNodeEdge, SemanticGraphEdge, SemanticGraphNode

from ..contracts import LocalSubgraph, SemanticGraphEdge, SemanticGraphNode


EDGE_WEIGHTS: dict[str, float] = {
    "setup_for": 4.0,
    "payoff_of": 4.0,
    "reaction_to": 3.5,
    "answers": 3.5,
    "contradicts": 3.5,
    "challenges": 3.0,
    "escalates": 3.0,
    "supports": 2.5,
    "elaborates": 2.5,
    "next_turn": 1.5,
    "prev_turn": 1.5,
    "overlaps_with": 1.0,
    "callback_to": 0.75,
    "topic_recurrence": 0.5,
}


@dataclass(slots=True)
class _CandidateNeighbor:
    node: SemanticGraphNode
    score: float
    hop_depth: int


def _gap_ms(left: SemanticGraphNode, right: SemanticGraphNode) -> int:
    if right.start_ms <= left.end_ms:
        return 0
    return right.start_ms - left.end_ms


def _time_bonus(gap_ms: int) -> float | None:
    if gap_ms <= 1500:
        return 2.0
    if gap_ms <= 4000:
        return 1.0
    if gap_ms <= 8000:
        return 0.5
    if gap_ms <= 15000:
        return 0.0
    return None


def _closure_bonus(edge_type: str, candidate: SemanticGraphNode, included_nodes: list[SemanticGraphNode]) -> float:
    included_types = {node.node_type for node in included_nodes}
    if edge_type == "payoff_of" and "setup_payoff" in included_types:
        return 1.5
    if edge_type == "answers" and "qa_exchange" in included_types:
        return 1.5
    if edge_type == "reaction_to" and included_types & {"claim", "reveal", "setup_payoff"}:
        return 1.0
    if edge_type in {"contradicts", "challenges"} and included_types & {"claim", "explanation", "challenge_exchange"}:
        return 1.0
    return 0.0


def _adjacency_bonus(candidate_idx: int, included_indices: set[int]) -> float:
    if candidate_idx - 1 in included_indices or candidate_idx + 1 in included_indices:
        return 0.5
    if any((candidate_idx - 1) < idx < (candidate_idx + 1) for idx in included_indices):
        return 0.5
    return 0.0


def _build_local_edges_for_node(node_id: str, included_node_ids: set[str], edges: list[SemanticGraphEdge]) -> tuple[list[LocalSubgraphNodeEdge], list[LocalSubgraphNodeEdge]]:
    inbound: list[LocalSubgraphNodeEdge] = []
    outbound: list[LocalSubgraphNodeEdge] = []
    for edge in edges:
        if edge.source_node_id == node_id and edge.target_node_id in included_node_ids:
            outbound.append(LocalSubgraphNodeEdge(edge_type=edge.edge_type, target_node_id=edge.target_node_id))
        if edge.target_node_id == node_id and edge.source_node_id in included_node_ids:
            inbound.append(LocalSubgraphNodeEdge(edge_type=edge.edge_type, target_node_id=edge.source_node_id))
    return inbound, outbound


def _serialize_subgraph(subgraph_id: str, seed: dict, included_nodes: list[SemanticGraphNode], edges: list[SemanticGraphEdge]) -> LocalSubgraph:
    ordered_nodes = sorted(included_nodes, key=lambda node: (node.start_ms, node.end_ms, node.node_id))
    included_node_ids = {node.node_id for node in ordered_nodes}
    local_nodes: list[LocalSubgraphNode] = []
    for node in ordered_nodes:
        inbound_edges, outbound_edges = _build_local_edges_for_node(node.node_id, included_node_ids, edges)
        local_nodes.append(
            LocalSubgraphNode(
                node_id=node.node_id,
                start_ms=node.start_ms,
                end_ms=node.end_ms,
                duration_ms=node.end_ms - node.start_ms,
                node_type=node.node_type,
                node_flags=list(node.node_flags),
                summary=node.summary,
                transcript_excerpt=node.transcript_text,
                word_count=len(node.transcript_text.split()),
                emotion_labels=list(node.evidence.emotion_labels),
                audio_events=list(node.evidence.audio_events),
                inbound_edges=inbound_edges,
                outbound_edges=outbound_edges,
            )
        )
    return LocalSubgraph(
        subgraph_id=subgraph_id,
        seed_node_id=seed["node_id"],
        source_prompt_ids=list(seed.get("source_prompt_ids") or []),
        start_ms=ordered_nodes[0].start_ms,
        end_ms=ordered_nodes[-1].end_ms,
        nodes=local_nodes,
    )


def build_local_subgraphs(*, seeds: list[dict], nodes: list[SemanticGraphNode], edges: list[SemanticGraphEdge], config: object | None = None) -> list[LocalSubgraph]:
    """Build deterministic local candidate subgraphs around seed nodes."""
    cfg = config if isinstance(config, Phase4SubgraphConfig) else Phase4SubgraphConfig()
    node_by_id = {node.node_id: node for node in nodes}
    node_index = {node.node_id: idx for idx, node in enumerate(sorted(nodes, key=lambda item: (item.start_ms, item.end_ms, item.node_id)))}
    edge_lookup: dict[frozenset[str], list[SemanticGraphEdge]] = {}
    for edge in edges:
        edge_lookup.setdefault(frozenset({edge.source_node_id, edge.target_node_id}), []).append(edge)

    built: list[tuple[LocalSubgraph, float, set[str]]] = []
    for seed_idx, seed in enumerate(seeds, start=1):
        seed_node = node_by_id.get(seed["node_id"])
        if seed_node is None:
            continue
        included = {seed_node.node_id: seed_node}
        hop_depths = {seed_node.node_id: 0}
        subgraph_score = 0.0

        while True:
            if len(included) >= cfg.max_node_count:
                break
            current_nodes = list(included.values())
            current_duration_ms = max(node.end_ms for node in current_nodes) - min(node.start_ms for node in current_nodes)
            if current_duration_ms >= cfg.max_duration_s * 1000:
                break

            included_indices = {node_index[node_id] for node_id in included}
            candidates: list[_CandidateNeighbor] = []
            for candidate in nodes:
                if candidate.node_id in included:
                    continue
                best_score: float | None = None
                best_hop_depth: int | None = None
                for included_node in current_nodes:
                    pair_edges = edge_lookup.get(frozenset({included_node.node_id, candidate.node_id}), [])
                    if not pair_edges:
                        continue
                    strongest_weight = max(EDGE_WEIGHTS.get(edge.edge_type, 0.0) for edge in pair_edges)
                    strongest_edge_type = max(pair_edges, key=lambda edge: EDGE_WEIGHTS.get(edge.edge_type, 0.0)).edge_type
                    gap_ms = _gap_ms(included_node, candidate) if included_node.start_ms <= candidate.start_ms else _gap_ms(candidate, included_node)
                    time_bonus = _time_bonus(gap_ms)
                    if time_bonus is None:
                        continue
                    next_hop_depth = hop_depths[included_node.node_id] + 1
                    if next_hop_depth > cfg.max_hop_depth:
                        continue
                    score = (
                        strongest_weight
                        + time_bonus
                        + _closure_bonus(strongest_edge_type, candidate, current_nodes)
                        + _adjacency_bonus(node_index[candidate.node_id], included_indices)
                        - (0.75 if next_hop_depth == 2 else 0.0)
                    )
                    if best_score is None or score > best_score:
                        best_score = score
                        best_hop_depth = next_hop_depth
                if best_score is not None and best_hop_depth is not None and best_score >= cfg.min_expansion_score:
                    projected_nodes = [*current_nodes, candidate]
                    projected_duration_ms = max(node.end_ms for node in projected_nodes) - min(node.start_ms for node in projected_nodes)
                    if projected_duration_ms > cfg.max_duration_s * 1000:
                        continue
                    candidates.append(_CandidateNeighbor(node=candidate, score=best_score, hop_depth=best_hop_depth))

            if not candidates:
                break
            candidates.sort(key=lambda item: (-item.score, item.node.start_ms, item.node.node_id))
            chosen = candidates[0]
            included[chosen.node.node_id] = chosen.node
            hop_depths[chosen.node.node_id] = chosen.hop_depth
            subgraph_score += chosen.score

        local_subgraph = _serialize_subgraph(
            subgraph_id=f"sg_{len(built) + 1:04d}",
            seed=seed,
            included_nodes=list(included.values()),
            edges=edges,
        )
        built.append((local_subgraph, subgraph_score, set(included.keys())))

    deduped: list[tuple[LocalSubgraph, float, set[str]]] = []
    for candidate in built:
        subgraph, score, node_ids = candidate
        replaced = False
        for idx, (existing_subgraph, existing_score, existing_node_ids) in enumerate(deduped):
            overlap_ratio = len(node_ids & existing_node_ids) / max(1, len(node_ids | existing_node_ids))
            if overlap_ratio > cfg.subgraph_overlap_dedupe_threshold:
                if score > existing_score:
                    deduped[idx] = candidate
                replaced = True
                break
        if not replaced:
            deduped.append(candidate)

    return [subgraph for subgraph, _, _ in deduped]
