from __future__ import annotations

from ..contracts import SemanticGraphEdge

COMPATIBLE_EDGE_GROUPS = [
    {"supports", "elaborates"},
    {"reaction_to", "payoff_of"},
    {"setup_for", "escalates"},
]

EDGE_PRECEDENCE = {
    "answers": 100,
    "contradicts": 95,
    "callback_to": 90,
    "payoff_of": 85,
    "setup_for": 84,
    "reaction_to": 80,
    "challenges": 75,
    "escalates": 70,
    "supports": 65,
    "elaborates": 60,
    "topic_recurrence": 55,
}


def _compatible(edge_type: str, other_edge_type: str) -> bool:
    if edge_type == other_edge_type:
        return True
    return any({edge_type, other_edge_type}.issubset(group) for group in COMPATIBLE_EDGE_GROUPS)


def _collapse_duplicate_group(group: list[SemanticGraphEdge]) -> SemanticGraphEdge:
    first = group[0]
    confidences = [edge.confidence for edge in group if edge.confidence is not None]
    batch_ids: list[str] = []
    for edge in group:
        for batch_id in edge.batch_ids:
            if batch_id not in batch_ids:
                batch_ids.append(batch_id)
    rationale = next((edge.rationale for edge in group if edge.rationale), None)
    return SemanticGraphEdge(
        source_node_id=first.source_node_id,
        target_node_id=first.target_node_id,
        edge_type=first.edge_type,
        rationale=rationale,
        confidence=(sum(confidences) / len(confidences)) if confidences else None,
        support_count=len(group),
        batch_ids=batch_ids,
    )


def _winner_sort_key(edge: SemanticGraphEdge) -> tuple[int, float, int, str]:
    return (
        edge.support_count or 0,
        edge.confidence if edge.confidence is not None else -1.0,
        EDGE_PRECEDENCE.get(edge.edge_type, 0),
        edge.edge_type,
    )


def reconcile_semantic_edges(*, edges: list[SemanticGraphEdge]) -> list[SemanticGraphEdge]:
    """Canonicalize, dedupe, and resolve semantic graph edges deterministically."""
    grouped_exact: dict[tuple[str, str, str], list[SemanticGraphEdge]] = {}
    for edge in edges:
        grouped_exact.setdefault(
            (edge.source_node_id, edge.target_node_id, edge.edge_type),
            [],
        ).append(edge)

    collapsed = [
        _collapse_duplicate_group(group)
        for _, group in sorted(grouped_exact.items(), key=lambda item: item[0])
    ]

    by_pair: dict[tuple[str, str], list[SemanticGraphEdge]] = {}
    for edge in collapsed:
        by_pair.setdefault((edge.source_node_id, edge.target_node_id), []).append(edge)

    reconciled: list[SemanticGraphEdge] = []
    for _, pair_edges in sorted(by_pair.items(), key=lambda item: item[0]):
        pair_edges.sort(key=lambda edge: (_winner_sort_key(edge)), reverse=True)
        kept: list[SemanticGraphEdge] = []
        for edge in pair_edges:
            incompatible_with = [existing for existing in kept if not _compatible(edge.edge_type, existing.edge_type)]
            if not incompatible_with:
                kept.append(edge)
                continue

            if all(_winner_sort_key(edge) > _winner_sort_key(existing) for existing in incompatible_with):
                kept = [existing for existing in kept if _compatible(edge.edge_type, existing.edge_type)]
                kept.append(edge)
        reconciled.extend(kept)

    reconciled.sort(key=lambda edge: (edge.source_node_id, edge.target_node_id, edge.edge_type))
    return reconciled
