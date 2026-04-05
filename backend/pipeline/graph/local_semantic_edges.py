from __future__ import annotations

from ..contracts import SemanticGraphEdge, SemanticGraphNode

LOCAL_EDGE_TYPES = {
    "answers",
    "challenges",
    "contradicts",
    "supports",
    "elaborates",
    "setup_for",
    "payoff_of",
    "reaction_to",
    "escalates",
}


def build_local_semantic_edges(*, nodes: list[SemanticGraphNode], gemini_responses: list[dict] | None = None) -> list[SemanticGraphEdge]:
    """Build local rhetorical graph edges from overlapping node neighborhoods."""
    if gemini_responses is None:
        raise ValueError("gemini_responses is required")

    node_ids = {node.node_id for node in nodes}
    edges: list[SemanticGraphEdge] = []
    for response in gemini_responses:
        batch_id = str(response.get("batch_id") or "").strip()
        if not batch_id:
            raise ValueError("batch_id is required for local semantic edge batches")
        target_node_ids = list(response.get("target_node_ids") or [])
        context_node_ids = list(response.get("context_node_ids") or target_node_ids)
        target_node_id_set = set(target_node_ids)
        context_node_id_set = set(context_node_ids)
        if not target_node_id_set:
            raise ValueError("target_node_ids are required for local semantic edge batches")
        if not target_node_id_set.issubset(node_ids):
            raise ValueError("target block references unknown node ids")
        if not context_node_id_set.issubset(node_ids):
            raise ValueError("context block references unknown node ids")
        if not target_node_id_set.issubset(context_node_id_set):
            raise ValueError("target block must be contained within the context block")

        for raw_edge in list(response.get("edges") or []):
            source_node_id = str(raw_edge.get("source_node_id") or "")
            target_node_id = str(raw_edge.get("target_node_id") or "")
            edge_type = str(raw_edge.get("edge_type") or "")
            if source_node_id not in target_node_id_set:
                raise ValueError("local semantic edges must originate from the target block")
            if target_node_id not in context_node_id_set:
                raise ValueError("local semantic edges must stay within the provided context block")
            if edge_type not in LOCAL_EDGE_TYPES:
                raise ValueError(f"invalid local semantic edge type: {edge_type}")

            edges.append(
                SemanticGraphEdge(
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    edge_type=edge_type,
                    rationale=str(raw_edge.get("rationale") or "").strip() or None,
                    confidence=raw_edge.get("confidence"),
                    batch_ids=[batch_id],
                )
            )
    return edges
