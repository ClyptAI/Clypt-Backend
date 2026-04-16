from __future__ import annotations

from .._embedding_utils import cosine_similarity
from ..contracts import SemanticGraphEdge, SemanticGraphNode

SEMANTIC_LONG_RANGE_WEIGHT = 0.75
MULTIMODAL_LONG_RANGE_WEIGHT = 0.25


def shortlist_long_range_pairs(*, nodes: list[SemanticGraphNode], top_k: int) -> list[dict]:
    """Shortlist likely callback/topic recurrence pairs using node embeddings."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    ordered_nodes = sorted(nodes, key=lambda node: (node.start_ms, node.end_ms, node.node_id))
    shortlisted: list[dict] = []
    for idx, earlier_node in enumerate(ordered_nodes):
        later_candidates: list[tuple[float, float, float, SemanticGraphNode]] = []
        for later_node in ordered_nodes[idx + 1 :]:
            semantic_similarity = cosine_similarity(
                earlier_node.semantic_embedding,
                later_node.semantic_embedding,
            )
            multimodal_similarity = cosine_similarity(
                earlier_node.multimodal_embedding,
                later_node.multimodal_embedding,
            )
            if semantic_similarity == float("-inf") or multimodal_similarity == float("-inf"):
                continue
            similarity = (
                (SEMANTIC_LONG_RANGE_WEIGHT * semantic_similarity)
                + (MULTIMODAL_LONG_RANGE_WEIGHT * multimodal_similarity)
            )
            later_candidates.append((similarity, semantic_similarity, multimodal_similarity, later_node))
        later_candidates.sort(key=lambda item: (-item[0], item[3].start_ms, item[3].node_id))
        for similarity, semantic_similarity, multimodal_similarity, later_node in later_candidates[:top_k]:
            shortlisted.append(
                {
                    "earlier_node_id": earlier_node.node_id,
                    "later_node_id": later_node.node_id,
                    "similarity": float(similarity),
                    "semantic_similarity": float(semantic_similarity),
                    "multimodal_similarity": float(multimodal_similarity),
                }
            )
    return shortlisted


def build_long_range_edges(*, candidate_pairs: list[dict], llm_response: dict | None = None) -> list[SemanticGraphEdge]:
    """Build callback/topic recurrence edges from LLM-adjudicated pairs."""
    if llm_response is None:
        raise ValueError("llm_response is required")

    pair_set = {
        (str(pair.get("later_node_id") or ""), str(pair.get("earlier_node_id") or ""))
        for pair in candidate_pairs
    }
    edges: list[SemanticGraphEdge] = []
    for raw_edge in list(llm_response.get("edges") or []):
        source_node_id = str(raw_edge.get("source_node_id") or "")
        target_node_id = str(raw_edge.get("target_node_id") or "")
        edge_type = str(raw_edge.get("edge_type") or "")
        if edge_type not in {"callback_to", "topic_recurrence"}:
            raise ValueError(f"invalid long-range edge type: {edge_type}")
        if (source_node_id, target_node_id) not in pair_set:
            raise ValueError("Qwen returned an edge for a non-shortlisted candidate long-range pair")

        edges.append(
            SemanticGraphEdge(
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                edge_type=edge_type,
                rationale=str(raw_edge.get("rationale") or "").strip() or None,
                confidence=raw_edge.get("confidence"),
            )
        )
    return edges
