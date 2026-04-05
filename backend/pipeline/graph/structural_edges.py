from __future__ import annotations

from ..contracts import SemanticGraphEdge, SemanticGraphNode


def build_structural_edges(*, nodes: list[SemanticGraphNode]) -> list[SemanticGraphEdge]:
    """Build deterministic next/prev/overlap edges from canonical nodes."""
    ordered_nodes = sorted(nodes, key=lambda node: (node.start_ms, node.end_ms, node.node_id))
    edges: list[SemanticGraphEdge] = []

    for idx, node in enumerate(ordered_nodes):
        if idx + 1 < len(ordered_nodes):
            next_node = ordered_nodes[idx + 1]
            edges.append(
                SemanticGraphEdge(
                    source_node_id=node.node_id,
                    target_node_id=next_node.node_id,
                    edge_type="next_turn",
                )
            )
            edges.append(
                SemanticGraphEdge(
                    source_node_id=next_node.node_id,
                    target_node_id=node.node_id,
                    edge_type="prev_turn",
                )
            )

    for i, left in enumerate(ordered_nodes):
        for right in ordered_nodes[i + 1 :]:
            if right.start_ms > left.end_ms:
                break
            if max(left.start_ms, right.start_ms) <= min(left.end_ms, right.end_ms):
                edges.append(
                    SemanticGraphEdge(
                        source_node_id=left.node_id,
                        target_node_id=right.node_id,
                        edge_type="overlaps_with",
                    )
                )
                edges.append(
                    SemanticGraphEdge(
                        source_node_id=right.node_id,
                        target_node_id=left.node_id,
                        edge_type="overlaps_with",
                    )
                )

    return edges
