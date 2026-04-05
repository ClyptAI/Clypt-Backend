from __future__ import annotations

from backend.pipeline.contracts import SemanticGraphNode, SemanticNodeEvidence
from backend.pipeline.graph.structural_edges import build_structural_edges
from backend.pipeline.semantics.boundary_reconciliation import reconcile_boundary_nodes


def _node(
    node_id: str,
    start_ms: int,
    end_ms: int,
    source_turn_ids: list[str],
    *,
    node_type: str = "claim",
    summary: str = "summary",
) -> SemanticGraphNode:
    return SemanticGraphNode(
        node_id=node_id,
        node_type=node_type,
        start_ms=start_ms,
        end_ms=end_ms,
        source_turn_ids=source_turn_ids,
        word_ids=[],
        transcript_text=" ".join(source_turn_ids),
        node_flags=[],
        summary=summary,
        evidence=SemanticNodeEvidence(),
    )


def test_reconcile_boundary_nodes_keeps_distinct_nodes_when_requested():
    left_nodes = [
        _node("node_left", 0, 1000, ["t_000001", "t_000002"]),
        _node("node_overlap_a", 1100, 1600, ["t_000003"]),
    ]
    right_nodes = [
        _node("node_overlap_b", 1100, 1600, ["t_000003"]),
        _node("node_right", 1700, 2200, ["t_000004"]),
    ]

    reconciled = reconcile_boundary_nodes(
        left_batch_nodes=left_nodes,
        right_batch_nodes=right_nodes,
        gemini_response={
            "resolution": "keep_both",
            "nodes": [
                {
                    "existing_node_id": "node_overlap_a",
                    "source_turn_ids": ["t_000003"],
                    "node_type": "claim",
                    "node_flags": [],
                    "summary": "Left version",
                },
                {
                    "existing_node_id": "node_right",
                    "source_turn_ids": ["t_000004"],
                    "node_type": "claim",
                    "node_flags": [],
                    "summary": "Right version",
                },
            ],
        },
    )

    assert [node.node_id for node in reconciled] == ["node_overlap_a", "node_right"]


def test_reconcile_boundary_nodes_merges_overlapping_boundary_nodes():
    left_nodes = [
        _node("node_overlap_a", 1100, 1600, ["t_000003"]),
    ]
    right_nodes = [
        _node("node_overlap_b", 1100, 2200, ["t_000003", "t_000004"], node_type="explanation"),
    ]

    reconciled = reconcile_boundary_nodes(
        left_batch_nodes=left_nodes,
        right_batch_nodes=right_nodes,
        gemini_response={
            "resolution": "merge",
            "merged_node": {
                "source_turn_ids": ["t_000003", "t_000004"],
                "node_type": "explanation",
                "node_flags": ["high_resonance_candidate"],
                "summary": "Merged boundary node",
            },
        },
    )

    assert len(reconciled) == 1
    node = reconciled[0]
    assert node.source_turn_ids == ["t_000003", "t_000004"]
    assert node.start_ms == 1100
    assert node.end_ms == 2200
    assert node.node_type == "explanation"
    assert node.node_flags == ["high_resonance_candidate"]


def test_build_structural_edges_draws_next_prev_and_overlap_edges():
    nodes = [
        _node("node_1", 0, 1000, ["t_000001"]),
        _node("node_2", 900, 1800, ["t_000002"]),
        _node("node_3", 2000, 2600, ["t_000003"]),
    ]

    edges = build_structural_edges(nodes=nodes)
    edge_tuples = {(edge.source_node_id, edge.target_node_id, edge.edge_type) for edge in edges}

    assert ("node_1", "node_2", "next_turn") in edge_tuples
    assert ("node_2", "node_1", "prev_turn") in edge_tuples
    assert ("node_2", "node_3", "next_turn") in edge_tuples
    assert ("node_3", "node_2", "prev_turn") in edge_tuples
    assert ("node_1", "node_2", "overlaps_with") in edge_tuples
    assert ("node_2", "node_1", "overlaps_with") in edge_tuples
    assert ("node_2", "node_3", "overlaps_with") not in edge_tuples
