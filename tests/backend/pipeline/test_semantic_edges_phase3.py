from __future__ import annotations

import pytest

from backend.pipeline.contracts import SemanticGraphEdge, SemanticGraphNode, SemanticNodeEvidence
from backend.pipeline.graph.local_semantic_edges import build_local_semantic_edges
from backend.pipeline.graph.long_range_edges import build_long_range_edges, shortlist_long_range_pairs
from backend.pipeline.graph.reconcile_edges import reconcile_semantic_edges


def _node(
    node_id: str,
    start_ms: int,
    end_ms: int,
    *,
    node_type: str = "claim",
    summary: str = "summary",
    semantic_embedding: list[float] | None = None,
    multimodal_embedding: list[float] | None = None,
) -> SemanticGraphNode:
    return SemanticGraphNode(
        node_id=node_id,
        node_type=node_type,
        start_ms=start_ms,
        end_ms=end_ms,
        source_turn_ids=[f"turn_{node_id}"],
        word_ids=[],
        transcript_text=node_id,
        node_flags=[],
        summary=summary,
        evidence=SemanticNodeEvidence(),
        semantic_embedding=semantic_embedding,
        multimodal_embedding=multimodal_embedding,
    )


def _edge(source: str, target: str, edge_type: str, *, rationale: str | None = None, confidence: float | None = None, batch_ids: list[str] | None = None) -> SemanticGraphEdge:
    return SemanticGraphEdge(
        source_node_id=source,
        target_node_id=target,
        edge_type=edge_type,
        rationale=rationale,
        confidence=confidence,
        batch_ids=batch_ids or [],
    )


def test_build_local_semantic_edges_validates_target_block_and_allowed_types():
    nodes = [
        _node("node_1", 0, 5000),
        _node("node_2", 6000, 10000, node_type="qa_exchange"),
        _node("node_3", 11000, 15000, node_type="reaction_beat"),
        _node("node_4", 16000, 20000),
    ]

    edges = build_local_semantic_edges(
        nodes=nodes,
        gemini_responses=[
            {
                "batch_id": "batch_01",
                "target_node_ids": ["node_2", "node_3"],
                "context_node_ids": ["node_1", "node_2", "node_3", "node_4"],
                "edges": [
                    {
                        "source_node_id": "node_2",
                        "target_node_id": "node_1",
                        "edge_type": "answers",
                        "rationale": "Directly answers the earlier question.",
                        "confidence": 0.82,
                    },
                    {
                        "source_node_id": "node_3",
                        "target_node_id": "node_2",
                        "edge_type": "reaction_to",
                        "rationale": "Reaction beat to the answer.",
                        "confidence": 0.76,
                    },
                ],
            }
        ],
    )

    assert {(edge.source_node_id, edge.target_node_id, edge.edge_type) for edge in edges} == {
        ("node_2", "node_1", "answers"),
        ("node_3", "node_2", "reaction_to"),
    }
    assert edges[0].batch_ids == ["batch_01"]


def test_build_local_semantic_edges_rejects_invalid_source_or_edge_type():
    nodes = [
        _node("node_1", 0, 5000),
        _node("node_2", 6000, 10000),
        _node("node_3", 11000, 15000),
    ]

    with pytest.raises(ValueError, match="target block"):
        build_local_semantic_edges(
            nodes=nodes,
            gemini_responses=[
                {
                    "batch_id": "batch_01",
                    "target_node_ids": ["node_2"],
                    "context_node_ids": ["node_1", "node_2", "node_3"],
                    "edges": [
                        {
                            "source_node_id": "node_1",
                            "target_node_id": "node_2",
                            "edge_type": "answers",
                            "rationale": "Invalid because source is halo only.",
                            "confidence": 0.8,
                        }
                    ],
                }
            ],
        )

    with pytest.raises(ValueError, match="local semantic edge type"):
        build_local_semantic_edges(
            nodes=nodes,
            gemini_responses=[
                {
                    "batch_id": "batch_02",
                    "target_node_ids": ["node_2"],
                    "context_node_ids": ["node_1", "node_2", "node_3"],
                    "edges": [
                        {
                            "source_node_id": "node_2",
                            "target_node_id": "node_1",
                            "edge_type": "callback_to",
                            "rationale": "Invalid local long-range type.",
                            "confidence": 0.7,
                        }
                    ],
                }
            ],
        )


def test_shortlist_long_range_pairs_returns_top_k_later_neighbors_only():
    nodes = [
        _node("node_1", 0, 5000, semantic_embedding=[1.0, 0.0], multimodal_embedding=[1.0, 0.0]),
        _node("node_2", 6000, 10000, semantic_embedding=[0.0, 1.0], multimodal_embedding=[0.0, 1.0]),
        _node("node_3", 11000, 15000, semantic_embedding=[0.9, 0.1], multimodal_embedding=[0.9, 0.1]),
        _node("node_4", 16000, 20000, semantic_embedding=[0.85, 0.15], multimodal_embedding=[0.85, 0.15]),
    ]

    candidate_pairs = shortlist_long_range_pairs(nodes=nodes, top_k=1)

    assert [(pair["earlier_node_id"], pair["later_node_id"]) for pair in candidate_pairs] == [
        ("node_1", "node_3"),
        ("node_2", "node_4"),
        ("node_3", "node_4"),
    ]


def test_shortlist_long_range_pairs_uses_multimodal_enrichment():
    nodes = [
        _node("node_1", 0, 5000, semantic_embedding=[1.0, 0.0], multimodal_embedding=[1.0, 0.0]),
        _node("node_2", 6000, 10000, semantic_embedding=[0.97, 0.03], multimodal_embedding=[0.0, 1.0]),
        _node("node_3", 11000, 15000, semantic_embedding=[0.95, 0.05], multimodal_embedding=[1.0, 0.0]),
    ]

    candidate_pairs = shortlist_long_range_pairs(nodes=nodes, top_k=1)

    assert [(pair["earlier_node_id"], pair["later_node_id"]) for pair in candidate_pairs] == [
        ("node_1", "node_3"),
        ("node_2", "node_3"),
    ]
    assert candidate_pairs[0]["semantic_similarity"] < candidate_pairs[1]["semantic_similarity"]
    assert candidate_pairs[0]["multimodal_similarity"] > candidate_pairs[1]["multimodal_similarity"]


def test_build_long_range_edges_validates_pairs_and_preserves_later_to_earlier_direction():
    candidate_pairs = [
        {
            "earlier_node_id": "node_1",
            "later_node_id": "node_4",
            "similarity": 0.92,
        },
        {
            "earlier_node_id": "node_2",
            "later_node_id": "node_5",
            "similarity": 0.88,
        },
    ]

    edges = build_long_range_edges(
        candidate_pairs=candidate_pairs,
        gemini_response={
            "edges": [
                {
                    "source_node_id": "node_4",
                    "target_node_id": "node_1",
                    "edge_type": "callback_to",
                    "rationale": "Explicitly refers back to the earlier joke.",
                    "confidence": 0.91,
                },
                {
                    "source_node_id": "node_5",
                    "target_node_id": "node_2",
                    "edge_type": "topic_recurrence",
                    "rationale": "Returns to the same topic later.",
                    "confidence": 0.73,
                },
            ]
        },
    )

    assert {(edge.source_node_id, edge.target_node_id, edge.edge_type) for edge in edges} == {
        ("node_4", "node_1", "callback_to"),
        ("node_5", "node_2", "topic_recurrence"),
    }

    with pytest.raises(ValueError, match="candidate long-range pair"):
        build_long_range_edges(
            candidate_pairs=candidate_pairs,
            gemini_response={
                "edges": [
                    {
                        "source_node_id": "node_1",
                        "target_node_id": "node_4",
                        "edge_type": "callback_to",
                        "rationale": "Wrong direction.",
                        "confidence": 0.5,
                    }
                ]
            },
        )


def test_reconcile_semantic_edges_dedupes_and_resolves_conflicts():
    reconciled = reconcile_semantic_edges(
        edges=[
            _edge("node_2", "node_1", "answers", rationale="Answers it.", confidence=0.7, batch_ids=["b1"]),
            _edge("node_2", "node_1", "answers", rationale="Answers it clearly.", confidence=0.9, batch_ids=["b2"]),
            _edge("node_3", "node_2", "supports", rationale="Adds evidence.", confidence=0.6, batch_ids=["b1"]),
            _edge("node_3", "node_2", "elaborates", rationale="Adds detail.", confidence=0.65, batch_ids=["b2"]),
            _edge("node_4", "node_2", "challenges", rationale="Pushback.", confidence=0.8, batch_ids=["b1"]),
            _edge("node_4", "node_2", "contradicts", rationale="Directly negates.", confidence=0.8, batch_ids=["b2"]),
            _edge("node_5", "node_1", "topic_recurrence", rationale="Same topic later.", confidence=0.8, batch_ids=["b1"]),
            _edge("node_5", "node_1", "callback_to", rationale="Explicit callback.", confidence=0.8, batch_ids=["b2"]),
        ]
    )

    edge_map = {(edge.source_node_id, edge.target_node_id, edge.edge_type): edge for edge in reconciled}

    answers = edge_map[("node_2", "node_1", "answers")]
    assert answers.support_count == 2
    assert answers.batch_ids == ["b1", "b2"]
    assert answers.confidence == pytest.approx(0.8)

    assert ("node_3", "node_2", "supports") in edge_map
    assert ("node_3", "node_2", "elaborates") in edge_map
    assert ("node_4", "node_2", "contradicts") in edge_map
    assert ("node_4", "node_2", "challenges") not in edge_map
    assert ("node_5", "node_1", "callback_to") in edge_map
    assert ("node_5", "node_1", "topic_recurrence") not in edge_map
