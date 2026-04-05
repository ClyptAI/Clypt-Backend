from __future__ import annotations

from backend.pipeline.candidates.build_local_subgraphs import build_local_subgraphs
from backend.pipeline.candidates.seed_retrieval import retrieve_seed_nodes
from backend.pipeline.config import Phase4SubgraphConfig
from backend.pipeline.contracts import SemanticGraphEdge, SemanticGraphNode, SemanticNodeEvidence


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


def _edge(source: str, target: str, edge_type: str) -> SemanticGraphEdge:
    return SemanticGraphEdge(
        source_node_id=source,
        target_node_id=target,
        edge_type=edge_type,
    )


def test_retrieve_seed_nodes_dedupes_by_node_and_retains_prompt_provenance():
    nodes = [
        _node("node_a", 0, 5000, semantic_embedding=[1.0, 0.0], multimodal_embedding=[1.0, 0.0]),
        _node("node_b", 25000, 30000, semantic_embedding=[0.9, 0.1], multimodal_embedding=[0.9, 0.1]),
        _node("node_c", 60000, 65000, semantic_embedding=[0.0, 1.0], multimodal_embedding=[0.0, 1.0]),
    ]
    prompts = [
        {"prompt_id": "p1", "text": "hook", "embedding": [1.0, 0.0]},
        {"prompt_id": "p2", "text": "shareable", "embedding": [0.95, 0.05]},
    ]

    seeds = retrieve_seed_nodes(prompts=prompts, nodes=nodes, top_k_per_prompt=2)

    assert seeds[0]["node_id"] == "node_a"
    assert seeds[0]["source_prompt_ids"] == ["p1", "p2"]
    assert seeds[1]["node_id"] == "node_b"


def test_retrieve_seed_nodes_enforces_two_seed_diversity_cap_per_20s_window():
    nodes = [
        _node("node_a", 0, 5000, semantic_embedding=[1.0, 0.0], multimodal_embedding=[1.0, 0.0]),
        _node("node_b", 7000, 12000, semantic_embedding=[0.98, 0.02], multimodal_embedding=[0.98, 0.02]),
        _node("node_c", 15000, 19000, semantic_embedding=[0.96, 0.04], multimodal_embedding=[0.96, 0.04]),
        _node("node_d", 30000, 34000, semantic_embedding=[0.95, 0.05], multimodal_embedding=[0.95, 0.05]),
    ]
    prompts = [
        {"prompt_id": "p1", "text": "hook", "embedding": [1.0, 0.0]},
    ]

    seeds = retrieve_seed_nodes(prompts=prompts, nodes=nodes, top_k_per_prompt=4)

    assert [seed["node_id"] for seed in seeds] == ["node_a", "node_b", "node_d"]


def test_retrieve_seed_nodes_requires_multimodal_enrichment_in_final_ranking():
    nodes = [
        _node(
            "node_semantic_only",
            0,
            5000,
            semantic_embedding=[1.0, 0.0],
            multimodal_embedding=[0.0, 1.0],
        ),
        _node(
            "node_enriched",
            10000,
            15000,
            semantic_embedding=[0.96, 0.04],
            multimodal_embedding=[1.0, 0.0],
        ),
    ]
    prompts = [
        {"prompt_id": "p1", "text": "hook", "embedding": [1.0, 0.0]},
    ]

    seeds = retrieve_seed_nodes(prompts=prompts, nodes=nodes, top_k_per_prompt=2)

    assert [seed["node_id"] for seed in seeds] == ["node_enriched", "node_semantic_only"]
    assert seeds[0]["semantic_similarity"] < seeds[1]["semantic_similarity"]
    assert seeds[0]["multimodal_similarity"] > seeds[1]["multimodal_similarity"]


def test_build_local_subgraphs_prefers_local_high_weight_neighbors_and_dedupes_overlaps():
    nodes = [
        _node("node_1", 0, 5000, node_type="setup_payoff", summary="setup"),
        _node("node_2", 5200, 9000, node_type="reaction_beat", summary="reaction"),
        _node("node_3", 9500, 14000, node_type="reveal", summary="payoff"),
        _node("node_4", 50000, 56000, node_type="claim", summary="far away"),
    ]
    edges = [
        _edge("node_1", "node_2", "reaction_to"),
        _edge("node_2", "node_1", "reaction_to"),
        _edge("node_1", "node_3", "payoff_of"),
        _edge("node_3", "node_1", "setup_for"),
        _edge("node_1", "node_4", "callback_to"),
        _edge("node_4", "node_1", "callback_to"),
        _edge("node_1", "node_2", "next_turn"),
        _edge("node_2", "node_1", "prev_turn"),
        _edge("node_2", "node_3", "next_turn"),
        _edge("node_3", "node_2", "prev_turn"),
    ]
    seeds = [
        {"node_id": "node_1", "source_prompt_ids": ["p1"], "retrieval_score": 0.99},
        {"node_id": "node_2", "source_prompt_ids": ["p2"], "retrieval_score": 0.80},
    ]

    subgraphs = build_local_subgraphs(
        seeds=seeds,
        nodes=nodes,
        edges=edges,
        config=Phase4SubgraphConfig(
            max_duration_s=45,
            max_node_count=10,
            max_hop_depth=2,
            max_branching_factor=3,
            min_expansion_score=3.0,
            seed_top_k_per_prompt=5,
            subgraph_overlap_dedupe_threshold=0.70,
            candidate_node_overlap_threshold=0.70,
            candidate_span_iou_threshold=0.70,
        ),
    )

    assert len(subgraphs) == 1
    subgraph = subgraphs[0]
    assert subgraph.seed_node_id == "node_1"
    assert [node.node_id for node in subgraph.nodes] == ["node_1", "node_2", "node_3"]
    assert subgraph.start_ms == 0
    assert subgraph.end_ms == 14000
