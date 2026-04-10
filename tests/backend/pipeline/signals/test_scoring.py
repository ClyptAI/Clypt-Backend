from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.pipeline.contracts import ClipCandidate, SemanticGraphNode, SemanticNodeEvidence
from backend.pipeline.signals.contracts import (
    CandidateSignalLink,
    ExternalSignalCluster,
    NodeSignalLink,
    SignalPromptSpec,
)
from backend.pipeline.signals.scoring import apply_signal_scoring


def test_apply_signal_scoring_uses_union_source_coverage_and_keeps_general_only_when_not_meaningful() -> None:
    nodes = [
        SemanticGraphNode(
            node_id="node-1",
            node_type="claim",
            start_ms=0,
            end_ms=3000,
            transcript_text="node 1",
            summary="node 1",
            evidence=SemanticNodeEvidence(),
            semantic_embedding=[1.0, 0.0],
            multimodal_embedding=[1.0, 0.0],
        ),
        SemanticGraphNode(
            node_id="node-2",
            node_type="reveal",
            start_ms=2000,
            end_ms=5000,
            transcript_text="node 2",
            summary="node 2",
            evidence=SemanticNodeEvidence(),
            semantic_embedding=[1.0, 0.0],
            multimodal_embedding=[1.0, 0.0],
        ),
    ]
    clusters = [
        ExternalSignalCluster(
            cluster_id="comment_cluster_001",
            cluster_type="comment",
            summary_text="audience noticed the same beat",
            member_signal_ids=["signal-1", "signal-2"],
            cluster_weight=0.0,
            embedding=[1.0, 0.0],
            metadata={},
        )
    ]
    node_links = [
        NodeSignalLink(
            node_id="node-1",
            cluster_id="comment_cluster_001",
            link_type="direct",
            hop_distance=0,
            time_offset_ms=0,
            similarity=1.0,
            link_score=0.0,
            evidence={},
        ),
        NodeSignalLink(
            node_id="node-2",
            cluster_id="comment_cluster_001",
            link_type="direct",
            hop_distance=0,
            time_offset_ms=0,
            similarity=1.0,
            link_score=0.0,
            evidence={},
        ),
    ]
    prompt_specs = [
        SignalPromptSpec(
            prompt_id="general_prompt_001",
            text="Find the strongest moment",
            prompt_source_type="general",
        ),
        SignalPromptSpec(
            prompt_id="comment_prompt_001",
            text="Find the payoff moment",
            prompt_source_type="comment",
            source_cluster_id="comment_cluster_001",
            source_cluster_type="comment",
        ),
    ]
    candidates = [
        ClipCandidate(
            clip_id="clip-1",
            node_ids=["node-1", "node-2"],
            start_ms=0,
            end_ms=10_000,
            score=4.0,
            rationale="base clip",
            source_prompt_ids=["general_prompt_001", "comment_prompt_001"],
            seed_node_id="node-1",
            subgraph_id="sg-1",
            query_aligned=True,
        )
    ]
    cfg = SimpleNamespace(
        cluster_freq_ref=30,
        cluster_mean_weight=0.45,
        cluster_max_weight=0.25,
        cluster_freq_weight=0.30,
        hop_decay_1=0.75,
        hop_decay_2=0.55,
        time_window_ms=30_000,
        epsilon=1e-6,
        coverage_weight=0.30,
        direct_ratio_weight=0.15,
        cluster_cap=1.0,
        total_cap=10.0,
        meaningful_min_cluster_contrib=0.10,
        meaningful_min_source_coverage=0.55,
        agreement_bonus_tier1=0.25,
        agreement_bonus_tier2=0.50,
        agreement_cap=1.0,
    )

    result = apply_signal_scoring(
        candidates=candidates,
        nodes=nodes,
        clusters=clusters,
        node_links=node_links,
        prompt_specs=prompt_specs,
        cfg=cfg,
    )

    assert len(result.candidates) == 1
    scored_candidate = result.candidates[0]
    assert scored_candidate.pool_rank == 1
    assert scored_candidate.score_breakdown is not None
    assert scored_candidate.score_breakdown["external_signal_score"] == pytest.approx(0.7004597, rel=1e-5)
    assert scored_candidate.score_breakdown["agreement_bonus"] == 0.0
    assert scored_candidate.score == pytest.approx(4.7004597, rel=1e-5)

    assert len(result.candidate_signal_links) == 1
    link = result.candidate_signal_links[0]
    assert link.coverage_ms == 6000
    assert link.agreement_flags == ["general"]
    assert link.bonus_applied == 0.0
