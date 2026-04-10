from __future__ import annotations

import pytest

from backend.pipeline.contracts import SemanticGraphEdge, SemanticGraphNode, SemanticNodeEvidence
from backend.pipeline.signals.cluster import cluster_signals
from backend.pipeline.signals.contracts import ExternalSignal, ExternalSignalCluster, SignalPromptSpec
from backend.pipeline.signals.linking import build_node_signal_links
from backend.pipeline.signals.llm_runtime import SignalLLMCallError


def _signal(signal_id: str, text: str, embedding: list[float], engagement_score: float) -> ExternalSignal:
    return ExternalSignal(
        signal_id=signal_id,
        signal_type="comment_top",
        source_platform="youtube",
        source_id=signal_id,
        text=text,
        engagement_score=engagement_score,
        metadata={},
    )


def test_cluster_signals_groups_similar_signals_and_uses_best_preview_text() -> None:
    signals = [
        _signal("signal-1", "most relevant", [1.0, 0.0], 10.0),
        _signal("signal-2", "less relevant", [0.98, 0.02], 3.0),
        _signal("signal-3", "separate topic", [0.0, 1.0], 8.0),
    ]

    clusters = cluster_signals(
        signals=signals,
        embeddings=[[1.0, 0.0], [0.98, 0.02], [0.0, 1.0]],
        cluster_type="comment",
        similarity_threshold=0.95,
    )

    assert len(clusters) == 2
    assert clusters[0].cluster_id == "comment_cluster_001"
    assert clusters[0].summary_text == "most relevant | less relevant"
    assert set(clusters[0].member_signal_ids) == {"signal-1", "signal-2"}
    assert clusters[1].member_signal_ids == ["signal-3"]


def test_cluster_signals_rejects_embedding_length_mismatch() -> None:
    signal = _signal("signal-1", "text", [1.0, 0.0], 1.0)

    with pytest.raises(ValueError, match="length mismatch"):
        cluster_signals(signals=[signal], embeddings=[], cluster_type="comment", similarity_threshold=0.5)


def test_build_node_signal_links_returns_direct_and_inferred_links(monkeypatch: pytest.MonkeyPatch) -> None:
    nodes = [
        SemanticGraphNode(
            node_id="node-1",
            node_type="claim",
            start_ms=0,
            end_ms=2000,
            transcript_text="node 1",
            summary="node 1",
            evidence=SemanticNodeEvidence(),
            semantic_embedding=[1.0, 0.0],
            multimodal_embedding=[1.0, 0.0],
        ),
        SemanticGraphNode(
            node_id="node-2",
            node_type="reveal",
            start_ms=1500,
            end_ms=3000,
            transcript_text="node 2",
            summary="node 2",
            evidence=SemanticNodeEvidence(),
            semantic_embedding=[0.95, 0.05],
            multimodal_embedding=[0.95, 0.05],
        ),
    ]
    edges = [
        SemanticGraphEdge(
            source_node_id="node-1",
            target_node_id="node-2",
            edge_type="payoff_of",
            rationale="node 2 pays off node 1",
            confidence=0.9,
            support_count=2,
            batch_ids=["batch-1"],
        )
    ]
    clusters = [
        ExternalSignalCluster(
            cluster_id="comment_cluster_001",
            cluster_type="comment",
            summary_text="audience noticed the payoff",
            member_signal_ids=["signal-1"],
            cluster_weight=0.0,
            embedding=[1.0, 0.0],
            metadata={},
        )
    ]
    prompt_specs = [
        SignalPromptSpec(
            prompt_id="comment_prompt_001",
            text="Find the payoff moment",
            prompt_source_type="comment",
            source_cluster_id="comment_cluster_001",
            source_cluster_type="comment",
        )
    ]

    class _FakeLLM:
        def generate_json(self, **kwargs):
            self.calls = getattr(self, "calls", [])
            self.calls.append(kwargs)
            return {"node_ids": ["node-1"], "reason": "core moment"}

    llm_client = _FakeLLM()

    links = build_node_signal_links(
        clusters=clusters,
        prompt_specs=prompt_specs,
        prompt_embeddings={"comment_prompt_001": [1.0, 0.0]},
        nodes=nodes,
        edges=edges,
        llm_client=llm_client,
        model="gemini-3-flash",
        thinking_level="minimal",
        max_hops=2,
        time_window_ms=10_000,
    )

    assert [link.link_type for link in links] == ["direct", "inferred"]
    assert links[0].node_id == "node-1"
    assert links[0].hop_distance == 0
    assert links[1].node_id == "node-2"
    assert links[1].hop_distance == 1
    assert links[1].evidence["edge_type"] == "payoff_of"


def test_build_node_signal_links_fails_hard_when_callpoint5_selects_no_valid_nodes() -> None:
    nodes = [
        SemanticGraphNode(
            node_id="node-1",
            node_type="claim",
            start_ms=0,
            end_ms=2000,
            transcript_text="node 1",
            summary="node 1",
            evidence=SemanticNodeEvidence(),
            semantic_embedding=[1.0, 0.0],
            multimodal_embedding=[1.0, 0.0],
        )
    ]
    clusters = [
        ExternalSignalCluster(
            cluster_id="comment_cluster_001",
            cluster_type="comment",
            summary_text="audience noticed the payoff",
            member_signal_ids=["signal-1"],
            cluster_weight=0.0,
            embedding=[1.0, 0.0],
            metadata={},
        )
    ]
    prompt_specs = [
        SignalPromptSpec(
            prompt_id="comment_prompt_001",
            text="Find the payoff moment",
            prompt_source_type="comment",
            source_cluster_id="comment_cluster_001",
            source_cluster_type="comment",
        )
    ]

    class _FakeLLM:
        def generate_json(self, **kwargs):
            return {"node_ids": ["missing-node"], "reason": "bad span"}

    with pytest.raises(SignalLLMCallError, match="callpoint_5_resolve_cluster_span"):
        build_node_signal_links(
            clusters=clusters,
            prompt_specs=prompt_specs,
            prompt_embeddings={"comment_prompt_001": [1.0, 0.0]},
            nodes=nodes,
            edges=[],
            llm_client=_FakeLLM(),
            model="gemini-3-flash",
            thinking_level="minimal",
            max_hops=2,
            time_window_ms=10_000,
        )
