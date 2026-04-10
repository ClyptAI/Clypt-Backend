from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from backend.pipeline.config import V31Config
from backend.pipeline.contracts import (
    ClipCandidate,
    SemanticGraphEdge,
    SemanticGraphNode,
    SemanticNodeEvidence,
)
from backend.pipeline.signals.contracts import (
    CandidateSignalLink,
    ExternalSignal,
    ExternalSignalCluster,
    NodeSignalLink,
    SignalPipelineOutput,
    SignalPromptSpec,
)
from backend.pipeline.signals.scoring import SignalScoringResult
from backend.runtime.phase14_live import V31LivePhase14Runner


class _Repo:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    def _mark(self, name: str, count: int) -> None:
        self.calls[name] = self.calls.get(name, 0) + count

    def write_phase_metric(self, record):
        return record

    def write_external_signals(self, *, run_id, signals):
        self._mark("write_external_signals", len(signals))

    def write_external_signal_clusters(self, *, run_id, clusters):
        self._mark("write_external_signal_clusters", len(clusters))

    def write_prompt_source_links(self, *, run_id, links):
        self._mark("write_prompt_source_links", len(links))

    def write_node_signal_links(self, *, run_id, links):
        self._mark("write_node_signal_links", len(links))

    def write_candidates(self, *, run_id, candidates):
        self._mark("write_candidates", len(candidates))

    def write_candidate_signal_links(self, *, run_id, links):
        self._mark("write_candidate_signal_links", len(links))

    def write_subgraph_provenance(self, *, run_id, provenance):
        self._mark("write_subgraph_provenance", len(provenance))


def test_phase14_live_run_phase4_writes_all_signal_tables(monkeypatch, tmp_path: Path) -> None:
    from backend.runtime import phase14_live

    repo = _Repo()
    runner = V31LivePhase14Runner(
        config=V31Config(output_root=tmp_path),
        llm_client=object(),
        embedding_client=object(),
        repository=repo,
    )

    node = SemanticGraphNode(
        node_id="node-1",
        node_type="claim",
        start_ms=0,
        end_ms=1000,
        transcript_text="hello",
        summary="summary",
        evidence=SemanticNodeEvidence(),
        semantic_embedding=[1.0, 0.0],
        multimodal_embedding=[1.0, 0.0],
    )
    edge = SemanticGraphEdge(
        source_node_id="node-1",
        target_node_id="node-1",
        edge_type="next_turn",
        rationale="self",
        confidence=1.0,
        support_count=1,
        batch_ids=["b1"],
    )

    candidate = ClipCandidate(
        clip_id="clip-1",
        node_ids=["node-1"],
        start_ms=0,
        end_ms=1000,
        score=1.0,
        rationale="r",
        source_prompt_ids=["general_prompt_001", "comment_prompt_001"],
        seed_node_id="node-1",
        subgraph_id="sg-1",
        query_aligned=True,
        pool_rank=1,
        score_breakdown={"base_score": 1.0},
    )

    signal_output = SignalPipelineOutput(
        external_signals=[
            ExternalSignal(
                signal_id="comment_top:1",
                signal_type="comment_top",
                source_platform="youtube",
                source_id="1",
                text="great moment",
                engagement_score=1.0,
                metadata={"quality": "high_signal", "like_count": 10, "reply_count": 2},
            )
        ],
        clusters=[
            ExternalSignalCluster(
                cluster_id="comment_cluster_001",
                cluster_type="comment",
                summary_text="great moment",
                member_signal_ids=["comment_top:1"],
                cluster_weight=1.0,
                embedding=[1.0, 0.0],
                metadata={},
            )
        ],
        prompt_specs=[
            SignalPromptSpec(
                prompt_id="comment_prompt_001",
                text="Find the great moment",
                prompt_source_type="comment",
                source_cluster_id="comment_cluster_001",
                source_cluster_type="comment",
            )
        ],
    )

    monkeypatch.setattr(phase14_live, "generate_meta_prompts_live", lambda **kwargs: ["Find the strongest hook"])
    monkeypatch.setattr(
        phase14_live,
        "embed_prompt_texts_live",
        lambda **kwargs: [
            {"prompt_id": "general_prompt_001", "embedding": [1.0, 0.0]},
            {"prompt_id": "comment_prompt_001", "embedding": [1.0, 0.0]},
        ],
    )
    monkeypatch.setattr(
        phase14_live,
        "retrieve_seed_nodes",
        lambda **kwargs: [{"prompt_id": "general_prompt_001", "node_id": "node-1", "retrieval_score": 0.9}],
    )
    monkeypatch.setattr(
        phase14_live,
        "build_local_subgraphs",
        lambda **kwargs: [
            SimpleNamespace(
                subgraph_id="sg-1",
                seed_node_id="node-1",
                source_prompt_ids=["general_prompt_001", "comment_prompt_001"],
                nodes=[SimpleNamespace(node_id="node-1")],
                model_dump=lambda mode="json": {
                    "subgraph_id": "sg-1",
                    "nodes": [{"node_id": "node-1"}],
                },
            )
        ],
    )
    monkeypatch.setattr(phase14_live, "run_subgraph_reviews", lambda **kwargs: ([SimpleNamespace(candidates=[candidate])], {"ok": True}))
    monkeypatch.setattr(phase14_live, "dedupe_clip_candidates", lambda candidates: candidates)
    monkeypatch.setattr(
        phase14_live,
        "run_candidate_pool_review",
        lambda **kwargs: SimpleNamespace(
            ranked_candidates=[
                SimpleNamespace(
                    candidate_temp_id="clip-1",
                    pool_rank=1,
                    score=1.25,
                    score_breakdown={"base_score": 1.0},
                    rationale="pool",
                )
            ]
        ),
    )
    monkeypatch.setattr(
        phase14_live,
        "build_node_signal_links",
        lambda **kwargs: [
            NodeSignalLink(
                node_id="node-1",
                cluster_id="comment_cluster_001",
                link_type="direct",
                hop_distance=0,
                time_offset_ms=0,
                similarity=1.0,
                link_score=0.8,
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        phase14_live,
        "apply_signal_scoring",
        lambda **kwargs: SignalScoringResult(
            candidates=[
                candidate.model_copy(
                    update={
                        "score": 1.5,
                        "external_signal_score": 0.2,
                        "agreement_bonus": 0.04,
                        "external_attribution_json": {"cluster_links": [], "agreement_flags": ["general", "comment"]},
                    }
                )
            ],
            candidate_signal_links=[
                CandidateSignalLink(
                    clip_id="clip-1",
                    cluster_id="comment_cluster_001",
                    cluster_type="comment",
                    aggregated_link_score=0.2,
                    coverage_ms=1000,
                    direct_node_count=1,
                    inferred_node_count=0,
                    agreement_flags=["general", "comment"],
                    bonus_applied=0.04,
                    evidence={},
                )
            ],
        ),
    )
    monkeypatch.setattr(phase14_live, "explain_candidate_attribution_with_llm", lambda **kwargs: "boosted by comments")

    summary = runner.run_phase_4(
        run_id="run_001",
        job_id="job_1",
        attempt=1,
        paths=runner.build_run_paths(run_id="run_001"),
        source_url="https://youtube.com/watch?v=abc",
        canonical_timeline=SimpleNamespace(turns=[SimpleNamespace(end_ms=1000)]),
        nodes=[node],
        edges=[edge],
        extra_prompt_texts=[],
        signal_output=signal_output,
    )

    assert summary["final_candidate_count"] == 1
    assert repo.calls["write_external_signals"] == 1
    assert repo.calls["write_external_signal_clusters"] == 1
    assert repo.calls["write_prompt_source_links"] >= 1
    assert repo.calls["write_node_signal_links"] == 1
    assert repo.calls["write_candidates"] == 1
    assert repo.calls["write_candidate_signal_links"] == 1
    assert repo.calls["write_subgraph_provenance"] == 1
