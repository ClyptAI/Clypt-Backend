from __future__ import annotations

import threading
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

    monkeypatch.setattr(
        phase14_live,
        "generate_meta_prompts_live",
        lambda **kwargs: (["Find the strongest hook"], {}),
    )
    monkeypatch.setattr(
        phase14_live,
        "embed_prompt_texts_live",
        lambda **kwargs: (
            [
                {"prompt_id": "general_prompt_001", "embedding": [1.0, 0.0]},
                {"prompt_id": "comment_prompt_001", "embedding": [1.0, 0.0]},
            ],
            {"latency_ms": 0.0},
        ),
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
        "run_candidate_pool_review_with_debug",
        lambda **kwargs: (
            SimpleNamespace(
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
            {
                "candidate_count": 1,
                "kept_candidate_count": 1,
                "dropped_candidate_count": 0,
                "max_pool_rank": 1,
                "latency_ms": 0.0,
                "prompt_chars": 0,
                "prompt_token_estimate": 0,
                "payload_chars": 0,
                "response_chars": 0,
            },
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
    from backend.pipeline.signals.responses import SignalsCandidateAttributionResponse

    monkeypatch.setattr(
        phase14_live,
        "explain_candidate_attribution_with_llm",
        lambda **kwargs: SignalsCandidateAttributionResponse(explanation="boosted by comments"),
    )

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


def test_live_phase14_runner_invokes_phase6_after_phase4(monkeypatch, tmp_path: Path) -> None:
    from backend.runtime import phase14_live

    runner = V31LivePhase14Runner(
        config=V31Config(output_root=tmp_path),
        llm_client=object(),
        embedding_client=object(),
    )

    candidate = ClipCandidate(
        clip_id="clip-1",
        node_ids=["node-1"],
        start_ms=0,
        end_ms=1000,
        score=1.0,
        rationale="r",
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

    def _run_phase_1(self, **kwargs):
        return {
            "canonical_timeline": SimpleNamespace(turns=[SimpleNamespace(end_ms=1000)]),
            "speech_emotion_timeline": SimpleNamespace(),
            "audio_event_timeline": SimpleNamespace(),
            "shot_tracklet_index": SimpleNamespace(tracklets=[]),
            "tracklet_geometry": SimpleNamespace(tracklets=[]),
        }

    def _run_phase_2(self, **kwargs):
        return {"nodes": [node]}

    def _run_phase_3(self, **kwargs):
        return {"edges": [edge]}

    def _run_phase_4(self, **kwargs):
        return {
            "candidates": [candidate],
            "seed_count": 1,
            "subgraph_count": 1,
            "raw_candidate_count": 1,
            "deduped_candidate_count": 1,
            "final_candidate_count": 1,
        }

    phase6_calls: list[dict[str, object]] = []

    def _run_phase_6(self, **kwargs):
        phase6_calls.append(kwargs)
        return {
            "clip_count": 1,
            "artifact_paths": {
                "caption_plan": str(kwargs["paths"].caption_plan),
                "publish_metadata": str(kwargs["paths"].publish_metadata),
                "render_plan": str(kwargs["paths"].render_plan),
            },
        }

    monkeypatch.setattr(type(runner), "run_phase_1", _run_phase_1)
    monkeypatch.setattr(type(runner), "run_phase_2", _run_phase_2)
    monkeypatch.setattr(type(runner), "run_phase_3", _run_phase_3)
    monkeypatch.setattr(type(runner), "run_phase_4", _run_phase_4)
    monkeypatch.setattr(type(runner), "run_phase_6", _run_phase_6, raising=False)
    monkeypatch.setattr(phase14_live, "start_comments_future", lambda **kwargs: None)
    monkeypatch.setattr(phase14_live, "start_trends_future", lambda **kwargs: None)

    summary = runner.run(
        run_id="run_phase6",
        source_url="https://www.youtube.com/watch?v=abc123xyz00",
        phase1_outputs=SimpleNamespace(
            phase1_audio={"local_video_path": "/tmp/source.mp4"},
            diarization_payload={"turns": []},
            phase1_visual={"video_metadata": {"fps": 30.0}, "shot_changes": [], "tracks": []},
            emotion2vec_payload={"segments": []},
            yamnet_payload={"events": []},
        ),
        job_id="job-1",
        attempt=1,
    )

    assert phase6_calls
    assert phase6_calls[0]["phase4_result"]["candidates"] == [candidate]
    assert summary.artifact_paths["caption_plan"].endswith("caption_plan.json")
    assert summary.artifact_paths["publish_metadata"].endswith("publish_metadata.json")
    assert summary.artifact_paths["render_plan"].endswith("render_plan.json")
    assert "caption_plan" in summary.metadata["phase6_artifacts"]


def test_live_phase14_run_phase6_uses_persisted_source_context(monkeypatch, tmp_path: Path) -> None:
    from backend.runtime import phase14_live

    runner = V31LivePhase14Runner(
        config=V31Config(output_root=tmp_path),
        llm_client=object(),
        embedding_client=object(),
    )
    paths = runner.build_run_paths(run_id="run_phase6_context")
    paths.source_context.write_text(
        (
            "{\n"
            '  "source_url": "https://www.youtube.com/watch?v=abc123xyz00",\n'
            '  "youtube_video_id": "abc123xyz00",\n'
            '  "source_title": "Persisted Title",\n'
            '  "source_description": "Persisted description",\n'
            '  "channel_id": "channel_123",\n'
            '  "channel_title": "Persisted Channel",\n'
            '  "published_at": "2026-04-19T00:00:00+00:00",\n'
            '  "default_audio_language": "en",\n'
            '  "category_id": "22",\n'
            '  "tags": ["persisted"],\n'
            '  "thumbnails": {"default": {"url": "https://example.com/thumb.jpg"}}\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    candidate = ClipCandidate(
        clip_id="clip-1",
        node_ids=["node-1"],
        start_ms=0,
        end_ms=1000,
        score=1.0,
        rationale="r",
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
    captured: dict[str, object] = {}

    def _fake_compile_phase6_artifacts(**kwargs):
        captured.update(kwargs)
        return {
            "artifact_paths": {
                "source_context": str(paths.source_context),
                "caption_plan": str(paths.caption_plan),
                "publish_metadata": str(paths.publish_metadata),
                "render_plan": str(paths.render_plan),
            }
        }

    monkeypatch.setattr(phase14_live, "compile_phase6_artifacts", _fake_compile_phase6_artifacts)
    monkeypatch.setattr(
        phase14_live,
        "resolve_youtube_video_id",
        lambda _: (_ for _ in ()).throw(AssertionError("phase6 should use persisted source_context")),
    )

    result = runner.run_phase_6(
        paths=paths,
        source_url="https://www.youtube.com/watch?v=abc123xyz00",
        canonical_timeline=SimpleNamespace(turns=[]),
        shot_tracklet_index=SimpleNamespace(tracklets=[]),
        tracklet_geometry=SimpleNamespace(tracklets=[]),
        nodes=[node],
        phase4_result={"candidates": [candidate]},
    )

    assert result["artifact_paths"]["source_context"] == str(paths.source_context)
    assert captured["source_context"]["source_title"] == "Persisted Title"
    assert captured["source_context"]["youtube_video_id"] == "abc123xyz00"


def test_live_phase14_run_phase6_requires_persisted_source_context_for_candidates(tmp_path: Path) -> None:
    runner = V31LivePhase14Runner(
        config=V31Config(output_root=tmp_path),
        llm_client=object(),
        embedding_client=object(),
    )
    paths = runner.build_run_paths(run_id="run_phase6_missing_context")
    candidate = ClipCandidate(
        clip_id="clip-1",
        node_ids=["node-1"],
        start_ms=0,
        end_ms=1000,
        score=1.0,
        rationale="r",
    )

    try:
        runner.run_phase_6(
            paths=paths,
            source_url="https://www.youtube.com/watch?v=abc123xyz00",
            canonical_timeline=SimpleNamespace(turns=[]),
            shot_tracklet_index=SimpleNamespace(tracklets=[]),
            tracklet_geometry=SimpleNamespace(tracklets=[]),
            nodes=[],
            phase4_result={"candidates": [candidate]},
        )
    except ValueError as exc:
        assert "persisted source_context" in str(exc)
    else:
        raise AssertionError("expected run_phase_6 to require persisted source_context for candidates")


def test_live_phase14_run_phase6_invokes_render_executor_and_exposes_outputs(monkeypatch, tmp_path: Path) -> None:
    from backend.runtime import phase14_live

    class _RenderExecutor:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "outputs": [
                    {
                        "clip_id": "clip-1",
                        "video_gcs_uri": "gs://bucket/phase14/run_phase6_render/render_outputs/clip-1.mp4",
                    }
                ]
            }

    render_executor = _RenderExecutor()
    runner = V31LivePhase14Runner(
        config=V31Config(output_root=tmp_path),
        llm_client=object(),
        embedding_client=object(),
        render_executor=render_executor,
    )
    paths = runner.build_run_paths(run_id="run_phase6_render")
    paths.source_context.write_text(
        (
            "{\n"
            '  "source_url": "https://www.youtube.com/watch?v=abc123xyz00",\n'
            '  "youtube_video_id": "abc123xyz00",\n'
            '  "source_title": "Persisted Title",\n'
            '  "source_description": "Persisted description",\n'
            '  "channel_id": "channel_123",\n'
            '  "channel_title": "Persisted Channel",\n'
            '  "published_at": "2026-04-19T00:00:00+00:00",\n'
            '  "default_audio_language": "en",\n'
            '  "category_id": "22",\n'
            '  "tags": ["persisted"],\n'
            '  "thumbnails": {"default": {"url": "https://example.com/thumb.jpg"}}\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    candidate = ClipCandidate(
        clip_id="clip-1",
        node_ids=["node-1"],
        start_ms=0,
        end_ms=1000,
        score=1.0,
        rationale="r",
    )

    monkeypatch.setattr(
        phase14_live,
        "compile_phase6_artifacts",
        lambda **kwargs: {
            "artifact_paths": {
                "source_context": str(paths.source_context),
                "caption_plan": str(paths.caption_plan),
                "publish_metadata": str(paths.publish_metadata),
                "render_plan": str(paths.render_plan),
                "captions_clip-1.ass": str(paths.captions_ass("clip-1")),
            }
        },
    )

    result = runner.run_phase_6(
        paths=paths,
        source_url="https://www.youtube.com/watch?v=abc123xyz00",
        phase1_outputs=SimpleNamespace(phase1_audio={"video_gcs_uri": "gs://bucket/source.mp4"}),
        canonical_timeline=SimpleNamespace(turns=[]),
        shot_tracklet_index=SimpleNamespace(tracklets=[]),
        tracklet_geometry=SimpleNamespace(tracklets=[]),
        nodes=[],
        phase4_result={"candidates": [candidate]},
    )

    assert render_executor.calls
    assert result["artifact_paths"]["rendered_clip-1.mp4"].endswith("clip-1.mp4")


def test_phase14_live_uses_phase_specific_concurrency_knobs(monkeypatch, tmp_path: Path) -> None:
    from backend.runtime import phase14_live

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
        source_prompt_ids=["general_prompt_001"],
        seed_node_id="node-1",
        subgraph_id="sg-1",
        query_aligned=True,
        pool_rank=1,
        score_breakdown={"base_score": 1.0},
    )

    runner = V31LivePhase14Runner(
        config=V31Config(
            output_root=tmp_path,
            phase2_merge_max_concurrent=8,
            phase3_local_max_concurrent=8,
            phase4_subgraph_max_concurrent=10,
        ),
        llm_client=object(),
        embedding_client=object(),
        node_media_preparer=lambda **kwargs: [],
    )
    runner.config.signals.max_concurrent = 6

    seen: dict[str, int] = {}

    def _fake_phase2(**kwargs):
        seen["phase2_merge_max_concurrent"] = kwargs["max_concurrent"]
        return [node], [], []

    def _fake_local_edges(**kwargs):
        seen["phase3_local_max_concurrent"] = kwargs["max_concurrent"]
        return [], []

    def _fake_subgraph_reviews(**kwargs):
        seen["phase4_subgraph_max_concurrent"] = kwargs["max_concurrent"]
        return [SimpleNamespace(candidates=[candidate])], []

    monkeypatch.setattr(
        phase14_live,
        "run_merge_classify_and_reconcile",
        _fake_phase2,
    )
    monkeypatch.setattr(
        phase14_live,
        "embed_text_semantic_nodes_live",
        lambda **kwargs: (
            [[1.0, 0.0]],
            {"node_count": 1, "semantic_payload_chars": 5, "semantic_duration_ms": 0.0},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        phase14_live,
        "embed_multimodal_media_live",
        lambda **kwargs: (
            [[1.0, 0.0]],
            {"multimodal_item_count": 0, "multimodal_duration_ms": 0.0},
        ),
        raising=False,
    )
    runner.run_phase_2(
        paths=runner.build_run_paths(run_id="run_phase2"),
        phase1_outputs=SimpleNamespace(),
        canonical_timeline=SimpleNamespace(),
        speech_emotion_timeline=None,
        audio_event_timeline=None,
    )

    monkeypatch.setattr(
        phase14_live,
        "run_local_semantic_edge_batches",
        _fake_local_edges,
    )
    monkeypatch.setattr(
        phase14_live,
        "run_long_range_edge_adjudication",
        lambda **kwargs: ([], {"diagnostics": {}, "shards": []}),
    )

    runner.run_phase_3(
        paths=runner.build_run_paths(run_id="run_phase3"),
        nodes=[node],
        long_range_top_k=2,
    )

    monkeypatch.setattr(phase14_live, "generate_meta_prompts_live", lambda **kwargs: (["Find the strongest hook"], {}))
    monkeypatch.setattr(
        phase14_live,
        "embed_prompt_texts_live",
        lambda **kwargs: ([{"prompt_id": "general_prompt_001", "embedding": [1.0, 0.0]}], {"latency_ms": 0.0}),
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
                source_prompt_ids=["general_prompt_001"],
                start_ms=0,
                end_ms=1000,
                nodes=[SimpleNamespace(node_id="node-1")],
                model_dump=lambda mode="json": {"subgraph_id": "sg-1", "nodes": [{"node_id": "node-1"}]},
            )
        ],
    )
    monkeypatch.setattr(
        phase14_live,
        "run_subgraph_reviews",
        _fake_subgraph_reviews,
    )
    monkeypatch.setattr(phase14_live, "dedupe_clip_candidates", lambda candidates: candidates)
    monkeypatch.setattr(
        phase14_live,
        "run_candidate_pool_review_with_debug",
        lambda **kwargs: (
            SimpleNamespace(
                ranked_candidates=[
                    SimpleNamespace(
                        candidate_temp_id="clip-1",
                        pool_rank=1,
                        score=1.0,
                        score_breakdown={"base_score": 1.0},
                        rationale="pool",
                    )
                ]
            ),
            {
                "candidate_count": 1,
                "kept_candidate_count": 1,
                "dropped_candidate_count": 0,
                "max_pool_rank": 1,
                "latency_ms": 0.0,
                "prompt_chars": 0,
                "prompt_token_estimate": 0,
                "payload_chars": 0,
                "response_chars": 0,
            },
        ),
    )
    def _fake_node_signal_links(**kwargs):
        seen["signals_max_concurrent"] = kwargs["max_concurrent"]
        return []

    monkeypatch.setattr(phase14_live, "build_node_signal_links", _fake_node_signal_links)
    monkeypatch.setattr(
        phase14_live,
        "apply_signal_scoring",
        lambda **kwargs: SignalScoringResult(
            candidates=[candidate],
            candidate_signal_links=[],
        ),
    )

    runner.run_phase_4(
        run_id="run_phase4",
        job_id="job_1",
        attempt=1,
        paths=runner.build_run_paths(run_id="run_phase4"),
        source_url="https://youtube.com/watch?v=abc",
        canonical_timeline=SimpleNamespace(turns=[SimpleNamespace(end_ms=1000)]),
        nodes=[node],
        edges=[edge],
        extra_prompt_texts=[],
        signal_output=SignalPipelineOutput(),
    )

    assert seen["phase2_merge_max_concurrent"] == 8
    assert seen["phase3_local_max_concurrent"] == 8
    assert seen["phase4_subgraph_max_concurrent"] == 10
    assert seen["signals_max_concurrent"] == 6


def test_phase14_live_run_phase2_starts_semantic_embeddings_before_media_prep_finishes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from backend.runtime import phase14_live

    node = SemanticGraphNode(
        node_id="node-1",
        node_type="claim",
        start_ms=0,
        end_ms=1000,
        source_turn_ids=["t_1"],
        word_ids=["w_1"],
        transcript_text="hello",
        node_flags=[],
        summary="summary",
        evidence=SemanticNodeEvidence(),
        semantic_embedding=[0.0, 0.0],
        multimodal_embedding=[0.0, 0.0],
    )

    runner = V31LivePhase14Runner(
        config=V31Config(output_root=tmp_path),
        llm_client=object(),
        embedding_client=object(),
        storage_client=object(),
    )

    semantic_started = threading.Event()
    semantic_can_finish = threading.Event()

    monkeypatch.setattr(
        phase14_live,
        "run_merge_classify_and_reconcile",
        lambda **kwargs: ([node], [], []),
    )

    def _fake_prepare_media(**kwargs):
        assert semantic_started.wait(timeout=1), "semantic embedding did not start before media prep finished"
        semantic_can_finish.set()
        return [{"file_uri": "gs://bucket/node-1.mp4", "mime_type": "video/mp4"}]

    def _fake_embed_text_semantic_nodes_live(**kwargs):
        semantic_started.set()
        assert semantic_can_finish.wait(timeout=1), "media prep did not overlap semantic embedding"
        return [[0.1, 0.2]], {"node_count": 1, "semantic_payload_chars": 5, "semantic_duration_ms": 0.0}

    def _fake_embed_multimodal_media_live(**kwargs):
        return [[0.3, 0.4]], {"multimodal_item_count": 1, "multimodal_duration_ms": 0.0}

    monkeypatch.setattr(phase14_live, "prepare_node_media_embeddings", _fake_prepare_media)
    monkeypatch.setattr(
        phase14_live,
        "embed_text_semantic_nodes_live",
        _fake_embed_text_semantic_nodes_live,
        raising=False,
    )
    monkeypatch.setattr(
        phase14_live,
        "embed_multimodal_media_live",
        _fake_embed_multimodal_media_live,
        raising=False,
    )
    result = runner.run_phase_2(
        paths=runner.build_run_paths(run_id="run_phase2_overlap"),
        phase1_outputs=SimpleNamespace(phase1_audio={"local_video_path": "/tmp/source.mp4"}),
        canonical_timeline=SimpleNamespace(),
        speech_emotion_timeline=None,
        audio_event_timeline=None,
    )

    assert result["nodes"][0].semantic_embedding == [0.1, 0.2]
    assert result["nodes"][0].multimodal_embedding == [0.3, 0.4]


def test_phase14_live_run_phase2_pipelines_batch_embedding_and_preserves_node_order(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from backend.runtime import phase14_live

    nodes = [
        SemanticGraphNode(
            node_id="node-1",
            node_type="claim",
            start_ms=0,
            end_ms=1000,
            source_turn_ids=["t_1"],
            word_ids=["w_1"],
            transcript_text="hello",
            node_flags=[],
            summary="summary",
            evidence=SemanticNodeEvidence(),
            semantic_embedding=[0.0, 0.0],
            multimodal_embedding=[0.0, 0.0],
        ),
        SemanticGraphNode(
            node_id="node-2",
            node_type="claim",
            start_ms=1500,
            end_ms=2500,
            source_turn_ids=["t_2"],
            word_ids=["w_2"],
            transcript_text="world",
            node_flags=[],
            summary="summary",
            evidence=SemanticNodeEvidence(),
            semantic_embedding=[0.0, 0.0],
            multimodal_embedding=[0.0, 0.0],
        ),
        SemanticGraphNode(
            node_id="node-3",
            node_type="claim",
            start_ms=7000,
            end_ms=8000,
            source_turn_ids=["t_3"],
            word_ids=["w_3"],
            transcript_text="again",
            node_flags=[],
            summary="summary",
            evidence=SemanticNodeEvidence(),
            semantic_embedding=[0.0, 0.0],
            multimodal_embedding=[0.0, 0.0],
        ),
    ]

    runner = V31LivePhase14Runner(
        config=V31Config(output_root=tmp_path),
        llm_client=object(),
        embedding_client=object(),
        node_media_preparer=object(),
    )

    first_batch_embedding_started = threading.Event()

    class _BatchPreparer:
        def prepare_batch(self, *, nodes, paths, phase1_outputs, batch_id, object_prefix):
            if batch_id == "batch_0001":
                assert first_batch_embedding_started.wait(timeout=1), (
                    "first batch embedding did not start before second batch finished"
                )
            media = [
                {
                    "node_id": node.node_id,
                    "file_uri": f"gs://bucket/{batch_id}/{node.node_id}.mp4",
                    "mime_type": "video/mp4",
                    "local_path": "",
                }
                for node in nodes
            ]
            return SimpleNamespace(
                media=media,
                metadata={"batch_id": batch_id, "node_count": len(nodes)},
            )

    runner.node_media_preparer = _BatchPreparer()

    monkeypatch.setattr(
        phase14_live,
        "run_merge_classify_and_reconcile",
        lambda **kwargs: (nodes, [], []),
    )
    monkeypatch.setattr(
        phase14_live,
        "embed_text_semantic_nodes_live",
        lambda **kwargs: (
            [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
            {"node_count": 3, "semantic_payload_chars": 15, "semantic_duration_ms": 0.0},
        ),
        raising=False,
    )

    def _fake_embed_multimodal_media_live(*, multimodal_media, embedding_client):
        node_ids = [item["node_id"] for item in multimodal_media]
        if node_ids == ["node-1", "node-2"]:
            first_batch_embedding_started.set()
            return [[1.0, 1.0], [2.0, 2.0]], {"multimodal_item_count": 2, "multimodal_duration_ms": 5.0}
        return [[3.0, 3.0]], {"multimodal_item_count": 1, "multimodal_duration_ms": 5.0}

    monkeypatch.setattr(
        phase14_live,
        "embed_multimodal_media_live",
        _fake_embed_multimodal_media_live,
        raising=False,
    )

    result = runner.run_phase_2(
        paths=runner.build_run_paths(run_id="run_phase2_pipelined"),
        phase1_outputs=SimpleNamespace(phase1_audio={"video_gcs_uri": "gs://bucket/source.mp4"}),
        canonical_timeline=SimpleNamespace(),
        speech_emotion_timeline=None,
        audio_event_timeline=None,
    )

    assert [node.node_id for node in result["nodes"]] == ["node-1", "node-2", "node-3"]
    assert [node.multimodal_embedding for node in result["nodes"]] == [
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ]


def test_phase14_live_run_phase3_overlaps_local_and_long_range(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from backend.runtime import phase14_live

    node = SemanticGraphNode(
        node_id="node-1",
        node_type="claim",
        start_ms=0,
        end_ms=1000,
        source_turn_ids=["t_1"],
        word_ids=["w_1"],
        transcript_text="hello",
        node_flags=[],
        summary="summary",
        evidence=SemanticNodeEvidence(),
        semantic_embedding=[0.1, 0.2],
        multimodal_embedding=[0.3, 0.4],
    )

    runner = V31LivePhase14Runner(
        config=V31Config(output_root=tmp_path),
        llm_client=object(),
        embedding_client=object(),
    )

    local_started = threading.Event()
    long_range_started = threading.Event()

    def _fake_local_edges(**kwargs):
        local_started.set()
        assert long_range_started.wait(timeout=1), "long-range lane did not start before local-edge lane finished"
        return [], []

    def _fake_long_range(**kwargs):
        long_range_started.set()
        assert local_started.wait(timeout=1), "local-edge lane did not start before long-range lane finished"
        return [], {"diagnostics": {}, "shards": []}

    monkeypatch.setattr(phase14_live, "run_local_semantic_edge_batches", _fake_local_edges)
    monkeypatch.setattr(phase14_live, "run_long_range_edge_adjudication", _fake_long_range)

    result = runner.run_phase_3(
        paths=runner.build_run_paths(run_id="run_phase3_overlap"),
        nodes=[node],
        long_range_top_k=2,
    )

    assert "edges" in result
