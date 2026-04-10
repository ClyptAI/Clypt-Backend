from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from pathlib import Path


UTC = timezone.utc


def _phase_metric(*, run_id: str, phase_name: str, status: str, started_at: datetime) -> object:
    return SimpleNamespace(
        run_id=run_id,
        phase_name=phase_name,
        status=status,
        started_at=started_at,
        ended_at=started_at,
        duration_ms=1.0,
        error_payload=None,
        query_version="graph-v2",
        metadata={},
    )


class _FakeRepository:
    def __init__(self) -> None:
        self.phase_metrics: list[object] = []
        self.nodes: list[object] = []
        self.edges: list[object] = []
        self.candidates: list[object] = []
        self.written_phase_metrics: list[object] = []

    def list_phase_metrics(self, *, run_id: str) -> list[object]:
        return list(self.phase_metrics)

    def list_nodes(self, *, run_id: str) -> list[object]:
        return list(self.nodes)

    def list_edges(self, *, run_id: str) -> list[object]:
        return list(self.edges)

    def list_candidates(self, *, run_id: str) -> list[object]:
        return list(self.candidates)

    def write_phase_metric(self, record):
        self.written_phase_metrics.append(record)
        return record


def _build_runner(tmp_path: Path, repository: _FakeRepository, log_events: list[dict[str, object]]):
    from backend.pipeline.config import V31Config
    from backend.runtime.phase14_live import V31LivePhase14Runner

    return V31LivePhase14Runner(
        config=V31Config(output_root=tmp_path),
        llm_client=object(),
        embedding_client=object(),
        repository=repository,
        query_version="graph-v2",
        log_event=lambda **payload: log_events.append(payload),
    )


def _phase1_outputs() -> object:
    return SimpleNamespace(
        phase1_audio={"local_video_path": "/tmp/source_video.mp4"},
        diarization_payload={"turns": []},
        phase1_visual={"video_metadata": {"fps": 30.0}, "shot_changes": [], "tracks": []},
        emotion2vec_payload={"segments": []},
        yamnet_payload={"events": []},
    )


def _semantic_node(node_id: str):
    from backend.pipeline.contracts import SemanticGraphNode

    return SemanticGraphNode(
        node_id=node_id,
        node_type="claim",
        start_ms=0,
        end_ms=1000,
        source_turn_ids=["t_1"],
        word_ids=["w_1"],
        transcript_text="hello",
        node_flags=[],
        summary="hello",
        evidence={"emotion_labels": [], "audio_events": []},
        semantic_embedding=[0.1, 0.2],
        multimodal_embedding=[0.3, 0.4],
    )


def _semantic_edge():
    from backend.pipeline.contracts import SemanticGraphEdge

    return SemanticGraphEdge(
        source_node_id="node_1",
        target_node_id="node_2",
        edge_type="next_turn",
        rationale="adjacent",
        confidence=1.0,
        support_count=1,
        batch_ids=[],
    )


def _candidate(clip_id: str):
    from backend.pipeline.contracts import ClipCandidate

    return ClipCandidate(
        clip_id=clip_id,
        node_ids=["node_1"],
        start_ms=0,
        end_ms=1000,
        score=1.0,
        rationale="test",
    )


def test_live_phase14_runner_skips_phase2_when_phase2_already_succeeded(tmp_path: Path, monkeypatch):
    repository = _FakeRepository()
    repository.phase_metrics = [
        _phase_metric(run_id="run_001", phase_name="phase2", status="succeeded", started_at=datetime(2026, 4, 8, 12, 0, tzinfo=UTC))
    ]
    repository.nodes = [_semantic_node("node_1")]
    repository.edges = [_semantic_edge()]
    repository.candidates = [_candidate("clip_1")]
    log_events: list[dict[str, object]] = []
    runner = _build_runner(tmp_path, repository, log_events)

    def _run_phase_1(self, **kwargs):
        return {
            "canonical_timeline": SimpleNamespace(turns=[SimpleNamespace(end_ms=1000)]),
            "speech_emotion_timeline": SimpleNamespace(),
            "audio_event_timeline": SimpleNamespace(),
        }

    def _run_phase_2(self, **kwargs):
        raise AssertionError("phase2 should be skipped when it already succeeded")

    phase3_calls: list[dict[str, object]] = []

    def _run_phase_3(self, **kwargs):
        phase3_calls.append(kwargs)
        assert kwargs["nodes"] == repository.nodes
        return {"edges": repository.edges}

    phase4_calls: list[dict[str, object]] = []

    def _run_phase_4(self, **kwargs):
        phase4_calls.append(kwargs)
        assert kwargs["nodes"] == repository.nodes
        assert kwargs["edges"] == repository.edges
        return {"final_candidate_count": 1, "seed_count": 0, "subgraph_count": 0, "raw_candidate_count": 0, "deduped_candidate_count": 0}

    monkeypatch.setattr(type(runner), "run_phase_1", _run_phase_1)
    monkeypatch.setattr(type(runner), "run_phase_2", _run_phase_2)
    monkeypatch.setattr(type(runner), "run_phase_3", _run_phase_3)
    monkeypatch.setattr(type(runner), "run_phase_4", _run_phase_4)

    summary = runner.run(
        run_id="run_001",
        source_url="https://example.com/video",
        phase1_outputs=_phase1_outputs(),
    )

    assert summary.metadata["candidate_count"] == 1
    assert phase3_calls and phase4_calls
    skip_events = [event for event in log_events if event["event"] == "phase_skipped_resume"]
    assert any(event["phase"] == "phase2" and event["reason"] == "phase2_already_succeeded" for event in skip_events)


def test_live_phase14_runner_skips_phase2_and_phase3_when_phase3_already_succeeded(tmp_path: Path, monkeypatch):
    repository = _FakeRepository()
    repository.phase_metrics = [
        _phase_metric(run_id="run_002", phase_name="phase2", status="succeeded", started_at=datetime(2026, 4, 8, 12, 0, tzinfo=UTC)),
        _phase_metric(run_id="run_002", phase_name="phase3", status="succeeded", started_at=datetime(2026, 4, 8, 12, 1, tzinfo=UTC)),
    ]
    repository.nodes = [_semantic_node("node_2")]
    repository.edges = [_semantic_edge()]
    log_events: list[dict[str, object]] = []
    runner = _build_runner(tmp_path, repository, log_events)

    def _run_phase_1(self, **kwargs):
        return {
            "canonical_timeline": SimpleNamespace(turns=[SimpleNamespace(end_ms=1000)]),
            "speech_emotion_timeline": SimpleNamespace(),
            "audio_event_timeline": SimpleNamespace(),
        }

    def _run_phase_2(self, **kwargs):
        raise AssertionError("phase2 should be skipped when phase3 already succeeded")

    def _run_phase_3(self, **kwargs):
        raise AssertionError("phase3 should be skipped when phase3 already succeeded")

    phase4_calls: list[dict[str, object]] = []

    def _run_phase_4(self, **kwargs):
        phase4_calls.append(kwargs)
        assert kwargs["nodes"] == repository.nodes
        assert kwargs["edges"] == repository.edges
        return {"final_candidate_count": 2, "seed_count": 0, "subgraph_count": 0, "raw_candidate_count": 0, "deduped_candidate_count": 0}

    monkeypatch.setattr(type(runner), "run_phase_1", _run_phase_1)
    monkeypatch.setattr(type(runner), "run_phase_2", _run_phase_2)
    monkeypatch.setattr(type(runner), "run_phase_3", _run_phase_3)
    monkeypatch.setattr(type(runner), "run_phase_4", _run_phase_4)

    summary = runner.run(
        run_id="run_002",
        source_url="https://example.com/video",
        phase1_outputs=_phase1_outputs(),
    )

    assert summary.metadata["candidate_count"] == 2
    assert phase4_calls
    skip_events = [event for event in log_events if event["event"] == "phase_skipped_resume"]
    assert any(event["phase"] == "phase2" for event in skip_events)
    assert any(event["phase"] == "phase3" for event in skip_events)


def test_live_phase14_runner_short_circuits_when_phase4_already_succeeded(tmp_path: Path, monkeypatch):
    repository = _FakeRepository()
    repository.phase_metrics = [
        _phase_metric(run_id="run_003", phase_name="phase2", status="succeeded", started_at=datetime(2026, 4, 8, 12, 0, tzinfo=UTC)),
        _phase_metric(run_id="run_003", phase_name="phase3", status="succeeded", started_at=datetime(2026, 4, 8, 12, 1, tzinfo=UTC)),
        _phase_metric(run_id="run_003", phase_name="phase4", status="succeeded", started_at=datetime(2026, 4, 8, 12, 2, tzinfo=UTC)),
    ]
    repository.nodes = [_semantic_node("node_3")]
    repository.edges = [_semantic_edge()]
    repository.candidates = [_candidate("clip_3"), _candidate("clip_4")]
    log_events: list[dict[str, object]] = []
    runner = _build_runner(tmp_path, repository, log_events)

    def _unexpected_call(self, **kwargs):
        raise AssertionError("heavy phase should be skipped when phase4 already succeeded")

    monkeypatch.setattr(type(runner), "run_phase_1", _unexpected_call)
    monkeypatch.setattr(type(runner), "run_phase_2", _unexpected_call)
    monkeypatch.setattr(type(runner), "run_phase_3", _unexpected_call)
    monkeypatch.setattr(type(runner), "run_phase_4", _unexpected_call)

    summary = runner.run(
        run_id="run_003",
        source_url="https://example.com/video",
        phase1_outputs=_phase1_outputs(),
    )

    assert summary.metadata["candidate_count"] == 2
    skip_events = [event for event in log_events if event["event"] == "phase_skipped_resume"]
    assert {event["phase"] for event in skip_events} >= {"phase2", "phase3", "phase4"}
