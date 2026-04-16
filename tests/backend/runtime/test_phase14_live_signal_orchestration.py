from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.pipeline.signals.contracts import SignalPipelineOutput, SignalPromptSpec


@dataclass(slots=True)
class _FakeFuture:
    result_value: object
    events: list[str]
    label: str

    def result(self):
        self.events.append(f"{self.label}_result")
        return self.result_value

    def cancel(self):
        self.events.append(f"{self.label}_cancel")
        return True


class _FakeRepository:
    def write_phase_metric(self, record):
        return record


def _phase1_outputs() -> object:
    return SimpleNamespace(
        phase1_audio={"local_video_path": "/tmp/source_video.mp4"},
        diarization_payload={"turns": []},
        phase1_visual={"video_metadata": {"fps": 30.0}, "shot_changes": [], "tracks": []},
        emotion2vec_payload={"segments": []},
        yamnet_payload={"events": []},
    )


def _build_runner(
    tmp_path: Path,
    repository: object | None = None,
    *,
    enable_comment_signals: bool = False,
    enable_trend_signals: bool = False,
):
    from backend.pipeline.config import V31Config
    from backend.runtime.phase14_live import V31LivePhase14Runner

    runner = V31LivePhase14Runner(
        config=V31Config(output_root=tmp_path),
        llm_client=object(),
        embedding_client=object(),
        repository=repository,
        query_version="graph-v2",
    )
    runner.config.signals.enable_comment_signals = enable_comment_signals
    runner.config.signals.enable_trend_signals = enable_trend_signals
    return runner


def test_phase14_runner_starts_comments_before_phase1_and_trends_after_phase2(tmp_path: Path, monkeypatch):
    from backend.runtime import phase14_live

    runner = _build_runner(
        tmp_path,
        _FakeRepository(),
        enable_comment_signals=True,
        enable_trend_signals=True,
    )
    events: list[str] = []

    def fake_start_comments_future(*, executor, cfg, llm_client, embedding_client, source_url, signal_event_logger=None):
        events.append("comments_submit")
        return _FakeFuture(
            SignalPipelineOutput(
                prompt_specs=[
                    SignalPromptSpec(
                        prompt_id="comment_prompt_001",
                        text="Find the audience joke callback.",
                        prompt_source_type="comment",
                        source_cluster_id="cluster_comment_1",
                        source_cluster_type="comment",
                    )
                ]
            ),
            events,
            "comments",
        )

    def fake_start_trends_future(*, executor, cfg, llm_client, embedding_client, nodes, source_url, signal_event_logger=None):
        events.append("trends_submit")
        return _FakeFuture(
            SignalPipelineOutput(
                prompt_specs=[
                    SignalPromptSpec(
                        prompt_id="trend_prompt_001",
                        text="Find the trending topic reference.",
                        prompt_source_type="trend",
                        source_cluster_id="cluster_trend_1",
                        source_cluster_type="trend",
                    )
                ]
            ),
            events,
            "trends",
        )

    def fake_run_phase_1(self, **kwargs):
        events.append("phase1")
        return {
            "canonical_timeline": SimpleNamespace(turns=[SimpleNamespace(end_ms=1000)]),
            "speech_emotion_timeline": SimpleNamespace(),
            "audio_event_timeline": SimpleNamespace(),
        }

    def fake_run_phase_2(self, **kwargs):
        events.append("phase2")
        return {"nodes": [SimpleNamespace(node_id="node_1")]}

    def fake_run_phase_3(self, **kwargs):
        events.append("phase3")
        return {"edges": []}

    def fake_run_phase_4(self, **kwargs):
        events.append("phase4")
        assert len(kwargs["signal_output"].prompt_specs) == 2
        return {
            "final_candidate_count": 0,
            "seed_count": 0,
            "subgraph_count": 0,
            "raw_candidate_count": 0,
            "deduped_candidate_count": 0,
        }

    monkeypatch.setattr(phase14_live, "start_comments_future", fake_start_comments_future)
    monkeypatch.setattr(phase14_live, "start_trends_future", fake_start_trends_future)
    monkeypatch.setattr(type(runner), "run_phase_1", fake_run_phase_1)
    monkeypatch.setattr(type(runner), "run_phase_2", fake_run_phase_2)
    monkeypatch.setattr(type(runner), "run_phase_3", fake_run_phase_3)
    monkeypatch.setattr(type(runner), "run_phase_4", fake_run_phase_4)

    runner.run(
        run_id="run_001",
        source_url="https://www.youtube.com/watch?v=abc123",
        phase1_outputs=_phase1_outputs(),
    )

    assert events[:4] == ["comments_submit", "phase1", "phase2", "trends_submit"]
    assert events.index("comments_result") > events.index("phase3")
    assert events.index("trends_result") > events.index("phase3")
    assert events.index("phase4") > events.index("trends_result")


def test_phase14_runner_starts_phase3_local_lane_before_phase2_finishes(tmp_path: Path, monkeypatch):
    from backend.runtime import phase14_live

    runner = _build_runner(
        tmp_path,
        _FakeRepository(),
        enable_comment_signals=True,
        enable_trend_signals=True,
    )
    events: list[str] = []

    def fake_start_comments_future(*, executor, cfg, llm_client, embedding_client, source_url, signal_event_logger=None):
        events.append("comments_submit")
        return _FakeFuture(SignalPipelineOutput(), events, "comments")

    def fake_start_trends_future(*, executor, cfg, llm_client, embedding_client, nodes, source_url, signal_event_logger=None):
        events.append("trends_submit")
        return _FakeFuture(SignalPipelineOutput(), events, "trends")

    def fake_run_phase_1(self, **kwargs):
        events.append("phase1")
        return {
            "canonical_timeline": SimpleNamespace(turns=[SimpleNamespace(end_ms=1000)]),
            "speech_emotion_timeline": SimpleNamespace(),
            "audio_event_timeline": SimpleNamespace(),
        }

    def fake_run_phase_2(self, **kwargs):
        events.append("phase2")
        raw_nodes_ready_callback = kwargs["raw_nodes_ready_callback"]
        raw_nodes_ready_callback([SimpleNamespace(node_id="node_1", start_ms=0, end_ms=1000)])
        return {"nodes": [SimpleNamespace(node_id="node_1")]}

    def fake_phase3_local_lane(self, **kwargs):
        events.append("phase3_local_prefetch")
        return [], [], 0.0

    def fake_run_phase_3(self, **kwargs):
        events.append("phase3")
        return {"edges": []}

    def fake_run_phase_4(self, **kwargs):
        events.append("phase4")
        return {
            "final_candidate_count": 0,
            "seed_count": 0,
            "subgraph_count": 0,
            "raw_candidate_count": 0,
            "deduped_candidate_count": 0,
        }

    monkeypatch.setattr(phase14_live, "start_comments_future", fake_start_comments_future)
    monkeypatch.setattr(phase14_live, "start_trends_future", fake_start_trends_future)
    monkeypatch.setattr(type(runner), "run_phase_1", fake_run_phase_1)
    monkeypatch.setattr(type(runner), "run_phase_2", fake_run_phase_2)
    monkeypatch.setattr(type(runner), "_run_phase_3_local_lane", fake_phase3_local_lane, raising=False)
    monkeypatch.setattr(type(runner), "run_phase_3", fake_run_phase_3)
    monkeypatch.setattr(type(runner), "run_phase_4", fake_run_phase_4)

    runner.run(
        run_id="run_overlap_001",
        source_url="https://www.youtube.com/watch?v=abc123",
        phase1_outputs=_phase1_outputs(),
    )

    assert events.index("phase3_local_prefetch") < events.index("trends_submit")
    assert events.index("phase3_local_prefetch") < events.index("phase3")


def test_phase14_runner_uses_general_prompts_only_when_augmentation_is_empty(tmp_path: Path, monkeypatch):
    from backend.runtime import phase14_live

    runner = _build_runner(
        tmp_path,
        _FakeRepository(),
        enable_comment_signals=True,
        enable_trend_signals=True,
    )
    captured: dict[str, object] = {}

    def fake_start_comments_future(*, executor, cfg, llm_client, embedding_client, source_url, signal_event_logger=None):
        return _FakeFuture(SignalPipelineOutput(), [], "comments")

    def fake_start_trends_future(*, executor, cfg, llm_client, embedding_client, nodes, source_url, signal_event_logger=None):
        return _FakeFuture(SignalPipelineOutput(), [], "trends")

    def fake_run_phase_1(self, **kwargs):
        return {
            "canonical_timeline": SimpleNamespace(turns=[SimpleNamespace(end_ms=1000)]),
            "speech_emotion_timeline": SimpleNamespace(),
            "audio_event_timeline": SimpleNamespace(),
        }

    def fake_run_phase_2(self, **kwargs):
        return {"nodes": [SimpleNamespace(node_id="node_1")]}

    def fake_run_phase_3(self, **kwargs):
        return {"edges": []}

    def fake_run_phase_4(self, **kwargs):
        captured["signal_output"] = kwargs["signal_output"]
        return {
            "final_candidate_count": 0,
            "seed_count": 0,
            "subgraph_count": 0,
            "raw_candidate_count": 0,
            "deduped_candidate_count": 0,
        }

    monkeypatch.setattr(phase14_live, "start_comments_future", fake_start_comments_future)
    monkeypatch.setattr(phase14_live, "start_trends_future", fake_start_trends_future)
    monkeypatch.setattr(type(runner), "run_phase_1", fake_run_phase_1)
    monkeypatch.setattr(type(runner), "run_phase_2", fake_run_phase_2)
    monkeypatch.setattr(type(runner), "run_phase_3", fake_run_phase_3)
    monkeypatch.setattr(type(runner), "run_phase_4", fake_run_phase_4)

    runner.run(
        run_id="run_002",
        source_url="https://www.youtube.com/watch?v=abc123",
        phase1_outputs=_phase1_outputs(),
    )

    signal_output = captured["signal_output"]
    assert isinstance(signal_output, SignalPipelineOutput)
    assert signal_output.external_signals == []
    assert signal_output.clusters == []
    assert signal_output.prompt_specs == []


def test_phase14_runner_builds_prompt_source_links_and_subgraph_provenance(tmp_path: Path):
    runner = _build_runner(tmp_path, _FakeRepository())
    from backend.pipeline.signals.contracts import SignalPromptSpec

    prompt_specs = [
        SignalPromptSpec(
            prompt_id="general_prompt_001",
            text="Find the strongest hook.",
            prompt_source_type="general",
        ),
        SignalPromptSpec(
            prompt_id="comment_prompt_001",
            text="Find the audience callback.",
            prompt_source_type="comment",
            source_cluster_id="cluster_comment_1",
            source_cluster_type="comment",
        ),
        SignalPromptSpec(
            prompt_id="trend_prompt_001",
            text="Find the trend reference.",
            prompt_source_type="trend",
            source_cluster_id="cluster_trend_1",
            source_cluster_type="trend",
        ),
    ]
    prompt_links = runner._build_prompt_source_links(run_id="run_003", prompt_specs=prompt_specs)
    assert [link.prompt_id for link in prompt_links] == [
        "general_prompt_001",
        "comment_prompt_001",
        "trend_prompt_001",
    ]
    assert [link.prompt_source_type for link in prompt_links] == ["general", "comment", "trend"]
    assert prompt_links[1].source_cluster_id == "cluster_comment_1"
    assert prompt_links[2].source_cluster_type == "trend"

    subgraph = SimpleNamespace(
        subgraph_id="sg_1",
        seed_node_id="node_1",
        source_prompt_ids=["general_prompt_001", "comment_prompt_001", "trend_prompt_001"],
        nodes=[SimpleNamespace(node_id="node_1"), SimpleNamespace(node_id="node_2")],
    )
    provenance = runner._build_subgraph_provenance(
        run_id="run_003",
        subgraphs=[subgraph],
        prompt_specs=prompt_specs,
    )
    assert provenance[0].seed_source_set == ["general", "comment", "trend"]
    assert provenance[0].seed_prompt_ids == [
        "general_prompt_001",
        "comment_prompt_001",
        "trend_prompt_001",
    ]
    assert provenance[0].source_cluster_ids == ["cluster_comment_1", "cluster_trend_1"]
    assert provenance[0].support_summary["source_type_counts"] == {"general": 1, "comment": 1, "trend": 1}


def test_phase14_runner_cancels_comments_future_when_phase2_fails(tmp_path: Path, monkeypatch):
    from backend.runtime import phase14_live

    runner = _build_runner(
        tmp_path,
        _FakeRepository(),
        enable_comment_signals=True,
        enable_trend_signals=True,
    )
    events: list[str] = []

    def fake_start_comments_future(*, executor, cfg, llm_client, embedding_client, source_url, signal_event_logger=None):
        events.append("comments_submit")
        return _FakeFuture(SignalPipelineOutput(), events, "comments")

    def fake_start_trends_future(*, executor, cfg, llm_client, embedding_client, nodes, source_url, signal_event_logger=None):
        events.append("trends_submit")
        return _FakeFuture(SignalPipelineOutput(), events, "trends")

    def fake_run_phase_1(self, **kwargs):
        return {
            "canonical_timeline": SimpleNamespace(turns=[SimpleNamespace(end_ms=1000)]),
            "speech_emotion_timeline": SimpleNamespace(),
            "audio_event_timeline": SimpleNamespace(),
        }

    def fake_run_phase_2(self, **kwargs):
        raise RuntimeError("phase2 boom")

    monkeypatch.setattr(phase14_live, "start_comments_future", fake_start_comments_future)
    monkeypatch.setattr(phase14_live, "start_trends_future", fake_start_trends_future)
    monkeypatch.setattr(type(runner), "run_phase_1", fake_run_phase_1)
    monkeypatch.setattr(type(runner), "run_phase_2", fake_run_phase_2)

    with pytest.raises(RuntimeError, match="phase2 boom"):
        runner.run(
            run_id="run_cancel_001",
            source_url="https://www.youtube.com/watch?v=abc123",
            phase1_outputs=_phase1_outputs(),
        )

    assert "comments_cancel" in events
    assert "trends_submit" not in events


def test_phase14_runner_cancels_comments_and_trends_when_phase3_fails(tmp_path: Path, monkeypatch):
    from backend.runtime import phase14_live

    runner = _build_runner(
        tmp_path,
        _FakeRepository(),
        enable_comment_signals=True,
        enable_trend_signals=True,
    )
    events: list[str] = []

    def fake_start_comments_future(*, executor, cfg, llm_client, embedding_client, source_url, signal_event_logger=None):
        events.append("comments_submit")
        return _FakeFuture(SignalPipelineOutput(), events, "comments")

    def fake_start_trends_future(*, executor, cfg, llm_client, embedding_client, nodes, source_url, signal_event_logger=None):
        events.append("trends_submit")
        return _FakeFuture(SignalPipelineOutput(), events, "trends")

    def fake_run_phase_1(self, **kwargs):
        return {
            "canonical_timeline": SimpleNamespace(turns=[SimpleNamespace(end_ms=1000)]),
            "speech_emotion_timeline": SimpleNamespace(),
            "audio_event_timeline": SimpleNamespace(),
        }

    def fake_run_phase_2(self, **kwargs):
        return {"nodes": [SimpleNamespace(node_id="node_1")]}

    def fake_run_phase_3(self, **kwargs):
        raise RuntimeError("phase3 boom")

    monkeypatch.setattr(phase14_live, "start_comments_future", fake_start_comments_future)
    monkeypatch.setattr(phase14_live, "start_trends_future", fake_start_trends_future)
    monkeypatch.setattr(type(runner), "run_phase_1", fake_run_phase_1)
    monkeypatch.setattr(type(runner), "run_phase_2", fake_run_phase_2)
    monkeypatch.setattr(type(runner), "run_phase_3", fake_run_phase_3)

    with pytest.raises(RuntimeError, match="phase3 boom"):
        runner.run(
            run_id="run_cancel_002",
            source_url="https://www.youtube.com/watch?v=abc123",
            phase1_outputs=_phase1_outputs(),
        )

    assert "trends_submit" in events
    assert "comments_cancel" in events
    assert "trends_cancel" in events
