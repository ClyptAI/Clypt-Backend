from __future__ import annotations

from pathlib import Path

from backend.pipeline.config import V31Config
from backend.pipeline.signals.contracts import SignalPipelineOutput, SignalPromptSpec
from backend.runtime.phase14_live import V31LivePhase14Runner


class _FakeRepository:
    def write_phase_metric(self, record):
        return record


class _Future:
    def __init__(self, output: SignalPipelineOutput, events: list[str], label: str) -> None:
        self._output = output
        self._events = events
        self._label = label

    def result(self):
        self._events.append(f"{self._label}:result")
        return self._output

    def cancel(self):
        self._events.append(f"{self._label}:cancel")
        return True


def _runner(tmp_path: Path) -> V31LivePhase14Runner:
    return V31LivePhase14Runner(
        config=V31Config(output_root=tmp_path),
        llm_client=object(),
        embedding_client=object(),
        repository=_FakeRepository(),
    )


def test_join_signal_outputs_waits_both_enabled_futures(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    events: list[str] = []
    comments_future = _Future(
        SignalPipelineOutput(prompt_specs=[SignalPromptSpec(prompt_id="c1", text="c", prompt_source_type="comment", source_cluster_id="cc", source_cluster_type="comment")]),
        events,
        "comments",
    )
    trends_future = _Future(
        SignalPipelineOutput(prompt_specs=[SignalPromptSpec(prompt_id="t1", text="t", prompt_source_type="trend", source_cluster_id="tc", source_cluster_type="trend")]),
        events,
        "trends",
    )

    comments_output, trends_output = runner._join_signal_outputs(
        run_id="run_001",
        job_id="job_1",
        attempt=1,
        comments_future=comments_future,
        comments_future_started_at=None,
        trends_future=trends_future,
        trends_future_started_at=None,
    )

    assert comments_output.prompt_specs[0].prompt_id == "c1"
    assert trends_output.prompt_specs[0].prompt_id == "t1"
    assert events == ["comments:result", "trends:result"]


def test_join_signal_outputs_waits_single_enabled_future(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    events: list[str] = []
    comments_future = _Future(SignalPipelineOutput(), events, "comments")

    comments_output, trends_output = runner._join_signal_outputs(
        run_id="run_001",
        job_id="job_1",
        attempt=1,
        comments_future=comments_future,
        comments_future_started_at=None,
        trends_future=None,
        trends_future_started_at=None,
    )

    assert comments_output == SignalPipelineOutput()
    assert trends_output == SignalPipelineOutput()
    assert events == ["comments:result"]
