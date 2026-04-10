from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

from backend.pipeline.signals.contracts import SignalPipelineOutput, SignalPromptSpec
from backend.pipeline.signals.runtime import (
    collect_signal_outputs,
    merge_signal_outputs,
    start_comments_future,
    start_trends_future,
    wait_enabled_signal_futures,
)


class _FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    def submit(self, fn, /, *args, **kwargs):  # noqa: ANN001
        self.calls.append((fn, args, kwargs))
        future: Future[SignalPipelineOutput] = Future()
        future.set_result(SignalPipelineOutput())
        return future


def test_start_comments_and_trends_future_respect_enable_flags() -> None:
    executor = _FakeExecutor()
    disabled_cfg = SimpleNamespace(enable_comment_signals=False, enable_trend_signals=False)

    assert start_comments_future(
        executor=executor,
        cfg=disabled_cfg,
        llm_client=object(),
        embedding_client=object(),
        source_url="https://youtube.com/watch?v=abc",
    ) is None
    assert start_trends_future(
        executor=executor,
        cfg=disabled_cfg,
        llm_client=object(),
        embedding_client=object(),
        nodes=[],
        source_url="https://youtube.com/watch?v=abc",
    ) is None
    assert executor.calls == []

    enabled_cfg = SimpleNamespace(enable_comment_signals=True, enable_trend_signals=True)

    comments_future = start_comments_future(
        executor=executor,
        cfg=enabled_cfg,
        llm_client="llm",
        embedding_client="embed",
        source_url="https://youtube.com/watch?v=abc",
    )
    trends_future = start_trends_future(
        executor=executor,
        cfg=enabled_cfg,
        llm_client="llm",
        embedding_client="embed",
        nodes=["node-1"],
        source_url="https://youtube.com/watch?v=abc",
    )

    assert comments_future is not None
    assert trends_future is not None
    assert len(executor.calls) == 2


def test_wait_enabled_signal_futures_supports_one_or_both_futures() -> None:
    comments_future: Future[SignalPipelineOutput] = Future()
    comments_future.set_result(
        SignalPipelineOutput(metadata={"source": "comments"}, prompt_specs=[SignalPromptSpec(prompt_id="c", text="c", prompt_source_type="general")])
    )
    trends_future: Future[SignalPipelineOutput] = Future()
    trends_future.set_result(SignalPipelineOutput(metadata={"source": "trends"}))

    comments_output, trends_output = wait_enabled_signal_futures(
        comments_future=comments_future,
        trends_future=None,
    )
    assert comments_output.metadata["source"] == "comments"
    assert trends_output.prompt_specs == []

    comments_output, trends_output = wait_enabled_signal_futures(
        comments_future=comments_future,
        trends_future=trends_future,
    )
    assert comments_output.metadata["source"] == "comments"
    assert trends_output.metadata["source"] == "trends"


def test_collect_signal_outputs_and_merge_degrade_to_general_only() -> None:
    comments_future: Future[SignalPipelineOutput] = Future()
    comments_future.set_result(SignalPipelineOutput(metadata={"comments": True}))
    trends_future: Future[SignalPipelineOutput] = Future()
    trends_future.set_result(SignalPipelineOutput(metadata={"trends": True}))

    merged = collect_signal_outputs(comments_future=comments_future, trends_future=trends_future)
    assert merged.external_signals == []
    assert merged.clusters == []
    assert merged.prompt_specs == []
    assert merged.metadata["comment_cluster_count"] == 0
    assert merged.metadata["trend_cluster_count"] == 0

    empty_merged = merge_signal_outputs(
        comments=SignalPipelineOutput(),
        trends=SignalPipelineOutput(),
    )
    assert empty_merged.prompt_specs == []
    assert empty_merged.external_signals == []
    assert empty_merged.clusters == []

