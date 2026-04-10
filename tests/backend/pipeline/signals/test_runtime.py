from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

from backend.pipeline.signals.contracts import ExternalSignal, ExternalSignalCluster
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


def test_build_comments_output_uses_callpoint10_consolidated_text_for_downstream(monkeypatch) -> None:
    from backend.pipeline.signals import runtime as signal_runtime

    class _FakeCommentsClient:
        def __init__(self, api_key: str, base_url: str) -> None:
            self.api_key = api_key
            self.base_url = base_url

        def fetch_threads(self, *, video_id: str, order: str, max_pages: int):
            return (
                [
                    {
                        "id": "thread-1",
                        "snippet": {
                            "topLevelComment": {
                                "id": "top-1",
                                "snippet": {
                                    "textDisplay": "TOP COMMENT TEXT",
                                },
                            },
                            "totalReplyCount": 1,
                        },
                    }
                ],
                1,
            )

        def fetch_replies(self, *, parent_id: str, max_replies: int):
            return [{"id": "reply-1", "snippet": {"textDisplay": "reply"}}]

    seen_texts: list[str] = []

    monkeypatch.setattr(signal_runtime, "YouTubeCommentsClient", _FakeCommentsClient)
    monkeypatch.setattr(signal_runtime, "resolve_youtube_video_id", lambda _: "video-1")
    monkeypatch.setattr(signal_runtime, "collapse_same_author_spam", lambda items: items)
    monkeypatch.setattr(signal_runtime, "target_top_threads", lambda **kwargs: 1)
    monkeypatch.setattr(
        signal_runtime,
        "consolidate_thread_with_llm",
        lambda **kwargs: {
            "thread_summary": "summary mentions callback",
            "moment_hints": ["hint one", "hint two"],
        },
    )
    monkeypatch.setattr(
        signal_runtime,
        "to_external_signals_from_threads",
        lambda **kwargs: [
            ExternalSignal(
                signal_id="comment_top:top-1",
                signal_type="comment_top",
                source_platform="youtube",
                source_id="top-1",
                text="RAW TOP TEXT",
                engagement_score=1.0,
                metadata={"thread_id": "thread-1"},
            )
        ],
    )

    def _classify(**kwargs):
        seen_texts.append(kwargs["signal"].text)
        return {"quality": "high_signal", "reason": "good"}

    monkeypatch.setattr(signal_runtime, "classify_comment_with_llm", _classify)
    monkeypatch.setattr(
        signal_runtime,
        "cluster_signals",
        lambda **kwargs: [
            ExternalSignalCluster(
                cluster_id="comment_cluster_001",
                cluster_type="comment",
                summary_text="cluster",
                member_signal_ids=["comment_top:top-1"],
                cluster_weight=0.0,
                embedding=[1.0, 0.0],
                metadata={},
            )
        ],
    )
    monkeypatch.setattr(signal_runtime, "generate_cluster_prompt_with_llm", lambda **kwargs: "Find callback")

    class _EmbedClient:
        def embed_texts(self, texts, task_type: str):
            assert task_type == "RETRIEVAL_QUERY"
            return [[1.0, 0.0] for _ in texts]

    cfg = SimpleNamespace(
        llm_fail_fast=True,
        youtube_api_key="key",
        youtube_base_url="https://example.test",
        comment_order="relevance",
        max_comment_pages=1,
        comment_top_threads_min=1,
        comment_top_threads_max=40,
        comment_max_replies_per_thread=0,
        comment_cluster_sim_threshold=0.82,
        llm=SimpleNamespace(
            model_10="gemini-3-flash",
            thinking_10="minimal",
            model_3="gemini-3.1-flash-lite",
            thinking_3="low",
            model_1="gemini-3-flash",
            thinking_1="low",
        ),
    )

    output = signal_runtime._build_comments_output(
        cfg=cfg,
        llm_client=object(),
        embedding_client=_EmbedClient(),
        source_url="https://www.youtube.com/watch?v=abc123",
        signal_event_logger=None,
    )

    assert len(output.external_signals) == 1
    assert seen_texts
    assert "Top comment: TOP COMMENT TEXT" in seen_texts[0]
    assert "Thread summary: summary mentions callback" in seen_texts[0]
    assert output.external_signals[0].metadata["raw_text"] == "RAW TOP TEXT"
