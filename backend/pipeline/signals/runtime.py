from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from backend.pipeline.contracts import SemanticGraphNode

from .cluster import cluster_signals
from .comments_client import (
    YouTubeCommentsClient,
    collapse_same_author_spam,
    resolve_youtube_video_id,
    target_top_threads,
    to_external_signals_from_threads,
)
from .contracts import ExternalSignal, SignalPipelineOutput, SignalPromptSpec
from .llm_runtime import (
    adjudicate_trend_relevance_with_llm,
    classify_comment_with_llm,
    consolidate_thread_with_llm,
    generate_cluster_prompt_with_llm,
    synthesize_trend_queries_with_llm,
)
from .trends_client import TrendSpygClient, to_external_signals_from_trends


def start_comments_future(
    *,
    executor: ThreadPoolExecutor,
    cfg: Any,
    llm_client: Any,
    embedding_client: Any,
    source_url: str,
) -> Future[SignalPipelineOutput] | None:
    if not bool(cfg.enable_comment_signals):
        return None
    return executor.submit(
        _build_comments_output,
        cfg=cfg,
        llm_client=llm_client,
        embedding_client=embedding_client,
        source_url=source_url,
    )


def start_trends_future(
    *,
    executor: ThreadPoolExecutor,
    cfg: Any,
    llm_client: Any,
    embedding_client: Any,
    nodes: list[SemanticGraphNode],
    source_url: str,
) -> Future[SignalPipelineOutput] | None:
    if not bool(cfg.enable_trend_signals):
        return None
    return executor.submit(
        _build_trends_output,
        cfg=cfg,
        llm_client=llm_client,
        embedding_client=embedding_client,
        nodes=nodes,
        source_url=source_url,
    )


def wait_enabled_signal_futures(
    *,
    comments_future: Future[SignalPipelineOutput] | None,
    trends_future: Future[SignalPipelineOutput] | None,
) -> tuple[SignalPipelineOutput, SignalPipelineOutput]:
    comments_output = SignalPipelineOutput()
    trends_output = SignalPipelineOutput()
    if comments_future is not None:
        comments_output = comments_future.result()
    if trends_future is not None:
        trends_output = trends_future.result()
    return comments_output, trends_output


def merge_signal_outputs(*, comments: SignalPipelineOutput, trends: SignalPipelineOutput) -> SignalPipelineOutput:
    return SignalPipelineOutput(
        external_signals=[*comments.external_signals, *trends.external_signals],
        clusters=[*comments.clusters, *trends.clusters],
        prompt_specs=[*comments.prompt_specs, *trends.prompt_specs],
        metadata={
            **(comments.metadata or {}),
            **(trends.metadata or {}),
            "comment_cluster_count": len(comments.clusters),
            "trend_cluster_count": len(trends.clusters),
        },
    )


def collect_signal_outputs(
    *,
    comments_future: Future[SignalPipelineOutput] | None,
    trends_future: Future[SignalPipelineOutput] | None,
) -> SignalPipelineOutput:
    comments_output, trends_output = wait_enabled_signal_futures(
        comments_future=comments_future,
        trends_future=trends_future,
    )
    return merge_signal_outputs(comments=comments_output, trends=trends_output)


def _build_comments_output(
    *,
    cfg: Any,
    llm_client: Any,
    embedding_client: Any,
    source_url: str,
) -> SignalPipelineOutput:
    if not bool(cfg.llm_fail_fast):
        raise ValueError("CLYPT_SIGNAL_LLM_FAIL_FAST must be enabled for signal pipelines")
    video_id = resolve_youtube_video_id(source_url)
    if not video_id:
        raise ValueError("unable to resolve youtube_video_id from source URL for comments signal pipeline")
    if not cfg.youtube_api_key:
        raise ValueError("CLYPT_YOUTUBE_DATA_API_KEY (or YOUTUBE_API_KEY) is required when comments signals are enabled")

    comments_client = YouTubeCommentsClient(api_key=cfg.youtube_api_key, base_url=cfg.youtube_base_url)
    threads, total_threads = comments_client.fetch_threads(
        video_id=video_id,
        order=cfg.comment_order,
        max_pages=cfg.max_comment_pages,
    )
    threads = collapse_same_author_spam(threads)
    top_threads_count = target_top_threads(
        total_threads=total_threads,
        min_threads=cfg.comment_top_threads_min,
        max_threads=cfg.comment_top_threads_max,
    )
    selected_threads = list(threads[:top_threads_count])

    # Fetch full replies and enrich each selected thread.
    for thread in selected_threads:
        top_comment = (thread.get("snippet") or {}).get("topLevelComment") or {}
        parent_id = str(top_comment.get("id") or "")
        if not parent_id:
            continue
        full_replies = comments_client.fetch_replies(
            parent_id=parent_id,
            max_replies=cfg.comment_max_replies_per_thread,
        )
        thread["replies"] = {"comments": full_replies}
    replies_total = sum(
        len(((thread.get("replies") or {}).get("comments") or []))
        for thread in selected_threads
    )

    # Callpoint #10: per-thread consolidation.
    thread_intents: list[dict[str, Any]] = []
    for thread in selected_threads:
        consolidated = consolidate_thread_with_llm(
            llm_client=llm_client,
            model=cfg.llm.model_10,
            thinking_level=cfg.llm.thinking_10,
            thread_payload=thread,
            fail_fast=cfg.llm_fail_fast,
        )
        thread_intents.append(consolidated)

    signals = to_external_signals_from_threads(thread_items=selected_threads, include_replies=True)

    # Callpoint #3: quality filter.
    filtered_signals: list[ExternalSignal] = []
    for signal in signals:
        classified = classify_comment_with_llm(
            llm_client=llm_client,
            model=cfg.llm.model_3,
            thinking_level=cfg.llm.thinking_3,
            signal=signal,
            fail_fast=cfg.llm_fail_fast,
        )
        quality = str(classified.get("quality") or "").strip().lower()
        signal.metadata["quality"] = quality
        signal.metadata["quality_reason"] = str(classified.get("reason") or "")
        if quality in {"spam", "low_signal"}:
            continue
        filtered_signals.append(signal)

    if not filtered_signals:
        return SignalPipelineOutput(metadata={"youtube_video_id": video_id, "threads_total": total_threads})

    embeddings = embedding_client.embed_texts(
        [signal.text for signal in filtered_signals],
        task_type="RETRIEVAL_QUERY",
    )
    clusters = cluster_signals(
        signals=filtered_signals,
        embeddings=embeddings,
        cluster_type="comment",
        similarity_threshold=float(cfg.comment_cluster_sim_threshold),
    )
    prompt_specs: list[SignalPromptSpec] = []
    for idx, cluster in enumerate(clusters, start=1):
        prompt_text = generate_cluster_prompt_with_llm(
            llm_client=llm_client,
            model=cfg.llm.model_1,
            thinking_level=cfg.llm.thinking_1,
            cluster=cluster,
            fail_fast=cfg.llm_fail_fast,
        )
        prompt_specs.append(
            SignalPromptSpec(
                prompt_id=f"comment_prompt_{idx:03d}",
                text=prompt_text,
                prompt_source_type="comment",
                source_cluster_id=cluster.cluster_id,
                source_cluster_type="comment",
            )
        )

    return SignalPipelineOutput(
        external_signals=filtered_signals,
        clusters=clusters,
        prompt_specs=prompt_specs,
        metadata={
            "youtube_video_id": video_id,
            "threads_total": total_threads,
            "threads_selected": len(selected_threads),
            "replies_total": replies_total,
            "thread_intents": thread_intents,
        },
    )


def _build_trends_output(
    *,
    cfg: Any,
    llm_client: Any,
    embedding_client: Any,
    nodes: list[SemanticGraphNode],
    source_url: str,
) -> SignalPipelineOutput:
    if not bool(cfg.llm_fail_fast):
        raise ValueError("CLYPT_SIGNAL_LLM_FAIL_FAST must be enabled for signal pipelines")
    if not nodes:
        return SignalPipelineOutput()

    video_context = {
        "source_url": source_url,
        "node_summaries": [
            {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "summary": node.summary,
                "start_ms": node.start_ms,
                "end_ms": node.end_ms,
            }
            for node in nodes[:120]
        ],
    }
    queries = synthesize_trend_queries_with_llm(
        llm_client=llm_client,
        model=cfg.llm.model_9,
        thinking_level=cfg.llm.thinking_9,
        video_context=video_context,
        fail_fast=cfg.llm_fail_fast,
    )
    if not queries:
        return SignalPipelineOutput(metadata={"trend_queries": []})

    trend_client = TrendSpygClient(max_items=int(cfg.trend_max_items))
    retained_signals: list[ExternalSignal] = []
    trend_items_count = 0
    for query in queries:
        trend_items = trend_client.fetch_related(query=query)
        trend_items_count += len(trend_items)
        for item in trend_items:
            adjudication = adjudicate_trend_relevance_with_llm(
                llm_client=llm_client,
                model=cfg.llm.model_2,
                thinking_level=cfg.llm.thinking_2,
                trend_item=item,
                video_context=video_context,
                fail_fast=cfg.llm_fail_fast,
            )
            keep = bool(adjudication.get("keep"))
            relevance = float(adjudication.get("relevance") or 0.0)
            if not keep and relevance < float(cfg.trend_relevance_threshold):
                continue
            signals = to_external_signals_from_trends(query=query, items=[item])
            for signal in signals:
                signal.metadata["relevance"] = relevance
                signal.metadata["relevance_reason"] = str(adjudication.get("reason") or "")
                retained_signals.append(signal)

    if not retained_signals:
        return SignalPipelineOutput(metadata={"trend_queries": queries})

    embeddings = embedding_client.embed_texts([signal.text for signal in retained_signals], task_type="RETRIEVAL_QUERY")
    clusters = cluster_signals(
        signals=retained_signals,
        embeddings=embeddings,
        cluster_type="trend",
        similarity_threshold=float(cfg.comment_cluster_sim_threshold),
    )

    prompt_specs: list[SignalPromptSpec] = []
    for idx, cluster in enumerate(clusters, start=1):
        prompt_text = generate_cluster_prompt_with_llm(
            llm_client=llm_client,
            model=cfg.llm.model_1,
            thinking_level=cfg.llm.thinking_1,
            cluster=cluster,
            fail_fast=cfg.llm_fail_fast,
        )
        prompt_specs.append(
            SignalPromptSpec(
                prompt_id=f"trend_prompt_{idx:03d}",
                text=prompt_text,
                prompt_source_type="trend",
                source_cluster_id=cluster.cluster_id,
                source_cluster_type="trend",
            )
        )

    return SignalPipelineOutput(
        external_signals=retained_signals,
        clusters=clusters,
        prompt_specs=prompt_specs,
        metadata={
            "trend_queries": queries,
            "trend_items_count": trend_items_count,
            "trend_retained_count": len(retained_signals),
        },
    )


__all__ = [
    "collect_signal_outputs",
    "merge_signal_outputs",
    "resolve_youtube_video_id",
    "start_comments_future",
    "start_trends_future",
    "wait_enabled_signal_futures",
]
