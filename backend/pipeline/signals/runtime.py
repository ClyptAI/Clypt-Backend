from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Callable, TypeVar

from backend.pipeline.config import SignalConfig
from backend.pipeline.contracts import SemanticGraphNode
from backend.providers.protocols import EmbeddingClient, LLMGenerateJsonClient

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
    adjudicate_trend_relevance_with_llm_batch,
    classify_comments_with_llm_batch,
    consolidate_thread_with_llm,
    generate_cluster_prompt_with_llm,
    synthesize_trend_queries_with_llm,
)
from .trends_client import TrendSpygClient, to_external_signals_from_trends

_COMMENT_CLASSIFY_BATCH_SIZE = 12
_TREND_RELEVANCE_BATCH_SIZE = 12
_T = TypeVar("_T")


def _chunked(items: list[_T], size: int) -> list[list[_T]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if not items:
        return []
    return [items[i : i + size] for i in range(0, len(items), size)]


def _signal_max_workers(*, cfg: SignalConfig, task_count: int) -> int:
    if task_count <= 0:
        return 1
    configured = int(getattr(cfg, "max_concurrent", 8) or 8)
    return max(1, min(configured, task_count))


def start_comments_future(
    *,
    executor: ThreadPoolExecutor,
    cfg: SignalConfig,
    llm_client: LLMGenerateJsonClient,
    embedding_client: EmbeddingClient,
    source_url: str,
    signal_event_logger: Callable[..., None] | None = None,
) -> Future[SignalPipelineOutput] | None:
    if not bool(cfg.enable_comment_signals):
        return None
    return executor.submit(
        _build_comments_output,
        cfg=cfg,
        llm_client=llm_client,
        embedding_client=embedding_client,
        source_url=source_url,
        signal_event_logger=signal_event_logger,
    )


def start_trends_future(
    *,
    executor: ThreadPoolExecutor,
    cfg: SignalConfig,
    llm_client: LLMGenerateJsonClient,
    embedding_client: EmbeddingClient,
    nodes: list[SemanticGraphNode],
    source_url: str,
    signal_event_logger: Callable[..., None] | None = None,
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
        signal_event_logger=signal_event_logger,
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
    cfg: SignalConfig,
    llm_client: LLMGenerateJsonClient,
    embedding_client: EmbeddingClient,
    source_url: str,
    signal_event_logger: Callable[..., None] | None = None,
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

    # Fetch full replies and enrich each selected thread (parallel fan-out).
    def _fetch_replies_for_thread(thread: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        top_comment = (thread.get("snippet") or {}).get("topLevelComment") or {}
        parent_id = str(top_comment.get("id") or "")
        if not parent_id:
            return "", []
        full_replies = comments_client.fetch_replies(
            parent_id=parent_id,
            max_replies=cfg.comment_max_replies_per_thread,
        )
        return parent_id, full_replies

    with ThreadPoolExecutor(
        max_workers=_signal_max_workers(cfg=cfg, task_count=len(selected_threads))
    ) as pool:
        futures = {
            pool.submit(_fetch_replies_for_thread, thread): idx
            for idx, thread in enumerate(selected_threads)
        }
        for future in as_completed(futures):
            idx = futures[future]
            parent_id, full_replies = future.result()
            if not parent_id:
                continue
            selected_threads[idx]["replies"] = {"comments": full_replies}
    replies_total = sum(
        len(((thread.get("replies") or {}).get("comments") or []))
        for thread in selected_threads
    )

    # Callpoint #10: per-thread consolidation (parallel).
    thread_intents: list[dict[str, Any] | None] = [None] * len(selected_threads)
    with ThreadPoolExecutor(
        max_workers=_signal_max_workers(cfg=cfg, task_count=len(selected_threads))
    ) as pool:
        futures = {
            pool.submit(
                consolidate_thread_with_llm,
                llm_client=llm_client,
                model=cfg.llm.model_10,
                thread_payload=thread,
                fail_fast=cfg.llm_fail_fast,
                event_logger=signal_event_logger,
            ): idx
            for idx, thread in enumerate(selected_threads)
        }
        for future in as_completed(futures):
            idx = futures[future]
            thread_intents[idx] = future.result()
    thread_intents = [item or {} for item in thread_intents]

    signals = to_external_signals_from_threads(thread_items=selected_threads, include_replies=True)
    thread_signal_text_by_id = {
        str(thread.get("id") or ""): _build_thread_signal_text(thread=thread, consolidated=consolidated)
        for thread, consolidated in zip(selected_threads, thread_intents, strict=False)
        if str(thread.get("id") or "")
    }
    for signal in signals:
        thread_id = str((signal.metadata or {}).get("thread_id") or "")
        merged_text = thread_signal_text_by_id.get(thread_id)
        if not merged_text:
            continue
        signal.metadata["raw_text"] = signal.text
        signal.metadata["thread_signal_text"] = merged_text
        signal.text = merged_text

    # Callpoint #3: quality filter (micro-batched + parallel).
    filtered_signals: list[ExternalSignal] = []
    signal_batches = _chunked(signals, _COMMENT_CLASSIFY_BATCH_SIZE)
    classified_batches: list[list[dict[str, Any]] | None] = [None] * len(signal_batches)
    with ThreadPoolExecutor(
        max_workers=_signal_max_workers(cfg=cfg, task_count=len(signal_batches))
    ) as pool:
        futures = {
            pool.submit(
                classify_comments_with_llm_batch,
                llm_client=llm_client,
                model=cfg.llm.model_3,
                signals=batch,
                fail_fast=cfg.llm_fail_fast,
                event_logger=signal_event_logger,
            ): idx
            for idx, batch in enumerate(signal_batches)
        }
        for future in as_completed(futures):
            idx = futures[future]
            classified_batches[idx] = future.result()
    for batch_signals, batch_classifications in zip(
        signal_batches,
        [item or [] for item in classified_batches],
        strict=True,
    ):
        for signal, classified in zip(batch_signals, batch_classifications, strict=True):
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
    if clusters:
        prompt_texts: list[str | None] = [None] * len(clusters)
        with ThreadPoolExecutor(
            max_workers=_signal_max_workers(cfg=cfg, task_count=len(clusters))
        ) as pool:
            futures = {
                pool.submit(
                    generate_cluster_prompt_with_llm,
                    llm_client=llm_client,
                    model=cfg.llm.model_1,
                    cluster=cluster,
                    fail_fast=cfg.llm_fail_fast,
                    event_logger=signal_event_logger,
                ): idx
                for idx, cluster in enumerate(clusters)
            }
            for future in as_completed(futures):
                idx = futures[future]
                prompt_texts[idx] = future.result()
        for idx, (cluster, prompt_text) in enumerate(zip(clusters, prompt_texts, strict=True), start=1):
            prompt_specs.append(
                SignalPromptSpec(
                    prompt_id=f"comment_prompt_{idx:03d}",
                    text=str(prompt_text or "").strip(),
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
    cfg: SignalConfig,
    llm_client: LLMGenerateJsonClient,
    embedding_client: EmbeddingClient,
    nodes: list[SemanticGraphNode],
    source_url: str,
    signal_event_logger: Callable[..., None] | None = None,
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
        video_context=video_context,
        fail_fast=cfg.llm_fail_fast,
        event_logger=signal_event_logger,
    )
    if not queries:
        return SignalPipelineOutput(metadata={"trend_queries": []})

    trend_client = TrendSpygClient(max_items=int(cfg.trend_max_items))
    retained_signals: list[ExternalSignal] = []
    trend_items_count = 0
    # Fetch related trend items per query in parallel.
    trend_items_by_query: list[list[dict[str, Any]] | None] = [None] * len(queries)
    with ThreadPoolExecutor(
        max_workers=_signal_max_workers(cfg=cfg, task_count=len(queries))
    ) as pool:
        futures = {
            pool.submit(trend_client.fetch_related, query=query): idx
            for idx, query in enumerate(queries)
        }
        for future in as_completed(futures):
            idx = futures[future]
            trend_items_by_query[idx] = future.result()
    trend_items_by_query = [items or [] for items in trend_items_by_query]
    trend_items_count = sum(len(items) for items in trend_items_by_query)

    # Callpoint #2: adjudicate relevance in micro-batches with bounded concurrency.
    batch_jobs: list[tuple[int, list[dict[str, Any]]]] = []
    for query_idx, trend_items in enumerate(trend_items_by_query):
        for batch in _chunked(trend_items, _TREND_RELEVANCE_BATCH_SIZE):
            batch_jobs.append((query_idx, batch))

    batch_results: list[list[dict[str, Any]] | None] = [None] * len(batch_jobs)
    if batch_jobs:
        with ThreadPoolExecutor(
            max_workers=_signal_max_workers(cfg=cfg, task_count=len(batch_jobs))
        ) as pool:
            futures = {
                pool.submit(
                    adjudicate_trend_relevance_with_llm_batch,
                    llm_client=llm_client,
                    model=cfg.llm.model_2,
                    trend_items=batch,
                    video_context=video_context,
                    fail_fast=cfg.llm_fail_fast,
                    event_logger=signal_event_logger,
                ): idx
                for idx, (_, batch) in enumerate(batch_jobs)
            }
            for future in as_completed(futures):
                idx = futures[future]
                batch_results[idx] = future.result()

    # Rebuild per-query adjudications in original item order.
    adjudications_by_query: list[list[dict[str, Any]]] = [[] for _ in queries]
    for (query_idx, _batch_items), adjudications in zip(batch_jobs, [item or [] for item in batch_results], strict=True):
        adjudications_by_query[query_idx].extend(adjudications)

    for query, trend_items, adjudications in zip(queries, trend_items_by_query, adjudications_by_query, strict=True):
        if len(adjudications) != len(trend_items):
            raise ValueError(
                "trend adjudication count mismatch for query "
                f"{query!r}: expected {len(trend_items)} got {len(adjudications)}"
            )
        for item, adjudication in zip(trend_items, adjudications, strict=True):
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
    if clusters:
        prompt_texts: list[str | None] = [None] * len(clusters)
        with ThreadPoolExecutor(
            max_workers=_signal_max_workers(cfg=cfg, task_count=len(clusters))
        ) as pool:
            futures = {
                pool.submit(
                    generate_cluster_prompt_with_llm,
                    llm_client=llm_client,
                    model=cfg.llm.model_1,
                    cluster=cluster,
                    fail_fast=cfg.llm_fail_fast,
                    event_logger=signal_event_logger,
                ): idx
                for idx, cluster in enumerate(clusters)
            }
            for future in as_completed(futures):
                idx = futures[future]
                prompt_texts[idx] = future.result()
        for idx, (cluster, prompt_text) in enumerate(zip(clusters, prompt_texts, strict=True), start=1):
            prompt_specs.append(
                SignalPromptSpec(
                    prompt_id=f"trend_prompt_{idx:03d}",
                    text=str(prompt_text or "").strip(),
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


def _build_thread_signal_text(*, thread: dict[str, Any], consolidated: dict[str, Any]) -> str:
    top_comment = _top_comment_text(thread)
    summary = str(consolidated.get("thread_summary") or "").strip()
    hints = [
        str(item).strip()
        for item in list(consolidated.get("moment_hints") or [])
        if str(item).strip()
    ]
    lines: list[str] = []
    if top_comment:
        lines.append(f"Top comment: {top_comment}")
    if summary:
        lines.append(f"Thread summary: {summary}")
    if hints:
        lines.append(f"Moment hints: {' | '.join(hints[:8])}")
    return "\n".join(lines).strip() or top_comment or summary


def _top_comment_text(thread: dict[str, Any]) -> str:
    top_comment = ((thread.get("snippet") or {}).get("topLevelComment") or {})
    top_snippet = top_comment.get("snippet") or {}
    text = top_snippet.get("textDisplay") or top_snippet.get("textOriginal") or ""
    return str(text).strip()


__all__ = [
    "collect_signal_outputs",
    "merge_signal_outputs",
    "resolve_youtube_video_id",
    "start_comments_future",
    "start_trends_future",
    "wait_enabled_signal_futures",
]
