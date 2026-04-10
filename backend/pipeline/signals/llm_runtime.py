from __future__ import annotations

import json
import logging
import time
from typing import Any

from .contracts import ExternalSignal, ExternalSignalCluster

logger = logging.getLogger(__name__)


def _compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


THREAD_CONSOLIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "thread_summary": {"type": "string"},
        "moment_hints": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["thread_summary", "moment_hints"],
}

COMMENT_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "quality": {"type": "string", "enum": ["high_signal", "contextual", "low_signal", "spam"]},
        "reason": {"type": "string"},
    },
    "required": ["quality"],
}

CLUSTER_PROMPT_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
    },
    "required": ["prompt"],
}

TREND_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["queries"],
}

TREND_RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "keep": {"type": "boolean"},
        "relevance": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["keep", "relevance"],
}

CLUSTER_SPAN_SCHEMA = {
    "type": "object",
    "properties": {
        "node_ids": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["node_ids"],
}

ATTRIBUTION_EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
    },
    "required": ["explanation"],
}


def _call_json(
    *,
    llm_client: Any,
    callpoint_id: str,
    prompt: str,
    model: str,
    thinking_level: str,
    response_schema: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    logger.info(
        "[signals_llm_call_start] callpoint=%s model=%s thinking=%s",
        callpoint_id,
        model,
        thinking_level,
    )
    try:
        response = llm_client.generate_json(
            prompt=prompt,
            model=model,
            temperature=0.0,
            response_schema=response_schema,
            max_output_tokens=32768,
            thinking_level=thinking_level,
        )
    except Exception:
        logger.exception(
            "[signals_llm_call_error] callpoint=%s model=%s thinking=%s",
            callpoint_id,
            model,
            thinking_level,
        )
        raise
    logger.info(
        "[signals_llm_call_done] callpoint=%s model=%s thinking=%s latency_ms=%.1f",
        callpoint_id,
        model,
        thinking_level,
        (time.perf_counter() - started) * 1000.0,
    )
    return response


def consolidate_thread_with_llm(
    *,
    llm_client: Any,
    model: str,
    thinking_level: str,
    thread_payload: dict[str, Any],
) -> dict[str, Any]:
    prompt = (
        "You are consolidating a YouTube comment thread (top comment + replies) into compact intent guidance.\n"
        "Return strict JSON.\n"
        f"Thread payload:\n{_compact(thread_payload)}"
    )
    return _call_json(
        llm_client=llm_client,
        callpoint_id="10",
        prompt=prompt,
        model=model,
        thinking_level=thinking_level,
        response_schema=THREAD_CONSOLIDATION_SCHEMA,
    )


def classify_comment_with_llm(
    *,
    llm_client: Any,
    model: str,
    thinking_level: str,
    signal: ExternalSignal,
) -> dict[str, Any]:
    prompt = (
        "Classify this audience signal quality for clip-seeding usefulness.\n"
        "Return quality in {high_signal, contextual, low_signal, spam}.\n"
        f"Signal:\n{_compact(signal.model_dump(mode='json'))}"
    )
    return _call_json(
        llm_client=llm_client,
        callpoint_id="3",
        prompt=prompt,
        model=model,
        thinking_level=thinking_level,
        response_schema=COMMENT_CLASSIFICATION_SCHEMA,
    )


def generate_cluster_prompt_with_llm(
    *,
    llm_client: Any,
    model: str,
    thinking_level: str,
    cluster: ExternalSignalCluster,
) -> str:
    prompt = (
        "You are writing one retrieval prompt to find the referenced video moment.\n"
        "Write one sentence starting with 'Find'.\n"
        f"Cluster:\n{_compact(cluster.model_dump(mode='json'))}"
    )
    response = _call_json(
        llm_client=llm_client,
        callpoint_id="1",
        prompt=prompt,
        model=model,
        thinking_level=thinking_level,
        response_schema=CLUSTER_PROMPT_SCHEMA,
    )
    value = str(response.get("prompt") or "").strip()
    if not value:
        raise ValueError("cluster prompt generation returned empty prompt")
    return value


def synthesize_trend_queries_with_llm(
    *,
    llm_client: Any,
    model: str,
    thinking_level: str,
    video_context: dict[str, Any],
) -> list[str]:
    prompt = (
        "Generate concise search/trend queries for this video context.\n"
        f"Context:\n{_compact(video_context)}"
    )
    response = _call_json(
        llm_client=llm_client,
        callpoint_id="9",
        prompt=prompt,
        model=model,
        thinking_level=thinking_level,
        response_schema=TREND_QUERY_SCHEMA,
    )
    queries = [str(item).strip() for item in list(response.get("queries") or [])]
    return [query for query in queries if query]


def adjudicate_trend_relevance_with_llm(
    *,
    llm_client: Any,
    model: str,
    thinking_level: str,
    trend_item: dict[str, Any],
    video_context: dict[str, Any],
) -> dict[str, Any]:
    prompt = (
        "Decide if this trend signal is relevant to the target video context.\n"
        "Return keep boolean and relevance score 0..1.\n"
        f"Video context:\n{_compact(video_context)}\n"
        f"Trend item:\n{_compact(trend_item)}"
    )
    return _call_json(
        llm_client=llm_client,
        callpoint_id="2",
        prompt=prompt,
        model=model,
        thinking_level=thinking_level,
        response_schema=TREND_RELEVANCE_SCHEMA,
    )


def resolve_cluster_span_with_llm(
    *,
    llm_client: Any,
    model: str,
    thinking_level: str,
    cluster: ExternalSignalCluster,
    neighborhood_payload: dict[str, Any],
) -> list[str]:
    prompt = (
        "Select the node_ids that best capture the core moment for this external signal cluster.\n"
        "Return only node_ids from the provided neighborhood, in chronological order.\n"
        f"Cluster:\n{_compact(cluster.model_dump(mode='json'))}\n"
        f"Neighborhood:\n{_compact(neighborhood_payload)}"
    )
    response = _call_json(
        llm_client=llm_client,
        callpoint_id="5",
        prompt=prompt,
        model=model,
        thinking_level=thinking_level,
        response_schema=CLUSTER_SPAN_SCHEMA,
    )
    node_ids = [str(item) for item in list(response.get("node_ids") or []) if str(item)]
    return node_ids


def explain_candidate_attribution_with_llm(
    *,
    llm_client: Any,
    model: str,
    thinking_level: str,
    evidence_payload: dict[str, Any],
) -> str:
    prompt = (
        "Write a concise explanation for why this clip candidate was externally boosted.\n"
        "One or two sentences, grounded only in provided evidence.\n"
        f"Evidence:\n{_compact(evidence_payload)}"
    )
    response = _call_json(
        llm_client=llm_client,
        callpoint_id="11",
        prompt=prompt,
        model=model,
        thinking_level=thinking_level,
        response_schema=ATTRIBUTION_EXPLANATION_SCHEMA,
    )
    explanation = str(response.get("explanation") or "").strip()
    return explanation


__all__ = [
    "adjudicate_trend_relevance_with_llm",
    "classify_comment_with_llm",
    "consolidate_thread_with_llm",
    "explain_candidate_attribution_with_llm",
    "generate_cluster_prompt_with_llm",
    "resolve_cluster_span_with_llm",
    "synthesize_trend_queries_with_llm",
]
