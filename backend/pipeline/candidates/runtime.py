from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import time
from typing import Any

from ..signals.contracts import SignalPromptSpec
from .prompts import (
    META_PROMPT_GENERATION_SCHEMA,
    POOL_REVIEW_SCHEMA,
    SUBGRAPH_REVIEW_SCHEMA,
    build_meta_prompt_generation_prompt,
    build_pooled_candidate_review_prompt,
    build_subgraph_review_prompt,
)
from .review_candidate_pool import review_candidate_pool
from .review_subgraphs import review_local_subgraph
from ..contracts import ClipCandidate, LocalSubgraph, PooledCandidateReviewResponse, SemanticGraphNode, SubgraphReviewResponse


def _compact_subgraph_payload(subgraph: LocalSubgraph) -> dict[str, Any]:
    nodes_payload: list[dict[str, Any]] = []
    for node in subgraph.nodes:
        item: dict[str, Any] = {
            "node_id": node.node_id,
            "start_ms": node.start_ms,
            "end_ms": node.end_ms,
            "node_type": node.node_type,
            "summary": node.summary,
            "transcript_excerpt": node.transcript_excerpt,
        }
        if node.node_flags:
            item["node_flags"] = list(node.node_flags)
        if node.emotion_labels:
            item["emotion_labels"] = list(node.emotion_labels)
        if node.audio_events:
            item["audio_events"] = list(node.audio_events)
        if node.inbound_edges:
            item["inbound_edges"] = [edge.model_dump(mode="json") for edge in node.inbound_edges]
        if node.outbound_edges:
            item["outbound_edges"] = [edge.model_dump(mode="json") for edge in node.outbound_edges]
        nodes_payload.append(item)
    return {
        "subgraph_id": subgraph.subgraph_id,
        "seed_node_id": subgraph.seed_node_id,
        "source_prompt_ids": list(subgraph.source_prompt_ids),
        "start_ms": subgraph.start_ms,
        "end_ms": subgraph.end_ms,
        "nodes": nodes_payload,
    }


def _compact_provenance_payload(provenance_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if provenance_payload is None:
        return None
    compact: dict[str, Any] = {
        "subgraph_id": provenance_payload.get("subgraph_id"),
        "seed_source_set": list(provenance_payload.get("seed_source_set") or []),
        "seed_prompt_ids": list(provenance_payload.get("seed_prompt_ids") or []),
        "source_cluster_ids": list(provenance_payload.get("source_cluster_ids") or []),
        "support_summary": dict(provenance_payload.get("support_summary") or {}),
    }
    selection_reason = provenance_payload.get("selection_reason")
    if selection_reason:
        compact["selection_reason"] = selection_reason
    return compact


def _compact_candidate_payload(candidates: list[ClipCandidate]) -> dict[str, Any]:
    payload_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        item: dict[str, Any] = {
            "clip_id": candidate.clip_id,
            "node_ids": list(candidate.node_ids),
            "start_ms": candidate.start_ms,
            "end_ms": candidate.end_ms,
            "score": candidate.score,
            "rationale": candidate.rationale,
            "source_prompt_ids": list(candidate.source_prompt_ids),
            "seed_node_id": candidate.seed_node_id,
            "subgraph_id": candidate.subgraph_id,
        }
        if candidate.query_aligned is not None:
            item["query_aligned"] = candidate.query_aligned
        if candidate.external_signal_score is not None:
            item["external_signal_score"] = candidate.external_signal_score
        if candidate.agreement_bonus is not None:
            item["agreement_bonus"] = candidate.agreement_bonus
        payload_candidates.append(item)
    return {"candidates": payload_candidates}


def _meta_prompt_target_count(duration_s: float) -> int:
    """Return the target number of retrieval prompts for a given video duration."""
    minutes = duration_s / 60.0
    if minutes < 5:
        return 2
    if minutes <= 15:
        return 4
    if minutes <= 25:
        return 6
    if minutes <= 60:
        return 8
    return 10


def generate_meta_prompts_live(
    *,
    nodes: list[SemanticGraphNode],
    llm_client: Any,
    model: str | None = None,
    duration_s: float = 0.0,
    max_output_tokens: int = 32768,
    return_debug: bool = False,
) -> list[str] | tuple[list[str], dict[str, Any]]:
    """Generate video-specific retrieval prompts from the semantic node graph using Qwen.

    Sends condensed node summaries (type, flags, summary, timestamps) to Qwen and instructs it
    to produce exactly target_count queries tuned to this video's actual content and themes.
    target_count scales with video duration (same buckets as the legacy static pool).
    Raises on any LLM or parse failure — callers should not swallow this.
    """
    target_count = _meta_prompt_target_count(duration_s)
    node_summaries = [
        {
            "node_type": node.node_type,
            "node_flags": list(node.node_flags or []),
            "summary": node.summary,
            "start_ms": node.start_ms,
            "end_ms": node.end_ms,
        }
        for node in nodes
    ]
    prompt = build_meta_prompt_generation_prompt(
        node_summaries=node_summaries,
        target_count=target_count,
    )
    started = time.perf_counter()
    response = llm_client.generate_json(
        prompt=prompt,
        model=model,
        temperature=0.0,
        response_schema=META_PROMPT_GENERATION_SCHEMA,
        max_output_tokens=max_output_tokens,
    )
    prompts = response.get("prompts") or []
    if not prompts or not isinstance(prompts, list):
        raise ValueError(
            f"generate_meta_prompts_live: Qwen returned no prompts list. response={response!r}"
        )
    final_prompts = [str(p) for p in prompts if p]
    diagnostics = {
        "target_prompt_count": target_count,
        "returned_prompt_count": len(final_prompts),
        "node_summary_count": len(node_summaries),
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "prompt_chars": len(prompt),
        "prompt_token_estimate": max(1, round(len(prompt) / 4.0)),
        "payload_chars": len(json.dumps(node_summaries, ensure_ascii=True, separators=(",", ":"))),
        "response_chars": len(json.dumps(response, ensure_ascii=True, separators=(",", ":"))),
    }
    if return_debug:
        return final_prompts, diagnostics
    return final_prompts


def embed_prompt_texts_live(
    *,
    prompts: list[str | dict[str, Any] | SignalPromptSpec],
    embedding_client: Any,
    return_debug: bool = False,
) -> list[dict] | tuple[list[dict], dict[str, Any]]:
    normalized_prompts: list[dict[str, Any]] = []
    prompt_texts: list[str] = []
    for index, prompt in enumerate(prompts, start=1):
        if isinstance(prompt, SignalPromptSpec):
            item = prompt.model_dump(mode="json")
        elif isinstance(prompt, dict):
            item = dict(prompt)
        else:
            item = {"prompt_id": f"prompt_{index:03d}", "text": str(prompt)}
        item.setdefault("prompt_id", f"prompt_{index:03d}")
        item.setdefault("text", "")
        normalized_prompts.append(item)
        prompt_texts.append(str(item["text"]))

    started = time.perf_counter()
    vectors = embedding_client.embed_texts(
        prompt_texts,
        task_type="RETRIEVAL_QUERY",
    )
    embedded = [
        {**item, "embedding": vector}
        for item, vector in zip(normalized_prompts, vectors, strict=True)
    ]
    diagnostics = {
        "prompt_count": len(prompt_texts),
        "text_chars": sum(len(text) for text in prompt_texts),
        "latency_ms": (time.perf_counter() - started) * 1000.0,
    }
    if return_debug:
        return embedded, diagnostics
    return embedded


def run_subgraph_reviews(
    *,
    subgraphs: list[LocalSubgraph],
    llm_client: Any,
    model: str | None = None,
    max_concurrent: int = 5,
    subgraph_provenance_by_id: dict[str, dict[str, Any]] | None = None,
    max_output_tokens: int = 32768,
) -> tuple[list[SubgraphReviewResponse], list[dict[str, Any]]]:
    def _call_review(subgraph):
        started = time.perf_counter()
        payload = _compact_subgraph_payload(subgraph)
        provenance_payload = None
        if subgraph_provenance_by_id is not None:
            provenance_payload = _compact_provenance_payload(subgraph_provenance_by_id.get(subgraph.subgraph_id))
        prompt = build_subgraph_review_prompt(subgraph_payload=payload, provenance_payload=provenance_payload)
        response = llm_client.generate_json(
            prompt=prompt,
            model=model,
            temperature=0.0,
            response_schema=SUBGRAPH_REVIEW_SCHEMA,
            max_output_tokens=max_output_tokens,
        )
        review = review_local_subgraph(subgraph=subgraph, llm_response=response)
        latency_ms = (time.perf_counter() - started) * 1000.0
        prompt_chars = len(prompt)
        debug_payload = {
            "subgraph_id": subgraph.subgraph_id,
            "seed_node_id": subgraph.seed_node_id,
            "diagnostics": {
                "latency_ms": latency_ms,
                "node_count": len(subgraph.nodes),
                "span_ms": max(0, int(subgraph.end_ms) - int(subgraph.start_ms)),
                "prompt_chars": prompt_chars,
                # Heuristic estimator for quick operator diagnostics (not billing-accurate).
                "prompt_token_estimate": max(1, round(prompt_chars / 4.0)),
                "payload_chars": len(json.dumps(payload, ensure_ascii=True, separators=(",", ":"))),
                "provenance_payload_chars": (
                    len(json.dumps(provenance_payload, ensure_ascii=True, separators=(",", ":")))
                    if provenance_payload is not None
                    else 0
                ),
                "response_chars": len(json.dumps(response, ensure_ascii=True, separators=(",", ":"))),
                "reject_all": review.reject_all,
                "candidate_count": len(review.candidates),
                "invalid_structured_output": review.reject_reason.startswith("invalid_structured_output:"),
            },
            "prompt": prompt,
            "response": response,
        }
        return review, debug_payload

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = [pool.submit(_call_review, sg) for sg in subgraphs]
        results = [f.result() for f in futures]

    reviews = [r[0] for r in results]
    debug = [r[1] for r in results]
    return reviews, debug


def run_candidate_pool_review(
    *,
    candidates: list[ClipCandidate],
    llm_client: Any,
    model: str | None = None,
    max_output_tokens: int = 32768,
) -> PooledCandidateReviewResponse:
    pooled, _ = run_candidate_pool_review_with_debug(
        candidates=candidates,
        llm_client=llm_client,
        model=model,
        max_output_tokens=max_output_tokens,
    )
    return pooled


def run_candidate_pool_review_with_debug(
    *,
    candidates: list[ClipCandidate],
    llm_client: Any,
    model: str | None = None,
    max_output_tokens: int = 32768,
) -> tuple[PooledCandidateReviewResponse, dict[str, Any]]:
    if llm_client is None:
        raise ValueError("llm_client is required for pooled candidate review.")
    payload = _compact_candidate_payload(candidates)
    payload_chars = len(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
    prompt = build_pooled_candidate_review_prompt(candidate_payload=payload)
    started = time.perf_counter()
    response = llm_client.generate_json(
        prompt=prompt,
        model=model,
        temperature=0.0,
        response_schema=POOL_REVIEW_SCHEMA,
        max_output_tokens=max_output_tokens,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    pooled = review_candidate_pool(candidates=candidates, llm_response=response)
    max_pool_rank = max(
        (decision.pool_rank or 0) for decision in pooled.ranked_candidates
    ) if pooled.ranked_candidates else 0
    debug = {
        "candidate_count": len(candidates),
        "kept_candidate_count": len(pooled.ranked_candidates),
        "dropped_candidate_count": len(pooled.dropped_candidate_temp_ids),
        "max_pool_rank": max_pool_rank,
        "latency_ms": latency_ms,
        "prompt_chars": len(prompt),
        # Heuristic estimator for quick operator diagnostics (not billing-accurate).
        "prompt_token_estimate": max(1, round(len(prompt) / 4.0)),
        "payload_chars": payload_chars,
        "response_chars": len(json.dumps(response, ensure_ascii=True, separators=(",", ":"))),
    }
    return pooled, debug


__all__ = [
    "embed_prompt_texts_live",
    "generate_meta_prompts_live",
    "run_candidate_pool_review",
    "run_candidate_pool_review_with_debug",
    "run_subgraph_reviews",
]
