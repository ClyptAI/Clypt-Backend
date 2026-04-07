from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .prompts import (
    META_PROMPT_GENERATION_SCHEMA,
    POOL_REVIEW_SCHEMA,
    SUBGRAPH_REVIEW_SCHEMA,
    build_meta_prompt_generation_prompt,
    build_pooled_candidate_review_prompt,
    build_subgraph_review_prompt,
)
from .query_embeddings import embed_prompt_texts
from .review_candidate_pool import review_candidate_pool
from .review_subgraphs import review_local_subgraph
from ..contracts import ClipCandidate, LocalSubgraph, PooledCandidateReviewResponse, SemanticGraphNode, SubgraphReviewResponse


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
) -> list[str]:
    """Generate video-specific retrieval prompts from the semantic node graph using Gemini Flash.

    Sends condensed node summaries (type, flags, summary, timestamps) to Gemini and instructs it
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
    response = llm_client.generate_json(prompt=prompt, model=model, temperature=0.0, response_schema=META_PROMPT_GENERATION_SCHEMA, max_output_tokens=1024)
    prompts = response.get("prompts") or []
    if not prompts or not isinstance(prompts, list):
        raise ValueError(
            f"generate_meta_prompts_live: Gemini returned no prompts list. response={response!r}"
        )
    return [str(p) for p in prompts if p]


def embed_prompt_texts_live(*, prompts: list[str], embedding_client: Any) -> list[dict]:
    embedded = embed_prompt_texts(prompts=prompts)
    vectors = embedding_client.embed_texts(
        [item["text"] for item in embedded],
        task_type="RETRIEVAL_QUERY",
    )
    return [
        {
            **item,
            "embedding": vector,
        }
        for item, vector in zip(embedded, vectors, strict=True)
    ]


def run_subgraph_reviews(
    *,
    subgraphs: list[LocalSubgraph],
    llm_client: Any,
    model: str | None = None,
    max_concurrent: int = 5,
) -> tuple[list[SubgraphReviewResponse], list[dict[str, Any]]]:
    def _call_review(subgraph):
        payload = subgraph.model_dump(mode="json")
        prompt = build_subgraph_review_prompt(subgraph_payload=payload)
        response = llm_client.generate_json(prompt=prompt, model=model, temperature=0.0, response_schema=SUBGRAPH_REVIEW_SCHEMA, max_output_tokens=1024)
        return review_local_subgraph(subgraph=subgraph, gemini_response=response), {
            "subgraph_id": subgraph.subgraph_id,
            "prompt": prompt,
            "response": response,
        }

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
) -> PooledCandidateReviewResponse:
    if llm_client is None:
        raise ValueError("llm_client is required for pooled candidate review.")
    payload = {
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
    }
    prompt = build_pooled_candidate_review_prompt(candidate_payload=payload)
    response = llm_client.generate_json(prompt=prompt, model=model, temperature=0.0, response_schema=POOL_REVIEW_SCHEMA, max_output_tokens=4096)
    return review_candidate_pool(candidates=candidates, gemini_response=response)


__all__ = [
    "embed_prompt_texts_live",
    "generate_meta_prompts_live",
    "run_candidate_pool_review",
    "run_subgraph_reviews",
]
