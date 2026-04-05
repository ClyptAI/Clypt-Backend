from __future__ import annotations

from typing import Any

from .prompts import build_pooled_candidate_review_prompt, build_subgraph_review_prompt
from .query_embeddings import embed_prompt_texts
from .review_candidate_pool import review_candidate_pool
from .review_subgraphs import review_local_subgraph
from ..contracts import ClipCandidate, LocalSubgraph, PooledCandidateReviewResponse, SubgraphReviewResponse


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
) -> tuple[list[SubgraphReviewResponse], list[dict[str, Any]]]:
    reviews: list[SubgraphReviewResponse] = []
    debug: list[dict[str, Any]] = []
    for subgraph in subgraphs:
        payload = subgraph.model_dump(mode="json")
        prompt = build_subgraph_review_prompt(subgraph_payload=payload)
        response = llm_client.generate_json(prompt=prompt, model=model, temperature=0.0)
        reviews.append(review_local_subgraph(subgraph=subgraph, gemini_response=response))
        debug.append(
            {
                "subgraph_id": subgraph.subgraph_id,
                "prompt": prompt,
                "response": response,
            }
        )
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
    response = llm_client.generate_json(prompt=prompt, model=model, temperature=0.0)
    return review_candidate_pool(candidates=candidates, gemini_response=response)


__all__ = [
    "embed_prompt_texts_live",
    "run_candidate_pool_review",
    "run_subgraph_reviews",
]
