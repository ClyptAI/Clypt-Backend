from __future__ import annotations

from .._embedding_utils import cosine_similarity
from ..contracts import SemanticGraphNode


SEMANTIC_RETRIEVAL_WEIGHT = 0.75
MULTIMODAL_RETRIEVAL_WEIGHT = 0.25


def _prompt_id(prompt: object, index: int) -> str:
    if isinstance(prompt, dict) and prompt.get("prompt_id"):
        return str(prompt["prompt_id"])
    return f"prompt_{index + 1:04d}"


def _prompt_embedding(prompt: object) -> list[float]:
    if isinstance(prompt, dict):
        embedding = prompt.get("embedding")
        if isinstance(embedding, list):
            return [float(value) for value in embedding]
    raise ValueError("each prompt must include an embedding for deterministic seed retrieval")


def retrieve_seed_nodes(*, prompts: list[object], nodes: list[SemanticGraphNode], top_k_per_prompt: int = 5) -> list[dict]:
    """Retrieve and dedupe seed nodes across prompt families."""
    per_prompt_hits: list[dict] = []
    for prompt_index, prompt in enumerate(prompts):
        prompt_id = _prompt_id(prompt, prompt_index)
        prompt_embedding = _prompt_embedding(prompt)
        scored_nodes = []
        for node in nodes:
            if not node.semantic_embedding or not node.multimodal_embedding:
                continue
            semantic_similarity = cosine_similarity(prompt_embedding, node.semantic_embedding)
            multimodal_similarity = cosine_similarity(prompt_embedding, node.multimodal_embedding)
            if semantic_similarity == float("-inf") or multimodal_similarity == float("-inf"):
                continue
            retrieval_score = (
                (SEMANTIC_RETRIEVAL_WEIGHT * semantic_similarity)
                + (MULTIMODAL_RETRIEVAL_WEIGHT * multimodal_similarity)
            )
            scored_nodes.append((retrieval_score, semantic_similarity, multimodal_similarity, node))
        scored_nodes.sort(key=lambda item: (-item[0], item[3].start_ms, item[3].node_id))
        for retrieval_score, semantic_similarity, multimodal_similarity, node in scored_nodes[:top_k_per_prompt]:
            per_prompt_hits.append(
                {
                    "node_id": node.node_id,
                    "source_prompt_ids": [prompt_id],
                    "retrieval_score": float(retrieval_score),
                    "semantic_similarity": float(semantic_similarity),
                    "multimodal_similarity": float(multimodal_similarity),
                    "start_ms": node.start_ms,
                    "end_ms": node.end_ms,
                }
            )

    deduped: dict[str, dict] = {}
    for hit in per_prompt_hits:
        existing = deduped.get(hit["node_id"])
        if existing is None:
            deduped[hit["node_id"]] = hit
            continue
        existing["retrieval_score"] = max(existing["retrieval_score"], hit["retrieval_score"])
        existing["semantic_similarity"] = max(
            existing.get("semantic_similarity", float("-inf")),
            hit["semantic_similarity"],
        )
        existing["multimodal_similarity"] = max(
            existing.get("multimodal_similarity", float("-inf")),
            hit["multimodal_similarity"],
        )
        for prompt_id in hit["source_prompt_ids"]:
            if prompt_id not in existing["source_prompt_ids"]:
                existing["source_prompt_ids"].append(prompt_id)

    ordered_hits = sorted(
        deduped.values(),
        key=lambda item: (-item["retrieval_score"], item["start_ms"], item["node_id"]),
    )

    kept: list[dict] = []
    neighborhood_counts: dict[int, int] = {}
    for hit in ordered_hits:
        neighborhood_key = int(hit["start_ms"]) // 20_000
        if neighborhood_counts.get(neighborhood_key, 0) >= 2:
            continue
        neighborhood_counts[neighborhood_key] = neighborhood_counts.get(neighborhood_key, 0) + 1
        kept.append(hit)

    return kept
