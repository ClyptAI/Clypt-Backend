from __future__ import annotations

from typing import Any

from .boundary_reconciliation import reconcile_boundary_nodes
from .media_embeddings import prepare_node_media_embeddings
from .merge_and_classify import merge_and_classify_neighborhood
from .node_embeddings import build_semantic_embedding_payload
from .prompts import (
    build_boundary_reconciliation_prompt,
    build_merge_and_classify_prompt,
)
from .turn_neighborhoods import build_turn_neighborhoods
from ..contracts import AudioEventTimeline, CanonicalTimeline, SemanticGraphNode, SpeechEmotionTimeline


def run_merge_and_classify_batches(
    *,
    canonical_timeline: CanonicalTimeline,
    speech_emotion_timeline: SpeechEmotionTimeline | None,
    audio_event_timeline: AudioEventTimeline | None,
    llm_client: Any,
    target_turn_count: int = 8,
    halo_turn_count: int = 2,
    model: str | None = None,
) -> tuple[list[SemanticGraphNode], list[dict[str, Any]]]:
    neighborhoods = build_turn_neighborhoods(
        canonical_timeline=canonical_timeline,
        speech_emotion_timeline=speech_emotion_timeline,
        audio_event_timeline=audio_event_timeline,
        target_turn_count=target_turn_count,
        halo_turn_count=halo_turn_count,
    )
    nodes: list[SemanticGraphNode] = []
    debug: list[dict[str, Any]] = []
    for neighborhood in neighborhoods:
        prompt = build_merge_and_classify_prompt(neighborhood_payload=neighborhood)
        response = llm_client.generate_json(prompt=prompt, model=model, temperature=0.0)
        merged_nodes = merge_and_classify_neighborhood(
            neighborhood_payload=neighborhood,
            gemini_response=response,
        )
        nodes.extend(merged_nodes)
        debug.append(
            {
                "batch_id": neighborhood["batch_id"],
                "prompt": prompt,
                "response": response,
            }
        )
    return nodes, debug


def run_boundary_reconciliation(
    *,
    left_batch_nodes: list[SemanticGraphNode],
    right_batch_nodes: list[SemanticGraphNode],
    llm_client: Any,
    model: str | None = None,
) -> tuple[list[SemanticGraphNode], dict[str, Any]]:
    overlap_payload = {
        "left_batch_nodes": [node.model_dump(mode="json") for node in left_batch_nodes],
        "right_batch_nodes": [node.model_dump(mode="json") for node in right_batch_nodes],
    }
    prompt = build_boundary_reconciliation_prompt(overlap_payload=overlap_payload)
    response = llm_client.generate_json(prompt=prompt, model=model, temperature=0.0)
    reconciled = reconcile_boundary_nodes(
        left_batch_nodes=left_batch_nodes,
        right_batch_nodes=right_batch_nodes,
        gemini_response=response,
    )
    return reconciled, {
        "prompt": prompt,
        "response": response,
    }


def embed_semantic_nodes_live(
    *,
    nodes: list[SemanticGraphNode],
    embedding_client: Any,
    multimodal_media: list[dict[str, str]],
) -> list[SemanticGraphNode]:
    if len(multimodal_media) != len(nodes):
        raise ValueError("multimodal_media must align one-to-one with nodes")
    texts = [build_semantic_embedding_payload(node=node) for node in nodes]
    semantic_embeddings = embedding_client.embed_texts(
        texts,
        task_type="RETRIEVAL_DOCUMENT",
    )
    multimodal_embeddings = embedding_client.embed_media_uris(multimodal_media)
    embedded_nodes: list[SemanticGraphNode] = []
    for node, semantic_embedding, multimodal_embedding in zip(
        nodes,
        semantic_embeddings,
        multimodal_embeddings,
        strict=True,
    ):
        embedded_nodes.append(
            node.model_copy(
                update={
                    "semantic_embedding": semantic_embedding,
                    "multimodal_embedding": multimodal_embedding,
                }
            )
        )
    return embedded_nodes


__all__ = [
    "embed_semantic_nodes_live",
    "prepare_node_media_embeddings",
    "run_boundary_reconciliation",
    "run_merge_and_classify_batches",
]
