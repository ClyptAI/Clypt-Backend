from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .boundary_reconciliation import reconcile_boundary_nodes
from .media_embeddings import prepare_node_media_embeddings
from .merge_and_classify import merge_and_classify_neighborhood
from .node_embeddings import build_semantic_embedding_payload
from .prompts import (
    BOUNDARY_RECONCILIATION_SCHEMA,
    MERGE_AND_CLASSIFY_SCHEMA,
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
    """Merge/classify per neighborhood batch only (no boundary reconciliation).
    Used by tests and the injected path. Live path should use
    run_merge_classify_and_reconcile() instead."""
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
        response = llm_client.generate_json(prompt=prompt, model=model, temperature=0.0, response_schema=MERGE_AND_CLASSIFY_SCHEMA, max_output_tokens=2048)
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


def run_merge_classify_and_reconcile(
    *,
    canonical_timeline: CanonicalTimeline,
    speech_emotion_timeline: SpeechEmotionTimeline | None,
    audio_event_timeline: AudioEventTimeline | None,
    llm_client: Any,
    target_batch_count: int = 5,
    max_turns_per_batch: int = 25,
    model: str | None = None,
    max_concurrent: int = 5,
) -> tuple[list[SemanticGraphNode], list[dict[str, Any]], list[dict[str, Any]]]:
    """Full Phase 2A + 2B: merge/classify per neighborhood batch, then boundary
    reconciliation between every adjacent batch pair.  Calls are dispatched in
    parallel via ThreadPoolExecutor.

    Returns (final_nodes, merge_debug, boundary_debug).
    merge_debug: one entry per neighborhood batch.
    boundary_debug: one entry per reconciled batch seam (len = batches - 1).
    """
    total_turns = len(canonical_timeline.turns)
    batch_size = (
        min(math.ceil(total_turns / target_batch_count), max_turns_per_batch)
        if total_turns > 0
        else max_turns_per_batch
    )
    halo_size = max(1, batch_size // 7)

    neighborhoods = build_turn_neighborhoods(
        canonical_timeline=canonical_timeline,
        speech_emotion_timeline=speech_emotion_timeline,
        audio_event_timeline=audio_event_timeline,
        target_turn_count=batch_size,
        halo_turn_count=halo_size,
    )

    def _call_merge(neighborhood):
        prompt = build_merge_and_classify_prompt(neighborhood_payload=neighborhood)
        response = llm_client.generate_json(prompt=prompt, model=model, temperature=0.0, response_schema=MERGE_AND_CLASSIFY_SCHEMA, max_output_tokens=2048)
        return merge_and_classify_neighborhood(
            neighborhood_payload=neighborhood, gemini_response=response
        ), {"batch_id": neighborhood["batch_id"], "prompt": prompt, "response": response}

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        merge_futures = [pool.submit(_call_merge, n) for n in neighborhoods]
        batch_results = [f.result() for f in merge_futures]  # preserves order

    batch_nodes = [r[0] for r in batch_results]
    merge_debug = [r[1] for r in batch_results]

    if not batch_nodes:
        return [], merge_debug, []

    # Parallel boundary reconciliation at each seam
    def _call_boundary(idx):
        left = [batch_nodes[idx - 1][-1]]
        right = [batch_nodes[idx][0]]
        reconciled, debug = run_boundary_reconciliation(
            left_batch_nodes=left,
            right_batch_nodes=right,
            llm_client=llm_client,
            model=model,
        )
        return idx, reconciled, {
            "left_batch_id": neighborhoods[idx - 1]["batch_id"],
            "right_batch_id": neighborhoods[idx]["batch_id"],
            **debug,
        }

    seam_indices = range(1, len(batch_nodes))
    seam_results: dict[int, tuple] = {}
    if seam_indices:
        with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            seam_futures = {pool.submit(_call_boundary, i): i for i in seam_indices}
            for f in as_completed(seam_futures):
                idx, reconciled, debug = f.result()
                seam_results[idx] = (reconciled, debug)

    # Stitch in order
    final_nodes: list[SemanticGraphNode] = list(batch_nodes[0])
    boundary_debug: list[dict[str, Any]] = []
    for idx in range(1, len(batch_nodes)):
        next_nodes = batch_nodes[idx]
        if not next_nodes or not final_nodes:
            final_nodes.extend(next_nodes)
            continue
        reconciled, br_debug = seam_results[idx]
        boundary_debug.append(br_debug)
        final_nodes = [*final_nodes[:-1], *reconciled, *next_nodes[1:]]

    return final_nodes, merge_debug, boundary_debug


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
    response = llm_client.generate_json(prompt=prompt, model=model, temperature=0.0, response_schema=BOUNDARY_RECONCILIATION_SCHEMA, max_output_tokens=1024)
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
    "run_merge_classify_and_reconcile",
]
