from __future__ import annotations

import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .boundary_reconciliation import reconcile_boundary_nodes, should_skip_boundary_reconciliation
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
from backend.providers.protocols import EmbeddingClient, LLMGenerateJsonClient

logger = logging.getLogger(__name__)


def run_merge_and_classify_batches(
    *,
    canonical_timeline: CanonicalTimeline,
    speech_emotion_timeline: SpeechEmotionTimeline | None,
    audio_event_timeline: AudioEventTimeline | None,
    llm_client: LLMGenerateJsonClient,
    target_turn_count: int = 8,
    halo_turn_count: int = 2,
    merge_max_output_tokens: int = 32768,
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
    turn_word_ids_by_turn_id = {
        turn.turn_id: list(turn.word_ids)
        for turn in canonical_timeline.turns
    }
    nodes: list[SemanticGraphNode] = []
    debug: list[dict[str, Any]] = []
    for neighborhood in neighborhoods:
        prompt = build_merge_and_classify_prompt(neighborhood_payload=neighborhood)
        response = llm_client.generate_json(
            prompt=prompt,
            model=model,
            temperature=0.0,
            response_schema=MERGE_AND_CLASSIFY_SCHEMA,
            max_output_tokens=merge_max_output_tokens,
        )
        merged_nodes = merge_and_classify_neighborhood(
            neighborhood_payload=neighborhood,
            llm_response=response,
            turn_word_ids_by_turn_id=turn_word_ids_by_turn_id,
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
    llm_client: LLMGenerateJsonClient,
    target_batch_count: int = 5,
    max_turns_per_batch: int = 25,
    merge_max_output_tokens: int = 32768,
    boundary_max_output_tokens: int = 32768,
    model: str | None = None,
    max_concurrent: int = 5,
    boundary_max_concurrent: int = 10,
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
    turn_word_ids_by_turn_id = {
        turn.turn_id: list(turn.word_ids)
        for turn in canonical_timeline.turns
    }

    def _call_merge(neighborhood):
        started = time.perf_counter()
        prompt = build_merge_and_classify_prompt(neighborhood_payload=neighborhood)
        response = llm_client.generate_json(
            prompt=prompt,
            model=model,
            temperature=0.0,
            response_schema=MERGE_AND_CLASSIFY_SCHEMA,
            max_output_tokens=merge_max_output_tokens,
        )
        merged_nodes = merge_and_classify_neighborhood(
            neighborhood_payload=neighborhood,
            llm_response=response,
            turn_word_ids_by_turn_id=turn_word_ids_by_turn_id,
        )
        payload_chars = len(json.dumps(neighborhood, ensure_ascii=True, separators=(",", ":")))
        response_chars = len(json.dumps(response, ensure_ascii=True, separators=(",", ":")))
        prompt_chars = len(prompt)
        return merged_nodes, {
            "batch_id": neighborhood["batch_id"],
            "prompt": prompt,
            "response": response,
            "diagnostics": {
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "target_turn_count": len(neighborhood.get("target_turn_ids") or []),
                "context_turn_count": len(neighborhood.get("turns") or []),
                "merged_node_count": len(merged_nodes),
                "prompt_chars": prompt_chars,
                "prompt_token_estimate": max(1, round(prompt_chars / 4.0)),
                "payload_chars": payload_chars,
                "response_chars": response_chars,
            },
        }

    merge_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        merge_futures = [pool.submit(_call_merge, n) for n in neighborhoods]
        batch_results = [f.result() for f in merge_futures]  # preserves order
    merge_duration_s = time.perf_counter() - merge_started
    logger.info(
        "[phase2] merge/classify batches done in %.1f s (batches=%d)",
        merge_duration_s,
        len(neighborhoods),
    )

    batch_nodes = [r[0] for r in batch_results]
    merge_debug = [r[1] for r in batch_results]

    if not batch_nodes:
        return [], merge_debug, []

    # Parallel boundary reconciliation at each seam
    def _call_boundary(idx):
        left = [batch_nodes[idx - 1][-1]]
        right = [batch_nodes[idx][0]]
        heuristic = should_skip_boundary_reconciliation(left_node=left[0], right_node=right[0])
        if bool(heuristic.get("skip_llm")):
            return idx, [left[0], right[0]], {
                "left_batch_id": neighborhoods[idx - 1]["batch_id"],
                "right_batch_id": neighborhoods[idx]["batch_id"],
                "heuristic_skip": True,
                "heuristic_reason": str(heuristic.get("reason") or "heuristic_skip"),
                "prompt": None,
                "response": None,
                "diagnostics": {
                    "latency_ms": 0.0,
                    "left_node_count": len(left),
                    "right_node_count": len(right),
                    "reconciled_node_count": 2,
                    "prompt_chars": 0,
                    "prompt_token_estimate": 0,
                    "payload_chars": 0,
                    "response_chars": 0,
                    "heuristic_skip": True,
                    "heuristic_sent_to_llm": False,
                    **heuristic,
                },
            }
        reconciled, debug = run_boundary_reconciliation(
            left_batch_nodes=left,
            right_batch_nodes=right,
            llm_client=llm_client,
            model=model,
            max_output_tokens=boundary_max_output_tokens,
        )
        return idx, reconciled, {
            "left_batch_id": neighborhoods[idx - 1]["batch_id"],
            "right_batch_id": neighborhoods[idx]["batch_id"],
            "heuristic_skip": False,
            "heuristic_reason": str(heuristic.get("reason") or "llm_review"),
            **debug,
            "diagnostics": {
                **dict(debug.get("diagnostics") or {}),
                "heuristic_skip": False,
                "heuristic_sent_to_llm": True,
                **heuristic,
            },
        }

    seam_indices = range(1, len(batch_nodes))
    seam_results: dict[int, tuple] = {}
    if seam_indices:
        boundary_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=boundary_max_concurrent) as pool:
            seam_futures = {pool.submit(_call_boundary, i): i for i in seam_indices}
            for f in as_completed(seam_futures):
                idx, reconciled, debug = f.result()
                seam_results[idx] = (reconciled, debug)
        boundary_duration_s = time.perf_counter() - boundary_started
        logger.info(
            "[phase2] boundary reconciliation done in %.1f s (seams=%d)",
            boundary_duration_s,
            len(batch_nodes) - 1,
        )
        seams_skipped = sum(
            1
            for _, (_, debug) in seam_results.items()
            if bool((debug.get("diagnostics") or {}).get("heuristic_skip"))
        )
        logger.info(
            "[phase2] boundary seam gate: seams_total=%d seams_skipped_by_heuristic=%d seams_sent_to_llm=%d",
            len(batch_nodes) - 1,
            seams_skipped,
            max(0, (len(batch_nodes) - 1) - seams_skipped),
        )

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
    llm_client: LLMGenerateJsonClient,
    max_output_tokens: int = 32768,
    model: str | None = None,
) -> tuple[list[SemanticGraphNode], dict[str, Any]]:
    overlap_payload = {
        "left_batch_nodes": [node.model_dump(mode="json") for node in left_batch_nodes],
        "right_batch_nodes": [node.model_dump(mode="json") for node in right_batch_nodes],
    }
    started = time.perf_counter()
    prompt = build_boundary_reconciliation_prompt(overlap_payload=overlap_payload)
    response = llm_client.generate_json(
        prompt=prompt,
        model=model,
        temperature=0.0,
        response_schema=BOUNDARY_RECONCILIATION_SCHEMA,
        max_output_tokens=max_output_tokens,
    )
    reconciled = reconcile_boundary_nodes(
        left_batch_nodes=left_batch_nodes,
        right_batch_nodes=right_batch_nodes,
        llm_response=response,
    )
    return reconciled, {
        "prompt": prompt,
        "response": response,
        "diagnostics": {
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "left_node_count": len(left_batch_nodes),
            "right_node_count": len(right_batch_nodes),
            "reconciled_node_count": len(reconciled),
            "prompt_chars": len(prompt),
            "prompt_token_estimate": max(1, round(len(prompt) / 4.0)),
            "payload_chars": len(json.dumps(overlap_payload, ensure_ascii=True, separators=(",", ":"))),
            "response_chars": len(json.dumps(response, ensure_ascii=True, separators=(",", ":"))),
        },
    }


def embed_text_semantic_nodes_live(
    *,
    nodes: list[SemanticGraphNode],
    embedding_client: EmbeddingClient,
) -> tuple[list[list[float]], dict[str, Any]]:
    texts = [build_semantic_embedding_payload(node=node) for node in nodes]
    started = time.perf_counter()
    embeddings = embedding_client.embed_texts(
        texts,
        task_type="RETRIEVAL_DOCUMENT",
    )
    duration_s = time.perf_counter() - started
    logger.info(
        "[phase2] semantic embeddings done in %.1f s (nodes=%d)",
        duration_s,
        len(nodes),
    )
    return embeddings, {
        "node_count": len(nodes),
        "semantic_payload_chars": sum(len(text) for text in texts),
        "semantic_duration_ms": duration_s * 1000.0,
    }


def embed_multimodal_media_live(
    *,
    multimodal_media: list[dict[str, str]],
    embedding_client: EmbeddingClient,
) -> tuple[list[list[float]], dict[str, Any]]:
    started = time.perf_counter()
    embeddings = embedding_client.embed_media_uris(multimodal_media)
    duration_s = time.perf_counter() - started
    logger.info(
        "[phase2] multimodal embeddings done in %.1f s (nodes=%d)",
        duration_s,
        len(multimodal_media),
    )
    return embeddings, {
        "multimodal_item_count": len(multimodal_media),
        "multimodal_duration_ms": duration_s * 1000.0,
    }


def apply_node_embeddings(
    *,
    nodes: list[SemanticGraphNode],
    semantic_embeddings: list[list[float]],
    multimodal_embeddings: list[list[float]],
) -> list[SemanticGraphNode]:
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


def embed_semantic_nodes_live(
    *,
    nodes: list[SemanticGraphNode],
    embedding_client: EmbeddingClient,
    multimodal_media: list[dict[str, str]],
    return_diagnostics: bool = False,
) -> list[SemanticGraphNode] | tuple[list[SemanticGraphNode], dict[str, Any]]:
    if len(multimodal_media) != len(nodes):
        raise ValueError("multimodal_media must align one-to-one with nodes")
    futures: dict[Any, str] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures[pool.submit(embed_text_semantic_nodes_live, nodes=nodes, embedding_client=embedding_client)] = "semantic"
        futures[pool.submit(embed_multimodal_media_live, multimodal_media=multimodal_media, embedding_client=embedding_client)] = "multimodal"
        results: dict[str, tuple[list[list[float]], dict[str, Any]]] = {}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:  # pragma: no cover - passthrough
                raise RuntimeError(f"Phase 2 {name} embedding request failed.") from exc

    semantic_embeddings, semantic_diagnostics = results["semantic"]
    multimodal_embeddings, multimodal_diagnostics = results["multimodal"]
    embedded_nodes = apply_node_embeddings(
        nodes=nodes,
        semantic_embeddings=semantic_embeddings,
        multimodal_embeddings=multimodal_embeddings,
    )
    diagnostics = {
        "node_count": len(nodes),
        **semantic_diagnostics,
        **multimodal_diagnostics,
    }
    if return_diagnostics:
        return embedded_nodes, diagnostics
    return embedded_nodes


__all__ = [
    "apply_node_embeddings",
    "embed_multimodal_media_live",
    "embed_semantic_nodes_live",
    "embed_text_semantic_nodes_live",
    "prepare_node_media_embeddings",
    "run_boundary_reconciliation",
    "run_merge_and_classify_batches",
    "run_merge_classify_and_reconcile",
]
