from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .local_semantic_edges import build_local_semantic_edges
from .long_range_edges import build_long_range_edges, shortlist_long_range_pairs
from .prompts import LOCAL_SEMANTIC_EDGE_SCHEMA, LONG_RANGE_EDGE_SCHEMA, build_local_semantic_edge_prompt, build_long_range_edge_prompt
from .reconcile_edges import reconcile_semantic_edges
from ..contracts import SemanticGraphEdge, SemanticGraphNode


def _build_node_batches(
    *,
    nodes: list[SemanticGraphNode],
    target_batch_count: int = 5,
    max_nodes_per_batch: int = 15,
) -> list[dict[str, Any]]:
    total_nodes = len(nodes)
    node_count = (
        min(math.ceil(total_nodes / target_batch_count), max_nodes_per_batch)
        if total_nodes > 0
        else max_nodes_per_batch
    )
    halo_count = max(1, node_count // 7)
    if node_count <= 0:
        raise ValueError("computed node_count must be positive")
    if not nodes:
        return []

    ordered = sorted(nodes, key=lambda node: (node.start_ms, node.end_ms, node.node_id))
    batches: list[dict[str, Any]] = []
    for start_idx in range(0, len(ordered), node_count):
        target_nodes = ordered[start_idx : start_idx + node_count]
        if not target_nodes:
            continue
        left = max(0, start_idx - halo_count)
        right = min(len(ordered), start_idx + node_count + halo_count)
        context_nodes = ordered[left:right]
        batches.append(
            {
                "batch_id": f"edge_batch_{len(batches) + 1:04d}",
                "target_node_ids": [node.node_id for node in target_nodes],
                "context_node_ids": [node.node_id for node in context_nodes],
                "nodes": [node.model_dump(mode="json") for node in context_nodes],
            }
        )
    return batches


def run_local_semantic_edge_batches(
    *,
    nodes: list[SemanticGraphNode],
    llm_client: Any,
    target_batch_count: int = 5,
    max_nodes_per_batch: int = 15,
    model: str | None = None,
    max_concurrent: int = 5,
) -> tuple[list[SemanticGraphEdge], list[dict[str, Any]]]:
    batches = _build_node_batches(
        nodes=nodes,
        target_batch_count=target_batch_count,
        max_nodes_per_batch=max_nodes_per_batch,
    )

    def _call_batch(batch):
        prompt = build_local_semantic_edge_prompt(batch_payload=batch)
        response = llm_client.generate_json(prompt=prompt, model=model, temperature=0.0, response_schema=LOCAL_SEMANTIC_EDGE_SCHEMA, max_output_tokens=2048)
        return batch, prompt, response

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = [pool.submit(_call_batch, b) for b in batches]
        batch_call_results = [f.result() for f in futures]

    raw_responses: list[dict[str, Any]] = []
    debug: list[dict[str, Any]] = []
    for batch, prompt, response in batch_call_results:
        raw_responses.append(
            {
                "batch_id": batch["batch_id"],
                "target_node_ids": batch["target_node_ids"],
                "context_node_ids": batch["context_node_ids"],
                "edges": list(response.get("edges") or []),
            }
        )
        debug.append(
            {
                "batch_id": batch["batch_id"],
                "prompt": prompt,
                "response": response,
            }
        )
    return build_local_semantic_edges(nodes=nodes, gemini_responses=raw_responses), debug


def run_long_range_edge_adjudication(
    *,
    nodes: list[SemanticGraphNode],
    llm_client: Any,
    top_k: int = 3,
    model: str | None = None,
) -> tuple[list[SemanticGraphEdge], dict[str, Any]]:
    pairs = shortlist_long_range_pairs(nodes=nodes, top_k=top_k)
    prompt = build_long_range_edge_prompt(pair_payload={"candidate_pairs": pairs})
    response = llm_client.generate_json(prompt=prompt, model=model, temperature=0.0, response_schema=LONG_RANGE_EDGE_SCHEMA, max_output_tokens=1024)
    edges = build_long_range_edges(candidate_pairs=pairs, gemini_response=response)
    return edges, {
        "candidate_pairs": pairs,
        "prompt": prompt,
        "response": response,
    }


def reconcile_live_semantic_edges(*, edges: list[SemanticGraphEdge]) -> list[SemanticGraphEdge]:
    return reconcile_semantic_edges(edges=edges)


__all__ = [
    "reconcile_live_semantic_edges",
    "run_local_semantic_edge_batches",
    "run_long_range_edge_adjudication",
]
