from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from backend.providers.protocols import LLMGenerateJsonClient
from .local_semantic_edges import LOCAL_EDGE_TYPES, build_local_semantic_edges
from .long_range_edges import build_long_range_edges, shortlist_long_range_pairs
from .prompts import LOCAL_SEMANTIC_EDGE_SCHEMA, LONG_RANGE_EDGE_SCHEMA, build_local_semantic_edge_prompt, build_long_range_edge_prompt
from .reconcile_edges import reconcile_semantic_edges
from ..contracts import SemanticGraphEdge, SemanticGraphNode


def _serialize_node_for_llm(node: SemanticGraphNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "start_ms": node.start_ms,
        "end_ms": node.end_ms,
        "node_type": node.node_type,
        "node_flags": list(node.node_flags),
        "summary": node.summary,
        "transcript_text": node.transcript_text,
    }


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
                "nodes": [_serialize_node_for_llm(node) for node in context_nodes],
            }
        )
    return batches


def run_local_semantic_edge_batches(
    *,
    nodes: list[SemanticGraphNode],
    llm_client: LLMGenerateJsonClient,
    target_batch_count: int = 5,
    max_nodes_per_batch: int = 15,
    model: str | None = None,
    max_concurrent: int = 5,
    max_output_tokens: int = 32768,
) -> tuple[list[SemanticGraphEdge], list[dict[str, Any]]]:
    batches = _build_node_batches(
        nodes=nodes,
        target_batch_count=target_batch_count,
        max_nodes_per_batch=max_nodes_per_batch,
    )

    def _call_batch(batch):
        started = time.perf_counter()
        prompt = build_local_semantic_edge_prompt(batch_payload=batch)
        response = llm_client.generate_json(
            prompt=prompt,
            model=model,
            temperature=0.0,
            response_schema=LOCAL_SEMANTIC_EDGE_SCHEMA,
            max_output_tokens=max_output_tokens,
        )
        return batch, prompt, response, {
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "target_node_count": len(batch["target_node_ids"]),
            "context_node_count": len(batch["context_node_ids"]),
            "prompt_chars": len(prompt),
            "prompt_token_estimate": max(1, round(len(prompt) / 4.0)),
            "payload_chars": len(json.dumps(batch, ensure_ascii=True, separators=(",", ":"))),
            "response_chars": len(json.dumps(response, ensure_ascii=True, separators=(",", ":"))),
        }

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = [pool.submit(_call_batch, b) for b in batches]
        batch_call_results = [f.result() for f in futures]

    node_ids = {node.node_id for node in nodes}
    raw_responses: list[dict[str, Any]] = []
    debug: list[dict[str, Any]] = []
    for batch, prompt, response, diagnostics in batch_call_results:
        target_node_ids = list(batch["target_node_ids"])
        context_node_ids = list(batch["context_node_ids"])
        target_node_id_set = set(target_node_ids)
        context_node_id_set = set(context_node_ids)
        sanitized_edges: list[dict[str, Any]] = []
        dropped_edges: list[dict[str, Any]] = []
        for raw_edge in list(response.get("edges") or []):
            source_node_id = str(raw_edge.get("source_node_id") or "").strip()
            target_node_id = str(raw_edge.get("target_node_id") or "").strip()
            edge_type = str(raw_edge.get("edge_type") or "").strip()
            drop_reason: str | None = None
            if source_node_id not in target_node_id_set:
                drop_reason = "source_outside_target_block"
            elif target_node_id not in context_node_id_set:
                drop_reason = "target_outside_context_block"
            elif edge_type not in LOCAL_EDGE_TYPES:
                drop_reason = "invalid_edge_type"
            elif source_node_id not in node_ids or target_node_id not in node_ids:
                drop_reason = "unknown_node_id"
            if drop_reason is not None:
                dropped_edges.append(
                    {
                        "reason": drop_reason,
                        "edge": raw_edge,
                    }
                )
                continue
            sanitized_edges.append(
                {
                    "source_node_id": source_node_id,
                    "target_node_id": target_node_id,
                    "edge_type": edge_type,
                    "rationale": raw_edge.get("rationale"),
                    "confidence": raw_edge.get("confidence"),
                }
            )
        raw_responses.append(
            {
                "batch_id": batch["batch_id"],
                "target_node_ids": target_node_ids,
                "context_node_ids": context_node_ids,
                "edges": sanitized_edges,
            }
        )
        debug.append(
            {
                "batch_id": batch["batch_id"],
                "prompt": prompt,
                "response": response,
                "diagnostics": diagnostics,
                "sanitized_edge_count": len(sanitized_edges),
                "dropped_edge_count": len(dropped_edges),
                "dropped_edges": dropped_edges,
            }
        )
    return build_local_semantic_edges(nodes=nodes, llm_responses=raw_responses), debug


def _build_long_range_pair_shards(
    *,
    candidate_pairs: list[dict[str, Any]],
    pairs_per_shard: int,
) -> list[dict[str, Any]]:
    if pairs_per_shard <= 0:
        raise ValueError("pairs_per_shard must be positive")
    return [
        {
            "shard_id": f"long_range_shard_{idx + 1:04d}",
            "candidate_pairs": candidate_pairs[start_idx : start_idx + pairs_per_shard],
        }
        for idx, start_idx in enumerate(range(0, len(candidate_pairs), pairs_per_shard))
    ]


def run_long_range_edge_adjudication(
    *,
    nodes: list[SemanticGraphNode],
    llm_client: LLMGenerateJsonClient,
    top_k: int = 3,
    model: str | None = None,
    max_output_tokens: int = 32768,
    pairs_per_shard: int = 24,
    max_concurrent: int = 4,
) -> tuple[list[SemanticGraphEdge], dict[str, Any]]:
    pairs = shortlist_long_range_pairs(nodes=nodes, top_k=top_k)
    if not pairs:
        return [], {
            "candidate_pairs": [],
            "prompt": None,
            "response": None,
            "shards": [],
            "diagnostics": {
                "latency_ms": 0.0,
                "candidate_pair_count": 0,
                "accepted_edge_count": 0,
                "shard_count": 0,
                "prompt_chars": 0,
                "prompt_token_estimate": 0,
                "payload_chars": 0,
                "response_chars": 0,
            },
        }

    shards = _build_long_range_pair_shards(
        candidate_pairs=pairs,
        pairs_per_shard=pairs_per_shard,
    )
    started = time.perf_counter()

    def _call_shard(shard: dict[str, Any]):
        shard_started = time.perf_counter()
        shard_pairs = list(shard["candidate_pairs"])
        prompt = build_long_range_edge_prompt(pair_payload={"candidate_pairs": shard_pairs})
        response = llm_client.generate_json(
            prompt=prompt,
            model=model,
            temperature=0.0,
            response_schema=LONG_RANGE_EDGE_SCHEMA,
            max_output_tokens=max_output_tokens,
        )
        edges = build_long_range_edges(candidate_pairs=shard_pairs, llm_response=response)
        return {
            "shard_id": shard["shard_id"],
            "candidate_pairs": shard_pairs,
            "prompt": prompt,
            "response": response,
            "edges": edges,
            "diagnostics": {
                "latency_ms": (time.perf_counter() - shard_started) * 1000.0,
                "candidate_pair_count": len(shard_pairs),
                "accepted_edge_count": len(edges),
                "prompt_chars": len(prompt),
                "prompt_token_estimate": max(1, round(len(prompt) / 4.0)),
                "payload_chars": len(json.dumps({"candidate_pairs": shard_pairs}, ensure_ascii=True, separators=(",", ":"))),
                "response_chars": len(json.dumps(response, ensure_ascii=True, separators=(",", ":"))),
            },
        }

    with ThreadPoolExecutor(max_workers=max(1, max_concurrent)) as pool:
        futures = [pool.submit(_call_shard, shard) for shard in shards]
        shard_results = [future.result() for future in futures]

    edges: list[SemanticGraphEdge] = []
    total_prompt_chars = 0
    total_prompt_token_estimate = 0
    total_payload_chars = 0
    total_response_chars = 0
    shard_debug: list[dict[str, Any]] = []
    for shard_result in shard_results:
        edges.extend(shard_result["edges"])
        diagnostics = dict(shard_result["diagnostics"])
        total_prompt_chars += int(diagnostics.get("prompt_chars") or 0)
        total_prompt_token_estimate += int(diagnostics.get("prompt_token_estimate") or 0)
        total_payload_chars += int(diagnostics.get("payload_chars") or 0)
        total_response_chars += int(diagnostics.get("response_chars") or 0)
        shard_debug.append(
            {
                "shard_id": shard_result["shard_id"],
                "candidate_pairs": shard_result["candidate_pairs"],
                "prompt": shard_result["prompt"],
                "response": shard_result["response"],
                "diagnostics": diagnostics,
            }
        )

    return edges, {
        "candidate_pairs": pairs,
        "prompt": shard_results[0]["prompt"] if len(shard_results) == 1 else None,
        "response": shard_results[0]["response"] if len(shard_results) == 1 else None,
        "shards": shard_debug,
        "diagnostics": {
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "candidate_pair_count": len(pairs),
            "accepted_edge_count": len(edges),
            "shard_count": len(shard_results),
            "prompt_chars": total_prompt_chars,
            "prompt_token_estimate": total_prompt_token_estimate,
            "payload_chars": total_payload_chars,
            "response_chars": total_response_chars,
        },
    }


def reconcile_live_semantic_edges(*, edges: list[SemanticGraphEdge]) -> list[SemanticGraphEdge]:
    return reconcile_semantic_edges(edges=edges)


__all__ = [
    "reconcile_live_semantic_edges",
    "run_local_semantic_edge_batches",
    "run_long_range_edge_adjudication",
]
