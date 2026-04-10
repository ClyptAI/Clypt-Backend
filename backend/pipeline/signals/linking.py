from __future__ import annotations

from collections import defaultdict, deque
import math
from typing import Any

from backend.pipeline._embedding_utils import cosine_similarity
from backend.pipeline.candidates.build_local_subgraphs import build_local_subgraphs
from backend.pipeline.candidates.seed_retrieval import retrieve_seed_nodes
from backend.pipeline.config import Phase4SubgraphConfig
from backend.pipeline.contracts import SemanticGraphEdge, SemanticGraphNode

from .contracts import ExternalSignalCluster, NodeSignalLink, SignalPromptSpec
from .llm_runtime import SignalLLMCallError, resolve_cluster_span_with_llm


def build_node_signal_links(
    *,
    clusters: list[ExternalSignalCluster],
    prompt_specs: list[SignalPromptSpec],
    prompt_embeddings: dict[str, list[float]],
    nodes: list[SemanticGraphNode],
    edges: list[SemanticGraphEdge],
    llm_client: Any,
    model: str,
    thinking_level: str,
    max_hops: int,
    time_window_ms: int,
    fail_fast: bool = True,
) -> list[NodeSignalLink]:
    if not clusters:
        return []

    node_by_id = {node.node_id: node for node in nodes}
    cluster_by_id = {cluster.cluster_id: cluster for cluster in clusters}
    prompt_by_cluster = {
        prompt.source_cluster_id: prompt
        for prompt in prompt_specs
        if prompt.source_cluster_id and prompt.prompt_source_type in {"comment", "trend"}
    }
    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.source_node_id].append((edge.target_node_id, edge.edge_type))
        adjacency[edge.target_node_id].append((edge.source_node_id, edge.edge_type))

    output: list[NodeSignalLink] = []
    for cluster_id, cluster in cluster_by_id.items():
        prompt = prompt_by_cluster.get(cluster_id)
        if prompt is None:
            continue
        prompt_embedding = prompt_embeddings.get(prompt.prompt_id)
        if not prompt_embedding:
            continue

        seed_hits = retrieve_seed_nodes(
            prompts=[{"prompt_id": prompt.prompt_id, "embedding": prompt_embedding}],
            nodes=nodes,
            top_k_per_prompt=5,
        )
        if not seed_hits:
            continue

        seed_ids = [hit["node_id"] for hit in seed_hits]
        hit_by_id = {hit["node_id"]: hit for hit in seed_hits}
        neighborhood_ids = _build_neighborhood(seed_ids=seed_ids, nodes=nodes, limit=12)
        neighborhood_payload = {
            "node_ids": neighborhood_ids,
            "nodes": [
                {
                    "node_id": node_id,
                    "start_ms": node_by_id[node_id].start_ms,
                    "end_ms": node_by_id[node_id].end_ms,
                    "node_type": node_by_id[node_id].node_type,
                    "summary": node_by_id[node_id].summary,
                }
                for node_id in neighborhood_ids
                if node_id in node_by_id
            ],
        }

        selected_ids = resolve_cluster_span_with_llm(
            llm_client=llm_client,
            model=model,
            thinking_level=thinking_level,
            cluster=cluster,
            neighborhood_payload=neighborhood_payload,
            fail_fast=fail_fast,
        )
        direct_ids = [node_id for node_id in selected_ids if node_id in node_by_id]
        if not direct_ids:
            raise SignalLLMCallError(
                callpoint_id="5",
                message=(
                    f"callpoint_5_resolve_cluster_span returned no valid node_ids for cluster={cluster_id}"
                ),
            )

        for node_id in direct_ids:
            node = node_by_id[node_id]
            similarity = float(hit_by_id.get(node_id, {}).get("retrieval_score") or 0.0)
            if similarity == 0.0 and node.semantic_embedding:
                sim = cosine_similarity(prompt_embedding, node.semantic_embedding)
                similarity = 0.0 if sim == float("-inf") else float(sim)
            output.append(
                NodeSignalLink(
                    node_id=node_id,
                    cluster_id=cluster_id,
                    link_type="direct",
                    hop_distance=0,
                    time_offset_ms=0,
                    similarity=similarity,
                    evidence={
                        "prompt_id": prompt.prompt_id,
                        "method": "seed_and_span",
                    },
                )
            )

        if max_hops <= 0:
            continue

        expansion_seeds = [
            {
                "node_id": node_id,
                "source_prompt_ids": [prompt.prompt_id],
                "retrieval_score": float(hit_by_id.get(node_id, {}).get("retrieval_score") or 0.0),
            }
            for node_id in direct_ids
        ]
        subgraph_cfg = Phase4SubgraphConfig(
            max_duration_s=max(1, int(math.ceil(time_window_ms / 1000.0))),
            max_hop_depth=max_hops,
        )
        expansion_subgraphs = build_local_subgraphs(
            seeds=expansion_seeds,
            nodes=nodes,
            edges=edges,
            config=subgraph_cfg,
        )
        allowed_inferred_ids = {
            node.node_id
            for subgraph in expansion_subgraphs
            for node in subgraph.nodes
            if node.node_id not in direct_ids
        }

        visited = set(direct_ids)
        queue: deque[tuple[str, int, int]] = deque((node_id, 0, node_by_id[node_id].start_ms) for node_id in direct_ids)
        while queue:
            node_id, hop, anchor_ms = queue.popleft()
            if hop >= max_hops:
                continue
            for neighbor_id, edge_type in adjacency.get(node_id, []):
                if neighbor_id in visited:
                    continue
                if neighbor_id not in allowed_inferred_ids:
                    continue
                neighbor = node_by_id.get(neighbor_id)
                if neighbor is None:
                    continue
                time_offset = abs(int(neighbor.start_ms) - int(anchor_ms))
                if time_offset > time_window_ms:
                    continue
                visited.add(neighbor_id)
                next_hop = hop + 1
                queue.append((neighbor_id, next_hop, anchor_ms))
                similarity = 0.0
                if neighbor.semantic_embedding:
                    sim = cosine_similarity(prompt_embedding, neighbor.semantic_embedding)
                    similarity = 0.0 if sim == float("-inf") else float(sim)
                output.append(
                    NodeSignalLink(
                        node_id=neighbor_id,
                        cluster_id=cluster_id,
                        link_type="inferred",
                        hop_distance=next_hop,
                        time_offset_ms=int(time_offset),
                        similarity=similarity,
                        evidence={
                            "from_node": node_id,
                            "edge_type": edge_type,
                            "prompt_id": prompt.prompt_id,
                        },
                    )
                )

    return _dedupe_node_signal_links(output)


def _build_neighborhood(*, seed_ids: list[str], nodes: list[SemanticGraphNode], limit: int) -> list[str]:
    node_order = sorted(nodes, key=lambda node: (node.start_ms, node.end_ms, node.node_id))
    idx_by_id = {node.node_id: idx for idx, node in enumerate(node_order)}
    selected: set[str] = set(seed_ids)
    for seed_id in seed_ids:
        idx = idx_by_id.get(seed_id)
        if idx is None:
            continue
        for delta in range(-2, 3):
            candidate_idx = idx + delta
            if 0 <= candidate_idx < len(node_order):
                selected.add(node_order[candidate_idx].node_id)
            if len(selected) >= max(1, limit):
                break
    ordered = [node.node_id for node in node_order if node.node_id in selected]
    return ordered[: max(1, limit)]


def _dedupe_node_signal_links(links: list[NodeSignalLink]) -> list[NodeSignalLink]:
    best: dict[tuple[str, str], NodeSignalLink] = {}
    for link in links:
        key = (link.cluster_id, link.node_id)
        existing = best.get(key)
        if existing is None:
            best[key] = link
            continue
        if link.link_type == "direct" and existing.link_type != "direct":
            best[key] = link
            continue
        if link.link_type == existing.link_type and link.similarity > existing.similarity:
            best[key] = link
            continue
        if link.link_type == existing.link_type and link.hop_distance < existing.hop_distance:
            best[key] = link
    return sorted(
        best.values(),
        key=lambda item: (item.cluster_id, item.node_id, item.hop_distance, item.link_type),
    )


__all__ = ["build_node_signal_links"]
