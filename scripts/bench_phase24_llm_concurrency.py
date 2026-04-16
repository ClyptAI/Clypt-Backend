from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.pipeline.candidates.prompts import (
    META_PROMPT_GENERATION_SCHEMA,
    POOL_REVIEW_SCHEMA,
    SUBGRAPH_REVIEW_SCHEMA,
    build_meta_prompt_generation_prompt,
    build_pooled_candidate_review_prompt,
    build_subgraph_review_prompt,
)
from backend.pipeline.candidates.runtime import (
    _compact_candidate_payload,
    _compact_subgraph_payload,
)
from backend.pipeline.contracts import (
    ClipCandidate,
    LocalSubgraph,
    LocalSubgraphNode,
    SemanticGraphNode,
    SemanticNodeEvidence,
)
from backend.pipeline.graph.prompts import (
    LOCAL_SEMANTIC_EDGE_SCHEMA,
    LONG_RANGE_EDGE_SCHEMA,
    build_local_semantic_edge_prompt,
    build_long_range_edge_prompt,
)
from backend.pipeline.signals.contracts import SignalPromptSpec
from backend.pipeline.semantics.prompts import (
    MERGE_AND_CLASSIFY_SCHEMA,
    build_merge_and_classify_prompt,
)
from backend.providers.config import LocalGenerationSettings
from backend.providers.openai_local import LocalOpenAIQwenClient


@dataclass(slots=True)
class ScenarioSpec:
    name: str
    prompt: str
    response_schema: dict[str, Any]
    max_output_tokens: int


def _semantic_node(index: int) -> SemanticGraphNode:
    start_ms = index * 1800
    end_ms = start_ms + 1400
    node_type = [
        "claim",
        "explanation",
        "example",
        "reaction_beat",
        "setup_payoff",
        "reveal",
    ][index % 6]
    flags = ["high_resonance_candidate"] if index % 5 == 0 else []
    return SemanticGraphNode(
        node_id=f"node_{index:03d}",
        node_type=node_type,
        start_ms=start_ms,
        end_ms=end_ms,
        source_turn_ids=[f"turn_{index:03d}"],
        word_ids=[],
        transcript_text=(
            f"Node {index} transcript about a debate beat, callback, reveal, or explanation "
            f"that could matter for clip selection."
        ),
        node_flags=flags,
        summary=(
            f"Node {index} summarizes a concrete conversational beat with setup, payoff, "
            f"reaction, or explanation value."
        ),
        evidence=SemanticNodeEvidence(
            emotion_labels=["happy"] if index % 4 == 0 else [],
            audio_events=["laughter"] if index % 6 == 0 else [],
        ),
        semantic_embedding=[1.0, 0.0],
        multimodal_embedding=[1.0, 0.0],
    )


def _local_subgraph() -> LocalSubgraph:
    nodes: list[LocalSubgraphNode] = []
    for index in range(8):
        start_ms = index * 2200
        end_ms = start_ms + 1600
        nodes.append(
            LocalSubgraphNode(
                node_id=f"sg_node_{index:03d}",
                start_ms=start_ms,
                end_ms=end_ms,
                duration_ms=end_ms - start_ms,
                node_type=["claim", "setup_payoff", "reaction_beat", "explanation"][index % 4],
                node_flags=["high_resonance_candidate"] if index in {1, 5} else [],
                summary=f"Subgraph node {index} captures a specific moment worth evaluating.",
                transcript_excerpt=(
                    f"Transcript excerpt for subgraph node {index} with enough semantic detail "
                    f"to evaluate clip quality and standalone clarity."
                ),
                word_count=18,
                emotion_labels=["happy"] if index % 3 == 0 else [],
                audio_events=["applause"] if index == 5 else [],
                inbound_edges=[],
                outbound_edges=[],
            )
        )
    return LocalSubgraph(
        subgraph_id="sg_bench_0001",
        seed_node_id="sg_node_001",
        source_prompt_ids=["prompt_general_001", "prompt_comment_001"],
        start_ms=nodes[0].start_ms,
        end_ms=nodes[-1].end_ms,
        nodes=nodes,
    )


def _pool_candidates() -> list[ClipCandidate]:
    candidates: list[ClipCandidate] = []
    for index in range(12):
        start_ms = index * 3000
        end_ms = start_ms + 1800
        candidates.append(
            ClipCandidate(
                clip_id=f"cand_{index:03d}",
                node_ids=[f"node_{index:03d}"],
                start_ms=start_ms,
                end_ms=end_ms,
                score=0.95 - (index * 0.03),
                rationale=f"Candidate {index} has a strong hook and clean standalone shape.",
                source_prompt_ids=["prompt_general_001"],
                seed_node_id=f"node_{index:03d}",
                subgraph_id="sg_bench_0001",
                query_aligned=index % 2 == 0,
            )
        )
    return candidates


def _scenario_phase2_merge() -> ScenarioSpec:
    neighborhood_payload = {
        "batch_id": "merge_batch_0001",
        "target_turn_ids": [f"turn_{idx:03d}" for idx in range(1, 13)],
        "context_turns": [
            {
                "turn_id": f"turn_{idx:03d}",
                "speaker_id": f"SPEAKER_{idx % 2}",
                "start_ms": idx * 1800,
                "end_ms": (idx * 1800) + 1200,
                "transcript_text": f"Turn {idx} advances a topic, setup, explanation, or reaction beat.",
                "emotion_labels": ["happy"] if idx % 4 == 0 else [],
                "audio_events": ["laughter"] if idx % 6 == 0 else [],
            }
            for idx in range(1, 15)
        ],
    }
    return ScenarioSpec(
        name="phase2_merge",
        prompt=build_merge_and_classify_prompt(neighborhood_payload=neighborhood_payload),
        response_schema=MERGE_AND_CLASSIFY_SCHEMA,
        max_output_tokens=2048,
    )


def _scenario_phase3_local() -> ScenarioSpec:
    nodes = [_semantic_node(index) for index in range(24)]
    batch_payload = {
        "batch_id": "edge_batch_0001",
        "target_node_ids": [node.node_id for node in nodes[:18]],
        "context_node_ids": [node.node_id for node in nodes],
        "nodes": [
            {
                "node_id": node.node_id,
                "start_ms": node.start_ms,
                "end_ms": node.end_ms,
                "node_type": node.node_type,
                "node_flags": list(node.node_flags),
                "summary": node.summary,
                "transcript_text": node.transcript_text,
            }
            for node in nodes
        ],
    }
    return ScenarioSpec(
        name="phase3_local",
        prompt=build_local_semantic_edge_prompt(batch_payload=batch_payload),
        response_schema=LOCAL_SEMANTIC_EDGE_SCHEMA,
        max_output_tokens=1536,
    )


def _scenario_phase3_long_range() -> ScenarioSpec:
    candidate_pairs = [
        {
            "earlier_node_id": f"node_{idx:03d}",
            "later_node_id": f"node_{idx + 12:03d}",
            "similarity": round(0.92 - (idx * 0.01), 3),
            "semantic_similarity": round(0.90 - (idx * 0.01), 3),
            "multimodal_similarity": round(0.88 - (idx * 0.01), 3),
        }
        for idx in range(24)
    ]
    return ScenarioSpec(
        name="phase3_long_range",
        prompt=build_long_range_edge_prompt(pair_payload={"candidate_pairs": candidate_pairs}),
        response_schema=LONG_RANGE_EDGE_SCHEMA,
        max_output_tokens=1792,
    )


def _scenario_phase4_meta() -> ScenarioSpec:
    nodes = [_semantic_node(index) for index in range(18)]
    node_summaries = [
        {
            "node_type": node.node_type,
            "node_flags": list(node.node_flags),
            "summary": node.summary,
            "start_ms": node.start_ms,
            "end_ms": node.end_ms,
        }
        for node in nodes
    ]
    return ScenarioSpec(
        name="phase4_meta",
        prompt=build_meta_prompt_generation_prompt(node_summaries=node_summaries, target_count=8),
        response_schema=META_PROMPT_GENERATION_SCHEMA,
        max_output_tokens=1024,
    )


def _scenario_phase4_subgraph() -> ScenarioSpec:
    subgraph = _local_subgraph()
    provenance_payload = {
        "subgraph_id": subgraph.subgraph_id,
        "seed_source_set": ["general", "comment"],
        "seed_prompt_ids": ["prompt_general_001", "prompt_comment_001"],
        "source_cluster_ids": ["cluster_comment_001"],
        "support_summary": {
            "seed_prompt_count": 2,
            "source_type_counts": {"general": 1, "comment": 1},
            "node_count": len(subgraph.nodes),
        },
        "selection_reason": "retained_after_dedupe",
    }
    return ScenarioSpec(
        name="phase4_subgraph",
        prompt=build_subgraph_review_prompt(
            subgraph_payload=_compact_subgraph_payload(subgraph),
            provenance_payload=provenance_payload,
        ),
        response_schema=SUBGRAPH_REVIEW_SCHEMA,
        max_output_tokens=2048,
    )


def _scenario_phase4_pool() -> ScenarioSpec:
    return ScenarioSpec(
        name="phase4_pool",
        prompt=build_pooled_candidate_review_prompt(
            candidate_payload=_compact_candidate_payload(_pool_candidates())
        ),
        response_schema=POOL_REVIEW_SCHEMA,
        max_output_tokens=768,
    )


def _build_scenario(name: str) -> ScenarioSpec:
    builders = {
        "phase2_merge": _scenario_phase2_merge,
        "phase3_local": _scenario_phase3_local,
        "phase3_long_range": _scenario_phase3_long_range,
        "phase4_meta": _scenario_phase4_meta,
        "phase4_subgraph": _scenario_phase4_subgraph,
        "phase4_pool": _scenario_phase4_pool,
    }
    if name not in builders:
        raise ValueError(f"unsupported scenario: {name}")
    return builders[name]()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil((percentile / 100.0) * len(ordered)) - 1))
    return ordered[rank]


def _run_once(client: LocalOpenAIQwenClient, scenario: ScenarioSpec, model: str) -> dict[str, Any]:
    started = time.perf_counter()
    parsed = client.generate_json(
        prompt=scenario.prompt,
        model=model,
        temperature=0.0,
        response_schema=scenario.response_schema,
        max_output_tokens=scenario.max_output_tokens,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "ok": True,
        "latency_ms": elapsed_ms,
        "response_chars": len(json.dumps(parsed, ensure_ascii=True, separators=(",", ":"))),
    }


def _run_level(
    *,
    client: LocalOpenAIQwenClient,
    scenario: ScenarioSpec,
    model: str,
    concurrency: int,
    rounds: int,
) -> dict[str, Any]:
    total_requests = max(1, concurrency * rounds)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_run_once, client, scenario, model)
            for _ in range(total_requests)
        ]
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover - operational harness
                results.append({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"})
    wall_clock_s = time.perf_counter() - started
    latencies = [float(item["latency_ms"]) for item in results if item.get("ok")]
    errors = [str(item["error"]) for item in results if not item.get("ok")]
    return {
        "scenario": scenario.name,
        "concurrency": concurrency,
        "rounds": rounds,
        "request_count": total_requests,
        "success_count": len(latencies),
        "error_count": len(errors),
        "wall_clock_s": wall_clock_s,
        "requests_per_s": (len(results) / wall_clock_s) if wall_clock_s > 0 else 0.0,
        "latency_ms": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "max": max(latencies) if latencies else 0.0,
            "mean": statistics.fmean(latencies) if latencies else 0.0,
        },
        "errors": errors[:5],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep real SGLang concurrency for representative Phase 2-4 prompts.")
    parser.add_argument(
        "--scenario",
        default="phase4_subgraph",
        choices=[
            "phase2_merge",
            "phase3_local",
            "phase3_long_range",
            "phase4_meta",
            "phase4_subgraph",
            "phase4_pool",
        ],
        help="Representative prompt family to benchmark.",
    )
    parser.add_argument(
        "--concurrency-values",
        default="1,2,4,6,8",
        help="Comma-separated concurrency values to test.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Requests per worker level are concurrency * rounds.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = _build_scenario(args.scenario)
    concurrency_values = [
        int(value.strip())
        for value in args.concurrency_values.split(",")
        if value.strip()
    ]
    client = LocalOpenAIQwenClient(
        settings=LocalGenerationSettings(
            base_url=args.base_url,
            model=args.model,
            timeout_s=args.timeout_s,
            max_retries=1,
            initial_backoff_s=0.5,
            max_backoff_s=2.0,
            backoff_multiplier=2.0,
            jitter_ratio=0.0,
            enable_thinking=False,
            temperature=0.0,
            top_p=1.0,
            top_k=40,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
        )
    )

    results = [
        _run_level(
            client=client,
            scenario=scenario,
            model=args.model,
            concurrency=concurrency,
            rounds=args.rounds,
        )
        for concurrency in concurrency_values
    ]
    best = max(
        (item for item in results if item["error_count"] == 0),
        key=lambda item: (item["requests_per_s"], -item["latency_ms"]["p95"]),
        default=None,
    )
    output = {
        "scenario": scenario.name,
        "base_url": args.base_url,
        "model": args.model,
        "rounds": args.rounds,
        "prompt_chars": len(scenario.prompt),
        "prompt_token_estimate": max(1, round(len(scenario.prompt) / 4.0)),
        "results": results,
        "best": best,
    }
    print(json.dumps(output, indent=2))
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
