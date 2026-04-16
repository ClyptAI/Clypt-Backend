from __future__ import annotations

from backend.pipeline.candidates.runtime import _compact_candidate_payload, _compact_subgraph_payload
from backend.pipeline.candidates.prompts import build_pooled_candidate_review_prompt
from backend.pipeline.contracts import ClipCandidate, LocalSubgraph, LocalSubgraphNode


def test_pooled_review_prompt_warns_about_exact_candidate_accounting_failure() -> None:
    prompt = build_pooled_candidate_review_prompt(
        candidate_payload={
            "candidates": [
                {"clip_id": "cand_tmp_001", "summary": "first"},
                {"clip_id": "cand_tmp_002", "summary": "second"},
            ]
        }
    )

    assert "pooled review must account for every candidate exactly once" in prompt
    assert "Before returning, reconcile the full input clip_id set" in prompt
    assert "If any clip_id is missing, duplicated, or appears in both kept and dropped outputs" in prompt


def test_phase4_runtime_compacts_subgraph_and_candidate_payloads() -> None:
    subgraph_payload = _compact_subgraph_payload(
        LocalSubgraph(
            subgraph_id="sg_0001",
            seed_node_id="node_seed",
            source_prompt_ids=["prompt_001"],
            start_ms=0,
            end_ms=1200,
            nodes=[
                LocalSubgraphNode(
                    node_id="node_seed",
                    start_ms=0,
                    end_ms=1200,
                    duration_ms=1200,
                    node_type="claim",
                    node_flags=[],
                    summary="Strong claim.",
                    transcript_excerpt="Strong claim transcript.",
                    word_count=3,
                    emotion_labels=[],
                    audio_events=[],
                    inbound_edges=[],
                    outbound_edges=[],
                )
            ],
        )
    )
    assert "duration_ms" not in subgraph_payload["nodes"][0]
    assert "word_count" not in subgraph_payload["nodes"][0]
    assert subgraph_payload["nodes"][0]["summary"] == "Strong claim."

    candidate_payload = _compact_candidate_payload(
        [
            ClipCandidate(
                clip_id="cand_001",
                node_ids=["node_seed"],
                start_ms=0,
                end_ms=1200,
                score=0.91,
                rationale="Best clip.",
                source_prompt_ids=["prompt_001"],
                seed_node_id="node_seed",
                subgraph_id="sg_0001",
                query_aligned=True,
            )
        ]
    )
    assert candidate_payload == {
        "candidates": [
            {
                "clip_id": "cand_001",
                "node_ids": ["node_seed"],
                "start_ms": 0,
                "end_ms": 1200,
                "score": 0.91,
                "rationale": "Best clip.",
                "source_prompt_ids": ["prompt_001"],
                "seed_node_id": "node_seed",
                "subgraph_id": "sg_0001",
                "query_aligned": True,
            }
        ]
    }
