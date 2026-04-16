from __future__ import annotations

from backend.pipeline.candidates.prompts import SUBGRAPH_REVIEW_SCHEMA
from backend.pipeline.candidates.review_subgraphs import review_local_subgraph
from backend.pipeline.contracts import LocalSubgraph, LocalSubgraphNode


def _node(node_id: str, start_ms: int, end_ms: int) -> LocalSubgraphNode:
    return LocalSubgraphNode(
        node_id=node_id,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=end_ms - start_ms,
        node_type="claim",
        node_flags=[],
        summary=node_id,
        transcript_excerpt=node_id,
        word_count=4,
        emotion_labels=[],
        audio_events=[],
        inbound_edges=[],
        outbound_edges=[],
    )


def _subgraph() -> LocalSubgraph:
    return LocalSubgraph(
        subgraph_id="sg_schema_compat",
        seed_node_id="n1",
        source_prompt_ids=[],
        start_ms=0,
        end_ms=9000,
        nodes=[
            _node("n1", 0, 3000),
            _node("n2", 3000, 6000),
            _node("n3", 6000, 9000),
        ],
    )


def test_subgraph_review_schema_omits_dynamic_oneof() -> None:
    assert "oneOf" not in SUBGRAPH_REVIEW_SCHEMA


def test_review_local_subgraph_requires_reject_reason_when_reject_all_true() -> None:
    response = review_local_subgraph(
        subgraph=_subgraph(),
        llm_response={
            "subgraph_id": "sg_schema_compat",
            "seed_node_id": "n1",
            "reject_all": True,
            "reject_reason": "",
            "candidates": [],
        },
    )
    assert response.reject_all is True
    assert response.candidates == []
    assert "invalid_structured_output" in response.reject_reason
