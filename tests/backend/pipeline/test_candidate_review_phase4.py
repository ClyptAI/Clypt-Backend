from __future__ import annotations

import pytest

from backend.pipeline.candidates.dedupe_candidates import dedupe_clip_candidates
from backend.pipeline.candidates.review_candidate_pool import review_candidate_pool
from backend.pipeline.candidates.review_subgraphs import review_local_subgraph
from backend.pipeline.contracts import ClipCandidate, LocalSubgraph, LocalSubgraphNode


def _subgraph_node(
    node_id: str,
    start_ms: int,
    end_ms: int,
    *,
    node_type: str = "claim",
    summary: str | None = None,
) -> LocalSubgraphNode:
    return LocalSubgraphNode(
        node_id=node_id,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=end_ms - start_ms,
        node_type=node_type,
        node_flags=[],
        summary=summary or node_id,
        transcript_excerpt=node_id,
        word_count=3,
        emotion_labels=[],
        audio_events=[],
        inbound_edges=[],
        outbound_edges=[],
    )


def _clip_candidate(
    candidate_id: str | None,
    node_ids: list[str],
    start_ms: int,
    end_ms: int,
    *,
    score: float,
    rationale: str,
) -> ClipCandidate:
    return ClipCandidate(
        clip_id=candidate_id,
        node_ids=node_ids,
        start_ms=start_ms,
        end_ms=end_ms,
        score=score,
        rationale=rationale,
    )


def test_review_local_subgraph_validates_and_attaches_candidate_metadata():
    subgraph = LocalSubgraph(
        subgraph_id="sg_0001",
        seed_node_id="node_1",
        source_prompt_ids=["p1", "p2"],
        start_ms=0,
        end_ms=12000,
        nodes=[
            _subgraph_node("node_1", 0, 4000, node_type="setup_payoff"),
            _subgraph_node("node_2", 4500, 8000, node_type="reaction_beat"),
            _subgraph_node("node_3", 8200, 12000, node_type="reveal"),
        ],
    )

    response = review_local_subgraph(
        subgraph=subgraph,
        gemini_response={
            "subgraph_id": "sg_0001",
            "seed_node_id": "node_1",
            "reject_all": False,
            "reject_reason": "",
            "candidates": [
                {
                    "node_ids": ["node_1", "node_2", "node_3"],
                    "start_ms": 0,
                    "end_ms": 12000,
                    "score": 8.2,
                    "rationale": "Complete payoff with a clean reaction lift.",
                }
            ],
        },
    )

    assert response.reject_all is False
    assert response.subgraph_id == "sg_0001"
    assert response.seed_node_id == "node_1"
    assert len(response.candidates) == 1
    candidate = response.candidates[0]
    assert candidate.clip_id == "sg_0001_cand_01"
    assert candidate.source_prompt_ids == ["p1", "p2"]
    assert candidate.seed_node_id == "node_1"
    assert candidate.subgraph_id == "sg_0001"


def test_review_local_subgraph_treats_invalid_output_as_reject_all():
    subgraph = LocalSubgraph(
        subgraph_id="sg_0002",
        seed_node_id="node_1",
        source_prompt_ids=[],
        start_ms=0,
        end_ms=12000,
        nodes=[
            _subgraph_node("node_1", 0, 4000),
            _subgraph_node("node_2", 4500, 8000),
            _subgraph_node("node_3", 8200, 12000),
        ],
    )

    response = review_local_subgraph(
        subgraph=subgraph,
        gemini_response={
            "subgraph_id": "sg_0002",
            "seed_node_id": "node_1",
            "reject_all": False,
            "reject_reason": "",
            "candidates": [
                {
                    "node_ids": ["node_1", "node_3"],
                    "start_ms": 0,
                    "end_ms": 12000,
                    "score": 7.0,
                    "rationale": "Skips the middle node.",
                }
            ],
        },
    )

    assert response.reject_all is True
    assert response.candidates == []
    assert "invalid_structured_output" in response.reject_reason


def test_dedupe_clip_candidates_prefers_higher_scoring_duplicate():
    candidates = [
        _clip_candidate("cand_a", ["node_1", "node_2", "node_3"], 0, 12000, score=8.5, rationale="Complete payoff."),
        _clip_candidate("cand_b", ["node_1", "node_2", "node_3"], 0, 12000, score=7.8, rationale="Complete payoff."),
        _clip_candidate("cand_c", ["node_8"], 30000, 36000, score=6.0, rationale="Standalone reveal."),
    ]

    deduped = dedupe_clip_candidates(candidates=candidates)

    assert [candidate.clip_id for candidate in deduped] == ["cand_a", "cand_c"]


def test_dedupe_clip_candidates_uses_rationale_then_shorter_duration_tiebreak():
    candidates = [
        _clip_candidate("cand_a", ["node_1", "node_2"], 0, 10000, score=8.0, rationale="Self-contained payoff and complete reveal."),
        _clip_candidate("cand_b", ["node_1", "node_2"], 0, 10000, score=8.0, rationale="Interesting moment."),
        _clip_candidate("cand_c", ["node_4", "node_5", "node_6"], 20000, 30000, score=7.2, rationale="Strong complete clip."),
        _clip_candidate("cand_d", ["node_4", "node_5"], 20000, 29000, score=7.2, rationale="Strong complete clip."),
    ]

    deduped = dedupe_clip_candidates(candidates=candidates)

    assert [candidate.clip_id for candidate in deduped] == ["cand_a", "cand_d"]


def test_review_candidate_pool_validates_ranked_response_and_generates_temp_ids():
    candidates = [
        _clip_candidate(None, ["node_1", "node_2"], 0, 10000, score=8.1, rationale="Strong opener."),
        _clip_candidate(None, ["node_5"], 30000, 36000, score=6.4, rationale="Reaction beat."),
    ]

    response = review_candidate_pool(
        candidates=candidates,
        gemini_response={
            "ranked_candidates": [
                {
                    "candidate_temp_id": "cand_tmp_001",
                    "keep": True,
                    "pool_rank": 1,
                    "score": 8.8,
                    "score_breakdown": {"overall_clip_quality": 8.8},
                    "rationale": "Best standalone clip.",
                }
            ],
            "dropped_candidate_temp_ids": ["cand_tmp_002"],
        },
    )

    assert len(response.ranked_candidates) == 1
    assert response.ranked_candidates[0].candidate_temp_id == "cand_tmp_001"
    assert response.ranked_candidates[0].pool_rank == 1
    assert response.dropped_candidate_temp_ids == ["cand_tmp_002"]


def test_review_candidate_pool_fails_hard_on_invalid_output():
    candidates = [
        _clip_candidate(None, ["node_1"], 0, 5000, score=7.0, rationale="Reveal."),
        _clip_candidate(None, ["node_2"], 8000, 12000, score=6.5, rationale="Reaction."),
    ]

    with pytest.raises(ValueError, match="pool_rank"):
        review_candidate_pool(
            candidates=candidates,
            gemini_response={
                "ranked_candidates": [
                    {
                        "candidate_temp_id": "cand_tmp_001",
                        "keep": True,
                        "pool_rank": 2,
                        "score": 8.0,
                        "score_breakdown": {"overall_clip_quality": 8.0},
                        "rationale": "Winner.",
                    }
                ],
                "dropped_candidate_temp_ids": ["cand_tmp_002"],
            },
        )
