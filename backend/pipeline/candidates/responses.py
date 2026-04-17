from __future__ import annotations

from pydantic import Field

from ..contracts import StrictModel


class CandidatesMetaPromptGenerationResponse(StrictModel):
    prompts: list[str] = Field(default_factory=list)


class CandidatesSubgraphReviewCandidateResponse(StrictModel):
    node_ids: list[str] = Field(default_factory=list)
    start_ms: int
    end_ms: int
    score: float
    rationale: str


class CandidatesSubgraphReviewResponse(StrictModel):
    subgraph_id: str
    seed_node_id: str
    reject_all: bool
    reject_reason: str
    candidates: list[CandidatesSubgraphReviewCandidateResponse] = Field(default_factory=list)


class CandidatesRankedCandidateScoreBreakdownResponse(StrictModel):
    virality: float
    coherence: float
    engagement: float


class CandidatesRankedCandidateResponse(StrictModel):
    candidate_temp_id: str
    keep: bool
    pool_rank: int
    score: float
    score_breakdown: CandidatesRankedCandidateScoreBreakdownResponse
    rationale: str


class CandidatesPooledCandidateReviewResponse(StrictModel):
    ranked_candidates: list[CandidatesRankedCandidateResponse] = Field(default_factory=list)
    dropped_candidate_temp_ids: list[str] = Field(default_factory=list)


__all__ = [
    "CandidatesMetaPromptGenerationResponse",
    "CandidatesPooledCandidateReviewResponse",
    "CandidatesRankedCandidateResponse",
    "CandidatesRankedCandidateScoreBreakdownResponse",
    "CandidatesSubgraphReviewCandidateResponse",
    "CandidatesSubgraphReviewResponse",
]
