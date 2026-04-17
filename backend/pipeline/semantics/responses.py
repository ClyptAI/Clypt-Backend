from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import Field

from ..contracts import NodeFlag, NodeType, StrictModel

BoundarySkipReason = Literal[
    "ambiguous_default",
    "overlapping_turns",
    "large_time_gap",
    "non_adjacent_turn_gap",
    "clear_semantic_split",
]


class BoundarySkipDecision(StrictModel):
    skip_llm: bool
    reason: BoundarySkipReason
    time_gap_ms: int
    turn_gap: int | None = None
    summary_similarity: float
    transcript_similarity: float
    shared_flag_count: int
    shared_flags: list[NodeFlag] = Field(default_factory=list)
    overlap_turn_count: int
    same_node_type: bool


class SemanticsMergedNodeResponse(StrictModel):
    source_turn_ids: list[str] = Field(default_factory=list)
    node_type: NodeType
    node_flags: list[NodeFlag] = Field(default_factory=list)
    summary: str


class SemanticsMergeAndClassifyBatchResponse(StrictModel):
    merged_nodes: list[SemanticsMergedNodeResponse] = Field(default_factory=list)


class SemanticsMergeAndClassifyLiveResponse(StrictModel):
    merged_nodes: list[SemanticsMergedNodeResponse] = Field(default_factory=list)


class BoundaryReconciliationExistingNodeResponse(StrictModel):
    existing_node_id: str
    source_turn_ids: list[str] = Field(default_factory=list)
    node_type: NodeType
    node_flags: list[NodeFlag] = Field(default_factory=list)
    summary: str


class BoundaryReconciliationMergedNodeResponse(StrictModel):
    source_turn_ids: list[str] = Field(default_factory=list)
    node_type: NodeType
    node_flags: list[NodeFlag] = Field(default_factory=list)
    summary: str


class BoundaryReconciliationResponse(StrictModel):
    resolution: Literal["keep_both", "merge"]
    nodes: list[BoundaryReconciliationExistingNodeResponse] = Field(default_factory=list)
    merged_node: BoundaryReconciliationMergedNodeResponse | None = None


SemanticsMergeAndClassifyResponse: TypeAlias = (
    SemanticsMergeAndClassifyBatchResponse | SemanticsMergeAndClassifyLiveResponse
)


__all__ = [
    "BoundaryReconciliationExistingNodeResponse",
    "BoundaryReconciliationMergedNodeResponse",
    "BoundaryReconciliationResponse",
    "BoundarySkipDecision",
    "BoundarySkipReason",
    "SemanticsMergeAndClassifyBatchResponse",
    "SemanticsMergeAndClassifyLiveResponse",
    "SemanticsMergeAndClassifyResponse",
    "SemanticsMergedNodeResponse",
]
