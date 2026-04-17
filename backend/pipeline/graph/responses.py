from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..contracts import StrictModel


class LocalSemanticEdgeItem(StrictModel):
    source_node_id: str
    target_node_id: str
    edge_type: str
    rationale: str | None = None
    confidence: float | None = None


class LocalSemanticEdgeResponse(StrictModel):
    edges: list[LocalSemanticEdgeItem] = Field(default_factory=list)


class LocalSemanticEdgeBatchResponse(StrictModel):
    batch_id: str
    target_node_ids: list[str] = Field(default_factory=list)
    context_node_ids: list[str] = Field(default_factory=list)
    edges: list[LocalSemanticEdgeItem] = Field(default_factory=list)


class LongRangeEdgeItem(StrictModel):
    source_node_id: str
    target_node_id: str
    edge_type: Literal["callback_to", "topic_recurrence"]
    rationale: str | None = None
    confidence: float | None = None


class LongRangeEdgeResponse(StrictModel):
    edges: list[LongRangeEdgeItem] = Field(default_factory=list)


__all__ = [
    "LocalSemanticEdgeBatchResponse",
    "LocalSemanticEdgeItem",
    "LocalSemanticEdgeResponse",
    "LongRangeEdgeItem",
    "LongRangeEdgeResponse",
]
