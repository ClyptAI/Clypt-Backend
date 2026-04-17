from __future__ import annotations

from typing import Literal

from pydantic import Field

from backend.pipeline.contracts import StrictModel


class SignalsThreadConsolidationResponse(StrictModel):
    thread_summary: str
    moment_hints: list[str] = Field(default_factory=list)


class SignalsCommentClassificationResponse(StrictModel):
    quality: Literal["high_signal", "contextual", "low_signal", "spam"]
    reason: str | None = None


class SignalsCommentClassificationBatchResponse(StrictModel):
    results: list[SignalsCommentClassificationResponse] = Field(default_factory=list)


class SignalsClusterPromptResponse(StrictModel):
    prompt: str


class SignalsTrendQueryResponse(StrictModel):
    queries: list[str] = Field(default_factory=list)


class SignalsTrendRelevanceResponse(StrictModel):
    keep: bool
    relevance: float
    reason: str | None = None


class SignalsTrendRelevanceBatchResponse(StrictModel):
    results: list[SignalsTrendRelevanceResponse] = Field(default_factory=list)


class SignalsClusterSpanResponse(StrictModel):
    node_ids: list[str] = Field(default_factory=list)
    reason: str | None = None


class SignalsCandidateAttributionResponse(StrictModel):
    explanation: str
