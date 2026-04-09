from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator

RunStatus = Literal["PHASE1_DONE", "PHASE24_QUEUED", "PHASE24_RUNNING", "PHASE24_DONE", "FAILED"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]
Phase14RunStatus = RunStatus


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunRecord(StrictModel):
    run_id: str
    source_url: str | None = None
    source_video_gcs_uri: str | None = None
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class TimelineTurnRecord(StrictModel):
    run_id: str
    turn_id: str
    speaker_id: str
    start_ms: NonNegativeInt
    end_ms: NonNegativeInt
    word_ids: list[str] = Field(default_factory=list)
    transcript_text: str
    identification_match: str | None = None

    @model_validator(mode="after")
    def _check_time_order(self):
        if self.start_ms > self.end_ms:
            raise ValueError("start_ms must be <= end_ms")
        return self


class SemanticNodeRecord(StrictModel):
    run_id: str
    node_id: str
    node_type: str
    start_ms: NonNegativeInt
    end_ms: NonNegativeInt
    source_turn_ids: list[str] = Field(default_factory=list)
    word_ids: list[str] = Field(default_factory=list)
    transcript_text: str
    node_flags: list[str] = Field(default_factory=list)
    summary: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    semantic_embedding: list[float] | None = None
    multimodal_embedding: list[float] | None = None

    @model_validator(mode="after")
    def _check_time_order(self):
        if self.start_ms > self.end_ms:
            raise ValueError("start_ms must be <= end_ms")
        return self


class SemanticEdgeRecord(StrictModel):
    run_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str
    rationale: str | None = None
    confidence: float | None = None
    support_count: int | None = None
    batch_ids: list[str] = Field(default_factory=list)


class ClipCandidateRecord(StrictModel):
    run_id: str
    clip_id: str | None = None
    node_ids: list[str] = Field(default_factory=list)
    start_ms: NonNegativeInt
    end_ms: NonNegativeInt
    score: float
    rationale: str
    source_prompt_ids: list[str] = Field(default_factory=list)
    seed_node_id: str | None = None
    subgraph_id: str | None = None
    query_aligned: bool | None = None
    pool_rank: int | None = None
    score_breakdown: dict[str, float] | None = None

    @model_validator(mode="after")
    def _check_time_order(self):
        if self.start_ms > self.end_ms:
            raise ValueError("start_ms must be <= end_ms")
        return self


class PhaseMetricRecord(StrictModel):
    run_id: str
    phase_name: str
    status: str
    started_at: datetime
    ended_at: datetime | None = None
    duration_ms: float | None = None
    error_payload: dict[str, Any] | None = None
    query_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Phase24JobRecord(StrictModel):
    run_id: str
    status: JobStatus
    attempt_count: int = 0
    last_error: dict[str, Any] | None = None
    worker_name: str | None = None
    task_name: str | None = None
    locked_at: datetime | None = None
    updated_at: datetime
    completed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ClipCandidateRecord",
    "JobStatus",
    "Phase14RunStatus",
    "Phase24JobRecord",
    "PhaseMetricRecord",
    "RunRecord",
    "RunStatus",
    "SemanticEdgeRecord",
    "SemanticNodeRecord",
    "StrictModel",
    "TimelineTurnRecord",
]
