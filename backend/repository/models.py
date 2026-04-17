from __future__ import annotations

from datetime import datetime
from typing import Any

from backend.common.domain_enums import ClusterType, JobStatus, LinkType, PromptSourceType, RunStatus, SignalType, SourcePlatform
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator

Phase14RunStatus = RunStatus
ExternalSignalType = SignalType
ExternalClusterType = ClusterType
SignalLinkType = LinkType


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
    external_signal_score: float | None = None
    agreement_bonus: float | None = None
    external_attribution_json: dict[str, Any] | None = None

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


class PhaseSubstepRecord(StrictModel):
    run_id: str
    phase_name: str
    step_name: str
    step_key: str
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


class ExternalSignalRecord(StrictModel):
    run_id: str
    signal_id: str
    signal_type: SignalType
    source_platform: SourcePlatform
    source_id: str
    author_id: str | None = None
    text: str
    engagement_score: float
    published_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExternalSignalClusterRecord(StrictModel):
    run_id: str
    cluster_id: str
    cluster_type: ClusterType
    summary_text: str
    member_signal_ids: list[str] = Field(default_factory=list)
    cluster_weight: float
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeSignalLinkRecord(StrictModel):
    run_id: str
    node_id: str
    cluster_id: str
    link_type: LinkType
    hop_distance: int
    time_offset_ms: int
    similarity: float
    link_score: float
    evidence: dict[str, Any] = Field(default_factory=dict)


class CandidateSignalLinkRecord(StrictModel):
    run_id: str
    clip_id: str
    cluster_id: str
    cluster_type: ClusterType
    aggregated_link_score: float
    coverage_ms: int
    direct_node_count: int
    inferred_node_count: int
    agreement_flags: list[str] = Field(default_factory=list)
    bonus_applied: float
    evidence: dict[str, Any] = Field(default_factory=dict)


class PromptSourceLinkRecord(StrictModel):
    run_id: str
    prompt_id: str
    prompt_source_type: PromptSourceType
    source_cluster_id: str | None = None
    source_cluster_type: ClusterType | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_prompt_source_consistency(self):
        if self.prompt_source_type == "general":
            if self.source_cluster_id is not None or self.source_cluster_type is not None:
                raise ValueError("general prompt sources must not reference a source cluster")
        else:
            if self.source_cluster_id is None or self.source_cluster_type is None:
                raise ValueError("comment and trend prompt sources must reference a source cluster")
            if self.prompt_source_type != self.source_cluster_type:
                raise ValueError("prompt_source_type must match source_cluster_type")
        return self


class SubgraphProvenanceRecord(StrictModel):
    run_id: str
    subgraph_id: str
    seed_source_set: list[PromptSourceType] = Field(default_factory=list)
    seed_prompt_ids: list[str] = Field(default_factory=list)
    source_cluster_ids: list[str] = Field(default_factory=list)
    support_summary: dict[str, Any] = Field(default_factory=dict)
    canonical_selected: bool
    dedupe_overlap_ratio: float | None = None
    selection_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_seed_sources(self):
        if not self.seed_source_set:
            raise ValueError("seed_source_set must be non-empty")
        return self


__all__ = [
    "CandidateSignalLinkRecord",
    "ClipCandidateRecord",
    "ExternalClusterType",
    "ExternalSignalClusterRecord",
    "ExternalSignalRecord",
    "ExternalSignalType",
    "JobStatus",
    "Phase14RunStatus",
    "Phase24JobRecord",
    "PhaseMetricRecord",
    "PhaseSubstepRecord",
    "PromptSourceLinkRecord",
    "PromptSourceType",
    "RunRecord",
    "RunStatus",
    "SemanticEdgeRecord",
    "SemanticNodeRecord",
    "SignalLinkType",
    "SourcePlatform",
    "NodeSignalLinkRecord",
    "SubgraphProvenanceRecord",
    "StrictModel",
    "TimelineTurnRecord",
]
