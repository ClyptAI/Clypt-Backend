from __future__ import annotations

from typing import Any

from backend.common.domain_enums import ClusterType, LinkType, PromptSourceType, SignalType, SourcePlatform
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SignalPromptSpec(StrictModel):
    prompt_id: str
    text: str
    prompt_source_type: PromptSourceType
    source_cluster_id: str | None = None
    source_cluster_type: ClusterType | None = None


class ExternalSignal(StrictModel):
    signal_id: str
    signal_type: SignalType
    source_platform: SourcePlatform
    source_id: str
    author_id: str | None = None
    text: str
    engagement_score: float = 0.0
    published_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExternalSignalCluster(StrictModel):
    cluster_id: str
    cluster_type: ClusterType
    summary_text: str
    member_signal_ids: list[str] = Field(default_factory=list)
    cluster_weight: float = 0.0
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeSignalLink(StrictModel):
    node_id: str
    cluster_id: str
    link_type: LinkType
    hop_distance: int
    time_offset_ms: int
    similarity: float
    link_score: float = 0.0
    evidence: dict[str, Any] = Field(default_factory=dict)


class CandidateSignalLink(StrictModel):
    clip_id: str
    cluster_id: str
    cluster_type: ClusterType
    aggregated_link_score: float
    coverage_ms: int
    direct_node_count: int
    inferred_node_count: int
    agreement_flags: list[str] = Field(default_factory=list)
    bonus_applied: float = 0.0
    evidence: dict[str, Any] = Field(default_factory=dict)


class SubgraphProvenance(StrictModel):
    subgraph_id: str
    seed_source_set: list[PromptSourceType] = Field(default_factory=list)
    seed_prompt_ids: list[str] = Field(default_factory=list)
    source_cluster_ids: list[str] = Field(default_factory=list)
    support_summary: dict[str, Any] = Field(default_factory=dict)
    canonical_selected: bool = True
    dedupe_overlap_ratio: float | None = None
    selection_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SignalPipelineOutput(StrictModel):
    external_signals: list[ExternalSignal] = Field(default_factory=list)
    clusters: list[ExternalSignalCluster] = Field(default_factory=list)
    prompt_specs: list[SignalPromptSpec] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "CandidateSignalLink",
    "ClusterType",
    "ExternalSignal",
    "ExternalSignalCluster",
    "LinkType",
    "NodeSignalLink",
    "PromptSourceType",
    "SignalPipelineOutput",
    "SignalPromptSpec",
    "SignalType",
    "SourcePlatform",
    "SubgraphProvenance",
]
