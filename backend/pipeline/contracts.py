from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator


NodeType = Literal[
    "claim",
    "explanation",
    "example",
    "anecdote",
    "reaction_beat",
    "qa_exchange",
    "challenge_exchange",
    "setup_payoff",
    "reveal",
    "transition",
]

NodeFlag = Literal[
    "topic_pivot",
    "callback_candidate",
    "high_resonance_candidate",
    "backchannel_dense",
    "interruption_heavy",
    "overlap_heavy",
    "resumed_topic",
]

EdgeType = Literal[
    "next_turn",
    "prev_turn",
    "overlaps_with",
    "answers",
    "challenges",
    "contradicts",
    "supports",
    "elaborates",
    "setup_for",
    "payoff_of",
    "reaction_to",
    "callback_to",
    "topic_recurrence",
    "escalates",
]

EmotionLabel = Literal[
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TranscriptWord(StrictModel):
    word_id: str
    text: str
    start_ms: NonNegativeInt
    end_ms: NonNegativeInt
    speaker_id: str | None = None

    @model_validator(mode="after")
    def _check_time_order(self):
        if self.start_ms > self.end_ms:
            raise ValueError("start_ms must be <= end_ms")
        return self


class CanonicalTurn(StrictModel):
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


class CanonicalTimeline(StrictModel):
    words: list[TranscriptWord] = Field(default_factory=list)
    turns: list[CanonicalTurn] = Field(default_factory=list)
    source_video_url: str | None = None
    video_gcs_uri: str | None = None


class SpeechEmotionEvent(StrictModel):
    turn_id: str
    primary_emotion_label: EmotionLabel
    primary_emotion_score: float
    per_class_scores: dict[EmotionLabel, float]


class SpeechEmotionTimeline(StrictModel):
    events: list[SpeechEmotionEvent] = Field(default_factory=list)


class AudioEvent(StrictModel):
    event_label: str
    start_ms: NonNegativeInt
    end_ms: NonNegativeInt
    confidence: float | None = None

    @model_validator(mode="after")
    def _check_time_order(self):
        if self.start_ms > self.end_ms:
            raise ValueError("start_ms must be <= end_ms")
        return self


class AudioEventTimeline(StrictModel):
    events: list[AudioEvent] = Field(default_factory=list)


class ShotTrackletDescriptor(StrictModel):
    tracklet_id: str
    shot_id: str
    start_ms: NonNegativeInt
    end_ms: NonNegativeInt
    representative_thumbnail_uris: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_time_order(self):
        if self.start_ms > self.end_ms:
            raise ValueError("start_ms must be <= end_ms")
        return self


class ShotTrackletIndex(StrictModel):
    tracklets: list[ShotTrackletDescriptor] = Field(default_factory=list)


class TrackletGeometryPoint(StrictModel):
    frame_index: NonNegativeInt
    timestamp_ms: NonNegativeInt
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)


class TrackletGeometryEntry(StrictModel):
    tracklet_id: str
    shot_id: str
    points: list[TrackletGeometryPoint] = Field(default_factory=list)


class TrackletGeometry(StrictModel):
    tracklets: list[TrackletGeometryEntry] = Field(default_factory=list)


class SemanticNodeEvidence(StrictModel):
    emotion_labels: list[EmotionLabel] = Field(default_factory=list)
    audio_events: list[str] = Field(default_factory=list)


class SemanticGraphNode(StrictModel):
    node_id: str
    node_type: NodeType
    start_ms: NonNegativeInt
    end_ms: NonNegativeInt
    source_turn_ids: list[str] = Field(default_factory=list)
    word_ids: list[str] = Field(default_factory=list)
    transcript_text: str
    node_flags: list[NodeFlag] = Field(default_factory=list)
    summary: str
    evidence: SemanticNodeEvidence = Field(default_factory=SemanticNodeEvidence)
    semantic_embedding: list[float] | None = None
    multimodal_embedding: list[float] | None = None

    @model_validator(mode="after")
    def _check_time_order(self):
        if self.start_ms > self.end_ms:
            raise ValueError("start_ms must be <= end_ms")
        return self


class SemanticGraphEdge(StrictModel):
    source_node_id: str
    target_node_id: str
    edge_type: EdgeType
    rationale: str | None = None
    confidence: float | None = None
    support_count: int | None = None
    batch_ids: list[str] = Field(default_factory=list)


class LocalSubgraphNodeEdge(StrictModel):
    edge_type: EdgeType
    target_node_id: str


class LocalSubgraphNode(StrictModel):
    node_id: str
    start_ms: NonNegativeInt
    end_ms: NonNegativeInt
    duration_ms: NonNegativeInt
    node_type: NodeType
    node_flags: list[NodeFlag] = Field(default_factory=list)
    summary: str
    transcript_excerpt: str
    word_count: NonNegativeInt
    emotion_labels: list[EmotionLabel] = Field(default_factory=list)
    audio_events: list[str] = Field(default_factory=list)
    inbound_edges: list[LocalSubgraphNodeEdge] = Field(default_factory=list)
    outbound_edges: list[LocalSubgraphNodeEdge] = Field(default_factory=list)


class LocalSubgraph(StrictModel):
    subgraph_id: str
    seed_node_id: str
    source_prompt_ids: list[str] = Field(default_factory=list)
    start_ms: NonNegativeInt
    end_ms: NonNegativeInt
    nodes: list[LocalSubgraphNode] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_time_order(self):
        if self.start_ms > self.end_ms:
            raise ValueError("start_ms must be <= end_ms")
        return self


class ClipCandidate(StrictModel):
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


class SubgraphReviewResponse(StrictModel):
    subgraph_id: str
    seed_node_id: str
    reject_all: bool
    reject_reason: str = ""
    candidates: list[ClipCandidate] = Field(default_factory=list)


class RankedCandidateDecision(StrictModel):
    candidate_temp_id: str
    keep: bool
    pool_rank: int | None = None
    score: float
    score_breakdown: dict[str, float] | None = None
    rationale: str


class PooledCandidateReviewResponse(StrictModel):
    ranked_candidates: list[RankedCandidateDecision] = Field(default_factory=list)
    dropped_candidate_temp_ids: list[str] = Field(default_factory=list)


class Phase14RunSummary(StrictModel):
    run_id: str
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "AudioEvent",
    "AudioEventTimeline",
    "CanonicalTimeline",
    "CanonicalTurn",
    "ClipCandidate",
    "EdgeType",
    "EmotionLabel",
    "LocalSubgraph",
    "LocalSubgraphNode",
    "LocalSubgraphNodeEdge",
    "NodeFlag",
    "NodeType",
    "Phase14RunSummary",
    "PooledCandidateReviewResponse",
    "RankedCandidateDecision",
    "SemanticGraphEdge",
    "SemanticGraphNode",
    "SemanticNodeEvidence",
    "ShotTrackletDescriptor",
    "ShotTrackletIndex",
    "SpeechEmotionEvent",
    "SpeechEmotionTimeline",
    "SubgraphReviewResponse",
    "TrackletGeometry",
    "TrackletGeometryEntry",
    "TrackletGeometryPoint",
    "TranscriptWord",
]
