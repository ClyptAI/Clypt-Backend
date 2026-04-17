"""Pydantic response models that match the frontend TypeScript types in src/types/clypt.ts.

These are the API contract — they convert internal repository/pipeline models
to the exact JSON shape the frontend expects.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Primitives ────────────────────────────────────────────────────────────────

PhaseNumber = Literal[1, 2, 3, 4, 5, 6]
PhaseStatus = Literal["pending", "running", "completed", "failed", "needs_action"]
EmotionLabel = Literal[
    "angry", "disgusted", "fearful", "happy", "neutral",
    "other", "sad", "surprised", "unknown",
]


# ── Runs ────────────���─────────────────────────────────────────────────────────

class PhaseStatusEntry(BaseModel):
    phase: PhaseNumber
    name: str
    status: PhaseStatus
    elapsed_s: float | None = None
    summary: str | None = None
    artifact_keys: list[str] = Field(default_factory=list)


class RunMeta(BaseModel):
    run_id: str
    source_url: str
    created_at: str  # ISO 8601
    display_name: str | None = None


class RunDetail(RunMeta):
    phases: list[PhaseStatusEntry] = Field(default_factory=list)
    node_count: int | None = None
    edge_count: int | None = None
    clip_count: int | None = None


class RunListItem(RunMeta):
    latest_phase: PhaseNumber
    latest_status: PhaseStatus
    clip_count: int | None = None


class RunCreateRequest(BaseModel):
    source_url: str
    display_name: str | None = None


# ── Graph ─────────���───────────────────────────────────────────────────────────

class SemanticNodeEvidence(BaseModel):
    emotion_labels: list[str] = Field(default_factory=list)
    audio_events: list[str] = Field(default_factory=list)


class SemanticGraphNode(BaseModel):
    node_id: str
    node_type: str
    start_ms: int
    end_ms: int
    source_turn_ids: list[str] = Field(default_factory=list)
    word_ids: list[str] = Field(default_factory=list)
    transcript_text: str
    node_flags: list[str] = Field(default_factory=list)
    summary: str
    evidence: SemanticNodeEvidence = Field(default_factory=SemanticNodeEvidence)
    semantic_embedding: list[float] | None = None
    multimodal_embedding: list[float] | None = None


class SemanticGraphEdge(BaseModel):
    source_node_id: str
    target_node_id: str
    edge_type: str
    rationale: str | None = None
    confidence: float | None = None
    support_count: int | None = None
    batch_ids: list[str] = Field(default_factory=list)


# ── Clips ─────────────────────────────────────────────���───────────────────────

class ClipCandidate(BaseModel):
    clip_id: str | None = None
    node_ids: list[str] = Field(default_factory=list)
    start_ms: int
    end_ms: int
    score: float
    rationale: str
    source_prompt_ids: list[str] = Field(default_factory=list)
    seed_node_id: str | None = None
    subgraph_id: str | None = None
    query_aligned: bool | None = None
    pool_rank: int | None = None
    score_breakdown: dict[str, float] | None = None


# ── Timeline ───────────────────────────────────────��──────────────────────────

class TimelineSpeakerTurn(BaseModel):
    turn_id: str
    speaker_id: str
    start_ms: int
    end_ms: int
    transcript_text: str
    emotion_primary: EmotionLabel
    emotion_score: float
    emotion_secondary: list[dict[str, Any]] = Field(default_factory=list)


class TimelineSpeaker(BaseModel):
    speaker_id: str
    display_name: str
    turns: list[TimelineSpeakerTurn] = Field(default_factory=list)


class TimelineEmotionSegment(BaseModel):
    start_ms: int
    end_ms: int
    label: EmotionLabel


class TimelineAudioEvent(BaseModel):
    start_ms: int
    end_ms: int
    label: str
    confidence: float


class TimelineShot(BaseModel):
    shot_id: str
    start_ms: int
    end_ms: int


class TimelineShotTracklets(BaseModel):
    shot_id: str
    start_ms: int
    end_ms: int
    tracklet_letters: list[str] = Field(default_factory=list)


class TimelineBundle(BaseModel):
    duration_ms: int
    shots: list[TimelineShot] = Field(default_factory=list)
    shot_tracklets: list[TimelineShotTracklets] = Field(default_factory=list)
    speakers: list[TimelineSpeaker] = Field(default_factory=list)
    emotions: list[TimelineEmotionSegment] = Field(default_factory=list)
    audio_events: list[TimelineAudioEvent] = Field(default_factory=list)


# ── Grounding ────────────────────��────────────────────────────────────────────

class BoundingBoxRect(BaseModel):
    x: float
    y: float
    w: float
    h: float


class GroundingTracklet(BaseModel):
    id: str
    letter: str
    duration_pct: float


class GroundingBinding(BaseModel):
    tracklet_id: str
    speaker_id: int
    start_ms: int
    end_ms: int
    method: Literal["drag", "word", "range"]


class GroundingIntent(BaseModel):
    intent: Literal["Follow", "Reaction", "Split", "Wide", "Manual"]
    follow: int | None = None
    react_on: int | None = None
    react_follow: int | None = None
    split_left: int | None = None
    split_right: int | None = None
    wide_includes: list[int] | None = None
    crop_set: bool | None = None


class GroundingCropPosition(BaseModel):
    x_percent: float
    y_percent: float
    height_percent: float


class GroundingShotState(BaseModel):
    shot_idx: int
    rects: dict[str, BoundingBoxRect] = Field(default_factory=dict)
    user_tracklets: list[GroundingTracklet] = Field(default_factory=list)
    hidden_tracklet_ids: list[str] = Field(default_factory=list)
    bindings: list[GroundingBinding] | None = None
    intent: GroundingIntent | None = None
    manual_crop: GroundingCropPosition | None = None


class GroundingClipState(BaseModel):
    run_id: str
    clip_id: str
    shots: list[GroundingShotState] = Field(default_factory=list)
    updated_at: str  # ISO 8601


# ── Render ──────��─────────────────────────────────────────────────────────────

class RenderPreset(BaseModel):
    id: str
    platform: str
    label: str
    aspect_ratio: Literal["9:16", "1:1", "16:9"]
    width: int
    height: int
    frame_rate: int
    max_duration_s: int | None = None


class RenderJobStatus(BaseModel):
    clip_id: str
    status: Literal["queued", "rendering", "completed", "failed"]
    progress_pct: float | None = None
    output_url: str | None = None
    error: str | None = None


class RenderSubmitRequest(BaseModel):
    preset_id: str
