"""Typed containers for speaker binding orchestration."""

from __future__ import annotations

from typing import Literal, TypedDict

from typing_extensions import NotRequired


class DiarizedSpan(TypedDict):
    turn_id: NotRequired[str]
    speaker_id: str
    start_time_ms: int
    end_time_ms: int
    exclusive: NotRequired[bool]
    overlap: NotRequired[bool]
    confidence: NotRequired[float]
    source_turn_ids: NotRequired[list[str]]


class ScheduledSpan(TypedDict):
    span_id: str
    span_type: Literal["single", "overlap"]
    speaker_ids: list[str]
    start_time_ms: int
    end_time_ms: int
    context_start_time_ms: int
    context_end_time_ms: int
    exclusive: bool
    overlap: bool
    source_turn_ids: list[str]
    speaker_id: NotRequired[str]
    track_id: NotRequired[str]
    priority: NotRequired[int]


class EasySpanDecision(TypedDict):
    span_id: str
    decision: Literal["accept", "reject", "defer"]
    start_time_ms: int
    end_time_ms: int
    reason: NotRequired[str]
    score: NotRequired[float]


class LrasdSpanJob(TypedDict):
    span_id: str
    video_path: str
    audio_wav_path: str
    start_time_ms: int
    end_time_ms: int
    speaker_id: NotRequired[str]


class SpanLevelAssignment(TypedDict):
    span_id: str
    speaker_id: str
    start_time_ms: int
    end_time_ms: int
    track_id: NotRequired[str]
    local_track_id: NotRequired[str]
    word_count: NotRequired[int]
    ambiguous: NotRequired[bool]
