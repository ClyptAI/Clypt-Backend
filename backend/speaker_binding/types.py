"""Typed containers for speaker binding orchestration."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class DiarizedSpan(TypedDict):
    speaker_id: str
    start_time_ms: int
    end_time_ms: int
    exclusive: NotRequired[bool]
    overlap: NotRequired[bool]
    confidence: NotRequired[float]


class ScheduledSpan(TypedDict):
    span_id: str
    speaker_id: NotRequired[str]
    start_time_ms: int
    end_time_ms: int
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

