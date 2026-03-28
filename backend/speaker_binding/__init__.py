"""Speaker binding package seam."""

from __future__ import annotations

from typing import Any

from .types import (
    DiarizedSpan,
    EasySpanDecision,
    LrasdSpanJob,
    ScheduledSpan,
    SpanLevelAssignment,
)


def run_speaker_binding(
    *,
    worker: Any,
    video_path: str,
    audio_wav_path: str,
    tracks: list[dict],
    words: list[dict],
    frame_to_dets: dict[int, list[dict]] | None = None,
    track_to_dets: dict[str, list[dict]] | None = None,
    track_identity_features: dict[str, dict] | None = None,
    analysis_context: dict | None = None,
    track_id_remap: dict[str, str] | None = None,
) -> list[dict]:
    """Delegate speaker binding orchestration back to the worker for now."""
    return worker._run_speaker_binding_impl(
        video_path=video_path,
        audio_wav_path=audio_wav_path,
        tracks=tracks,
        words=words,
        frame_to_dets=frame_to_dets,
        track_to_dets=track_to_dets,
        track_identity_features=track_identity_features,
        analysis_context=analysis_context,
        track_id_remap=track_id_remap,
    )


__all__ = [
    "DiarizedSpan",
    "EasySpanDecision",
    "LrasdSpanJob",
    "ScheduledSpan",
    "SpanLevelAssignment",
    "run_speaker_binding",
]
