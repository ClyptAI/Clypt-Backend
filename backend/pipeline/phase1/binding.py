"""Speaker binding stage seam — shared analysis dict + diarization turns."""

from __future__ import annotations

from typing import Any, Mapping

from backend.pipeline.phase1.decode_cache import Phase1AnalysisContext


def build_speaker_binding_analysis_context(
    *,
    phase1_context: Phase1AnalysisContext | None,
    analysis_context_fallback: Mapping[str, Any] | None,
    audio_speaker_turns: list[dict],
) -> dict[str, Any]:
    """Merge cached analysis metadata with diarization turns for LR-ASD / binding."""
    base: dict[str, Any]
    if phase1_context is not None:
        base = phase1_context.as_dict()
    elif isinstance(analysis_context_fallback, dict):
        base = dict(analysis_context_fallback)
    else:
        base = {}
    out = dict(base)
    out["audio_speaker_turns"] = list(audio_speaker_turns)
    return out


__all__ = ["build_speaker_binding_analysis_context"]
