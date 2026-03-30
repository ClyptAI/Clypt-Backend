"""Face / identity feature pipeline seams (paths derive from shared analysis context)."""

from __future__ import annotations

from backend.pipeline.phase1.decode_cache import Phase1AnalysisContext


def analysis_video_path_for_faces(ctx: Phase1AnalysisContext) -> str:
    """Video path face workers should decode (shared proxy/H.264 path when prepared)."""
    return ctx.analysis_video_path


def letterbox_source_dims(ctx: Phase1AnalysisContext) -> tuple[int, int]:
    """Approximate source (pre-letterbox) dimensions from cached metadata if present."""
    d = ctx.as_dict()
    sm = d.get("source_meta") if isinstance(d.get("source_meta"), dict) else {}
    w = int(sm.get("width", 0) or 0)
    h = int(sm.get("height", 0) or 0)
    return w, h


__all__ = ["analysis_video_path_for_faces", "letterbox_source_dims"]
