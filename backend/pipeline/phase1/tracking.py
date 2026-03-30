"""Tracking stage seam — delegates to worker implementations with a shared context."""

from __future__ import annotations

from typing import Any

from backend.pipeline.phase1.decode_cache import Phase1AnalysisContext


def run_tracking_stage(worker: Any, video_path: str, ctx: Phase1AnalysisContext) -> tuple[list[dict], dict]:
    """Run YOLO/tracker + face pipeline using one shared :class:`Phase1AnalysisContext`."""
    mode = worker._select_tracking_mode()
    analysis_context = ctx.ensure_prepared(worker, tracking_mode=mode)
    execution_mode = "direct" if mode in {"direct", "shared_analysis_proxy"} else "chunked"
    if execution_mode == "direct":
        tracks, metrics = worker._run_tracking_direct(video_path, analysis_context=analysis_context)
    else:
        tracks, metrics = worker._run_tracking_chunked(video_path, analysis_context=analysis_context)

    metrics = dict(metrics or {})
    metrics.setdefault("tracking_mode", mode)
    metrics["analysis_context"] = analysis_context
    metrics["phase1_decode_context"] = ctx
    return tracks, metrics


__all__ = ["run_tracking_stage"]
