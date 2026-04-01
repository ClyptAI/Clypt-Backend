"""Tracking stage seam — delegates to worker implementations with a shared context."""

from __future__ import annotations

import os
from typing import Any

from backend.pipeline.phase1.decode_cache import Phase1AnalysisContext


def _safe_video_mb(path: str) -> float | None:
    try:
        if not path:
            return None
        return round(float(os.path.getsize(path)) / (1024.0 * 1024.0), 3)
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return None


def _decode_metrics_from_context(ctx: Phase1AnalysisContext, analysis_context: dict[str, Any]) -> dict[str, Any]:
    source_path = str(analysis_context.get("source_video_path") or ctx.source_video_path or "")
    prepared_path = str(analysis_context.get("prepared_video_path") or source_path)
    analysis_path = str(analysis_context.get("analysis_video_path") or prepared_path)
    source_meta = analysis_context.get("source_meta") if isinstance(analysis_context.get("source_meta"), dict) else {}
    analysis_meta = (
        analysis_context.get("analysis_meta")
        if isinstance(analysis_context.get("analysis_meta"), dict)
        else {}
    )
    source_frames = int(source_meta.get("total_frames", 0) or 0)
    analysis_frames = int(analysis_meta.get("total_frames", 0) or 0)
    source_duration_s = float(source_meta.get("duration_s", 0.0) or 0.0)
    analysis_duration_s = float(analysis_meta.get("duration_s", 0.0) or 0.0)
    source_video_mb = _safe_video_mb(source_path)
    analysis_video_mb = _safe_video_mb(analysis_path)
    decode_size_ratio = None
    if isinstance(source_video_mb, float) and source_video_mb > 0 and isinstance(analysis_video_mb, float):
        decode_size_ratio = round(analysis_video_mb / source_video_mb, 4)
    return {
        "decode_prepare_wallclock_ms": round(float(ctx.prepare_elapsed_ms), 3),
        "decode_cache_prepare_invocations": int(ctx.prepare_invocations),
        "decode_cache_hits": int(ctx.cache_hits),
        "decode_source_video_mb": source_video_mb,
        "decode_prepared_video_mb": _safe_video_mb(prepared_path),
        "decode_analysis_video_mb": analysis_video_mb,
        "decode_before_after_size_ratio": decode_size_ratio,
        "decode_proxy_active": bool(analysis_path and prepared_path and analysis_path != prepared_path),
        "decode_source_duration_s": round(source_duration_s, 3) if source_duration_s > 0 else None,
        "decode_analysis_duration_s": round(analysis_duration_s, 3) if analysis_duration_s > 0 else None,
        "decode_source_frame_count": source_frames if source_frames > 0 else None,
        "decode_analysis_frame_count": analysis_frames if analysis_frames > 0 else None,
    }


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
    metrics["phase1_decode_context"] = {
        "prepare_elapsed_ms": round(float(ctx.prepare_elapsed_ms), 3),
        "prepare_invocations": int(ctx.prepare_invocations),
        "cache_hits": int(ctx.cache_hits),
    }
    metrics.update(_decode_metrics_from_context(ctx, analysis_context))
    return tracks, metrics


__all__ = ["run_tracking_stage"]
