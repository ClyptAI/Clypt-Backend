"""Runtime metrics helpers for speaker binding."""

from __future__ import annotations


def new_lrasd_pipeline_metrics() -> dict:
    return {
        "lrasd_prep_queue_depth": 0,
        "lrasd_prep_queue_depth_max": 0,
        "lrasd_prep_wallclock_s": 0.0,
        "lrasd_infer_wallclock_s": 0.0,
        "lrasd_spans_processed": 0,
        "lrasd_spans_per_sec": 0.0,
        "lrasd_easy_cascade_skipped_jobs": 0,
    }


def finalize_lrasd_pipeline_metrics(metrics: dict | None) -> dict:
    finalized = new_lrasd_pipeline_metrics()
    if isinstance(metrics, dict):
        finalized.update(metrics)

    wallclock_s = float(finalized.get("lrasd_prep_wallclock_s", 0.0) or 0.0) + float(
        finalized.get("lrasd_infer_wallclock_s", 0.0) or 0.0
    )
    spans_processed = int(finalized.get("lrasd_spans_processed", 0) or 0)
    finalized["lrasd_spans_per_sec"] = (
        float(spans_processed) / wallclock_s if wallclock_s > 0.0 else 0.0
    )
    finalized["lrasd_prep_queue_depth"] = int(finalized.get("lrasd_prep_queue_depth", 0) or 0)
    finalized["lrasd_prep_queue_depth_max"] = int(
        finalized.get("lrasd_prep_queue_depth_max", 0) or 0
    )
    finalized["lrasd_easy_cascade_skipped_jobs"] = int(
        finalized.get("lrasd_easy_cascade_skipped_jobs", 0) or 0
    )
    finalized["lrasd_spans_processed"] = spans_processed
    return finalized


__all__ = ["finalize_lrasd_pipeline_metrics", "new_lrasd_pipeline_metrics"]
