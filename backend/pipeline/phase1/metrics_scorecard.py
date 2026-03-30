"""Pure Phase 1 benchmark / validation metrics (scorecard) from ledger dicts.

Used for spec validation: assignment coverage, candidate scoring presence,
unknown rate, overlap camera consistency proxy, and wallclock summaries.
"""

from __future__ import annotations

from typing import Any, Mapping

SCORECARD_VERSION = 1

# High-confidence binding rows: used only for the misassignment *proxy* (not ground truth).
_HIGH_CONFIDENCE_THRESHOLD = 0.85
_LOW_TOP_MARGIN_THRESHOLD = 0.05


def _as_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _extract_stage_wallclock_s(tracking_metrics: Mapping[str, Any]) -> dict[str, float]:
    """Collect ``*_wallclock_s`` floats from tracking metrics (pipeline substages)."""
    out: dict[str, float] = {}
    for key, val in tracking_metrics.items():
        if isinstance(key, str) and key.endswith("_wallclock_s"):
            fv = _safe_float(val)
            if fv is not None:
                out[key] = fv
    return dict(sorted(out.items()))


def _overlap_camera_consistency_ratio(decisions: list[Any]) -> float | None:
    """Fraction of overlap-follow decisions where camera target matches visibility rules."""
    rows = [d for d in decisions if isinstance(d, dict)]
    if not rows:
        return None
    ok = 0
    for d in rows:
        visible = [_as_str(x) for x in (d.get("visible_local_track_ids") or []) if _as_str(x)]
        stay_wide = bool(d.get("stay_wide"))
        target = _as_str(d.get("camera_target_local_track_id"))
        if stay_wide:
            ok += 1
            continue
        if len(visible) == 1:
            ok += 1 if target == visible[0] else 0
        elif len(visible) == 0:
            ok += 1 if not target else 0
        else:
            ok += 1 if (not target or target in visible) else 0
    return ok / len(rows)


def _with_scored_candidate_ratio(debug_rows: list[Any]) -> float | None:
    if not debug_rows:
        return None
    scored = 0
    for row in debug_rows:
        if not isinstance(row, dict):
            continue
        cands = row.get("candidates")
        if isinstance(cands, list) and len(cands) > 0:
            scored += 1
    return scored / len(debug_rows)


def _high_confidence_misassignment_proxy_ratio(debug_rows: list[Any]) -> float | None:
    """Proxy: high-confidence rows that look suspicious (ambiguous, low margin, or local mismatch)."""
    eligible = 0
    flagged = 0
    for row in debug_rows:
        if not isinstance(row, dict):
            continue
        conf = _safe_float(row.get("calibrated_confidence"))
        chosen_local = _as_str(row.get("chosen_local_track_id"))
        if conf is None or conf < _HIGH_CONFIDENCE_THRESHOLD or not chosen_local:
            continue
        eligible += 1
        ambiguous = bool(row.get("ambiguous"))
        margin = _safe_float(row.get("top_1_top_2_margin"))
        active_local = _as_str(row.get("active_audio_local_track_id"))
        low_margin = margin is not None and margin < _LOW_TOP_MARGIN_THRESHOLD
        local_mismatch = bool(active_local and chosen_local and active_local != chosen_local)
        if ambiguous or low_margin or local_mismatch:
            flagged += 1
    if eligible == 0:
        return None
    return flagged / eligible


def _tracking_metrics_summary(tm: Mapping[str, Any]) -> dict[str, Any]:
    """Small JSON-safe subset for manifests and logs (avoid huge nested blobs)."""
    keys = (
        "schema_pass_rate",
        "throughput_fps",
        "tracking_mode",
        "tracking_wallclock_s",
        "cluster_tracklets_wallclock_s",
        "lrasd_wallclock_s",
        "speaker_binding_wallclock_s",
        "speaker_binding_assignment_ratio",
        "idf1_proxy",
        "mota_proxy",
        "track_fragmentation_rate",
    )
    out: dict[str, Any] = {}
    for k in keys:
        if k not in tm:
            continue
        v = tm[k]
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, dict)):
            continue
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


def compute_phase1_scorecard(
    phase_1_audio: Mapping[str, Any],
    phase_1_visual: Mapping[str, Any],
    *,
    job_timings_ms: Mapping[str, Any] | None = None,
    tracking_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable scorecard from Phase 1 audio/visual ledgers.

    ``tracking_metrics`` defaults to ``phase_1_visual[\"tracking_metrics\"]`` when omitted.
    ``job_timings_ms`` is typically ``manifest.metadata.timings`` (ingest / processing / upload).
    """
    words = phase_1_audio.get("words") if isinstance(phase_1_audio.get("words"), list) else []
    word_list = list(words)
    n_words = len(word_list)

    unknown = 0
    assigned = 0
    for w in word_list:
        if not isinstance(w, dict):
            continue
        tid = _as_str(w.get("speaker_track_id"))
        if tid:
            assigned += 1
        else:
            unknown += 1

    if n_words == 0:
        assignment_coverage: float | None = None
        unknown_rate: float | None = None
    else:
        assignment_coverage = assigned / n_words
        unknown_rate = unknown / n_words

    debug_rows = (
        phase_1_audio.get("speaker_candidate_debug")
        if isinstance(phase_1_audio.get("speaker_candidate_debug"), list)
        else []
    )
    debug_list = list(debug_rows)

    overlap_decisions = (
        phase_1_audio.get("overlap_follow_decisions")
        if isinstance(phase_1_audio.get("overlap_follow_decisions"), list)
        else []
    )

    tm: Mapping[str, Any] = (
        tracking_metrics
        if isinstance(tracking_metrics, Mapping)
        else (phase_1_visual.get("tracking_metrics") or {})
    )
    if not isinstance(tm, Mapping):
        tm = {}

    timings = job_timings_ms if isinstance(job_timings_ms, Mapping) else {}
    ingest_ms = int(max(0, int(timings.get("ingest_ms", 0) or 0)))
    processing_ms = int(max(0, int(timings.get("processing_ms", 0) or 0)))
    upload_ms = int(max(0, int(timings.get("upload_ms", 0) or 0)))
    total_ms = ingest_ms + processing_ms + upload_ms

    return {
        "version": SCORECARD_VERSION,
        "assignment_coverage": assignment_coverage,
        "with_scored_candidate_ratio": _with_scored_candidate_ratio(debug_list),
        "unknown_rate": unknown_rate,
        "high_confidence_misassignment_proxy_ratio": _high_confidence_misassignment_proxy_ratio(debug_list),
        "overlap_camera_consistency_ratio": _overlap_camera_consistency_ratio(list(overlap_decisions)),
        "wallclock_ms": {
            "ingest_ms": ingest_ms,
            "processing_ms": processing_ms,
            "upload_ms": upload_ms,
            "total_ms": total_ms,
        },
        "stage_wallclock_s": _extract_stage_wallclock_s(tm),
        "tracking_metrics_summary": _tracking_metrics_summary(tm),
        "counts": {
            "word_count": n_words,
            "assigned_word_count": assigned,
            "unknown_word_count": unknown,
            "speaker_candidate_debug_rows": len(debug_list),
            "overlap_follow_decisions": len([d for d in overlap_decisions if isinstance(d, dict)]),
        },
    }


__all__ = [
    "SCORECARD_VERSION",
    "compute_phase1_scorecard",
]
