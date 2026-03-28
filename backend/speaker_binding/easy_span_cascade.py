"""Cheap classifier for single-speaker spans that can skip LR-ASD."""

from __future__ import annotations


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def classify_easy_span(
    span: dict | None,
    ranked_candidates: list[dict] | None,
    *,
    min_dominant_visibility: float = 0.75,
    min_continuity_support: float = 0.68,
    max_competitor_ratio: float = 0.55,
) -> dict:
    span = dict(span or {})
    speaker_ids = [
        str(speaker_id)
        for speaker_id in list(span.get("speaker_ids") or [])
        if str(speaker_id or "")
    ]
    if bool(span.get("overlap", False)) or str(span.get("span_type", "")) == "overlap":
        return {"decision": "hard", "reason": "overlap_span"}
    if len(speaker_ids) != 1:
        return {"decision": "hard", "reason": "not_single_speaker"}
    if bool(span.get("requires_lrasd", False)):
        return {"decision": "hard", "reason": "visual_discontinuity"}

    candidates = [dict(candidate) for candidate in (ranked_candidates or []) if isinstance(candidate, dict)]
    if not candidates:
        return {"decision": "hard", "reason": "no_visible_candidate"}

    winner = candidates[0]
    winner_local_track_id = str(winner.get("local_track_id", "") or "")
    if not winner_local_track_id:
        return {"decision": "hard", "reason": "missing_local_track"}
    if not bool(winner.get("candidate_survives", False)):
        return {"decision": "hard", "reason": "weak_body_continuity"}

    winner_visibility = _as_float(winner.get("speech_overlap_ratio"))
    winner_continuity = _as_float(winner.get("continuity_support_score"))
    winner_rank = _as_float(winner.get("rank_score"))
    runner_up = candidates[1] if len(candidates) > 1 else None
    runner_up_visibility = _as_float(None if runner_up is None else runner_up.get("speech_overlap_ratio"))
    runner_up_rank = _as_float(None if runner_up is None else runner_up.get("rank_score"))
    competitor_ratio = (
        0.0
        if winner_visibility <= 1e-6
        else float(runner_up_visibility / winner_visibility)
    )
    if winner_visibility < min_dominant_visibility or competitor_ratio > max_competitor_ratio:
        return {"decision": "hard", "reason": "weak_visibility_or_handoff"}
    if winner_continuity < min_continuity_support:
        return {"decision": "hard", "reason": "weak_body_continuity"}

    winning_score = max(0.0, min(1.0, (0.55 * winner_visibility) + (0.45 * winner_continuity)))
    winning_margin = (
        winner_rank
        if runner_up is None
        else max(0.0, winner_rank - runner_up_rank)
    )
    return {
        "decision": "easy",
        "decision_source": "easy_span",
        "speaker_id": speaker_ids[0],
        "local_track_id": winner_local_track_id,
        "support_ratio": float(winner_visibility),
        "continuity_support_score": float(winner_continuity),
        "competitor_ratio": float(competitor_ratio),
        "winning_score": float(winning_score),
        "winning_margin": float(winning_margin),
        "max_visible_candidates": int(len(candidates)),
    }


def build_easy_span_binding_rows(
    span: dict | None,
    decision: dict | None,
    *,
    global_track_id: str | None = None,
    source_turn_ranges: dict[str, dict] | None = None,
) -> list[dict]:
    span = dict(span or {})
    decision = dict(decision or {})
    if decision.get("decision") != "easy":
        return []

    source_turn_ids = [
        str(source_turn_id)
        for source_turn_id in list(span.get("source_turn_ids") or [])
        if str(source_turn_id or "")
    ] or [""]
    start_time_ms = int(span.get("context_start_time_ms", span.get("start_time_ms", 0)) or 0)
    end_time_ms = int(span.get("context_end_time_ms", span.get("end_time_ms", start_time_ms)) or start_time_ms)
    if end_time_ms < start_time_ms:
        start_time_ms, end_time_ms = end_time_ms, start_time_ms

    rows: list[dict] = []
    for source_turn_id in source_turn_ids:
        turn_range = dict((source_turn_ranges or {}).get(source_turn_id) or {})
        row_start_time_ms = int(turn_range.get("start_time_ms", start_time_ms) or start_time_ms)
        row_end_time_ms = int(turn_range.get("end_time_ms", end_time_ms) or end_time_ms)
        if row_end_time_ms < row_start_time_ms:
            row_start_time_ms, row_end_time_ms = row_end_time_ms, row_start_time_ms
        if turn_range:
            row_start_time_ms = max(int(start_time_ms), int(row_start_time_ms))
            row_end_time_ms = min(int(end_time_ms), int(row_end_time_ms))
        if row_end_time_ms <= row_start_time_ms:
            continue
        row = {
            "speaker_id": str(decision.get("speaker_id", "") or ""),
            "source_turn_id": source_turn_id,
            "span_id": str(span.get("span_id", "") or ""),
            "start_time_ms": int(row_start_time_ms),
            "end_time_ms": int(row_end_time_ms),
            "local_track_id": str(decision.get("local_track_id", "") or ""),
            "ambiguous": False,
            "decision_source": "easy_span",
            "winning_score": float(decision.get("winning_score", 0.0) or 0.0),
            "winning_margin": float(decision.get("winning_margin", 0.0) or 0.0),
            "support_ratio": float(decision.get("support_ratio", 0.0) or 0.0),
            "max_visible_candidates": int(decision.get("max_visible_candidates", 0) or 0),
        }
        if global_track_id not in (None, ""):
            row["track_id"] = str(global_track_id)
        rows.append(row)
    return rows


__all__ = ["build_easy_span_binding_rows", "classify_easy_span"]
