"""Structured JSON logs for Phase 1 stage boundaries (spec: job_id, stage, event, …)."""

from __future__ import annotations

import json
import os
from typing import Any

def resolve_phase1_worker_id() -> str:
    w = str(os.getenv("CLYPT_WORKER_ID", "") or "").strip()
    return w or "unknown"


def resolve_job_and_worker_ids(log_context: dict[str, Any] | None) -> tuple[str, str]:
    """Prefer job_id / worker_id from overlap-follow log context when present."""
    jid = ""
    wid = resolve_phase1_worker_id()
    if isinstance(log_context, dict):
        raw_j = log_context.get("job_id")
        if raw_j is not None and str(raw_j).strip():
            jid = str(raw_j).strip()
        raw_w = str(log_context.get("worker_id") or "").strip()
        if raw_w:
            wid = raw_w
    return jid, wid


def build_stage_log_payload(
    *,
    job_id: str,
    stage: str,
    event: str,
    decision_source: str,
    reason_code: str,
    elapsed_ms: int,
    worker_id: str,
    **extras: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "job_id": job_id or "",
        "stage": stage,
        "event": event,
        "decision_source": decision_source,
        "reason_code": reason_code,
        "elapsed_ms": int(max(0, elapsed_ms)),
        "worker_id": worker_id or "unknown",
    }
    for key, val in extras.items():
        if val is None:
            continue
        payload[key] = val
    return payload


def emit_stage_log(
    *,
    job_id: str,
    stage: str,
    event: str,
    decision_source: str,
    reason_code: str,
    elapsed_ms: int,
    worker_id: str,
    **extras: Any,
) -> None:
    payload = build_stage_log_payload(
        job_id=job_id,
        stage=stage,
        event=event,
        decision_source=decision_source,
        reason_code=reason_code,
        elapsed_ms=elapsed_ms,
        worker_id=worker_id,
        **extras,
    )
    print(json.dumps(payload, default=str, separators=(",", ":")))


def emit_overlap_follow_postpass_summary(
    *,
    job_id: str,
    worker_id: str,
    run_metadata: dict[str, Any],
) -> None:
    extras: dict[str, Any] = {}
    for key in ("reason_code_counts", "adjudication_path_counts", "fallback_category_counts"):
        val = run_metadata.get(key)
        if val:
            extras[key] = val
    rc = "ok"
    rcc = run_metadata.get("reason_code_counts")
    if isinstance(rcc, dict) and rcc:
        if int(rcc.get("gemini_unavailable", 0) or 0) > 0:
            rc = "gemini_unavailable"
        elif int(rcc.get("gemini_invalid_response", 0) or 0) > 0:
            rc = "gemini_invalid_response"
        elif int(rcc.get("low_overlap_evidence", 0) or 0) > 0:
            rc = "low_overlap_evidence"
        elif int(rcc.get("deterministic_selected", 0) or 0) > 0:
            rc = "deterministic_selected"
    emit_stage_log(
        job_id=job_id,
        stage="overlap_follow",
        event="postpass_summary",
        decision_source="overlap_adjudication",
        reason_code=rc,
        elapsed_ms=0,
        worker_id=worker_id,
        **extras,
    )


__all__ = [
    "build_stage_log_payload",
    "emit_overlap_follow_postpass_summary",
    "emit_stage_log",
    "resolve_job_and_worker_ids",
    "resolve_phase1_worker_id",
]
