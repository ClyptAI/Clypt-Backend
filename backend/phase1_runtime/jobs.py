from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from .models import Phase1JobCreatePayload, Phase1JobRecord
from .state_store import SQLiteJobStore

UTC = timezone.utc


def create_job(store: SQLiteJobStore, payload: Phase1JobCreatePayload) -> Phase1JobRecord:
    now = datetime.now(UTC).isoformat()
    job_id = f"job_{uuid4().hex}"
    return store.save_job(
        job_id=job_id,
        source_url=payload.source_url,
        source_path=payload.source_path,
        runtime_controls=payload.runtime_controls,
        status="queued",
        current_step="queued",
        progress_message="Queued for V3.1 Phase 1 worker",
        progress_pct=0.0,
        created_at=now,
        updated_at=now,
    )


def get_job(store: SQLiteJobStore, job_id: str) -> Phase1JobRecord | None:
    return store.get_job(job_id)


__all__ = [
    "create_job",
    "get_job",
]
