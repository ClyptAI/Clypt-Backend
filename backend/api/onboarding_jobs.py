from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class OnboardingJobStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_job(self, *, channel_id: str) -> dict[str, Any]:
        job_id = f"creator_job_{uuid.uuid4().hex[:12]}"
        now = self._now()
        job = {
            "job_id": job_id,
            "channel_id": channel_id,
            "status": "queued",
            "progress_pct": 0,
            "current_stage": "queued",
            "stage_label": "Queued",
            "stage_detail": "Waiting to start analysis",
            "created_at": now,
            "updated_at": now,
            "profile": None,
            "error": None,
        }
        self._write(job_id, job)
        return job

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        path = self._path(job_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def update_job(self, job_id: str, **fields: Any) -> dict[str, Any]:
        existing = self.get_job(job_id)
        if existing is None:
            raise FileNotFoundError(job_id)
        existing.update(fields)
        existing["updated_at"] = self._now()
        self._write(job_id, existing)
        return existing

    def mark_running(self, job_id: str, *, stage: str, progress_pct: int, detail: str) -> dict[str, Any]:
        return self.update_job(
            job_id,
            status="running",
            progress_pct=max(0, min(progress_pct, 100)),
            current_stage=stage,
            stage_label=self._stage_label(stage),
            stage_detail=detail,
        )

    def mark_succeeded(self, job_id: str, *, profile: dict[str, Any], channel: dict[str, Any], sources: list[dict[str, Any]]) -> dict[str, Any]:
        return self.update_job(
            job_id,
            status="succeeded",
            progress_pct=100,
            current_stage="complete",
            stage_label="Complete",
            stage_detail="Creator profile ready",
            profile=profile,
            channel=channel,
            recent_items_scanned=sources,
            error=None,
        )

    def mark_failed(self, job_id: str, *, stage: str, detail: str) -> dict[str, Any]:
        return self.update_job(
            job_id,
            status="failed",
            current_stage=stage,
            stage_label=self._stage_label(stage),
            stage_detail=detail,
            error=detail,
        )

    def _path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def _write(self, job_id: str, payload: dict[str, Any]) -> None:
        self._path(job_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _stage_label(stage: str) -> str:
        return stage.replace("_", " ").strip().title()
