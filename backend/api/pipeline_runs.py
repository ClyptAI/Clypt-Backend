from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PipelineRunStore:
    """Simple file-backed store for pipeline run state."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _run_path(self, run_id: str) -> Path:
        return self.root / run_id / "state.json"

    def _read(self, run_id: str) -> dict[str, Any] | None:
        path = self._run_path(run_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, run_id: str, data: dict[str, Any]) -> None:
        path = self._run_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def create_run(self, *, video_url: str, creator_id: str | None = None) -> dict[str, Any]:
        import uuid
        run_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        data: dict[str, Any] = {
            "run_id": run_id,
            "status": "pending",
            "video_url": video_url,
            "creator_id": creator_id,
            "created_at": now,
            "updated_at": now,
            "phase": None,
            "progress_pct": 0,
            "detail": None,
            "summary": None,
        }
        self._write(run_id, data)
        return data

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._read(run_id)

    def mark_running(self, run_id: str, *, phase: str | None = None, progress_pct: int = 0, detail: str | None = None) -> None:
        data = self._read(run_id)
        if data is None:
            return
        data["status"] = "running"
        data["phase"] = phase
        data["progress_pct"] = progress_pct
        data["detail"] = detail
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write(run_id, data)

    def mark_succeeded(self, run_id: str, *, summary: dict[str, Any] | None = None) -> None:
        data = self._read(run_id)
        if data is None:
            return
        data["status"] = "succeeded"
        data["progress_pct"] = 100
        data["phase"] = "complete"
        data["summary"] = summary
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write(run_id, data)

    def mark_failed(self, run_id: str, *, phase: str | None = None, detail: str | None = None) -> None:
        data = self._read(run_id)
        if data is None:
            return
        data["status"] = "failed"
        data["phase"] = phase
        data["detail"] = detail
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write(run_id, data)
