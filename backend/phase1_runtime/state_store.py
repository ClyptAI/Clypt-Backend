from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .models import Phase1JobRecord

UTC = timezone.utc


def _jsonable(value):
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


class SQLiteJobStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    source_url TEXT,
                    source_path TEXT,
                    runtime_controls_json TEXT,
                    status TEXT NOT NULL,
                    retries INTEGER NOT NULL DEFAULT 0,
                    claim_token TEXT,
                    result_json TEXT,
                    failure_json TEXT,
                    current_step TEXT,
                    progress_message TEXT,
                    progress_pct REAL,
                    log_path TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                )
                """
            )

    def healthcheck(self) -> bool:
        with self._connect() as connection:
            connection.execute("SELECT 1")
        return True

    def save_job(
        self,
        *,
        job_id: str,
        source_url: str | None,
        source_path: str | None,
        runtime_controls: dict | None,
        status: str,
        retries: int = 0,
        claim_token: str | None = None,
        result: dict | None = None,
        failure: dict | None = None,
        current_step: str | None = None,
        progress_message: str | None = None,
        progress_pct: float | None = None,
        log_path: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> Phase1JobRecord:
        now_iso = datetime.now(UTC).isoformat()
        created_at = created_at or now_iso
        updated_at = updated_at or now_iso
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, source_url, source_path, runtime_controls_json, status, retries, claim_token,
                    result_json, failure_json, current_step, progress_message, progress_pct, log_path,
                    created_at, updated_at, started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    source_url=excluded.source_url,
                    source_path=excluded.source_path,
                    runtime_controls_json=excluded.runtime_controls_json,
                    status=excluded.status,
                    retries=excluded.retries,
                    claim_token=excluded.claim_token,
                    result_json=excluded.result_json,
                    failure_json=excluded.failure_json,
                    current_step=excluded.current_step,
                    progress_message=excluded.progress_message,
                    progress_pct=excluded.progress_pct,
                    log_path=excluded.log_path,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    started_at=excluded.started_at,
                    completed_at=excluded.completed_at
                """,
                (
                    job_id,
                    source_url,
                    source_path,
                    json.dumps(runtime_controls) if runtime_controls is not None else None,
                    status,
                    retries,
                    claim_token,
                    json.dumps(_jsonable(result)) if result is not None else None,
                    json.dumps(_jsonable(failure)) if failure is not None else None,
                    current_step,
                    progress_message,
                    progress_pct,
                    log_path,
                    created_at,
                    updated_at,
                    started_at,
                    completed_at,
                ),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Phase1JobRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def claim_next_job(self) -> Phase1JobRecord | None:
        now = datetime.now(UTC).isoformat()
        claim_token = uuid4().hex
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            connection.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    retries = retries + 1,
                    claim_token = ?,
                    current_step = 'running',
                    progress_message = 'Worker claimed job',
                    progress_pct = 0.01,
                    updated_at = ?,
                    started_at = COALESCE(started_at, ?)
                WHERE job_id = ?
                """,
                (claim_token, now, now, row["job_id"]),
            )
            claimed = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)).fetchone()
            connection.commit()
        return self._row_to_job(claimed) if claimed else None

    def complete_job(self, *, job_id: str, claim_token: str, result: dict) -> Phase1JobRecord | None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'succeeded',
                    claim_token = NULL,
                    result_json = ?,
                    current_step = 'complete',
                    progress_message = 'Job succeeded',
                    progress_pct = 1.0,
                    updated_at = ?,
                    completed_at = ?
                WHERE job_id = ? AND claim_token = ? AND status = 'running'
                """,
                (json.dumps(_jsonable(result)), now, now, job_id, claim_token),
            )
            if connection.total_changes == 0:
                return None
        return self.get_job(job_id)

    def fail_job(self, *, job_id: str, claim_token: str, failure: dict) -> Phase1JobRecord | None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    claim_token = NULL,
                    failure_json = ?,
                    progress_message = ?,
                    updated_at = ?,
                    completed_at = ?
                WHERE job_id = ? AND claim_token = ? AND status = 'running'
                """,
                (json.dumps(_jsonable(failure)), failure.get("error_message") or "Job failed", now, now, job_id, claim_token),
            )
            if connection.total_changes == 0:
                return None
        return self.get_job(job_id)

    def _row_to_job(self, row: sqlite3.Row) -> Phase1JobRecord:
        return Phase1JobRecord(
            job_id=row["job_id"],
            source_url=row["source_url"],
            source_path=row["source_path"],
            runtime_controls=json.loads(row["runtime_controls_json"]) if row["runtime_controls_json"] else None,
            status=row["status"],
            retries=int(row["retries"]),
            claim_token=row["claim_token"],
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            failure=json.loads(row["failure_json"]) if row["failure_json"] else None,
            current_step=row["current_step"],
            progress_message=row["progress_message"],
            progress_pct=row["progress_pct"],
            log_path=row["log_path"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )


__all__ = ["SQLiteJobStore"]
