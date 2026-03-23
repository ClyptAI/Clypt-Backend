from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from backend.do_phase1_service.models import JobRecord


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
                    source_url TEXT NOT NULL,
                    runtime_controls_json TEXT,
                    status TEXT NOT NULL,
                    retries INTEGER NOT NULL DEFAULT 0,
                    claim_token TEXT,
                    manifest_json TEXT,
                    manifest_uri TEXT,
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
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "claim_token" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN claim_token TEXT")
            if "runtime_controls_json" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN runtime_controls_json TEXT")
            if "current_step" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN current_step TEXT")
            if "progress_message" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN progress_message TEXT")
            if "progress_pct" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN progress_pct REAL")
            if "log_path" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN log_path TEXT")

    def healthcheck(self) -> bool:
        with self._connect() as connection:
            connection.execute("SELECT 1")
        return True

    def claim_next_job(self, *, stale_after_seconds: int = 1800) -> JobRecord | None:
        now = datetime.now(UTC)
        stale_before = now - timedelta(seconds=stale_after_seconds)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'queued'
                   OR (status = 'running' AND updated_at <= ?)
                ORDER BY CASE WHEN status = 'queued' THEN 0 ELSE 1 END, created_at ASC
                LIMIT 1
                """,
                (stale_before.isoformat(),),
            ).fetchone()
            if row is None:
                connection.commit()
                return None

            started_at = row["started_at"] or now.isoformat()
            retries = int(row["retries"]) + 1
            claim_token = uuid4().hex
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, retries = ?, claim_token = ?, updated_at = ?, started_at = ?, completed_at = NULL
                WHERE job_id = ?
                """,
                ("running", retries, claim_token, now.isoformat(), started_at, row["job_id"]),
            )
            claimed = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)).fetchone()
            connection.commit()
        return self._row_to_job(claimed)

    def save_job(
        self,
        *,
        job_id: str,
        source_url: str,
        runtime_controls: dict | None = None,
        status: str,
        retries: int = 0,
        claim_token: str | None = None,
        manifest: dict | None = None,
        manifest_uri: str | None = None,
        failure: dict | None = None,
        current_step: str | None = None,
        progress_message: str | None = None,
        progress_pct: float | None = None,
        log_path: str | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> JobRecord:
        now = datetime.now(UTC)
        created = created_at or now
        updated = updated_at or now
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, source_url, runtime_controls_json, status, retries, claim_token, manifest_json, manifest_uri,
                    failure_json, current_step, progress_message, progress_pct, log_path,
                    created_at, updated_at, started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    source_url=excluded.source_url,
                    runtime_controls_json=excluded.runtime_controls_json,
                    status=excluded.status,
                    retries=excluded.retries,
                    claim_token=excluded.claim_token,
                    manifest_json=excluded.manifest_json,
                    manifest_uri=excluded.manifest_uri,
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
                    json.dumps(runtime_controls) if runtime_controls is not None else None,
                    status,
                    retries,
                    claim_token,
                    json.dumps(manifest) if manifest is not None else None,
                    manifest_uri,
                    json.dumps(failure) if failure is not None else None,
                    current_step,
                    progress_message,
                    progress_pct,
                    log_path,
                    created.isoformat(),
                    updated.isoformat(),
                    started_at.isoformat() if started_at else None,
                    completed_at.isoformat() if completed_at else None,
                ),
            )
        return self.get_job(job_id)

    def heartbeat_job(self, job_id: str, claim_token: str) -> JobRecord | None:
        now = datetime.now(UTC)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET updated_at = ?
                WHERE job_id = ? AND status = 'running' AND claim_token = ?
                """,
                (now.isoformat(), job_id, claim_token),
            )
            if connection.total_changes == 0:
                return None
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def update_job_progress(
        self,
        job_id: str,
        *,
        claim_token: str | None = None,
        current_step: str | None = None,
        progress_message: str | None = None,
        progress_pct: float | None = None,
        log_path: str | None = None,
    ) -> JobRecord | None:
        now = datetime.now(UTC)
        assignments: list[str] = ["updated_at = ?"]
        params: list[object] = [now.isoformat()]

        if current_step is not None:
            assignments.append("current_step = ?")
            params.append(current_step)
        if progress_message is not None:
            assignments.append("progress_message = ?")
            params.append(progress_message)
        if progress_pct is not None:
            assignments.append("progress_pct = ?")
            params.append(float(progress_pct))
        if log_path is not None:
            assignments.append("log_path = ?")
            params.append(log_path)
        if len(assignments) == 1:
            return self.get_job(job_id)

        where = ["job_id = ?"]
        params.append(job_id)
        if claim_token is not None:
            where.extend(["status = 'running'", "claim_token = ?"])
            params.append(claim_token)

        with self._connect() as connection:
            connection.execute(
                f"UPDATE jobs SET {', '.join(assignments)} WHERE {' AND '.join(where)}",
                tuple(params),
            )
            if connection.total_changes == 0:
                return None
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def complete_job(
        self,
        *,
        job_id: str,
        claim_token: str,
        manifest: dict,
        manifest_uri: str,
    ) -> JobRecord | None:
        now = datetime.now(UTC)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'succeeded',
                    manifest_json = ?,
                    manifest_uri = ?,
                    failure_json = NULL,
                    current_step = 'complete',
                    progress_message = 'Phase 1 manifest persisted',
                    progress_pct = 1.0,
                    updated_at = ?,
                    completed_at = ?,
                    claim_token = NULL
                WHERE job_id = ? AND status = 'running' AND claim_token = ?
                """,
                (
                    json.dumps(manifest),
                    manifest_uri,
                    now.isoformat(),
                    now.isoformat(),
                    job_id,
                    claim_token,
                ),
            )
            if connection.total_changes == 0:
                return None
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def fail_job(
        self,
        *,
        job_id: str,
        claim_token: str,
        error_type: str,
        error_message: str,
        failed_step: str | None = None,
    ) -> JobRecord | None:
        now = datetime.now(UTC)
        failure = json.dumps(
            {
                "error_type": error_type,
                "error_message": error_message,
                "failed_step": failed_step,
            }
        )
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    failure_json = ?,
                    current_step = COALESCE(?, current_step),
                    progress_message = ?,
                    updated_at = ?,
                    completed_at = ?,
                    claim_token = NULL
                WHERE job_id = ? AND status = 'running' AND claim_token = ?
                """,
                (
                    failure,
                    failed_step,
                    error_message,
                    now.isoformat(),
                    now.isoformat(),
                    job_id,
                    claim_token,
                ),
            )
            if connection.total_changes == 0:
                return None
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list_recoverable_jobs(self) -> list[JobRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at ASC"
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def list_recent_jobs(self, *, limit: int = 20) -> list[JobRecord]:
        limit = max(1, min(200, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            source_url=row["source_url"],
            runtime_controls=json.loads(row["runtime_controls_json"]) if row["runtime_controls_json"] else None,
            status=row["status"],
            retries=int(row["retries"]),
            claim_token=row["claim_token"],
            manifest=json.loads(row["manifest_json"]) if row["manifest_json"] else None,
            manifest_uri=row["manifest_uri"],
            failure=json.loads(row["failure_json"]) if row["failure_json"] else None,
            current_step=row["current_step"],
            progress_message=row["progress_message"],
            progress_pct=float(row["progress_pct"]) if row["progress_pct"] is not None else None,
            log_path=row["log_path"],
            created_at=_parse_db_datetime(row["created_at"]),
            updated_at=_parse_db_datetime(row["updated_at"]),
            started_at=_parse_db_datetime(row["started_at"]) if row["started_at"] else None,
            completed_at=_parse_db_datetime(row["completed_at"]) if row["completed_at"] else None,
        )
UTC = timezone.utc


def _parse_db_datetime(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)
