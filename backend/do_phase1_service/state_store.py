from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

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
                    status TEXT NOT NULL,
                    retries INTEGER NOT NULL DEFAULT 0,
                    manifest_json TEXT,
                    manifest_uri TEXT,
                    failure_json TEXT,
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
        source_url: str,
        status: str,
        retries: int = 0,
        manifest: dict | None = None,
        manifest_uri: str | None = None,
        failure: dict | None = None,
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
                    job_id, source_url, status, retries, manifest_json, manifest_uri,
                    failure_json, created_at, updated_at, started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    source_url=excluded.source_url,
                    status=excluded.status,
                    retries=excluded.retries,
                    manifest_json=excluded.manifest_json,
                    manifest_uri=excluded.manifest_uri,
                    failure_json=excluded.failure_json,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    started_at=excluded.started_at,
                    completed_at=excluded.completed_at
                """,
                (
                    job_id,
                    source_url,
                    status,
                    retries,
                    json.dumps(manifest) if manifest is not None else None,
                    manifest_uri,
                    json.dumps(failure) if failure is not None else None,
                    created.isoformat(),
                    updated.isoformat(),
                    started_at.isoformat() if started_at else None,
                    completed_at.isoformat() if completed_at else None,
                ),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list_recoverable_jobs(self) -> list[JobRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE status IN ('queued', 'running') ORDER BY created_at ASC"
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def pop_next_job(self) -> JobRecord | None:
        jobs = self.list_recoverable_jobs()
        return jobs[0] if jobs else None

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            source_url=row["source_url"],
            status=row["status"],
            retries=int(row["retries"]),
            manifest=json.loads(row["manifest_json"]) if row["manifest_json"] else None,
            manifest_uri=row["manifest_uri"],
            failure=json.loads(row["failure_json"]) if row["failure_json"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        )
