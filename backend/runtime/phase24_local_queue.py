from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


class Phase24LocalQueue:
    """Durable SQLite (WAL) queue for Phase 2–4 jobs in local development."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS phase24_jobs (
                    job_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    available_at REAL NOT NULL,
                    locked_at REAL,
                    worker_id TEXT,
                    payload_json TEXT NOT NULL,
                    last_error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_phase24_jobs_status_avail "
                "ON phase24_jobs (status, available_at, created_at)"
            )
            conn.commit()

    def enqueue(self, run_id: str, payload: dict[str, Any]) -> str:
        """Insert a queued job, or return the existing job_id for run_id."""
        now = time.time()
        payload_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT job_id FROM phase24_jobs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is not None:
                conn.commit()
                return str(row["job_id"])
            job_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO phase24_jobs (
                    job_id, run_id, status, attempt_count, available_at,
                    locked_at, worker_id, payload_json, last_error, created_at, updated_at
                ) VALUES (?, ?, 'queued', 0, ?, NULL, NULL, ?, NULL, ?, ?)
                """,
                (job_id, run_id, now, payload_json, now, now),
            )
            conn.commit()
            return job_id

    def claim_next(
        self,
        worker_id: str,
        lease_timeout_s: int,
        *,
        max_inflight: int | None = None,
        reclaim_expired_leases: bool = True,
    ) -> dict[str, Any] | None:
        """
        Optionally reclaim expired leases, then atomically claim the next queued job.
        Returns a row dict including payload (parsed) and attempt_count after increment.
        """
        now = time.time()
        lease_timeout_s = max(1, int(lease_timeout_s))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if reclaim_expired_leases:
                conn.execute(
                    """
                    UPDATE phase24_jobs
                    SET status = 'queued',
                        locked_at = NULL,
                        worker_id = NULL,
                        updated_at = ?
                    WHERE status = 'running'
                      AND locked_at IS NOT NULL
                      AND (? - locked_at) > ?
                    """,
                    (now, now, float(lease_timeout_s)),
                )
            if max_inflight is not None and int(max_inflight) > 0:
                running_count = conn.execute(
                    "SELECT COUNT(1) AS c FROM phase24_jobs WHERE status = 'running'"
                ).fetchone()
                if int(running_count["c"]) >= int(max_inflight):
                    conn.commit()
                    return None
            picked = conn.execute(
                """
                SELECT job_id FROM phase24_jobs
                WHERE status = 'queued' AND available_at <= ?
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if picked is None:
                conn.commit()
                return None
            jid = str(picked["job_id"])
            cur = conn.execute(
                """
                UPDATE phase24_jobs
                SET status = 'running',
                    attempt_count = attempt_count + 1,
                    locked_at = ?,
                    worker_id = ?,
                    updated_at = ?
                WHERE job_id = ? AND status = 'queued'
                RETURNING
                    job_id,
                    run_id,
                    status,
                    attempt_count,
                    available_at,
                    locked_at,
                    worker_id,
                    payload_json,
                    last_error,
                    created_at,
                    updated_at
                """,
                (now, worker_id, now, jid),
            )
            row = cur.fetchone()
            conn.commit()
            if row is None:
                return None
            out = dict(row)
            out["payload"] = json.loads(out.pop("payload_json"))
            return out

    def count_expired_running(self, lease_timeout_s: int) -> int:
        """Count running jobs with expired leases without reclaiming them."""
        now = time.time()
        timeout_s = max(1, int(lease_timeout_s))
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(1) AS c
                FROM phase24_jobs
                WHERE status = 'running'
                  AND locked_at IS NOT NULL
                  AND (? - locked_at) > ?
                """,
                (now, float(timeout_s)),
            ).fetchone()
            return int(row["c"]) if row is not None else 0

    def mark_succeeded(self, job_id: str) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE phase24_jobs
                SET status = 'succeeded',
                    locked_at = NULL,
                    worker_id = NULL,
                    last_error = NULL,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (now, job_id),
            )
            conn.commit()

    def mark_failed(
        self,
        job_id: str,
        *,
        error: str,
        retry: bool,
        retry_delay_s: float = 0.0,
    ) -> None:
        now = time.time()
        delay = max(0.0, float(retry_delay_s))
        with self._connect() as conn:
            if retry:
                conn.execute(
                    """
                    UPDATE phase24_jobs
                    SET status = 'queued',
                        last_error = ?,
                        available_at = ?,
                        locked_at = NULL,
                        worker_id = NULL,
                        updated_at = ?
                    WHERE job_id = ?
                    """,
                    (error[:8192], now + delay, now, job_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE phase24_jobs
                    SET status = 'failed',
                        last_error = ?,
                        locked_at = NULL,
                        worker_id = NULL,
                        updated_at = ?
                    WHERE job_id = ?
                    """,
                    (error[:8192], now, job_id),
                )
            conn.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM phase24_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(row)


__all__ = ["Phase24LocalQueue"]
