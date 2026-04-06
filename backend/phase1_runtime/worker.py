from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import io
import time
from typing import Any, Callable

from .state_store import SQLiteJobStore


class Phase1Worker:
    def __init__(
        self,
        *,
        store: SQLiteJobStore,
        run_job: Callable[..., dict[str, Any]],
        logs_root: str | Path | None = None,
    ):
        self.store = store
        self.run_job = run_job
        self.logs_root = Path(logs_root or "backend/outputs/v3_1_phase1_service/logs")

    def run_next_job_once(self) -> bool:
        job = self.store.claim_next_job()
        if job is None:
            return False
        self.logs_root.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_root / f"{job.job_id}.log"
        self.store.save_job(
            job_id=job.job_id,
            source_url=job.source_url,
            source_path=job.source_path,
            runtime_controls=job.runtime_controls,
            status=job.status,
            retries=job.retries,
            claim_token=job.claim_token,
            result=job.result,
            failure=job.failure,
            current_step=job.current_step,
            progress_message=job.progress_message,
            progress_pct=job.progress_pct,
            log_path=str(log_path),
            created_at=job.created_at,
            updated_at=job.updated_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )
        stream = io.StringIO()
        try:
            with redirect_stdout(stream), redirect_stderr(stream):
                result = self.run_job(
                    job_id=job.job_id,
                    source_url=job.source_url,
                    source_path=job.source_path,
                    runtime_controls=job.runtime_controls,
                )
            log_path.write_text(stream.getvalue(), encoding="utf-8")
            completed = self.store.complete_job(
                job_id=job.job_id,
                claim_token=job.claim_token or "",
                result=result,
            )
            if completed is None:
                raise RuntimeError(f"could not complete claimed job {job.job_id}")
            return True
        except Exception as exc:
            log_path.write_text(stream.getvalue(), encoding="utf-8")
            failure = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            failing_branch = getattr(exc, "failing_branch", None)
            branch_log_path = getattr(exc, "branch_log_path", None)
            if failing_branch is not None:
                failure["failing_branch"] = str(failing_branch)
            if branch_log_path is not None:
                failure["branch_log_path"] = str(branch_log_path)
            self.store.fail_job(
                job_id=job.job_id,
                claim_token=job.claim_token or "",
                failure=failure,
            )
            raise

    def run_forever(self, *, poll_interval_s: float = 2.0, stop_after_idle_loops: int | None = None) -> None:
        idle_loops = 0
        while True:
            processed = self.run_next_job_once()
            if processed:
                idle_loops = 0
                continue
            idle_loops += 1
            if stop_after_idle_loops is not None and idle_loops >= stop_after_idle_loops:
                return
            time.sleep(max(0.0, float(poll_interval_s)))


__all__ = ["Phase1Worker"]
