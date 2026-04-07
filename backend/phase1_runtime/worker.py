from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import logging
from pathlib import Path
import io
import time
from typing import Any, Callable

from .state_store import SQLiteJobStore


def _build_job_log_handler(log_path: Path) -> logging.Handler:
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    return handler


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

        class _TeeStream:
            def __init__(self, *streams: io.TextIOBase):
                self._streams = streams

            def write(self, data: str) -> int:
                for stream_obj in self._streams:
                    stream_obj.write(data)
                    stream_obj.flush()
                return len(data)

            def flush(self) -> None:
                for stream_obj in self._streams:
                    stream_obj.flush()

        try:
            with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
                tee = _TeeStream(stream, log_file)
                root_logger = logging.getLogger()
                job_log_handler = _build_job_log_handler(log_path)
                root_logger.addHandler(job_log_handler)
                try:
                    with redirect_stdout(tee), redirect_stderr(tee):
                        result = self.run_job(
                            job_id=job.job_id,
                            source_url=job.source_url,
                            source_path=job.source_path,
                            runtime_controls=job.runtime_controls,
                        )
                finally:
                    root_logger.removeHandler(job_log_handler)
                    job_log_handler.flush()
                    job_log_handler.close()
                tee.flush()
            completed = self.store.complete_job(
                job_id=job.job_id,
                claim_token=job.claim_token or "",
                result=result,
            )
            if completed is None:
                raise RuntimeError(f"could not complete claimed job {job.job_id}")
            return True
        except Exception as exc:
            if not log_path.exists():
                log_path.write_text(stream.getvalue(), encoding="utf-8")
            self.store.fail_job(
                job_id=job.job_id,
                claim_token=job.claim_token or "",
                failure={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
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
