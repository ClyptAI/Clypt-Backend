from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from backend.do_phase1_service.extract import run_extraction_job
from backend.do_phase1_service.jobs import mark_failed, mark_succeeded
from backend.do_phase1_service.state_store import SQLiteJobStore
from backend.do_phase1_service.storage import GCSStorage, StorageBackend


DEFAULT_STATE_ROOT = Path(os.getenv("DO_PHASE1_STATE_ROOT", "/var/lib/clypt/do_phase1_service"))
DEFAULT_DB_PATH = Path(os.getenv("DO_PHASE1_DB_PATH", str(DEFAULT_STATE_ROOT / "jobs.db")))
DEFAULT_OUTPUT_ROOT = Path(os.getenv("DO_PHASE1_OUTPUT_ROOT", str(DEFAULT_STATE_ROOT / "workdir")))
DEFAULT_RUNNING_STALE_AFTER_SECONDS = int(os.getenv("DO_PHASE1_RUNNING_STALE_AFTER_SECONDS", "1800"))
DEFAULT_LOOP_ERROR_BACKOFF_SECONDS = float(os.getenv("DO_PHASE1_LOOP_ERROR_BACKOFF_SECONDS", "2.0"))

logger = logging.getLogger(__name__)


def run_worker_once(
    store: SQLiteJobStore,
    *,
    output_root: str | Path,
    storage: StorageBackend | None = None,
    stale_after_seconds: int = DEFAULT_RUNNING_STALE_AFTER_SECONDS,
) -> bool:
    job = store.claim_next_job(stale_after_seconds=stale_after_seconds)
    if job is None:
        return False

    failed_step = "control_plane"
    try:
        failed_step = "storage_init"
        storage = storage or GCSStorage()

        failed_step = "extraction"
        manifest = run_extraction_job(
            source_url=job.source_url,
            job_id=job.job_id,
            output_dir=Path(output_root),
            storage=storage,
            attempt_count=job.retries,
        )

        failed_step = "persist_success"
        mark_succeeded(
            store,
            job.job_id,
            manifest=_manifest_payload(manifest),
            manifest_uri=manifest.manifest_uri,
        )
    except Exception as exc:
        _mark_job_failed_safely(
            store,
            job.job_id,
            error_type=type(exc).__name__,
            error_message=str(exc),
            failed_step=failed_step,
        )
        return True
    return True


def run_worker_loop(
    store: SQLiteJobStore | None = None,
    *,
    output_root: str | Path | None = None,
    storage: StorageBackend | None = None,
    poll_interval_seconds: float = 2.0,
) -> None:
    store = store or SQLiteJobStore(DEFAULT_DB_PATH)
    output_root = Path(output_root or DEFAULT_OUTPUT_ROOT)
    while True:
        try:
            processed = run_worker_once(
            store,
            output_root=output_root,
            storage=storage,
            stale_after_seconds=DEFAULT_RUNNING_STALE_AFTER_SECONDS,
        )
        except Exception:
            logger.exception("Phase 1 worker loop hit an unexpected control-plane error")
            time.sleep(DEFAULT_LOOP_ERROR_BACKOFF_SECONDS)
            processed = True
        if not processed:
            time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    run_worker_loop()


def _manifest_payload(manifest) -> dict:
    payload = manifest.model_dump(mode="json")
    payload.pop("manifest_uri", None)
    return payload


def _mark_job_failed_safely(
    store: SQLiteJobStore,
    job_id: str,
    *,
    error_type: str,
    error_message: str,
    failed_step: str,
) -> None:
    try:
        mark_failed(
            store,
            job_id,
            error_type=error_type,
            error_message=error_message,
            failed_step=failed_step,
        )
    except Exception:
        pass
