from __future__ import annotations

import logging
import os
import threading
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
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = float(os.getenv("DO_PHASE1_HEARTBEAT_INTERVAL_SECONDS", "30.0"))
DEFAULT_LOOP_ERROR_BACKOFF_SECONDS = float(os.getenv("DO_PHASE1_LOOP_ERROR_BACKOFF_SECONDS", "2.0"))

logger = logging.getLogger(__name__)


def run_worker_once(
    store: SQLiteJobStore,
    *,
    output_root: str | Path,
    storage: StorageBackend | None = None,
    stale_after_seconds: int = DEFAULT_RUNNING_STALE_AFTER_SECONDS,
    heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
) -> bool:
    job = store.claim_next_job(stale_after_seconds=stale_after_seconds)
    if job is None:
        return False
    if job.claim_token is None:
        logger.warning("Claimed Phase 1 job %s without a claim token; skipping", job.job_id)
        return True

    failed_step = "control_plane"
    heartbeat_stop = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    try:
        failed_step = "storage_init"
        storage = storage or GCSStorage()

        heartbeat_thread = _start_heartbeat_thread(
            store,
            job.job_id,
            claim_token=job.claim_token,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            stop_event=heartbeat_stop,
        )

        failed_step = "extraction"
        manifest = run_extraction_job(
            source_url=job.source_url,
            job_id=job.job_id,
            output_dir=Path(output_root),
            storage=storage,
        )

        heartbeat_stop.set()
        _join_heartbeat_thread(heartbeat_thread)
        failed_step = "persist_success"
        mark_succeeded(
            store,
            job.job_id,
            claim_token=job.claim_token,
            manifest=_manifest_payload(manifest),
            manifest_uri=manifest.manifest_uri,
        )
    except Exception as exc:
        heartbeat_stop.set()
        _join_heartbeat_thread(heartbeat_thread)
        _mark_job_failed_safely(
            store,
            job.job_id,
            claim_token=job.claim_token,
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
                heartbeat_interval_seconds=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
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
    claim_token: str,
    *,
    error_type: str,
    error_message: str,
    failed_step: str,
) -> None:
    try:
        mark_failed(
            store,
            job_id,
            claim_token=claim_token,
            error_type=error_type,
            error_message=error_message,
            failed_step=failed_step,
        )
    except Exception:
        pass


def _start_heartbeat_thread(
    store: SQLiteJobStore,
    job_id: str,
    claim_token: str,
    *,
    heartbeat_interval_seconds: float,
    stop_event: threading.Event,
) -> threading.Thread:
    def _heartbeat_loop() -> None:
        while not stop_event.wait(heartbeat_interval_seconds):
            try:
                heartbeat = store.heartbeat_job(job_id, claim_token)
                if heartbeat is None:
                    logger.info("Phase 1 job %s lost its active claim; stopping heartbeat", job_id)
                    return
            except Exception:
                logger.warning("Failed to heartbeat running Phase 1 job %s", job_id, exc_info=True)

    thread = threading.Thread(target=_heartbeat_loop, name=f"phase1-heartbeat-{job_id}", daemon=True)
    thread.start()
    return thread


def _join_heartbeat_thread(thread: threading.Thread | None) -> None:
    if thread is None:
        return
    thread.join(timeout=1.0)
