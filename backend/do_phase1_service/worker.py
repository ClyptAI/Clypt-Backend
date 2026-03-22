from __future__ import annotations

import os
import time
from pathlib import Path

from backend.do_phase1_service.extract import run_extraction_job
from backend.do_phase1_service.jobs import mark_failed, mark_running, mark_succeeded
from backend.do_phase1_service.state_store import SQLiteJobStore
from backend.do_phase1_service.storage import GCSStorage, StorageBackend


DEFAULT_STATE_ROOT = Path(os.getenv("DO_PHASE1_STATE_ROOT", "/var/lib/clypt/do_phase1_service"))
DEFAULT_DB_PATH = Path(os.getenv("DO_PHASE1_DB_PATH", str(DEFAULT_STATE_ROOT / "jobs.db")))
DEFAULT_OUTPUT_ROOT = Path(os.getenv("DO_PHASE1_OUTPUT_ROOT", str(DEFAULT_STATE_ROOT / "workdir")))


def run_worker_once(
    store: SQLiteJobStore,
    *,
    output_root: str | Path,
    storage: StorageBackend | None = None,
) -> bool:
    job = store.pop_next_job()
    if job is None:
        return False

    storage = storage or GCSStorage()
    mark_running(store, job.job_id)
    try:
        manifest = run_extraction_job(
            source_url=job.source_url,
            job_id=job.job_id,
            output_dir=Path(output_root),
            storage=storage,
        )
    except Exception as exc:
        mark_failed(
            store,
            job.job_id,
            error_type=type(exc).__name__,
            error_message=str(exc),
            failed_step="extraction",
        )
        return True

    mark_succeeded(
        store,
        job.job_id,
        manifest=_manifest_payload(manifest),
        manifest_uri=manifest.manifest_uri,
    )
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
        processed = run_worker_once(store, output_root=output_root, storage=storage)
        if not processed:
            time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    run_worker_loop()


def _manifest_payload(manifest) -> dict:
    payload = manifest.model_dump(mode="json")
    payload.pop("manifest_uri", None)
    return payload
