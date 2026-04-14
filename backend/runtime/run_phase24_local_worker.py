from __future__ import annotations

import logging
import os
import sys

from backend.providers import load_provider_settings
from backend.runtime.phase24_local_queue import Phase24LocalQueue
from backend.runtime.phase24_local_worker import Phase24LocalWorkerLoop
from backend.runtime.phase24_worker_app import build_default_phase24_worker_service


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    settings = load_provider_settings()
    if settings.phase24_local_queue.queue_backend != "local_sqlite":
        raise SystemExit(
            f"CLYPT_PHASE24_QUEUE_BACKEND must be local_sqlite for this worker "
            f"(got {settings.phase24_local_queue.queue_backend!r})."
        )
    queue = Phase24LocalQueue(path=settings.phase24_local_queue.path)
    service = build_default_phase24_worker_service()
    worker_id = os.environ.get("CLYPT_PHASE24_LOCAL_WORKER_ID", "phase24-local-worker")
    loop = Phase24LocalWorkerLoop(
        queue=queue,
        service=service,
        worker_id=worker_id,
        poll_interval_s=settings.phase24_local_queue.poll_interval_ms / 1000.0,
        lease_timeout_s=settings.phase24_local_queue.lease_timeout_s,
        max_requests_per_worker=settings.phase24_local_queue.max_requests_per_worker,
        max_inflight=settings.phase24_local_queue.max_inflight,
        fail_fast_preemption_threshold=settings.phase24_worker.fail_fast_preemption_threshold,
        admission_metrics_path=settings.phase24_worker.admission_metrics_path,
        block_on_phase1_active=settings.phase24_worker.block_on_phase1_active,
        max_vllm_queue_depth=settings.phase24_worker.max_vllm_queue_depth,
        max_vllm_decode_backlog=settings.phase24_worker.max_vllm_decode_backlog,
    )
    loop.run_forever()


if __name__ == "__main__":
    main()
