from __future__ import annotations

from typing import Any

from .phase24_local_queue import Phase24LocalQueue


class Phase24LocalDispatcherClient:
    """Enqueue Phase 2–4 work into the local SQLite queue."""

    def __init__(self, *, queue: Phase24LocalQueue) -> None:
        self._queue = queue
        self.backend = "local_sqlite"

    def enqueue_phase24(
        self,
        *,
        run_id: str,
        payload: dict[str, Any],
        worker_url: str | None = None,
    ) -> str:
        _ = worker_url
        job_id = self._queue.enqueue(run_id, payload)
        return f"local-sqlite:{job_id}"


__all__ = ["Phase24LocalDispatcherClient"]
