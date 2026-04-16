from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from backend.runtime.phase24_error_policy import (
    Phase24FailFastError,
    Phase24FailureClass,
    classify_phase24_exception,
)
from backend.runtime.phase24_worker_app import Phase24TaskPayload, Phase24WorkerService

from .phase24_local_queue import Phase24LocalQueue

logger = logging.getLogger(__name__)


class Phase24LocalWorkerLoop:
    def __init__(
        self,
        *,
        queue: Phase24LocalQueue,
        service: Phase24WorkerService,
        worker_id: str,
        poll_interval_s: float = 0.5,
        lease_timeout_s: int = 1800,
        max_requests_per_worker: int = 0,
        max_inflight: int = 1,
        reclaim_expired_leases: bool = False,
        fail_fast_on_stale_running: bool = True,
        fail_fast_preemption_threshold: int = 0,
        admission_metrics_path: str | None = None,
        block_on_phase1_active: bool = False,
        max_vllm_queue_depth: int = 0,
        max_vllm_decode_backlog: int = 0,
    ) -> None:
        self._queue = queue
        self._service = service
        self._worker_id = worker_id
        self._poll_interval_s = max(0.05, float(poll_interval_s))
        self._lease_timeout_s = max(1, int(lease_timeout_s))
        self._max_requests_per_worker = max(0, int(max_requests_per_worker))
        self._max_inflight = max(1, int(max_inflight))
        self._reclaim_expired_leases = bool(reclaim_expired_leases)
        self._fail_fast_on_stale_running = bool(fail_fast_on_stale_running)
        self._fail_fast_preemption_threshold = max(0, int(fail_fast_preemption_threshold))
        self._admission_metrics_path = Path(admission_metrics_path) if admission_metrics_path else None
        self._block_on_phase1_active = bool(block_on_phase1_active)
        self._max_vllm_queue_depth = max(0, int(max_vllm_queue_depth))
        self._max_vllm_decode_backlog = max(0, int(max_vllm_decode_backlog))
        self._processed_count = 0

    def _read_admission_metrics(self) -> dict[str, Any] | None:
        if self._admission_metrics_path is None:
            return None
        if not self._admission_metrics_path.exists():
            return None
        try:
            return json.loads(self._admission_metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "phase24 local admission metrics unreadable path=%s: %s",
                self._admission_metrics_path,
                exc,
            )
            return None

    def _assert_admission_health(self, metrics: dict[str, Any] | None) -> None:
        if self._fail_fast_preemption_threshold <= 0 or not metrics:
            return
        preemption_count = int(metrics.get("preemption_count") or 0)
        if preemption_count >= self._fail_fast_preemption_threshold:
            raise Phase24FailFastError(
                "preemption threshold breached: "
                f"count={preemption_count} threshold={self._fail_fast_preemption_threshold}"
            )

    def _admission_allows_dequeue(self, metrics: dict[str, Any] | None) -> bool:
        if not metrics:
            return True
        if self._block_on_phase1_active and bool(metrics.get("phase1_gpu_active")):
            return False
        if self._max_vllm_queue_depth > 0:
            queue_depth = int(metrics.get("vllm_queue_depth") or 0)
            if queue_depth > self._max_vllm_queue_depth:
                return False
        if self._max_vllm_decode_backlog > 0:
            decode_backlog = int(metrics.get("vllm_decode_backlog") or 0)
            if decode_backlog > self._max_vllm_decode_backlog:
                return False
        return True

    def run_once(self) -> bool:
        """Process at most one job. Returns True if a job was claimed (even if it failed)."""
        metrics = self._read_admission_metrics()
        self._assert_admission_health(metrics)
        if not self._admission_allows_dequeue(metrics):
            return False
        if self._fail_fast_on_stale_running:
            stale_count = self._queue.count_expired_running(self._lease_timeout_s)
            if stale_count > 0:
                raise Phase24FailFastError(
                    "stale running lease detected: "
                    f"count={stale_count} lease_timeout_s={self._lease_timeout_s} "
                    "(auto-reclaim disabled)"
                )
        row = self._queue.claim_next(
            self._worker_id,
            self._lease_timeout_s,
            max_inflight=self._max_inflight,
            reclaim_expired_leases=self._reclaim_expired_leases,
        )
        if row is None:
            return False
        job_id = str(row["job_id"])
        attempt = int(row["attempt_count"])
        payload_dict: dict[str, Any] = row["payload"]
        try:
            payload = Phase24TaskPayload.model_validate(payload_dict)
        except Exception as exc:
            err_text = f"{exc.__class__.__name__}: {exc}"
            self._queue.mark_failed(job_id, error=err_text, retry=False)
            logger.error(
                "phase24 local worker payload validation failed job_id=%s attempt=%s: %s",
                job_id,
                attempt,
                err_text,
            )
            self._processed_count += 1
            return True
        try:
            result = self._service.handle_task(payload, job_id=job_id, attempt=attempt)
        except Exception as exc:
            err_text = f"{exc.__class__.__name__}: {exc}"
            failure_class = classify_phase24_exception(exc)
            retry = (
                attempt < self._service.max_attempts
                and failure_class == Phase24FailureClass.TRANSIENT
            )
            logger.warning(
                "phase24 local worker job failed job_id=%s attempt=%s class=%s retry=%s: %s",
                job_id,
                attempt,
                failure_class.value,
                retry,
                err_text,
            )
            self._queue.mark_failed(
                job_id,
                error=err_text,
                retry=retry,
                retry_delay_s=0.0,
            )
            self._processed_count += 1
            if failure_class == Phase24FailureClass.FAIL_FAST:
                raise Phase24FailFastError(
                    f"phase24 local worker fail-fast crash job_id={job_id} attempt={attempt}: {err_text}"
                ) from exc
            return True
        terminal = isinstance(result, dict) and result.get("status") == "max_attempts_exceeded"
        if terminal:
            self._queue.mark_failed(job_id, error="max_attempts_exceeded", retry=False)
        else:
            self._queue.mark_succeeded(job_id)
        self._processed_count += 1
        return True

    def run_forever(self) -> None:
        while True:
            if 0 < self._max_requests_per_worker <= self._processed_count:
                logger.info(
                    "phase24 local worker exiting after max_requests_per_worker=%s",
                    self._max_requests_per_worker,
                )
                return
            try:
                worked = self.run_once()
            except Phase24FailFastError as exc:
                logger.error("phase24 local worker fail-fast stop: %s", exc)
                return
            if not worked:
                time.sleep(self._poll_interval_s)


__all__ = ["Phase24LocalWorkerLoop"]
