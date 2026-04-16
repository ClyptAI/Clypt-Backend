from __future__ import annotations

import time
import urllib.error
from pathlib import Path

import pytest

from backend.runtime.phase24_error_policy import Phase24FailFastError
from backend.runtime.phase24_local_queue import Phase24LocalQueue
from backend.runtime.phase24_local_worker import Phase24LocalWorkerLoop
from backend.runtime.phase24_worker_app import Phase24TaskPayload


def _payload_dict(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "source_url": "https://example.com/video",
        "query_version": "v1",
        "phase1_outputs": {
            "phase1_audio": {
                "source_audio": "https://example.com/video",
                "video_gcs_uri": "gs://bucket/video.mp4",
                "local_video_path": "/tmp/source_video.mp4",
            },
            "diarization_payload": {"words": [], "turns": []},
            "phase1_visual": {"video_metadata": {"fps": 30.0}, "shot_changes": [], "tracks": []},
            "emotion2vec_payload": {"segments": []},
            "yamnet_payload": {"events": []},
        },
        "phase4_extra_prompt_texts": [],
    }


class _RecordingService:
    max_attempts = 2

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def handle_task(self, payload: Phase24TaskPayload, *, job_id: str, attempt: int) -> dict:
        self.calls.append((job_id, attempt))
        if attempt == 1:
            raise TimeoutError("first attempt fails transiently")
        return {"run_id": payload.run_id, "status": "succeeded", "summary": {}}


def test_phase24_local_worker_loop_retries_then_succeeds(tmp_path: Path) -> None:
    queue = Phase24LocalQueue(tmp_path / "w.sqlite")
    job_id = queue.enqueue("run-w", _payload_dict("run-w"))
    service = _RecordingService()
    loop = Phase24LocalWorkerLoop(
        queue=queue,
        service=service,
        worker_id="unit-test",
        poll_interval_s=0.01,
        lease_timeout_s=60,
    )
    assert loop.run_once() is True
    assert queue.get_job(job_id)["status"] == "queued"
    assert service.calls == [(job_id, 1)]

    assert loop.run_once() is True
    assert queue.get_job(job_id)["status"] == "succeeded"
    assert service.calls == [(job_id, 1), (job_id, 2)]


class _TerminalService:
    max_attempts = 1

    def handle_task(self, payload: Phase24TaskPayload, *, job_id: str, attempt: int) -> dict:
        return {"run_id": payload.run_id, "status": "max_attempts_exceeded", "summary": {}}


def test_phase24_local_worker_marks_queue_failed_on_max_attempts_response(tmp_path: Path) -> None:
    queue = Phase24LocalQueue(tmp_path / "t.sqlite")
    job_id = queue.enqueue("run-t", _payload_dict("run-t"))
    loop = Phase24LocalWorkerLoop(
        queue=queue,
        service=_TerminalService(),
        worker_id="unit-test",
        poll_interval_s=0.01,
        lease_timeout_s=60,
    )
    assert loop.run_once() is True
    assert queue.get_job(job_id)["status"] == "failed"


def test_phase24_local_worker_marks_failed_on_invalid_payload(tmp_path: Path) -> None:
    queue = Phase24LocalQueue(tmp_path / "bad.sqlite")
    job_id = queue.enqueue("run-bad", {"run_id": "run-bad"})
    loop = Phase24LocalWorkerLoop(
        queue=queue,
        service=_TerminalService(),
        worker_id="unit-test",
        poll_interval_s=0.01,
        lease_timeout_s=60,
    )
    assert loop.run_once() is True
    assert queue.get_job(job_id)["status"] == "failed"


class _NoopService:
    max_attempts = 5

    def __init__(self) -> None:
        self.calls = 0

    def handle_task(self, payload: Phase24TaskPayload, *, job_id: str, attempt: int) -> dict:
        self.calls += 1
        return {"run_id": payload.run_id, "status": "succeeded", "summary": {}}


def test_phase24_local_worker_honors_max_requests_per_worker(tmp_path: Path) -> None:
    queue = Phase24LocalQueue(tmp_path / "maxreq.sqlite")
    queue.enqueue("run-1", _payload_dict("run-1"))
    queue.enqueue("run-2", _payload_dict("run-2"))
    service = _NoopService()
    loop = Phase24LocalWorkerLoop(
        queue=queue,
        service=service,
        worker_id="unit-test",
        poll_interval_s=0.01,
        lease_timeout_s=60,
        max_requests_per_worker=1,
    )
    loop.run_forever()
    assert service.calls == 1


class _NonTransientService:
    max_attempts = 3

    def handle_task(self, payload: Phase24TaskPayload, *, job_id: str, attempt: int) -> dict:
        raise ValueError("schema mismatch deterministic")


def test_phase24_local_worker_does_not_retry_non_transient_errors(tmp_path: Path) -> None:
    queue = Phase24LocalQueue(tmp_path / "noretry.sqlite")
    job_id = queue.enqueue("run-noretry", _payload_dict("run-noretry"))
    loop = Phase24LocalWorkerLoop(
        queue=queue,
        service=_NonTransientService(),
        worker_id="unit-test",
        poll_interval_s=0.01,
        lease_timeout_s=60,
    )
    assert loop.run_once() is True
    assert queue.get_job(job_id)["status"] == "failed"


def test_phase24_local_worker_failfast_stops_on_preemption_threshold(tmp_path: Path) -> None:
    queue = Phase24LocalQueue(tmp_path / "preempt.sqlite")
    queue.enqueue("run-preempt", _payload_dict("run-preempt"))
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"preemption_count": 5}', encoding="utf-8")
    service = _NoopService()
    loop = Phase24LocalWorkerLoop(
        queue=queue,
        service=service,
        worker_id="unit-test",
        poll_interval_s=0.01,
        lease_timeout_s=60,
        fail_fast_preemption_threshold=3,
        admission_metrics_path=str(metrics),
    )
    loop.run_forever()
    assert service.calls == 0


def test_phase24_local_worker_blocks_dequeue_when_phase1_active(tmp_path: Path) -> None:
    queue = Phase24LocalQueue(tmp_path / "phase1active.sqlite")
    job_id = queue.enqueue("run-phase1-active", _payload_dict("run-phase1-active"))
    metrics = tmp_path / "metrics_phase1.json"
    metrics.write_text('{"phase1_gpu_active": true}', encoding="utf-8")
    service = _NoopService()
    loop = Phase24LocalWorkerLoop(
        queue=queue,
        service=service,
        worker_id="unit-test",
        poll_interval_s=0.01,
        lease_timeout_s=60,
        max_requests_per_worker=1,
        block_on_phase1_active=True,
        admission_metrics_path=str(metrics),
    )
    assert loop.run_once() is False
    assert queue.get_job(job_id)["status"] == "queued"
    assert service.calls == 0


class _ConnectionRefusedService:
    max_attempts = 3

    def handle_task(self, payload: Phase24TaskPayload, *, job_id: str, attempt: int) -> dict:
        raise urllib.error.URLError("[Errno 111] Connection refused")


def test_phase24_local_worker_fail_fast_on_connection_refused(tmp_path: Path) -> None:
    queue = Phase24LocalQueue(tmp_path / "connrefused.sqlite")
    job_id = queue.enqueue("run-connrefused", _payload_dict("run-connrefused"))
    loop = Phase24LocalWorkerLoop(
        queue=queue,
        service=_ConnectionRefusedService(),
        worker_id="unit-test",
        poll_interval_s=0.01,
        lease_timeout_s=60,
    )
    with pytest.raises(Phase24FailFastError, match="fail-fast crash"):
        loop.run_once()
    row = queue.get_job(job_id)
    assert row is not None
    assert row["status"] == "failed"


def test_phase24_local_worker_fail_fast_on_stale_running_lease(tmp_path: Path) -> None:
    queue = Phase24LocalQueue(tmp_path / "stale.sqlite")
    queue.enqueue("run-stale", _payload_dict("run-stale"))
    claimed = queue.claim_next("worker-a", lease_timeout_s=1, reclaim_expired_leases=False)
    assert claimed is not None
    time.sleep(1.25)

    loop = Phase24LocalWorkerLoop(
        queue=queue,
        service=_NoopService(),
        worker_id="unit-test",
        poll_interval_s=0.01,
        lease_timeout_s=1,
    )
    with pytest.raises(Phase24FailFastError, match="stale running lease"):
        loop.run_once()
