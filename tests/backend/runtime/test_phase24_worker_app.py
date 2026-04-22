from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest


UTC = timezone.utc


class _FakeRepository:
    def __init__(self) -> None:
        self.run_record = None
        self.job_record = None

    def upsert_run(self, record):
        self.run_record = record
        return record

    def get_run(self, run_id: str):
        return self.run_record if self.run_record and self.run_record.run_id == run_id else None

    def upsert_phase24_job(self, record):
        self.job_record = record
        return record

    def get_phase24_job(self, run_id: str):
        return self.job_record if self.job_record and self.job_record.run_id == run_id else None

    def acquire_phase24_job_lease(
        self,
        *,
        run_id: str,
        job_id: str,
        worker_name: str,
        attempt: int,
        query_version: str,
        running_timeout_s: int = 1800,
    ):
        current = self.get_phase24_job(run_id)
        if current is not None and current.status == "succeeded":
            return {"acquired": False, "status": "succeeded"}
        if current is not None and current.status == "running":
            return {"acquired": False, "status": "running"}
        from backend.repository.models import Phase24JobRecord

        now = datetime(2026, 4, 8, 12, 0, tzinfo=UTC)
        self.job_record = Phase24JobRecord(
            run_id=run_id,
            status="running",
            attempt_count=attempt,
            last_error=None,
            worker_name=worker_name,
            task_name=job_id,
            locked_at=now,
            updated_at=now,
            completed_at=None,
            metadata={"query_version": query_version},
        )
        return {"acquired": True, "status": "running"}


class _FakeRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return type(
            "Summary",
            (),
            {
                "model_dump": lambda self, mode="json": {
                    "run_id": kwargs["run_id"],
                    "artifact_paths": {},
                    "metadata": {"candidate_count": 1},
                }
            },
        )()


def _build_payload() -> dict[str, object]:
    return {
        "run_id": "run_001",
        "source_url": "https://example.com/video",
        "query_version": "graph-v2",
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
        "phase4_extra_prompt_texts": ["find the strongest moment"],
    }


def test_phase24_worker_service_processes_task_and_updates_repository():
    from backend.runtime.phase24_worker_app import Phase24TaskPayload, Phase24WorkerService

    repository = _FakeRepository()
    runner = _FakeRunner()
    service = Phase24WorkerService(
        repository=repository,
        runner=runner,
        service_name="clypt-phase26-worker",
        environment="staging",
        default_query_version="graph-v1",
        max_attempts=3,
    )

    response = service.handle_task(
        payload=Phase24TaskPayload(**_build_payload()),
        job_id="task-001",
        attempt=3,
    )

    assert response["status"] == "succeeded"
    assert response["summary"]["metadata"]["candidate_count"] == 1
    assert len(runner.calls) == 1
    assert runner.calls[0]["run_id"] == "run_001"
    assert runner.calls[0]["job_id"] == "task-001"
    assert runner.calls[0]["attempt"] == 3
    assert repository.run_record is not None
    assert repository.run_record.status == "PHASE24_DONE"
    assert repository.job_record is not None
    assert repository.job_record.status == "succeeded"
    assert repository.job_record.attempt_count == 3
    assert repository.job_record.task_name == "task-001"


def test_phase24_worker_service_merges_payload_source_context_into_phase1_outputs():
    from backend.runtime.phase24_worker_app import Phase24TaskPayload, Phase24WorkerService

    repository = _FakeRepository()
    runner = _FakeRunner()
    service = Phase24WorkerService(
        repository=repository,
        runner=runner,
        service_name="clypt-phase26-worker",
        environment="staging",
        default_query_version="graph-v1",
        max_attempts=3,
    )

    payload_dict = _build_payload()
    payload_dict["source_context"] = {
        "source_url": "https://example.com/video",
        "youtube_video_id": "abc123xyz00",
        "source_title": "Persisted Title",
        "source_description": "Persisted description",
        "channel_id": "channel_123",
        "channel_title": "Persisted Channel",
        "published_at": "2026-04-19T00:00:00+00:00",
        "default_audio_language": "en",
        "category_id": "22",
        "tags": ["persisted"],
        "thumbnails": {"default": {"url": "https://example.com/thumb.jpg"}},
    }

    service.handle_task(
        payload=Phase24TaskPayload(**payload_dict),
        job_id="task-001",
        attempt=1,
    )

    assert runner.calls[0]["phase1_outputs"].source_context["source_title"] == "Persisted Title"


def test_phase24_worker_service_short_circuits_completed_jobs():
    from backend.repository.models import Phase24JobRecord
    from backend.runtime.phase24_worker_app import Phase24TaskPayload, Phase24WorkerService

    repository = _FakeRepository()
    repository.job_record = Phase24JobRecord(
        run_id="run_001",
        status="succeeded",
        attempt_count=1,
        last_error=None,
        worker_name="clypt-phase26-worker",
        task_name="task-001",
        locked_at=None,
        updated_at=datetime(2026, 4, 8, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 4, 8, 12, 1, tzinfo=UTC),
        metadata={"query_version": "graph-v2"},
    )
    runner = _FakeRunner()
    service = Phase24WorkerService(
        repository=repository,
        runner=runner,
        service_name="clypt-phase26-worker",
        environment="staging",
        default_query_version="graph-v1",
        max_attempts=3,
    )

    response = service.handle_task(
        payload=Phase24TaskPayload(**_build_payload()),
        job_id="task-001",
        attempt=1,
    )

    assert response["status"] == "already_succeeded"
    assert runner.calls == []


def test_phase24_worker_service_short_circuits_running_jobs():
    from backend.repository.models import Phase24JobRecord
    from backend.runtime.phase24_worker_app import Phase24TaskPayload, Phase24WorkerService

    repository = _FakeRepository()
    repository.job_record = Phase24JobRecord(
        run_id="run_001",
        status="running",
        attempt_count=1,
        last_error=None,
        worker_name="clypt-phase26-worker",
        task_name="task-001",
        locked_at=None,
        updated_at=datetime(2026, 4, 8, 12, 0, tzinfo=UTC),
        completed_at=None,
        metadata={"query_version": "graph-v2"},
    )
    runner = _FakeRunner()
    service = Phase24WorkerService(
        repository=repository,
        runner=runner,
        service_name="clypt-phase26-worker",
        environment="staging",
        default_query_version="graph-v1",
        max_attempts=3,
    )

    response = service.handle_task(
        payload=Phase24TaskPayload(**_build_payload()),
        job_id="task-001",
        attempt=1,
    )

    assert response["status"] == "already_running"
    assert runner.calls == []


def test_phase24_worker_app_marks_non_terminal_failures_as_queued():
    from backend.runtime.phase24_worker_app import Phase24TaskPayload, Phase24WorkerService

    class _FailingRunner(_FakeRunner):
        def run(self, **kwargs):
            raise TimeoutError("boom")

    repository = _FakeRepository()
    service = Phase24WorkerService(
        repository=repository,
        runner=_FailingRunner(),
        service_name="clypt-phase26-worker",
        environment="staging",
        default_query_version="graph-v1",
        max_attempts=3,
    )

    with pytest.raises(TimeoutError, match="boom"):
        service.handle_task(
            payload=Phase24TaskPayload(**_build_payload()),
            job_id="task-001",
            attempt=2,
        )

    assert repository.job_record is not None
    assert repository.job_record.status == "queued"
    assert repository.run_record is not None
    assert repository.run_record.status == "PHASE24_QUEUED"


def test_phase24_worker_app_marks_non_transient_failures_terminal():
    from backend.runtime.phase24_worker_app import Phase24TaskPayload, Phase24WorkerService

    class _FailingRunner(_FakeRunner):
        def run(self, **kwargs):
            raise ValueError("schema mismatch")

    repository = _FakeRepository()
    service = Phase24WorkerService(
        repository=repository,
        runner=_FailingRunner(),
        service_name="clypt-phase26-worker",
        environment="staging",
        default_query_version="graph-v1",
        max_attempts=3,
    )

    with pytest.raises(ValueError, match="schema mismatch"):
        service.handle_task(
            payload=Phase24TaskPayload(**_build_payload()),
            job_id="task-001",
            attempt=1,
        )

    assert repository.job_record is not None
    assert repository.job_record.status == "failed"
    assert repository.run_record is not None
    assert repository.run_record.status == "FAILED"


def test_phase24_worker_app_stops_when_attempt_exceeds_max_attempts():
    from backend.runtime.phase24_worker_app import Phase24TaskPayload, Phase24WorkerService

    repository = _FakeRepository()
    runner = _FakeRunner()
    service = Phase24WorkerService(
        repository=repository,
        runner=runner,
        service_name="clypt-phase26-worker",
        environment="staging",
        default_query_version="graph-v1",
        max_attempts=2,
    )

    result = service.handle_task(
        payload=Phase24TaskPayload(**_build_payload()),
        job_id="task-001",
        attempt=3,
    )

    assert result["status"] == "max_attempts_exceeded"
    assert runner.calls == []
    assert repository.job_record is not None
    assert repository.job_record.status == "failed"
    assert repository.run_record is not None
    assert repository.run_record.status == "FAILED"


def test_phase24_worker_app_loads_phase1_outputs_from_gcs_pointer(tmp_path):
    from backend.runtime.phase24_worker_app import Phase24TaskPayload, Phase24WorkerService

    class _FakeStorage:
        def __init__(self) -> None:
            self.download_calls: list[str] = []

        def download_file(self, *, gcs_uri: str, local_path):
            self.download_calls.append(gcs_uri)
            payload = _build_payload()["phase1_outputs"]
            local_path.write_text(json.dumps(payload), encoding="utf-8")

    class _RunnerWithStorage(_FakeRunner):
        def __init__(self) -> None:
            super().__init__()
            self.storage_client = _FakeStorage()
            self.node_media_preparer = object()

    repository = _FakeRepository()
    runner = _RunnerWithStorage()
    service = Phase24WorkerService(
        repository=repository,
        runner=runner,
        service_name="clypt-phase26-worker",
        environment="staging",
        default_query_version="graph-v1",
        max_attempts=3,
    )

    result = service.handle_task(
        payload=Phase24TaskPayload(
            run_id="run_001",
            source_url="https://example.com/video",
            source_video_gcs_uri="gs://bucket/video.mp4",
            phase1_outputs_gcs_uri="gs://bucket/phase24_inputs/run_001.json",
            query_version="graph-v2",
        ),
        job_id="task-001",
        attempt=1,
    )

    assert result["status"] == "succeeded"
    assert len(runner.calls) == 1
    assert (
        runner.calls[0]["phase1_outputs"].phase1_audio["video_gcs_uri"] == "gs://bucket/video.mp4"
    )
    assert runner.storage_client.download_calls == ["gs://bucket/phase24_inputs/run_001.json"]


def test_phase24_worker_app_failfast_on_p95_latency_threshold():
    from backend.runtime.phase24_error_policy import Phase24FailFastError
    from backend.runtime.phase24_worker_app import Phase24TaskPayload, Phase24WorkerService

    class _HighLatencyRunner(_FakeRunner):
        def run(self, **kwargs):
            return type(
                "Summary",
                (),
                {
                    "model_dump": lambda self, mode="json": {
                        "run_id": kwargs["run_id"],
                        "artifact_paths": {},
                        "metadata": {"subgraph_review_p95_latency_ms": 9000},
                    }
                },
            )()

    repository = _FakeRepository()
    service = Phase24WorkerService(
        repository=repository,
        runner=_HighLatencyRunner(),
        service_name="clypt-phase26-worker",
        environment="staging",
        default_query_version="graph-v1",
        max_attempts=3,
        fail_fast_p95_latency_ms=5000,
    )

    with pytest.raises(Phase24FailFastError):
        service.handle_task(
            payload=Phase24TaskPayload(**_build_payload()),
            job_id="task-001",
            attempt=1,
        )

    assert repository.job_record is not None
    assert repository.job_record.status == "failed"
    assert repository.run_record is not None
    assert repository.run_record.status == "FAILED"


def test_phase24_worker_app_failfast_on_in_run_preemption_threshold(tmp_path):
    from backend.runtime.phase24_error_policy import Phase24FailFastError
    from backend.runtime.phase24_worker_app import Phase24TaskPayload, Phase24WorkerService

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text('{"preemption_count": 5}', encoding="utf-8")

    class _RunnerWithEvent(_FakeRunner):
        def run(self, **kwargs):
            log_event = kwargs.get("log_event")
            if callable(log_event):
                log_event(
                    run_id=kwargs["run_id"],
                    job_id=kwargs["job_id"],
                    phase="phase3",
                    event="phase_progress",
                    attempt=kwargs["attempt"],
                    query_version="graph-v1",
                    status="running",
                )
            return super().run(**kwargs)

    repository = _FakeRepository()
    service = Phase24WorkerService(
        repository=repository,
        runner=_RunnerWithEvent(),
        service_name="clypt-phase26-worker",
        environment="staging",
        default_query_version="graph-v1",
        max_attempts=3,
        fail_fast_preemption_threshold=3,
        admission_metrics_path=str(metrics_path),
    )

    with pytest.raises(Phase24FailFastError):
        service.handle_task(
            payload=Phase24TaskPayload(**_build_payload()),
            job_id="task-001",
            attempt=1,
        )

    assert repository.job_record is not None
    assert repository.job_record.status == "failed"
    assert repository.run_record is not None
    assert repository.run_record.status == "FAILED"


def test_build_default_phase24_worker_service_uses_local_model_for_flash(monkeypatch):
    from types import SimpleNamespace

    from backend.runtime import phase24_worker_app as module

    captured: dict[str, object] = {}
    fake_runner = object()
    bootstrap_calls: list[str] = []

    class _FakeRunnerFactory:
        @staticmethod
        def from_env(**kwargs):
            captured.update(kwargs)
            return fake_runner

    class _FakeRepository:
        def bootstrap_schema(self) -> None:
            bootstrap_calls.append("bootstrapped")

    settings = SimpleNamespace(
        spanner=SimpleNamespace(),
        vertex=SimpleNamespace(
            generation_backend="local_openai",
            flash_model="gemini-3-flash-preview",
        ),
        local_generation=SimpleNamespace(
            model="Qwen/Qwen3.6-35B-A3B",
        ),
        storage=SimpleNamespace(),
        node_media_prep=SimpleNamespace(
            service_url="http://10.0.0.5:9100",
            auth_token="prep-token",
            timeout_s=3600.0,
            max_concurrency=8,
        ),
        phase24_worker=SimpleNamespace(
            query_version="v1",
            debug_snapshots=False,
            service_name="clypt-phase26-worker",
            environment="test",
            max_attempts=3,
            fail_fast_p95_latency_ms=0.0,
            fail_fast_preemption_threshold=0,
            admission_metrics_path=None,
        ),
    )

    monkeypatch.setattr(module, "load_provider_settings", lambda: settings)
    monkeypatch.setattr(
        module.SpannerPhase14Repository,
        "from_settings",
        lambda settings: _FakeRepository(),
    )
    monkeypatch.setattr(module, "LocalOpenAIQwenClient", lambda settings: "llm")
    monkeypatch.setattr(module, "VertexEmbeddingClient", lambda settings: "embed")
    monkeypatch.setattr(module, "GCSStorageClient", lambda settings: "storage")
    monkeypatch.setattr(module, "V31LivePhase14Runner", _FakeRunnerFactory)

    service = module.build_default_phase24_worker_service()

    assert service.runner is fake_runner
    assert bootstrap_calls == ["bootstrapped"]
    assert captured["flash_model"] == "Qwen/Qwen3.6-35B-A3B"
