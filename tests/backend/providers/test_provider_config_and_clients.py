from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


def test_load_provider_settings_uses_env_and_gcloud_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    monkeypatch.delenv("VERTEX_GEMINI_LOCATION", raising=False)
    monkeypatch.delenv("VERTEX_EMBEDDING_LOCATION", raising=False)
    monkeypatch.setenv("CLYPT_GCS_BUCKET", "bucket-a")
    monkeypatch.setenv("VIBEVOICE_VLLM_BASE_URL", "http://127.0.0.1:8000")

    def fake_run(cmd, check, capture_output, text):
        assert cmd[:4] == ["gcloud", "config", "get-value", "project"]
        return subprocess.CompletedProcess(cmd, 0, stdout="clypt-v3\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    settings = load_provider_settings()

    assert settings.vllm_vibevoice.base_url == "http://127.0.0.1:8000"
    assert settings.vllm_vibevoice.model == "vibevoice"
    assert settings.vibevoice.max_new_tokens == 32768
    assert settings.vibevoice.do_sample is False
    assert settings.vertex.project == "clypt-v3"
    assert settings.vertex.generation_location == "global"
    assert settings.vertex.embedding_location == "us-central1"
    assert settings.storage.gcs_bucket == "bucket-a"
    assert settings.vertex.generation_model
    assert settings.vertex.embedding_model
    assert settings.phase1_runtime.run_yamnet_on_gpu is True


def test_load_provider_settings_vllm_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "clypt-v3")
    monkeypatch.setenv("GCS_BUCKET", "bucket-a")
    monkeypatch.setenv("VIBEVOICE_VLLM_BASE_URL", "http://127.0.0.1:8000")
    monkeypatch.delenv("VIBEVOICE_BACKEND", raising=False)
    monkeypatch.delenv("VIBEVOICE_VLLM_MODEL", raising=False)

    settings = load_provider_settings()

    assert settings.vllm_vibevoice.base_url == "http://127.0.0.1:8000"
    assert settings.vllm_vibevoice.model == "vibevoice"
    assert settings.vibevoice.max_new_tokens == 32768
    assert settings.vibevoice.do_sample is False


def test_load_provider_settings_raises_without_gcs_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    monkeypatch.delenv("CLYPT_GCS_BUCKET", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "clypt-v3")
    monkeypatch.setenv("VIBEVOICE_VLLM_BASE_URL", "http://127.0.0.1:8000")

    with pytest.raises(ValueError, match="GCS_BUCKET"):
        load_provider_settings()


def test_load_provider_settings_reads_untracked_env_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    monkeypatch.delenv("CLYPT_GCS_BUCKET", raising=False)
    monkeypatch.delenv("VIBEVOICE_VLLM_MODEL", raising=False)
    (tmp_path / ".env.local").write_text(
        "\n".join(
            [
                "VIBEVOICE_BACKEND=vllm",
                "VIBEVOICE_VLLM_BASE_URL=http://127.0.0.1:8000",
                "VIBEVOICE_VLLM_MODEL=vibevoice",
                "VIBEVOICE_DO_SAMPLE=1",
                "GOOGLE_CLOUD_PROJECT=clypt-v3",
                "VERTEX_GEMINI_LOCATION=global",
                "VERTEX_EMBEDDING_LOCATION=us-central1",
                "GCS_BUCKET=clypt-storage-v3",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_provider_settings()

    assert settings.vllm_vibevoice.base_url == "http://127.0.0.1:8000"
    assert settings.vllm_vibevoice.model == "vibevoice"
    assert settings.vibevoice.do_sample is True
    assert settings.vertex.project == "clypt-v3"
    assert settings.vertex.generation_location == "global"
    assert settings.vertex.embedding_location == "us-central1"
    assert settings.storage.gcs_bucket == "clypt-storage-v3"
    assert settings.phase1_runtime.run_yamnet_on_gpu is True


def test_load_provider_settings_vibevoice_custom_hotwords(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "clypt-v3")
    monkeypatch.setenv("GCS_BUCKET", "bucket-a")
    monkeypatch.setenv("VIBEVOICE_VLLM_BASE_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("VIBEVOICE_HOTWORDS_CONTEXT", "hello, world")

    settings = load_provider_settings()

    assert settings.vibevoice.hotwords_context == "hello, world"


def test_load_provider_settings_rejects_non_vllm_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "clypt-v3")
    monkeypatch.setenv("GCS_BUCKET", "bucket-a")
    monkeypatch.setenv("VIBEVOICE_VLLM_BASE_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("VIBEVOICE_BACKEND", "native")

    with pytest.raises(ValueError, match="only 'vllm' is supported"):
        load_provider_settings()


def test_load_provider_settings_exposes_phase24_queue_and_spanner_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "clypt-v3")
    monkeypatch.setenv("GCS_BUCKET", "bucket-a")
    monkeypatch.setenv("VIBEVOICE_VLLM_BASE_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("CLYPT_PHASE24_TASKS_LOCATION", "us-central1")
    monkeypatch.setenv("CLYPT_PHASE24_TASKS_QUEUE", "phase24-queue")
    monkeypatch.setenv("CLYPT_PHASE24_WORKER_URL", "https://phase24-worker.example.com")
    monkeypatch.setenv("CLYPT_PHASE24_WORKER_SERVICE_ACCOUNT_EMAIL", "worker-sa@example.com")
    monkeypatch.setenv("CLYPT_SPANNER_INSTANCE", "phase14-instance")
    monkeypatch.setenv("CLYPT_SPANNER_DATABASE", "phase14-db")
    monkeypatch.setenv("CLYPT_SPANNER_DDL_OPERATION_TIMEOUT_S", "42")
    monkeypatch.setenv("CLYPT_PHASE24_WORKER_SERVICE_NAME", "phase24-worker")
    monkeypatch.setenv("CLYPT_PHASE24_ENVIRONMENT", "staging")
    monkeypatch.setenv("CLYPT_PHASE24_QUERY_VERSION", "graph-v2")
    monkeypatch.setenv("CLYPT_PHASE24_CONCURRENCY", "4")
    monkeypatch.setenv("CLYPT_DEBUG_SNAPSHOTS", "1")
    monkeypatch.setenv("CLYPT_PHASE24_MAX_ATTEMPTS", "5")

    settings = load_provider_settings()

    assert settings.cloud_tasks.project == "clypt-v3"
    assert settings.cloud_tasks.location == "us-central1"
    assert settings.cloud_tasks.queue == "phase24-queue"
    assert settings.cloud_tasks.worker_url == "https://phase24-worker.example.com"
    assert settings.cloud_tasks.service_account_email == "worker-sa@example.com"
    assert settings.spanner.project == "clypt-v3"
    assert settings.spanner.instance == "phase14-instance"
    assert settings.spanner.database == "phase14-db"
    assert settings.spanner.ddl_operation_timeout_s == 42.0
    assert settings.phase24_worker.service_name == "phase24-worker"
    assert settings.phase24_worker.environment == "staging"
    assert settings.phase24_worker.query_version == "graph-v2"
    assert settings.phase24_worker.concurrency == 4
    assert settings.phase24_worker.debug_snapshots is True
    assert settings.phase24_worker.max_attempts == 5


def test_load_provider_settings_exposes_vertex_retry_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "clypt-v3")
    monkeypatch.setenv("GCS_BUCKET", "bucket-a")
    monkeypatch.setenv("VIBEVOICE_VLLM_BASE_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("VERTEX_API_MAX_RETRIES", "9")
    monkeypatch.setenv("VERTEX_API_INITIAL_BACKOFF_S", "0.5")
    monkeypatch.setenv("VERTEX_API_MAX_BACKOFF_S", "12.0")
    monkeypatch.setenv("VERTEX_API_BACKOFF_MULTIPLIER", "1.7")
    monkeypatch.setenv("VERTEX_API_JITTER_RATIO", "0.05")
    monkeypatch.setenv("VERTEX_THINKING_BUDGET", "96")

    settings = load_provider_settings()

    assert settings.vertex.api_max_retries == 9
    assert settings.vertex.api_initial_backoff_s == 0.5
    assert settings.vertex.api_max_backoff_s == 12.0
    assert settings.vertex.api_backoff_multiplier == 1.7
    assert settings.vertex.api_jitter_ratio == 0.05
    assert settings.vertex.thinking_budget == 96


def test_phase24_task_queue_client_uses_run_id_for_idempotent_task_name() -> None:
    from backend.providers.config import CloudTasksSettings
    from backend.providers.task_queue import AlreadyExists, Phase24TaskQueueClient, _task_id_for_run_id

    captured: dict[str, object] = {}

    class _FakeTasksClient:
        def create_task(self, request):
            captured["request"] = request

            class _Response:
                name = request["task"]["name"]

            return _Response()

    queue_client = Phase24TaskQueueClient(
        settings=CloudTasksSettings(
            project="clypt-v3",
            location="us-central1",
            queue="clypt-phase24",
            worker_url="https://phase24-worker.example.com",
            service_account_email="worker-sa@example.com",
        ),
        tasks_client=_FakeTasksClient(),
    )

    task_name = queue_client.enqueue_phase24(
        run_id="run/001",
        payload={"run_id": "run/001", "source_url": "https://example.com/video"},
    )

    request = captured["request"]
    assert task_name == queue_client.task_name_for_run_id("run/001")
    assert task_name.endswith(_task_id_for_run_id("run/001"))
    assert request == {
        "parent": "projects/clypt-v3/locations/us-central1/queues/clypt-phase24",
        "task": {
            "name": task_name,
            "http_request": {
                "http_method": "POST",
                "url": "https://phase24-worker.example.com",
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(
                    {"run_id": "run/001", "source_url": "https://example.com/video"},
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
                "oidc_token": {
                    "service_account_email": "worker-sa@example.com",
                    "audience": "https://phase24-worker.example.com",
                },
            },
        },
    }


def test_phase24_task_queue_client_treats_already_exists_as_idempotent_success() -> None:
    from backend.providers.config import CloudTasksSettings
    from backend.providers.task_queue import AlreadyExists, Phase24TaskQueueClient, _task_id_for_run_id

    requested: dict[str, object] = {}

    class _FakeTasksClient:
        def create_task(self, request):
            requested["request"] = request
            raise AlreadyExists("task already exists")

    queue_client = Phase24TaskQueueClient(
        settings=CloudTasksSettings(
            project="clypt-v3",
            location="us-central1",
            queue="clypt-phase24",
            worker_url="https://phase24-worker.example.com",
        ),
        tasks_client=_FakeTasksClient(),
    )

    task_name = queue_client.enqueue_phase24(run_id="run-001", payload={"run_id": "run-001"})

    assert task_name == queue_client.task_name_for_run_id("run-001")
    assert task_name.endswith(_task_id_for_run_id("run-001"))
    assert requested["request"]["task"]["name"] == task_name


def test_phase24_task_queue_client_task_id_uses_hash_suffix_for_collisions_and_length() -> None:
    from backend.providers.task_queue import _task_id_for_run_id

    first = _task_id_for_run_id("run/abc")
    second = _task_id_for_run_id("run?abc")
    long_id = _task_id_for_run_id("run-" + ("a" * 1000))

    assert first != second
    assert first.startswith("phase24-run-abc-")
    assert second.startswith("phase24-run-abc-")
    assert len(long_id) <= 500
    assert long_id.startswith("phase24-run-")
    assert len(long_id.rsplit("-", 1)[-1]) == 12
