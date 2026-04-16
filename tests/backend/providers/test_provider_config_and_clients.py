from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest


def test_load_provider_settings_uses_env_and_gcloud_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    monkeypatch.delenv("GENAI_GENERATION_LOCATION", raising=False)
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
    assert settings.vllm_vibevoice.audio_mode == "url"
    assert settings.vibevoice.max_new_tokens == 32768
    assert settings.vibevoice.do_sample is False
    assert settings.vibevoice.top_p == 1.0
    assert settings.vibevoice.num_beams == 1
    assert settings.vibevoice.repetition_penalty == 1.03
    assert settings.vertex.project == "clypt-v3"
    assert settings.vertex.generation_location == "global"
    assert settings.vertex.embedding_location == "us-central1"
    assert settings.storage.gcs_bucket == "bucket-a"
    assert settings.vertex.generation_model
    assert settings.vertex.embedding_model
    assert settings.phase1_runtime.run_yamnet_on_gpu is False


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
    assert settings.vllm_vibevoice.audio_mode == "url"
    assert settings.vibevoice.max_new_tokens == 32768
    assert settings.vibevoice.do_sample is False
    assert settings.vibevoice.top_p == 1.0
    assert settings.vibevoice.num_beams == 1
    assert settings.vibevoice.repetition_penalty == 1.03


def test_load_provider_settings_reads_phase1_cloud_run_asr_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "clypt-v3")
    monkeypatch.setenv("GCS_BUCKET", "bucket-a")
    monkeypatch.setenv("CLYPT_PHASE1_ASR_BACKEND", "cloud_run_l4")
    monkeypatch.setenv("CLYPT_PHASE1_ASR_SERVICE_URL", "https://phase1-asr.example.com")
    monkeypatch.setenv("CLYPT_PHASE1_ASR_AUTH_MODE", "id_token")
    monkeypatch.setenv("CLYPT_PHASE1_ASR_AUDIENCE", "https://phase1-asr.example.com")
    monkeypatch.setenv("CLYPT_PHASE1_ASR_TIMEOUT_S", "901")
    monkeypatch.delenv("VIBEVOICE_VLLM_BASE_URL", raising=False)

    settings = load_provider_settings()

    assert settings.phase1_asr.backend == "cloud_run_l4"
    assert settings.phase1_asr.service_url == "https://phase1-asr.example.com"
    assert settings.phase1_asr.auth_mode == "id_token"
    assert settings.phase1_asr.audience == "https://phase1-asr.example.com"
    assert settings.phase1_asr.timeout_s == 901.0


def test_load_provider_settings_rejects_legacy_vllm_url_when_phase1_asr_is_remote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "clypt-v3")
    monkeypatch.setenv("GCS_BUCKET", "bucket-a")
    monkeypatch.setenv("CLYPT_PHASE1_ASR_BACKEND", "cloud_run_l4")
    monkeypatch.setenv("CLYPT_PHASE1_ASR_SERVICE_URL", "https://phase1-asr.example.com")
    monkeypatch.setenv("VIBEVOICE_VLLM_BASE_URL", "http://127.0.0.1:8000")

    with pytest.raises(ValueError, match="VIBEVOICE_VLLM_BASE_URL"):
        load_provider_settings()


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

    if importlib.util.find_spec("dotenv") is None:
        pytest.skip("python-dotenv is not installed in this test environment")

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
                "VIBEVOICE_DO_SAMPLE=1.0",
                "VIBEVOICE_TOP_P=0.91",
                "VIBEVOICE_NUM_BEAMS=4.0",
                "VIBEVOICE_REPETITION_PENALTY=1.03",
                "GOOGLE_CLOUD_PROJECT=clypt-v3",
                "GENAI_GENERATION_LOCATION=global",
                "VERTEX_EMBEDDING_LOCATION=us-central1",
                "GCS_BUCKET=clypt-storage-v3",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_provider_settings()

    assert settings.vllm_vibevoice.base_url == "http://127.0.0.1:8000"
    assert settings.vllm_vibevoice.model == "vibevoice"
    assert settings.vllm_vibevoice.audio_mode == "url"
    assert settings.vibevoice.do_sample is True
    assert settings.vibevoice.top_p == 0.91
    assert settings.vibevoice.num_beams == 4
    assert settings.vibevoice.repetition_penalty == 1.03
    assert settings.vertex.project == "clypt-v3"
    assert settings.vertex.generation_location == "global"
    assert settings.vertex.embedding_location == "us-central1"
    assert settings.storage.gcs_bucket == "clypt-storage-v3"
    assert settings.phase1_runtime.run_yamnet_on_gpu is False


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


def test_load_provider_settings_rejects_unknown_generation_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "clypt-v3")
    monkeypatch.setenv("GCS_BUCKET", "bucket-a")
    monkeypatch.setenv("VIBEVOICE_VLLM_BASE_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("GENAI_GENERATION_BACKEND", "vertex")

    with pytest.raises(ValueError, match="expected 'developer' or 'local_openai'"):
        load_provider_settings()


def test_load_provider_settings_defaults_local_openai_and_loads_local_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "clypt-v3")
    monkeypatch.setenv("GCS_BUCKET", "bucket-a")
    monkeypatch.setenv("VIBEVOICE_VLLM_BASE_URL", "http://127.0.0.1:8000")
    monkeypatch.delenv("GENAI_GENERATION_BACKEND", raising=False)
    monkeypatch.setenv("CLYPT_LOCAL_LLM_BASE_URL", "http://127.0.0.1:9000/v1")
    monkeypatch.setenv("CLYPT_LOCAL_LLM_MODEL", "my-qwen")
    monkeypatch.setenv("CLYPT_LOCAL_LLM_TIMEOUT_S", "120")
    monkeypatch.setenv("CLYPT_LOCAL_LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("CLYPT_LOCAL_LLM_INITIAL_BACKOFF_S", "0.5")
    monkeypatch.setenv("CLYPT_LOCAL_LLM_MAX_BACKOFF_S", "10")
    monkeypatch.setenv("CLYPT_LOCAL_LLM_BACKOFF_MULTIPLIER", "1.5")
    monkeypatch.setenv("CLYPT_LOCAL_LLM_JITTER_RATIO", "0.1")
    monkeypatch.setenv("CLYPT_LOCAL_LLM_TEMPERATURE", "0.9")
    monkeypatch.setenv("CLYPT_LOCAL_LLM_TOP_P", "0.85")
    monkeypatch.setenv("CLYPT_LOCAL_LLM_TOP_K", "30")
    monkeypatch.setenv("CLYPT_LOCAL_LLM_MIN_P", "0.05")
    monkeypatch.setenv("CLYPT_LOCAL_LLM_PRESENCE_PENALTY", "1.8")
    monkeypatch.setenv("CLYPT_LOCAL_LLM_REPETITION_PENALTY", "1.02")

    settings = load_provider_settings()

    assert settings.vertex.generation_backend == "local_openai"
    assert settings.local_generation.base_url == "http://127.0.0.1:9000/v1"
    assert settings.local_generation.model == "my-qwen"
    assert settings.local_generation.timeout_s == 120.0
    assert settings.local_generation.max_retries == 3
    assert settings.local_generation.initial_backoff_s == 0.5
    assert settings.local_generation.max_backoff_s == 10.0
    assert settings.local_generation.backoff_multiplier == 1.5
    assert settings.local_generation.jitter_ratio == 0.1
    assert not hasattr(settings.local_generation, "enable_thinking")
    assert settings.local_generation.temperature == 0.9
    assert settings.local_generation.top_p == 0.85
    assert settings.local_generation.top_k == 30
    assert settings.local_generation.min_p == 0.05
    assert settings.local_generation.presence_penalty == 1.8
    assert settings.local_generation.repetition_penalty == 1.02


def test_load_provider_settings_rejects_removed_local_llm_thinking_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "clypt-v3")
    monkeypatch.setenv("GCS_BUCKET", "bucket-a")
    monkeypatch.setenv("VIBEVOICE_VLLM_BASE_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("CLYPT_LOCAL_LLM_ENABLE_THINKING", "1")

    with pytest.raises(ValueError, match="CLYPT_LOCAL_LLM_ENABLE_THINKING"):
        load_provider_settings()


def test_local_openai_qwen_client_parses_json_and_validates_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock
    import urllib.request

    from backend.providers.config import LocalGenerationSettings
    from backend.providers.openai_local import LocalOpenAIQwenClient

    settings = LocalGenerationSettings(model="qwen", max_retries=0)
    captured_payloads: list[dict[str, Any]] = []

    monkeypatch.chdir(tmp_path)
    failure_dir = tmp_path / "local-openai-failures"
    monkeypatch.setenv("CLYPT_LOCAL_OPENAI_FAILURE_DIR", str(failure_dir))

    def _install_payload(payload: dict[str, Any]) -> None:
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value.read.return_value = json.dumps(
            payload
        ).encode("utf-8")
        mock_resp.__exit__.return_value = None

        def _fake_urlopen(req: urllib.request.Request, timeout: float):
            _ = timeout
            captured_payloads.append(json.loads(req.data.decode("utf-8")))
            return mock_resp

        monkeypatch.setattr("backend.providers.openai_local.urllib.request.urlopen", _fake_urlopen)

    def _install_response(content: str, *, finish_reason: str | None = None) -> None:
        payload = {"choices": [{"message": {"content": content}}]}
        if finish_reason is not None:
            payload["choices"][0]["finish_reason"] = finish_reason
        _install_payload(payload)

    _install_response('{"thread_summary": "x", "moment_hints": []}')
    client = LocalOpenAIQwenClient(settings=settings)
    assert "thinking_level" not in inspect.signature(client.generate_json).parameters
    schema = {
        "type": "object",
        "properties": {
            "thread_summary": {"type": "string"},
            "moment_hints": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["thread_summary", "moment_hints"],
    }
    out = client.generate_json(
        prompt="p",
        response_schema=schema,
        max_output_tokens=123,
    )
    assert out["thread_summary"] == "x"
    assert captured_payloads[0]["max_tokens"] == 123
    assert captured_payloads[0]["response_format"]["type"] == "json_schema"
    assert captured_payloads[0]["response_format"]["json_schema"]["strict"] is True
    assert (
        captured_payloads[0]["response_format"]["json_schema"]["schema"]["additionalProperties"]
        is False
    )
    assert captured_payloads[0]["chat_template_kwargs"]["enable_thinking"] is False
    assert captured_payloads[0]["top_k"] == 40
    assert captured_payloads[0]["min_p"] == 0.0
    assert "extra_body" not in captured_payloads[0]
    assert "JSON" in captured_payloads[0]["messages"][0]["content"]

    _install_response("not-json")
    with pytest.raises(ValueError, match="valid JSON"):
        client.generate_json(prompt="p", response_schema=schema)

    with pytest.raises(ValueError, match="response_schema is required"):
        client.generate_json(prompt="p")

    _install_response("{}")
    with pytest.raises(ValueError, match="missing required"):
        client.generate_json(prompt="p", response_schema=schema)

    _install_response("not-json", finish_reason="length")
    with pytest.raises(ValueError, match="finish_reason=length"):
        client.generate_json(prompt="p", response_schema=schema, max_output_tokens=7)

    _install_payload(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": None,
                        "reasoning_content": '{"thread_summary":"x","moment_hints":[]}',
                    },
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="missing message.content"):
        client.generate_json(prompt="p", response_schema=schema)
    artifacts = sorted(failure_dir.glob("*.response.json"))
    assert artifacts
    assert '"reasoning_content"' in artifacts[-1].read_text(encoding="utf-8")


def test_v31_config_reads_phase3_and_phase4_output_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.pipeline.config import V31Config

    monkeypatch.setenv("CLYPT_PHASE3_LOCAL_MAX_OUTPUT_TOKENS", "1536")
    monkeypatch.setenv("CLYPT_PHASE3_LONG_RANGE_MAX_OUTPUT_TOKENS", "1792")
    monkeypatch.setenv("CLYPT_PHASE3_LONG_RANGE_TOP_K", "2")
    monkeypatch.setenv("CLYPT_PHASE3_LONG_RANGE_PAIRS_PER_SHARD", "18")
    monkeypatch.setenv("CLYPT_PHASE3_LONG_RANGE_MAX_CONCURRENT", "3")
    monkeypatch.setenv("CLYPT_PHASE3_LOCAL_MAX_CONCURRENT", "8")
    monkeypatch.setenv("CLYPT_PHASE4_META_MAX_OUTPUT_TOKENS", "1024")
    monkeypatch.setenv("CLYPT_PHASE4_SUBGRAPH_MAX_OUTPUT_TOKENS", "2048")
    monkeypatch.setenv("CLYPT_PHASE4_SUBGRAPH_MAX_CONCURRENT", "10")
    monkeypatch.setenv("CLYPT_PHASE4_POOL_MAX_OUTPUT_TOKENS", "768")

    config = V31Config()

    assert config.phase3_local_max_output_tokens == 1536
    assert config.phase3_long_range_max_output_tokens == 1792
    assert config.phase3_long_range_top_k == 2
    assert config.phase3_long_range_pairs_per_shard == 18
    assert config.phase3_long_range_max_concurrent == 3
    assert config.phase3_local_max_concurrent == 8
    assert config.phase4_meta_max_output_tokens == 1024
    assert config.phase4_subgraph_max_output_tokens == 2048
    assert config.phase4_subgraph_max_concurrent == 10
    assert config.phase4_pool_max_output_tokens == 768


def test_v31_config_reads_phase2_and_signal_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.pipeline.config import V31Config

    monkeypatch.setenv("CLYPT_PHASE2_MERGE_MAX_CONCURRENT", "8")
    monkeypatch.setenv("CLYPT_PHASE2_BOUNDARY_MAX_CONCURRENT", "10")
    monkeypatch.setenv("CLYPT_SIGNAL_MAX_CONCURRENT", "6")

    config = V31Config()

    assert config.phase2_merge_max_concurrent == 8
    assert config.phase2_boundary_max_concurrent == 10
    assert config.signals.max_concurrent == 6


def test_v31_config_defaults_match_recommended_bench_caps() -> None:
    from backend.pipeline.config import V31Config

    config = V31Config()

    # Defaults reflect the bench-validated max-in-flight caps (2026-04-16 Qwen3.6 on H200).
    assert config.phase2_merge_max_concurrent == 16
    assert config.phase2_boundary_max_concurrent == 16
    assert config.phase3_local_max_concurrent == 24
    assert config.phase3_long_range_max_concurrent == 24
    assert config.phase4_subgraph_max_concurrent == 16


def test_v31_config_rejects_renamed_concurrency_envs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.pipeline.config import V31Config

    monkeypatch.setenv("CLYPT_PHASE2_MAX_CONCURRENT", "8")

    with pytest.raises(ValueError, match="CLYPT_PHASE2_MERGE_MAX_CONCURRENT"):
        V31Config()


def test_v31_config_rejects_removed_global_concurrency_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.pipeline.config import V31Config

    monkeypatch.setenv("CLYPT_GEMINI_MAX_CONCURRENT", "2")

    with pytest.raises(ValueError, match="CLYPT_GEMINI_MAX_CONCURRENT"):
        V31Config()


def test_v31_config_rejects_removed_thinking_envs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.pipeline.config import V31Config

    monkeypatch.setenv("CLYPT_PHASE3_LOCAL_THINKING_LEVEL", "minimal")

    with pytest.raises(ValueError, match="CLYPT_PHASE3_LOCAL_THINKING_LEVEL"):
        V31Config()


def test_build_default_phase24_worker_service_enforces_local_openai_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.providers.config import (
        LocalGenerationSettings,
        Phase24MediaPrepSettings,
        ProviderSettings,
        StorageSettings,
        VertexSettings,
        VibeVoiceSettings,
        VibeVoiceVLLMSettings,
    )
    from backend.providers.openai_local import LocalOpenAIQwenClient
    from backend.runtime import phase24_worker_app

    base = dict(
        vibevoice=VibeVoiceSettings(),
        vllm_vibevoice=VibeVoiceVLLMSettings(base_url="http://127.0.0.1:8000"),
        local_generation=LocalGenerationSettings(model="local-qwen"),
        storage=StorageSettings(gcs_bucket="bucket-a"),
    )

    captured: dict[str, Any] = {}

    class _FakeRepo:
        def bootstrap_schema(self) -> None:
            return None

        @classmethod
        def from_settings(cls, **kwargs):
            return cls()

    class _FakeRunner:
        @classmethod
        def from_env(cls, **kwargs):
            captured["llm_client"] = kwargs["llm_client"]
            captured["node_media_preparer"] = kwargs.get("node_media_preparer")
            return object()

    monkeypatch.setattr(phase24_worker_app, "SpannerPhase14Repository", _FakeRepo)
    monkeypatch.setattr(phase24_worker_app, "V31LivePhase14Runner", _FakeRunner)
    monkeypatch.setattr(phase24_worker_app, "GCSStorageClient", lambda **kwargs: object())
    monkeypatch.setattr(phase24_worker_app, "VertexEmbeddingClient", lambda **kwargs: object())

    settings_local = ProviderSettings(
        **base,
        vertex=VertexSettings(project="clypt-v3", generation_backend="local_openai"),
    )
    monkeypatch.setattr(phase24_worker_app, "load_provider_settings", lambda: settings_local)
    phase24_worker_app.build_default_phase24_worker_service()
    assert isinstance(captured["llm_client"], LocalOpenAIQwenClient)
    assert captured["llm_client"].settings.model == "local-qwen"
    assert captured["node_media_preparer"] is None

    settings_remote = ProviderSettings(
        **base,
        vertex=VertexSettings(project="clypt-v3", generation_backend="local_openai"),
        phase24_media_prep=Phase24MediaPrepSettings(
            backend="cloud_run_l4",
            service_url="https://media-prep.example.com",
            auth_mode="none",
        ),
    )
    monkeypatch.setattr(phase24_worker_app, "load_provider_settings", lambda: settings_remote)
    phase24_worker_app.build_default_phase24_worker_service()
    assert callable(captured["node_media_preparer"])

    settings_dev = ProviderSettings(
        **base,
        vertex=VertexSettings(
            project="clypt-v3",
            generation_backend="developer",
            gemini_api_key="test-key",
        ),
    )
    monkeypatch.setattr(phase24_worker_app, "load_provider_settings", lambda: settings_dev)
    with pytest.raises(ValueError, match="only GENAI_GENERATION_BACKEND=local_openai"):
        phase24_worker_app.build_default_phase24_worker_service()

    settings_invalid_media = ProviderSettings(
        **base,
        vertex=VertexSettings(project="clypt-v3", generation_backend="local_openai"),
        phase24_media_prep=Phase24MediaPrepSettings(backend="mystery"),
    )
    monkeypatch.setattr(phase24_worker_app, "load_provider_settings", lambda: settings_invalid_media)
    with pytest.raises(ValueError, match="backend=local or cloud_run_l4"):
        phase24_worker_app.build_default_phase24_worker_service()


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
    monkeypatch.setenv("CLYPT_PHASE24_FAILFAST_PREEMPTION_THRESHOLD", "7")
    monkeypatch.setenv("CLYPT_PHASE24_FAILFAST_P95_LATENCY_MS", "4500")
    monkeypatch.setenv("CLYPT_PHASE24_ADMISSION_METRICS_PATH", "/tmp/admission_metrics.json")
    monkeypatch.setenv("CLYPT_PHASE24_BLOCK_ON_PHASE1_ACTIVE", "1")
    monkeypatch.setenv("CLYPT_PHASE24_MEDIA_PREP_BACKEND", "cloud_run_l4")
    monkeypatch.setenv("CLYPT_PHASE24_MEDIA_PREP_SERVICE_URL", "https://phase24-media-prep.example.com")
    monkeypatch.setenv("CLYPT_PHASE24_MEDIA_PREP_AUTH_MODE", "id_token")
    monkeypatch.setenv("CLYPT_PHASE24_MEDIA_PREP_AUDIENCE", "https://phase24-media-prep.example.com")
    monkeypatch.setenv("CLYPT_PHASE24_MEDIA_PREP_TIMEOUT_S", "901")
    monkeypatch.setenv("CLYPT_PHASE24_QUEUE_BACKEND", "local_sqlite")
    monkeypatch.setenv("CLYPT_PHASE24_LOCAL_QUEUE_PATH", "backend/outputs/phase24.sqlite")
    monkeypatch.setenv("CLYPT_PHASE24_LOCAL_POLL_INTERVAL_MS", "700")
    monkeypatch.setenv("CLYPT_PHASE24_LOCAL_LEASE_TIMEOUT_S", "1200")
    monkeypatch.setenv("CLYPT_PHASE24_LOCAL_MAX_INFLIGHT", "3")
    monkeypatch.setenv("CLYPT_PHASE24_LOCAL_MAX_REQUESTS_PER_WORKER", "11")

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
    assert settings.phase24_worker.fail_fast_preemption_threshold == 7
    assert settings.phase24_worker.fail_fast_p95_latency_ms == 4500.0
    assert settings.phase24_worker.admission_metrics_path == "/tmp/admission_metrics.json"
    assert settings.phase24_worker.block_on_phase1_active is True
    assert settings.phase24_media_prep.backend == "cloud_run_l4"
    assert settings.phase24_media_prep.service_url == "https://phase24-media-prep.example.com"
    assert settings.phase24_media_prep.auth_mode == "id_token"
    assert settings.phase24_media_prep.audience == "https://phase24-media-prep.example.com"
    assert settings.phase24_media_prep.timeout_s == 901.0
    assert settings.phase24_local_queue.queue_backend == "local_sqlite"
    assert str(settings.phase24_local_queue.path).endswith("backend/outputs/phase24.sqlite")
    assert settings.phase24_local_queue.poll_interval_ms == 700
    assert settings.phase24_local_queue.lease_timeout_s == 1200
    assert settings.phase24_local_queue.max_inflight == 3
    assert settings.phase24_local_queue.max_requests_per_worker == 11


def test_load_provider_settings_exposes_split_generation_and_embedding_retry_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "clypt-v3")
    monkeypatch.setenv("GCS_BUCKET", "bucket-a")
    monkeypatch.setenv("VIBEVOICE_VLLM_BASE_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("GENAI_GENERATION_API_MAX_RETRIES", "9")
    monkeypatch.setenv("GENAI_GENERATION_API_INITIAL_BACKOFF_S", "0.5")
    monkeypatch.setenv("GENAI_GENERATION_API_MAX_BACKOFF_S", "12.0")
    monkeypatch.setenv("GENAI_GENERATION_API_BACKOFF_MULTIPLIER", "1.7")
    monkeypatch.setenv("GENAI_GENERATION_API_JITTER_RATIO", "0.05")
    monkeypatch.setenv("VERTEX_EMBEDDING_API_MAX_RETRIES", "3")
    monkeypatch.setenv("VERTEX_EMBEDDING_API_INITIAL_BACKOFF_S", "0.2")
    monkeypatch.setenv("VERTEX_EMBEDDING_API_MAX_BACKOFF_S", "6.0")
    monkeypatch.setenv("VERTEX_EMBEDDING_API_BACKOFF_MULTIPLIER", "1.4")
    monkeypatch.setenv("VERTEX_EMBEDDING_API_JITTER_RATIO", "0.01")

    settings = load_provider_settings()

    assert settings.vertex.generation_api_max_retries == 9
    assert settings.vertex.generation_api_initial_backoff_s == 0.5
    assert settings.vertex.generation_api_max_backoff_s == 12.0
    assert settings.vertex.generation_api_backoff_multiplier == 1.7
    assert settings.vertex.generation_api_jitter_ratio == 0.05
    assert settings.vertex.embedding_api_max_retries == 3
    assert settings.vertex.embedding_api_initial_backoff_s == 0.2
    assert settings.vertex.embedding_api_max_backoff_s == 6.0
    assert settings.vertex.embedding_api_backoff_multiplier == 1.4
    assert settings.vertex.embedding_api_jitter_ratio == 0.01


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
