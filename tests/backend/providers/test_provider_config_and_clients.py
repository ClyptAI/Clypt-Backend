from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_load_provider_settings_uses_env_and_gcloud_fallback(tmp_path: Path, monkeypatch):
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    monkeypatch.delenv("VERTEX_GEMINI_LOCATION", raising=False)
    monkeypatch.delenv("VERTEX_EMBEDDING_LOCATION", raising=False)
    monkeypatch.delenv("PYANNOTE_API_KEY", raising=False)
    monkeypatch.setenv("CLYPT_GCS_BUCKET", "bucket-a")
    monkeypatch.setenv("VIBEVOICE_BACKEND", "hf")

    def fake_run(cmd, check, capture_output, text):
        assert cmd[:4] == ["gcloud", "config", "get-value", "project"]
        return subprocess.CompletedProcess(cmd, 0, stdout="clypt-v3\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    settings = load_provider_settings()

    assert settings.vibevoice.backend == "hf"
    assert settings.vibevoice.model_id == "microsoft/VibeVoice-ASR-HF"
    assert settings.vibevoice.flash_attention is True
    assert settings.vibevoice.liger_kernel is True
    assert settings.vertex.project == "clypt-v3"
    assert settings.vertex.generation_location == "global"
    assert settings.vertex.embedding_location == "us-central1"
    assert settings.storage.gcs_bucket == "bucket-a"
    assert settings.vertex.generation_model
    assert settings.vertex.embedding_model
    assert settings.phase1_runtime.yamnet_branch_device == "cpu"


def test_load_provider_settings_native_defaults(tmp_path: Path, monkeypatch):
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "clypt-v3")
    monkeypatch.setenv("GCS_BUCKET", "bucket-a")
    monkeypatch.delenv("VIBEVOICE_BACKEND", raising=False)

    settings = load_provider_settings()

    assert settings.vibevoice.backend == "native"
    assert settings.vibevoice.model_id == "microsoft/VibeVoice-ASR"
    assert settings.vibevoice.max_new_tokens == 32768
    assert settings.vibevoice.do_sample is False


def test_load_provider_settings_raises_without_gcs_bucket(tmp_path: Path, monkeypatch):
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    monkeypatch.delenv("CLYPT_GCS_BUCKET", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "clypt-v3")

    with pytest.raises(ValueError, match="GCS_BUCKET"):
        load_provider_settings()


def test_load_provider_settings_reads_untracked_env_local(tmp_path: Path, monkeypatch):
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYANNOTE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    monkeypatch.delenv("CLYPT_GCS_BUCKET", raising=False)
    monkeypatch.delenv("VIBEVOICE_MODEL_ID", raising=False)
    (tmp_path / ".env.local").write_text(
        "\n".join(
            [
                "VIBEVOICE_BACKEND=hf",
                "VIBEVOICE_MODEL_ID=microsoft/VibeVoice-ASR-HF",
                "VIBEVOICE_FLASH_ATTN=0",
                "VIBEVOICE_LIGER=0",
                "GOOGLE_CLOUD_PROJECT=clypt-v3",
                "VERTEX_GEMINI_LOCATION=global",
                "VERTEX_EMBEDDING_LOCATION=us-central1",
                "GCS_BUCKET=clypt-storage-v3",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_provider_settings()

    assert settings.vibevoice.backend == "hf"
    assert settings.vibevoice.model_id == "microsoft/VibeVoice-ASR-HF"
    assert settings.vibevoice.flash_attention is False
    assert settings.vibevoice.liger_kernel is False
    assert settings.vertex.project == "clypt-v3"
    assert settings.vertex.generation_location == "global"
    assert settings.vertex.embedding_location == "us-central1"
    assert settings.storage.gcs_bucket == "clypt-storage-v3"
    assert settings.phase1_runtime.yamnet_branch_device == "cpu"


def test_load_provider_settings_vibevoice_custom_hotwords(tmp_path: Path, monkeypatch):
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "clypt-v3")
    monkeypatch.setenv("GCS_BUCKET", "bucket-a")
    monkeypatch.setenv("VIBEVOICE_BACKEND", "hf")
    monkeypatch.setenv("VIBEVOICE_HOTWORDS_CONTEXT", "hello, world")

    settings = load_provider_settings()

    assert settings.vibevoice.hotwords_context == "hello, world"


def test_load_provider_settings_vibevoice_system_prompt(tmp_path: Path, monkeypatch):
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "clypt-v3")
    monkeypatch.setenv("GCS_BUCKET", "bucket-a")
    monkeypatch.setenv("VIBEVOICE_BACKEND", "hf")
    monkeypatch.setenv("VIBEVOICE_SYSTEM_PROMPT", "Custom system prompt")

    settings = load_provider_settings()

    assert settings.vibevoice.system_prompt == "Custom system prompt"


def test_load_provider_settings_phase1_runtime_parallel_knobs(tmp_path: Path, monkeypatch):
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "clypt-v3")
    monkeypatch.setenv("GCS_BUCKET", "bucket-a")
    monkeypatch.setenv("CLYPT_PHASE1_BRANCH_TIMEOUT_S", "99.5")
    monkeypatch.setenv("CLYPT_PHASE1_BRANCH_POLL_INTERVAL_S", "0.25")
    monkeypatch.setenv("CLYPT_PHASE1_PARALLEL_ENABLED", "0")

    settings = load_provider_settings()

    assert settings.phase1_runtime.branch_timeout_s == 99.5
    assert settings.phase1_runtime.branch_poll_interval_s == 0.25
    assert settings.phase1_runtime.phase1_parallel_enabled is False
    assert settings.phase1_runtime.phase1_parallel_gpu_branch_limit == 2
    assert settings.phase1_runtime.yamnet_branch_device == "cpu"


def test_default_phase1_job_runner_uses_parallel_sidecar_runtime(tmp_path: Path, monkeypatch):
    from backend.phase1_runtime import factory

    parallel_calls: list[dict[str, object]] = []
    yamnet_devices: list[str] = []

    def fake_parallel_phase1_sidecars(
        *,
        source_url: str,
        video_gcs_uri: str,
        workspace,
        branch_timeout_s: float,
        poll_interval_s: float,
    ):
        parallel_calls.append(
            {
                "source_url": source_url,
                "video_gcs_uri": video_gcs_uri,
                "workspace": workspace,
                "branch_timeout_s": branch_timeout_s,
                "poll_interval_s": poll_interval_s,
            }
        )
        return {"ok": True}

    monkeypatch.setattr(factory, "run_parallel_phase1_sidecars", fake_parallel_phase1_sidecars)
    monkeypatch.setattr(
        factory,
        "load_provider_settings",
        lambda: SimpleNamespace(
            vibevoice=SimpleNamespace(
                backend="native",
                native_venv_python="",
                model_id="model",
                flash_attention=True,
                liger_kernel=True,
                hotwords_context="",
                system_prompt="",
                max_new_tokens=1024,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
                num_beams=1,
                attn_implementation="flash_attention_2",
                subprocess_timeout_s=7200,
            ),
            vertex=SimpleNamespace(project="clypt-v3"),
            storage=SimpleNamespace(gcs_bucket="bucket-a"),
            phase1_runtime=SimpleNamespace(
                working_root=tmp_path,
                keep_workdir=False,
                branch_timeout_s=45.0,
                branch_poll_interval_s=0.3,
                phase1_parallel_enabled=True,
                phase1_parallel_gpu_branch_limit=2,
                yamnet_branch_device="cpu",
            ),
        ),
    )
    monkeypatch.setattr(factory, "VibeVoiceASRProvider", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(factory, "ForcedAlignmentProvider", lambda: object())
    monkeypatch.setattr(factory, "VertexEmbeddingClient", lambda settings: object())
    monkeypatch.setattr(factory, "VertexGeminiClient", lambda settings: object())
    monkeypatch.setattr(factory, "GCSStorageClient", lambda settings: object())
    monkeypatch.setattr(
        factory,
        "V31LivePhase14Runner",
        SimpleNamespace(from_env=lambda **kwargs: object()),
    )
    monkeypatch.setattr(factory, "VisualPipelineConfig", SimpleNamespace(from_env=lambda: object()))
    monkeypatch.setattr(factory, "SimpleVisualExtractor", lambda visual_config: object())
    monkeypatch.setattr(factory, "Emotion2VecPlusProvider", lambda: object())

    def fake_yamnet_provider(*, device: str, **kwargs):
        yamnet_devices.append(device)
        return SimpleNamespace(device=device, **kwargs)

    monkeypatch.setattr(factory, "YAMNetProvider", fake_yamnet_provider)

    runner = factory.build_default_phase1_job_runner()

    assert runner.run_phase1_sidecars.__name__ == "run_parallel_phase1_sidecars"

    sentinel_workspace = object()
    result = runner.run_phase1_sidecars(
        source_url="https://youtube.com/watch?v=demo",
        video_gcs_uri="gs://bucket/source.mp4",
        workspace=sentinel_workspace,
    )

    assert result == {"ok": True}
    assert parallel_calls == [
        {
            "source_url": "https://youtube.com/watch?v=demo",
            "video_gcs_uri": "gs://bucket/source.mp4",
            "workspace": sentinel_workspace,
            "branch_timeout_s": 45.0,
            "poll_interval_s": 0.3,
        }
    ]
    assert yamnet_devices == ["cpu"]


def test_default_phase1_job_runner_keeps_serial_fallback_when_parallel_disabled(
    tmp_path: Path, monkeypatch
):
    from backend.phase1_runtime import factory

    parallel_calls: list[dict[str, object]] = []
    yamnet_devices: list[str] = []

    def fake_parallel_phase1_sidecars(**kwargs):
        parallel_calls.append(dict(kwargs))
        return {"ok": True}

    monkeypatch.setattr(factory, "run_parallel_phase1_sidecars", fake_parallel_phase1_sidecars)
    monkeypatch.setattr(
        factory,
        "load_provider_settings",
        lambda: SimpleNamespace(
            vibevoice=SimpleNamespace(
                backend="native",
                native_venv_python="",
                model_id="model",
                flash_attention=True,
                liger_kernel=True,
                hotwords_context="",
                system_prompt="",
                max_new_tokens=1024,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
                num_beams=1,
                attn_implementation="flash_attention_2",
                subprocess_timeout_s=7200,
            ),
            vertex=SimpleNamespace(project="clypt-v3"),
            storage=SimpleNamespace(gcs_bucket="bucket-a"),
            phase1_runtime=SimpleNamespace(
                working_root=tmp_path,
                keep_workdir=False,
                branch_timeout_s=45.0,
                branch_poll_interval_s=0.3,
                phase1_parallel_enabled=False,
                phase1_parallel_gpu_branch_limit=2,
                yamnet_branch_device="gpu",
            ),
        ),
    )
    monkeypatch.setattr(factory, "VibeVoiceASRProvider", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(factory, "ForcedAlignmentProvider", lambda: object())
    monkeypatch.setattr(factory, "VertexEmbeddingClient", lambda settings: object())
    monkeypatch.setattr(factory, "VertexGeminiClient", lambda settings: object())
    monkeypatch.setattr(factory, "GCSStorageClient", lambda settings: object())
    monkeypatch.setattr(
        factory,
        "V31LivePhase14Runner",
        SimpleNamespace(from_env=lambda **kwargs: object()),
    )
    monkeypatch.setattr(factory, "VisualPipelineConfig", SimpleNamespace(from_env=lambda: object()))
    monkeypatch.setattr(factory, "SimpleVisualExtractor", lambda visual_config: object())
    monkeypatch.setattr(factory, "Emotion2VecPlusProvider", lambda: object())

    def fake_yamnet_provider(*, device: str, **kwargs):
        yamnet_devices.append(device)
        return SimpleNamespace(device=device, **kwargs)

    monkeypatch.setattr(factory, "YAMNetProvider", fake_yamnet_provider)

    runner = factory.build_default_phase1_job_runner()

    assert runner.run_phase1_sidecars.__name__ == "_run_phase1_sidecars"
    assert parallel_calls == []
    assert yamnet_devices == ["gpu"]
