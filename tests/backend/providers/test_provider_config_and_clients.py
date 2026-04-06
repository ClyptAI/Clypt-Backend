from __future__ import annotations

import subprocess
from pathlib import Path

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
    assert settings.phase1_runtime.run_yamnet_on_gpu is True


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
    assert settings.phase1_runtime.run_yamnet_on_gpu is True


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
