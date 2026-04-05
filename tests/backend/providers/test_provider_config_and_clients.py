from __future__ import annotations

import json
import subprocess
from pathlib import Path

import httpx
import pytest


def test_load_provider_settings_uses_env_and_gcloud_fallback(tmp_path: Path, monkeypatch):
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    monkeypatch.delenv("VERTEX_GEMINI_LOCATION", raising=False)
    monkeypatch.delenv("VERTEX_EMBEDDING_LOCATION", raising=False)
    monkeypatch.setenv("PYANNOTE_API_KEY", "secret-key")
    monkeypatch.setenv("CLYPT_GCS_BUCKET", "bucket-a")

    def fake_run(cmd, check, capture_output, text):
        assert cmd[:4] == ["gcloud", "config", "get-value", "project"]
        return subprocess.CompletedProcess(cmd, 0, stdout="clypt-v3\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    settings = load_provider_settings()

    assert settings.pyannote.api_key == "secret-key"
    assert settings.vertex.project == "clypt-v3"
    assert settings.vertex.generation_location == "global"
    assert settings.vertex.embedding_location == "us-central1"
    assert settings.storage.gcs_bucket == "bucket-a"
    assert settings.vertex.generation_model
    assert settings.vertex.embedding_model
    assert settings.phase1_runtime.run_yamnet_on_gpu is True


def test_load_provider_settings_requires_pyannote_key(tmp_path: Path, monkeypatch):
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYANNOTE_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "clypt-v3")
    monkeypatch.setenv("CLYPT_GCS_BUCKET", "bucket-a")

    with pytest.raises(ValueError, match="PYANNOTE_API_KEY"):
        load_provider_settings()


def test_load_provider_settings_reads_untracked_env_local(tmp_path: Path, monkeypatch):
    from backend.providers.config import load_provider_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYANNOTE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    monkeypatch.delenv("CLYPT_GCS_BUCKET", raising=False)
    (tmp_path / ".env.local").write_text(
        "\n".join(
            [
                "PYANNOTE_API_KEY=envlocal-secret",
                "GOOGLE_CLOUD_PROJECT=clypt-v3",
                "VERTEX_GEMINI_LOCATION=global",
                "VERTEX_EMBEDDING_LOCATION=us-central1",
                "GCS_BUCKET=clypt-storage-v3",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_provider_settings()

    assert settings.pyannote.api_key == "envlocal-secret"
    assert settings.vertex.project == "clypt-v3"
    assert settings.vertex.generation_location == "global"
    assert settings.vertex.embedding_location == "us-central1"
    assert settings.storage.gcs_bucket == "clypt-storage-v3"
    assert settings.phase1_runtime.run_yamnet_on_gpu is True


def test_pyannote_cloud_client_submits_and_polls_until_success():
    from backend.providers.config import PyannoteSettings
    from backend.providers.pyannote import PyannoteCloudClient

    requests: list[tuple[str, str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        requests.append((request.method, str(request.url), body))
        if request.method == "POST" and request.url.path == "/v1/diarize":
            return httpx.Response(200, json={"jobId": "job_123"})
        if request.method == "GET" and request.url.path == "/v1/jobs/job_123":
            count = sum(1 for method, url, _ in requests if method == "GET" and url.endswith("/v1/jobs/job_123"))
            if count == 1:
                return httpx.Response(200, json={"jobId": "job_123", "status": "running"})
            return httpx.Response(
                200,
                json={
                    "jobId": "job_123",
                    "status": "succeeded",
                    "output": {
                        "diarization": [{"speaker": "S1", "start": 0.0, "end": 1.0}],
                        "wordLevelTranscription": [{"word": "hi", "start": 0.0, "end": 0.2, "speaker": "S1"}],
                    },
                },
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    client = PyannoteCloudClient(
        settings=PyannoteSettings(api_key="secret", poll_interval_s=0.0, timeout_s=5.0),
        http_client=httpx.Client(transport=transport, base_url="https://api.pyannote.ai"),
    )

    output = client.run_diarize(media_url="gs://bucket/video.mp4")

    assert output["diarization"][0]["speaker"] == "S1"
    assert requests[0][0] == "POST"
    assert requests[0][1].endswith("/v1/diarize")
    assert requests[0][2]["url"] == "gs://bucket/video.mp4"
    assert requests[0][2]["transcription"] is True
    assert requests[0][2]["model"] == "precision-2"
    assert requests[0][2]["transcriptionConfig"]["model"] == "parakeet-tdt-0.6b-v3"


def test_pyannote_cloud_client_identify_includes_voiceprints():
    from backend.providers.config import PyannoteSettings
    from backend.providers.pyannote import PyannoteCloudClient

    requests: list[tuple[str, str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        requests.append((request.method, str(request.url), body))
        if request.method == "POST" and request.url.path == "/v1/identify":
            return httpx.Response(200, json={"jobId": "job_identify"})
        if request.method == "GET" and request.url.path == "/v1/jobs/job_identify":
            return httpx.Response(
                200,
                json={
                    "jobId": "job_identify",
                    "status": "succeeded",
                    "output": {"identification": [{"match": "Sam"}], "voiceprints": [{"match": "Sam"}]},
                },
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    client = PyannoteCloudClient(
        settings=PyannoteSettings(api_key="secret", poll_interval_s=0.0, timeout_s=5.0),
        http_client=httpx.Client(transport=transport, base_url="https://api.pyannote.ai"),
    )

    output = client.run_identify(
        media_url="gs://bucket/video.mp4",
        voiceprint_ids=["vp_sam", "vp_rick"],
    )

    assert output["identification"][0]["match"] == "Sam"
    assert requests[0][2]["voiceprints"] == ["vp_sam", "vp_rick"]
