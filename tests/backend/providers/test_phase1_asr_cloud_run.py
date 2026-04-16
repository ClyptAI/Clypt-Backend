from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import types
from urllib.error import HTTPError

import pytest


def test_cloud_run_vibevoice_provider_posts_asr_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from backend.providers.config import Phase1ASRSettings
    from backend.providers.phase1_asr_cloud_run import CloudRunVibeVoiceProvider

    captured: dict[str, object] = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "turns": [
                        {"Start": 0.0, "End": 1.0, "Speaker": 0, "Content": "hello world"},
                    ]
                }
            ).encode("utf-8")

    def fake_fetch_id_token(_req, audience):
        captured["audience"] = audience
        return "token-123"

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse()

    google_module = types.ModuleType("google")
    google_auth_module = types.ModuleType("google.auth")
    google_auth_transport_module = types.ModuleType("google.auth.transport")
    google_auth_transport_requests_module = types.ModuleType("google.auth.transport.requests")
    google_auth_transport_requests_module.Request = lambda: object()
    google_oauth2_module = types.ModuleType("google.oauth2")
    google_oauth2_id_token_module = types.ModuleType("google.oauth2.id_token")
    google_oauth2_id_token_module.fetch_id_token = fake_fetch_id_token
    google_oauth2_module.id_token = google_oauth2_id_token_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.auth", google_auth_module)
    monkeypatch.setitem(sys.modules, "google.auth.transport", google_auth_transport_module)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", google_auth_transport_requests_module)
    monkeypatch.setitem(sys.modules, "google.oauth2", google_oauth2_module)
    monkeypatch.setitem(sys.modules, "google.oauth2.id_token", google_oauth2_id_token_module)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    provider = CloudRunVibeVoiceProvider(
        settings=Phase1ASRSettings(
            backend="cloud_run_l4",
            service_url="https://phase1-asr.example.com",
            auth_mode="id_token",
            timeout_s=321.0,
        ),
        hotwords_context="default hotwords",
        repetition_penalty=1.03,
    )

    turns = provider.run(
        audio_path=tmp_path / "ignored.wav",
        context_info="custom hotwords",
        audio_gcs_uri="gs://bucket/canonical/audio.wav",
    )

    assert turns == [{"Start": 0.0, "End": 1.0, "Speaker": 0, "Content": "hello world"}]
    assert captured["audience"] == "https://phase1-asr.example.com"
    assert captured["url"] == "https://phase1-asr.example.com/tasks/asr"
    assert captured["timeout"] == 321.0
    assert captured["headers"]["Authorization"] == "Bearer token-123"
    assert captured["body"] == {
        "audio_gcs_uri": "gs://bucket/canonical/audio.wav",
        "context_info": "custom hotwords",
        "generation_config": {
            "max_new_tokens": 32768,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.03,
            "num_beams": 1,
        },
    }


def test_cloud_run_vibevoice_provider_surfaces_http_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from backend.providers.config import Phase1ASRSettings
    from backend.providers.phase1_asr_cloud_run import CloudRunVibeVoiceProvider

    def fake_urlopen(req, timeout):
        raise HTTPError(
            req.full_url,
            503,
            "unavailable",
            hdrs=None,
            fp=io.BytesIO(b'{"detail":"backend unavailable"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    provider = CloudRunVibeVoiceProvider(
        settings=Phase1ASRSettings(
            backend="cloud_run_l4",
            service_url="https://phase1-asr.example.com",
            auth_mode="none",
        ),
        hotwords_context="default hotwords",
    )

    with pytest.raises(RuntimeError, match="HTTP 503"):
        provider.run(
            audio_path=tmp_path / "ignored.wav",
            audio_gcs_uri="gs://bucket/canonical/audio.wav",
        )
