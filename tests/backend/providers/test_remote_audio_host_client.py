"""Unit tests for the Phase 1 remote VibeVoice ASR client.

VibeVoice ASR lives on the RTX 6000 Ada host; NFA / emotion2vec+ / YAMNet
run in-process on the H200 after this call returns. These tests stub out
``urllib.request.urlopen`` so the client's retry / parse / re-emit logic
can be verified without any real network.
"""

from __future__ import annotations

import io
import json
from typing import Any
from urllib.error import HTTPError

import pytest

from backend.providers.audio_host_client import (
    RemoteVibeVoiceAsrClient,
    RemoteVibeVoiceAsrError,
    VibeVoiceAsrResponse,
)
from backend.providers.config import VibeVoiceAsrServiceSettings


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        return None

    def read(self) -> bytes:
        return self._body


def _settings() -> VibeVoiceAsrServiceSettings:
    return VibeVoiceAsrServiceSettings(
        service_url="http://rtx-box:9100",
        auth_token="secret-token",
    )


def _ok_payload(stage_events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "run_id": "run-1",
        "turns": [
            {"Speaker": 0, "Start": 0.0, "End": 1.0, "Content": "hello"},
        ],
        "stage_events": stage_events
        or [
            {
                "stage_name": "vibevoice_asr",
                "status": "succeeded",
                "duration_ms": 1234.5,
                "metadata": {"turn_count": 1},
            }
        ],
    }


def test_run_posts_payload_returns_typed_response_and_reemits_stage_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout):  # noqa: ARG001
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse(json.dumps(_ok_payload()).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    emitted: list[dict[str, Any]] = []

    def logger_fn(**kwargs: Any) -> None:
        emitted.append(kwargs)

    client = RemoteVibeVoiceAsrClient(settings=_settings(), max_retries=0)
    response = client.run(
        audio_gcs_uri="gs://bucket/audio.wav",
        source_url="https://example.com/video",
        video_gcs_uri="gs://bucket/video.mp4",
        run_id="run-1",
        stage_event_logger=logger_fn,
    )

    assert captured["url"] == "http://rtx-box:9100/tasks/vibevoice-asr"
    assert captured["method"] == "POST"
    header_dict = {k.lower(): v for k, v in captured["headers"].items()}
    assert header_dict["authorization"] == "Bearer secret-token"
    assert header_dict["content-type"].startswith("application/json")
    assert captured["body"] == {
        "audio_gcs_uri": "gs://bucket/audio.wav",
        "source_url": "https://example.com/video",
        "video_gcs_uri": "gs://bucket/video.mp4",
        "run_id": "run-1",
    }
    assert captured["timeout"] == pytest.approx(7200.0)

    assert isinstance(response, VibeVoiceAsrResponse)
    assert response.turns[0]["Content"] == "hello"

    # Stage events from the remote service are re-emitted through the logger.
    stage_names = [event["stage_name"] for event in emitted]
    assert stage_names == ["vibevoice_asr"]
    assert emitted[0]["status"] == "succeeded"
    assert emitted[0]["duration_ms"] == 1234.5
    assert emitted[0]["metadata"] == {"turn_count": 1}


def test_run_requires_audio_gcs_uri() -> None:
    client = RemoteVibeVoiceAsrClient(settings=_settings(), max_retries=0)
    with pytest.raises(ValueError, match="audio_gcs_uri"):
        client.run(audio_gcs_uri="")


def test_run_raises_remote_error_on_http_error_and_emits_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(req, timeout):  # noqa: ARG001
        raise HTTPError(
            url=req.full_url,
            code=500,
            msg="boom",
            hdrs=None,
            fp=io.BytesIO(b'{"detail": "cuda oom"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    emitted: list[dict[str, Any]] = []

    def logger_fn(**kwargs: Any) -> None:
        emitted.append(kwargs)

    client = RemoteVibeVoiceAsrClient(settings=_settings(), max_retries=0)
    with pytest.raises(RemoteVibeVoiceAsrError) as excinfo:
        client.run(
            audio_gcs_uri="gs://bucket/audio.wav",
            run_id="run-err",
            stage_event_logger=logger_fn,
        )
    assert excinfo.value.status_code == 500
    assert emitted and emitted[0]["stage_name"] == "vibevoice_asr"
    assert emitted[0]["status"] == "failed"
    assert emitted[0]["error_payload"]["code"] == "RemoteVibeVoiceAsrError"


def test_run_retries_transient_5xx_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_urlopen(req, timeout):  # noqa: ARG001
        calls["n"] += 1
        if calls["n"] == 1:
            raise HTTPError(
                url=req.full_url,
                code=503,
                msg="busy",
                hdrs=None,
                fp=io.BytesIO(b""),
            )
        return _FakeResponse(json.dumps(_ok_payload()).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("backend.providers.audio_host_client.time.sleep", lambda _s: None)

    client = RemoteVibeVoiceAsrClient(settings=_settings(), max_retries=2)
    response = client.run(audio_gcs_uri="gs://bucket/audio.wav", run_id="run-retry")
    assert calls["n"] == 2
    assert response.turns


def test_run_raises_on_invalid_response_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req, timeout):  # noqa: ARG001
        return _FakeResponse(json.dumps({"turns": "not-a-list"}).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RemoteVibeVoiceAsrError, match="invalid vibevoice-asr response shape"):
        RemoteVibeVoiceAsrClient(settings=_settings(), max_retries=0).run(
            audio_gcs_uri="gs://bucket/audio.wav",
        )


def test_healthcheck_returns_parsed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req, timeout):  # noqa: ARG001
        assert req.full_url == "http://rtx-box:9100/health"
        assert req.get_method() == "GET"
        return _FakeResponse(b'{"status": "ready"}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = RemoteVibeVoiceAsrClient(settings=_settings(), max_retries=0)
    assert client.healthcheck() == {"status": "ready"}


def test_supports_concurrent_visual_is_true() -> None:
    client = RemoteVibeVoiceAsrClient(settings=_settings())
    assert client.supports_concurrent_visual is True


def test_deprecated_aliases_still_importable() -> None:
    """The legacy class names must continue to resolve during the deprecation window."""
    from backend.providers.audio_host_client import (
        PhaseOneAudioResponse,
        RemoteAudioChainClient,
        RemoteAudioChainError,
    )
    from backend.providers.config import AudioHostSettings

    assert RemoteAudioChainClient is RemoteVibeVoiceAsrClient
    assert RemoteAudioChainError is RemoteVibeVoiceAsrError
    assert PhaseOneAudioResponse is VibeVoiceAsrResponse
    assert AudioHostSettings is VibeVoiceAsrServiceSettings
