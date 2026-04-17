"""Unit tests for the Phase 1 remote audio host client.

The Phase 1 audio chain lives exclusively on the RTX 6000 Ada box. These tests
stub out the underlying HTTP call at ``urllib.request.urlopen`` so the client's
retry/parse/re-emit logic can be verified without any real network.
"""

from __future__ import annotations

import io
import json
from typing import Any
from urllib.error import HTTPError

import pytest

from backend.providers.audio_host_client import (
    PhaseOneAudioResponse,
    RemoteAudioChainClient,
    RemoteAudioChainError,
)
from backend.providers.config import AudioHostSettings


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


def _settings() -> AudioHostSettings:
    return AudioHostSettings(
        service_url="http://rtx-box:9100",
        auth_token="secret-token",
    )


def _ok_payload(stage_events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "run_id": "run-1",
        "turns": [
            {"Speaker": 0, "Start": 0.0, "End": 1.0, "Content": "hello"},
        ],
        "diarization_payload": {"turns": [{"turn_id": "t_000001"}], "words": []},
        "emotion2vec_payload": {"segments": [{"turn_id": "t_000001"}]},
        "yamnet_payload": {"events": [{"start_ms": 0, "end_ms": 500}]},
        "stage_events": stage_events
        or [
            {
                "stage_name": "vibevoice_asr",
                "status": "succeeded",
                "duration_ms": 1234.5,
                "metadata": {"turn_count": 1},
            },
            {
                "stage_name": "forced_alignment",
                "status": "succeeded",
                "duration_ms": 500.0,
                "metadata": {"word_count": 3},
            },
            {
                "stage_name": "emotion2vec",
                "status": "succeeded",
                "duration_ms": 250.0,
                "metadata": {"segment_count": 1},
            },
            {
                "stage_name": "yamnet",
                "status": "succeeded",
                "duration_ms": 150.0,
                "metadata": {"event_count": 1},
            },
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

    client = RemoteAudioChainClient(settings=_settings(), max_retries=0)
    response = client.run(
        audio_gcs_uri="gs://bucket/audio.wav",
        source_url="https://example.com/video",
        video_gcs_uri="gs://bucket/video.mp4",
        run_id="run-1",
        stage_event_logger=logger_fn,
    )

    assert captured["url"] == "http://rtx-box:9100/tasks/phase1-audio"
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

    assert isinstance(response, PhaseOneAudioResponse)
    assert response.turns[0]["Content"] == "hello"
    assert response.diarization_payload["turns"][0]["turn_id"] == "t_000001"
    assert response.emotion2vec_payload["segments"][0]["turn_id"] == "t_000001"
    assert response.yamnet_payload["events"][0]["end_ms"] == 500

    stage_names = [event["stage_name"] for event in emitted]
    assert stage_names == ["vibevoice_asr", "forced_alignment", "emotion2vec", "yamnet"]
    assert all(event["status"] == "succeeded" for event in emitted)
    assert emitted[0]["duration_ms"] == 1234.5
    assert emitted[1]["metadata"] == {"word_count": 3}


def test_run_requires_audio_gcs_uri() -> None:
    client = RemoteAudioChainClient(settings=_settings(), max_retries=0)
    with pytest.raises(ValueError, match="audio_gcs_uri"):
        client.run(audio_gcs_uri="")


def test_run_raises_remote_audio_chain_error_on_http_error_and_emits_failure(
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

    client = RemoteAudioChainClient(settings=_settings(), max_retries=0)
    with pytest.raises(RemoteAudioChainError) as excinfo:
        client.run(
            audio_gcs_uri="gs://bucket/audio.wav",
            run_id="run-err",
            stage_event_logger=logger_fn,
        )
    assert excinfo.value.status_code == 500
    assert emitted and emitted[0]["stage_name"] == "audio_host_call"
    assert emitted[0]["status"] == "failed"
    assert emitted[0]["error_payload"]["code"] == "RemoteAudioChainError"


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

    client = RemoteAudioChainClient(settings=_settings(), max_retries=2)
    response = client.run(audio_gcs_uri="gs://bucket/audio.wav", run_id="run-retry")
    assert calls["n"] == 2
    assert response.turns


def test_run_raises_on_invalid_response_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req, timeout):  # noqa: ARG001
        return _FakeResponse(json.dumps({"turns": "not-a-list"}).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RemoteAudioChainError, match="invalid audio host response shape"):
        RemoteAudioChainClient(settings=_settings(), max_retries=0).run(
            audio_gcs_uri="gs://bucket/audio.wav",
        )


def test_healthcheck_returns_parsed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req, timeout):  # noqa: ARG001
        assert req.full_url == "http://rtx-box:9100/health"
        assert req.get_method() == "GET"
        return _FakeResponse(b'{"status": "ready"}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = RemoteAudioChainClient(settings=_settings(), max_retries=0)
    assert client.healthcheck() == {"status": "ready"}


def test_supports_concurrent_visual_is_true() -> None:
    client = RemoteAudioChainClient(settings=_settings())
    assert client.supports_concurrent_visual is True
