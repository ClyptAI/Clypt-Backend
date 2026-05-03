from __future__ import annotations

import json
from email.parser import BytesParser
from email.policy import default
from typing import Any

import pytest

from backend.providers.elevenlabs_scribe import (
    ElevenLabsScribeClient,
    ElevenLabsScribeError,
    ElevenLabsScribeSettings,
    ScribeRequestOptions,
    validate_scribe_response,
)


class _FakeResponse:
    def __init__(self, body: bytes, *, status: int = 200, headers: dict[str, str] | None = None) -> None:
        self._body = body
        self.status = status
        self.headers = headers or {}

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        return None

    def read(self) -> bytes:
        return self._body


def _settings() -> ElevenLabsScribeSettings:
    return ElevenLabsScribeSettings(api_key="scribe-key", max_retries=0)


def _parse_multipart(data: bytes, content_type: str) -> dict[str, str]:
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: " + content_type.encode("ascii") + b"\r\n\r\n" + data
    )
    return {
        part.get_param("name", header="content-disposition"): part.get_content()
        for part in message.iter_parts()
    }


def test_transcribe_posts_required_default_fields_and_omits_optional_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout):  # noqa: ARG001
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = dict(req.header_items())
        captured["body"] = req.data
        body = {
            "language_code": "en",
            "language_probability": 0.99,
            "text": "Hello",
            "words": [
                {
                    "type": "word",
                    "text": "Hello",
                    "start": 0.0,
                    "end": 0.4,
                    "speaker_id": "speaker_0",
                    "unknown_future_field": "preserved",
                }
            ],
            "future_top_level": {"kept": True},
        }
        return _FakeResponse(json.dumps(body).encode("utf-8"), headers={"request-id": "req-1"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = ElevenLabsScribeClient(settings=_settings(), boundary_factory=lambda: "boundary-x")
    result = client.transcribe(source_url="https://storage.googleapis.com/b/audio.wav?sig=abc")

    assert captured["url"] == "https://api.elevenlabs.io/v1/speech-to-text"
    assert captured["method"] == "POST"
    headers = {key.lower(): value for key, value in captured["headers"].items()}
    assert headers["xi-api-key"] == "scribe-key"
    assert headers["accept"] == "application/json"
    fields = _parse_multipart(captured["body"], headers["content-type"])
    assert fields == {
        "model_id": "scribe_v2",
        "source_url": "https://storage.googleapis.com/b/audio.wav?sig=abc",
        "diarize": "true",
        "tag_audio_events": "true",
        "timestamps_granularity": "word",
        "language_code": "en",
        "temperature": "0.0",
    }
    assert "num_speakers" not in fields
    assert "keyterms" not in fields
    assert "seed" not in fields
    assert "cloud_storage_url" not in fields
    assert "entity_detection" not in fields
    assert "redact" not in fields
    assert result.raw["future_top_level"] == {"kept": True}
    assert result.raw["words"][0]["unknown_future_field"] == "preserved"
    assert result.metrics["word_count"] == 1
    assert result.metrics["speaker_count"] == 1
    assert result.metrics["elevenlabs_request_id"] == "req-1"


def test_transcribe_includes_only_supplied_optional_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout):  # noqa: ARG001
        captured["headers"] = dict(req.header_items())
        captured["body"] = req.data
        return _FakeResponse(
            json.dumps(
                {
                    "words": [
                        {
                            "type": "word",
                            "text": "Hi",
                            "start": 0.0,
                            "end": 0.2,
                            "speaker_id": "speaker_0",
                        }
                    ]
                }
            ).encode("utf-8")
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = ElevenLabsScribeClient(settings=_settings(), boundary_factory=lambda: "boundary-y")
    client.transcribe(
        source_url="https://storage.googleapis.com/b/audio.wav?sig=abc",
        options=ScribeRequestOptions(
            num_speakers=2,
            diarization_threshold=0.7,
            keyterms=["Clypt", "Qwen"],
            seed=123,
            file_format="pcm_s16le_16",
        ),
    )

    headers = {key.lower(): value for key, value in captured["headers"].items()}
    fields = _parse_multipart(captured["body"], headers["content-type"])
    assert fields["num_speakers"] == "2"
    assert fields["diarization_threshold"] == "0.7"
    assert json.loads(fields["keyterms"]) == ["Clypt", "Qwen"]
    assert fields["seed"] == "123"
    assert fields["file_format"] == "pcm_s16le_16"


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ([], "object"),
        ({}, "words list"),
        ({"words": []}, "no type=word"),
        ({"words": [{"type": "word", "text": "x", "end": 1.0, "speaker_id": "speaker_0"}]}, "start/end"),
        ({"words": [{"type": "word", "text": "x", "start": 0.0, "end": 1.0}]}, "speaker_id"),
    ],
)
def test_validate_scribe_response_rejects_contract_violations(raw: Any, match: str) -> None:
    with pytest.raises(ElevenLabsScribeError, match=match):
        validate_scribe_response(raw)


def test_validate_scribe_response_preserves_non_word_tags_and_allows_empty_audio() -> None:
    raw = {
        "words": [
            {"type": "audio_event", "text": "(laughs)", "start": 0.1, "end": 0.5},
        ],
        "unexpected": "kept",
    }

    assert validate_scribe_response(raw, require_word_tokens=False) is raw


def test_transcribe_requires_https_signed_source_url() -> None:
    client = ElevenLabsScribeClient(settings=_settings())

    with pytest.raises(ValueError, match="signed HTTPS"):
        client.transcribe(source_url="gs://bucket/audio.wav")
