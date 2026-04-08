from __future__ import annotations

import json

import pytest

from backend.providers.vibevoice_vllm import VibeVoiceVLLMProvider


def test_vibevoice_vllm_provider_parses_turn_mode_response() -> None:
    provider = VibeVoiceVLLMProvider(
        base_url="http://127.0.0.1:8000",
        output_mode="turns",
    )
    payload = json.dumps(
        [
            {"Start": 0.0, "End": 1.2, "Speaker": 0, "Content": "Hello world"},
            {"start": 1.2, "end": 2.0, "speaker_id": 1, "text": "Again"},
        ]
    )
    turns = provider._parse_content(payload)

    assert turns == [
        {"Start": 0.0, "End": 1.2, "Speaker": 0, "Content": "Hello world"},
        {"Start": 1.2, "End": 2.0, "Speaker": 1, "Content": "Again"},
    ]


def test_vibevoice_vllm_provider_parses_word_mode_response() -> None:
    provider = VibeVoiceVLLMProvider(
        base_url="http://127.0.0.1:8000",
        output_mode="words",
    )
    payload = json.dumps(
        [
            {"start_ms": 100, "end_ms": 180, "speaker_id": 0, "word": "Hello"},
            {"start": 0.18, "end": 0.32, "speaker": "SPEAKER_0", "text": "world"},
            {"start_ms": 340, "end_ms": 450, "speaker_id": 1, "word": "Again"},
        ]
    )
    words = provider._parse_content(payload)

    assert words == [
        {"start_ms": 100, "end_ms": 180, "speaker_id": 0, "word": "Hello"},
        {"start_ms": 180, "end_ms": 320, "speaker_id": 0, "word": "world"},
        {"start_ms": 340, "end_ms": 450, "speaker_id": 1, "word": "Again"},
    ]


def test_vibevoice_vllm_provider_rejects_unknown_output_mode() -> None:
    with pytest.raises(RuntimeError, match="VIBEVOICE_OUTPUT_MODE"):
        VibeVoiceVLLMProvider(
            base_url="http://127.0.0.1:8000",
            output_mode="segments",
        )
