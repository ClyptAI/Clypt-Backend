"""Unit tests for the RTX-side Phase 1 audio chain.

``run_audio_chain`` is the pure-Python glue that runs VibeVoice ASR → NeMo
Forced Aligner → emotion2vec+ → YAMNet serially on the RTX 6000 Ada host. It
used to live in ``backend.phase1_runtime.extract`` as ``_run_audio_chain``;
these tests are the successor to the three ``test_run_phase1_sidecars_*``
tests that previously covered the same logic in-process on the H200.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Fake providers mirror the ones the old extract.py tests used. Each records
# the call order + the arguments so we can assert the chain orders stages
# correctly and passes through the right intermediate data shapes.
# ---------------------------------------------------------------------------


class _FakeVibeVoice:
    def __init__(self, turns: list[dict[str, Any]] | None = None) -> None:
        self._turns = turns or [
            {"Start": 0.0, "End": 0.3, "Speaker": 0, "Content": "hello"},
        ]
        self.calls: list[dict[str, Any]] = []

    def run(self, *, audio_path: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append({"audio_path": audio_path, **kwargs})
        return list(self._turns)


class _FakeForcedAligner:
    def __init__(self, words: list[dict[str, Any]] | None = None) -> None:
        self._words = (
            words
            if words is not None
            else [
                {
                    "word_id": "w_000001",
                    "text": "hello",
                    "start_ms": 0,
                    "end_ms": 300,
                    "speaker_id": "SPEAKER_0",
                }
            ]
        )
        self.calls: list[dict[str, Any]] = []

    def run(self, *, audio_path: str, turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.calls.append({"audio_path": audio_path, "turn_count": len(turns)})
        return list(self._words)


class _FakeEmotion:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, *, audio_path: str, turns: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls.append({"audio_path": audio_path, "turn_count": len(turns)})
        first_turn_id = turns[0]["turn_id"] if turns else "t_000001"
        return {
            "segments": [
                {
                    "turn_id": first_turn_id,
                    "labels": ["neutral"],
                    "scores": [0.88],
                    "per_class_scores": {"neutral": 0.88},
                }
            ]
        }


class _FakeYamnet:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, *, audio_path: str) -> dict[str, Any]:
        self.calls.append({"audio_path": audio_path})
        return {"events": []}


def test_run_audio_chain_runs_stages_in_order_and_returns_merged_payload(tmp_path: Path):
    from backend.runtime.phase1_audio_service.audio_chain import run_audio_chain

    audio_path = tmp_path / "source_audio.wav"
    audio_path.write_text("audio-bytes", encoding="utf-8")

    call_order: list[str] = []
    stage_events: list[dict[str, Any]] = []

    vibevoice = _FakeVibeVoice()
    forced_aligner = _FakeForcedAligner()
    emotion = _FakeEmotion()
    yamnet = _FakeYamnet()

    # Wrap provider methods to record cross-stage ordering.
    orig_vv = vibevoice.run
    orig_fa = forced_aligner.run
    orig_em = emotion.run
    orig_yn = yamnet.run

    def _vv(**kwargs):
        call_order.append("vibevoice")
        return orig_vv(**kwargs)

    def _fa(**kwargs):
        call_order.append("forced_aligner")
        return orig_fa(**kwargs)

    def _em(**kwargs):
        call_order.append("emotion")
        return orig_em(**kwargs)

    def _yn(**kwargs):
        call_order.append("yamnet")
        return orig_yn(**kwargs)

    vibevoice.run = _vv  # type: ignore[method-assign]
    forced_aligner.run = _fa  # type: ignore[method-assign]
    emotion.run = _em  # type: ignore[method-assign]
    yamnet.run = _yn  # type: ignore[method-assign]

    result = run_audio_chain(
        audio_path=audio_path,
        vibevoice_provider=vibevoice,
        forced_aligner=forced_aligner,
        emotion_provider=emotion,
        yamnet_provider=yamnet,
        stage_event_recorder=lambda **event: stage_events.append(event),
    )

    # Stages fire in the canonical VibeVoice → NFA → emotion2vec+ → YAMNet order.
    assert call_order == ["vibevoice", "forced_aligner", "emotion", "yamnet"]

    # Each provider was invoked with the audio file path.
    audio_path_str = str(audio_path)
    assert vibevoice.calls[0]["audio_path"] == audio_path_str
    assert forced_aligner.calls[0]["audio_path"] == audio_path_str
    assert emotion.calls[0]["audio_path"] == audio_path_str
    assert yamnet.calls[0]["audio_path"] == audio_path_str

    # Merged payload preserves ASR turns, diarization turns/words, emotion
    # segments and YAMNet events in the documented response shape.
    assert result["turns"][0]["Content"] == "hello"
    assert result["diarization_payload"]["turns"][0]["speaker_id"] == "SPEAKER_0"
    assert result["diarization_payload"]["words"][0]["text"] == "hello"
    assert result["emotion2vec_payload"]["segments"][0]["labels"] == ["neutral"]
    assert result["yamnet_payload"]["events"] == []

    # Stage events trail covers all four audio stages at status=succeeded.
    names = [e["stage_name"] for e in stage_events]
    assert names == ["vibevoice_asr", "forced_alignment", "emotion2vec", "yamnet"]
    assert all(e["status"] == "succeeded" for e in stage_events)


def test_run_audio_chain_fails_when_forced_alignment_returns_zero_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from backend.runtime.phase1_audio_service.audio_chain import run_audio_chain

    monkeypatch.delenv("CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT", raising=False)

    audio_path = tmp_path / "source_audio.wav"
    audio_path.write_text("audio-bytes", encoding="utf-8")

    with pytest.raises(RuntimeError, match="forced-alignment produced 0 words"):
        run_audio_chain(
            audio_path=audio_path,
            vibevoice_provider=_FakeVibeVoice(),
            forced_aligner=_FakeForcedAligner(words=[]),
            emotion_provider=_FakeEmotion(),
            yamnet_provider=_FakeYamnet(),
        )


def test_run_audio_chain_can_bypass_zero_words_with_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from backend.runtime.phase1_audio_service.audio_chain import run_audio_chain

    monkeypatch.setenv("CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT", "0")

    audio_path = tmp_path / "source_audio.wav"
    audio_path.write_text("audio-bytes", encoding="utf-8")

    result = run_audio_chain(
        audio_path=audio_path,
        vibevoice_provider=_FakeVibeVoice(),
        forced_aligner=_FakeForcedAligner(words=[]),
        emotion_provider=_FakeEmotion(),
        yamnet_provider=_FakeYamnet(),
    )

    # When forced alignment is bypassed, vibevoice_merge synthesizes fallback
    # word entries from the ASR turn content so downstream Phase 2-4 still has
    # word-level granularity.
    words = result["diarization_payload"]["words"]
    assert words
    assert words[0]["text"] == "hello"


def test_run_audio_chain_forwards_audio_gcs_uri_to_vibevoice_when_supported(tmp_path: Path):
    """When audio_gcs_uri is supplied, it is forwarded to VibeVoice's kwargs."""
    from backend.runtime.phase1_audio_service.audio_chain import run_audio_chain

    audio_path = tmp_path / "source_audio.wav"
    audio_path.write_text("audio-bytes", encoding="utf-8")

    class _VibeVoiceWithUri:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def run(self, *, audio_path: str, audio_gcs_uri: str | None = None) -> list[dict[str, Any]]:
            self.calls.append({"audio_path": audio_path, "audio_gcs_uri": audio_gcs_uri})
            return [{"Start": 0.0, "End": 0.3, "Speaker": 0, "Content": "hello"}]

    vibevoice = _VibeVoiceWithUri()

    result = run_audio_chain(
        audio_path=audio_path,
        vibevoice_provider=vibevoice,
        forced_aligner=_FakeForcedAligner(),
        emotion_provider=_FakeEmotion(),
        yamnet_provider=_FakeYamnet(),
        audio_gcs_uri="gs://bucket/source.wav",
    )

    assert len(vibevoice.calls) == 1
    assert vibevoice.calls[0]["audio_gcs_uri"] == "gs://bucket/source.wav"
    assert result["turns"][0]["Content"] == "hello"


def test_run_audio_chain_falls_back_when_vibevoice_rejects_audio_gcs_uri_kwarg(tmp_path: Path):
    """Older VibeVoice providers that don't accept audio_gcs_uri should still work."""
    from backend.runtime.phase1_audio_service.audio_chain import run_audio_chain

    audio_path = tmp_path / "source_audio.wav"
    audio_path.write_text("audio-bytes", encoding="utf-8")

    class _VibeVoiceNoUri:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def run(self, *, audio_path: str) -> list[dict[str, Any]]:
            self.calls.append({"audio_path": audio_path})
            return [{"Start": 0.0, "End": 0.3, "Speaker": 0, "Content": "hello"}]

    vibevoice = _VibeVoiceNoUri()

    result = run_audio_chain(
        audio_path=audio_path,
        vibevoice_provider=vibevoice,
        forced_aligner=_FakeForcedAligner(),
        emotion_provider=_FakeEmotion(),
        yamnet_provider=_FakeYamnet(),
        audio_gcs_uri="gs://bucket/source.wav",
    )

    # The chain retried without the audio_gcs_uri kwarg and succeeded.
    assert len(vibevoice.calls) == 1
    assert result["turns"][0]["Content"] == "hello"
