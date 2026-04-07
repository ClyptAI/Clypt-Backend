from __future__ import annotations

import builtins
import wave
from pathlib import Path

import pytest


def test_probe_requires_torchaudio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from backend.providers.vibevoice import _probe_audio_duration_s

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "torchaudio":
            raise ImportError("torchaudio missing for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="torchaudio is required"):
        _probe_audio_duration_s(tmp_path / "missing.wav")


def _write_minimal_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 200)


def test_probe_short_wav_matches_torchaudio(tmp_path: Path) -> None:
    pytest.importorskip("torchaudio")
    from backend.providers.vibevoice import _probe_audio_duration_s

    p = tmp_path / "t.wav"
    _write_minimal_wav(p)
    d = _probe_audio_duration_s(p)
    assert d is not None
    assert abs(d - 200 / 16000.0) < 0.001


def test_validate_torchaudio_runtime_rejects_missing_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.providers import vibevoice

    class _FakeTorchAudio:
        __version__ = "broken"

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "torchaudio":
            return _FakeTorchAudio()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="torchaudio.info"):
        vibevoice.validate_torchaudio_runtime()
