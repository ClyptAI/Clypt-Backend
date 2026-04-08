from __future__ import annotations

import builtins

import pytest


def test_validate_torchaudio_runtime_rejects_missing_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.providers import audio_runtime

    class _FakeTorchAudio:
        __version__ = "broken"

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "torchaudio":
            return _FakeTorchAudio()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="torchaudio.info"):
        audio_runtime.validate_torchaudio_runtime()
