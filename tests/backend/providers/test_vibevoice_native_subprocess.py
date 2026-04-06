from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

import pytest


def _write_minimal_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 200)


def test_native_load_requires_venv_python() -> None:
    from backend.providers.vibevoice import VibeVoiceASRProvider

    p = VibeVoiceASRProvider(backend="native", native_venv_python=None)
    with pytest.raises(RuntimeError, match="VIBEVOICE_NATIVE_VENV_PYTHON"):
        p.load()


def test_native_subprocess_parses_stdout_and_normalizes_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.providers import vibevoice as vv
    from backend.providers.vibevoice import VibeVoiceASRProvider

    audio = tmp_path / "clip.wav"
    _write_minimal_wav(audio)

    fake_py = tmp_path / "fake_python"
    fake_py.write_text("#!/bin/true\n")
    fake_py.chmod(0o755)

    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        job = json.loads(str(kwargs.get("input") or "{}"))
        assert job["audio_path"] == str(audio.resolve())
        assert job["model_path"] == "microsoft/VibeVoice-ASR"
        payload = {
            "turns": [
                {"Start": 0.5, "End": 2.0, "Speaker": 1, "Content": "hello"},
            ],
            "error": None,
        }
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(payload), stderr="log line\n",
        )

    monkeypatch.setattr(vv.subprocess, "run", fake_run)
    monkeypatch.setattr(vv, "_probe_audio_duration_s", lambda _: 1.25)

    p = VibeVoiceASRProvider(
        backend="native",
        native_venv_python=str(fake_py),
        model_id="microsoft/VibeVoice-ASR",
    )
    p.load()
    turns = p.run(audio)

    assert len(turns) == 1
    assert turns[0]["Start"] == 0.5
    assert turns[0]["End"] == 2.0
    assert turns[0]["Speaker"] == 1
    assert turns[0]["Content"] == "hello"
    assert captured["cmd"] == [str(fake_py), "-m", "backend.runtime.vibevoice_native_worker"]


def test_native_subprocess_raises_on_worker_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.providers import vibevoice as vv
    from backend.providers.vibevoice import VibeVoiceASRProvider

    audio = tmp_path / "clip.wav"
    _write_minimal_wav(audio)

    fake_py = tmp_path / "fake_python"
    fake_py.write_text("#!/bin/true\n")
    fake_py.chmod(0o755)

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({"turns": [], "error": "CUDA OOM"}),
            stderr="",
        )

    monkeypatch.setattr(vv.subprocess, "run", fake_run)
    monkeypatch.setattr(vv, "_probe_audio_duration_s", lambda _: 1.25)

    p = VibeVoiceASRProvider(
        backend="native",
        native_venv_python=str(fake_py),
    )
    p.load()
    with pytest.raises(RuntimeError, match="CUDA OOM"):
        p.run(audio)
