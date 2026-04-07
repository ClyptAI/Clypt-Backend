from __future__ import annotations

import io
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


class _FakePipe(io.StringIO):
    def __init__(self, initial_value: str = "") -> None:
        super().__init__(initial_value)
        self.captured_value = ""

    def close(self) -> None:
        self.captured_value = self.getvalue()
        super().close()


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

    class FakePopen:
        def __init__(self, cmd: list[str], **kwargs: object) -> None:
            captured["cmd"] = cmd
            captured.update(kwargs)
            payload = {
                "turns": [
                    {"Start": 0.5, "End": 2.0, "Speaker": 1, "Content": "hello"},
                ],
                "error": None,
            }
            self.stdin = _FakePipe()
            self.stdout = io.StringIO(json.dumps(payload))
            self.stderr = io.StringIO("log line\n")
            self.returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            job = json.loads(self.stdin.captured_value or "{}")
            assert job["audio_path"] == str(audio.resolve())
            assert job["model_path"] == "microsoft/VibeVoice-ASR"
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(vv.subprocess, "Popen", FakePopen)
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
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"
    assert captured["env"]["PYTHONIOENCODING"] == "utf-8"


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

    class FakePopen:
        def __init__(self, cmd: list[str], **kwargs: object) -> None:
            self.stdin = _FakePipe()
            self.stdout = io.StringIO(json.dumps({"turns": [], "error": "CUDA OOM"}))
            self.stderr = io.StringIO("")
            self.returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(vv.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(vv, "_probe_audio_duration_s", lambda _: 1.25)

    p = VibeVoiceASRProvider(
        backend="native",
        native_venv_python=str(fake_py),
    )
    p.load()
    with pytest.raises(RuntimeError, match="CUDA OOM"):
        p.run(audio)


def test_native_subprocess_forces_utf8_stream_decoding(
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

    class FakePopen:
        def __init__(self, cmd: list[str], **kwargs: object) -> None:
            captured.update(kwargs)
            payload = {
                "turns": [
                    {"Start": 0.0, "End": 1.0, "Speaker": 0, "Content": "naive cafe — test"},
                ],
                "error": None,
            }
            self.stdin = _FakePipe()
            self.stdout = io.StringIO(json.dumps(payload, ensure_ascii=False))
            self.stderr = io.StringIO("π log line\n")
            self.returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(vv.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(vv, "_probe_audio_duration_s", lambda _: 1.25)

    p = VibeVoiceASRProvider(
        backend="native",
        native_venv_python=str(fake_py),
    )
    p.load()
    turns = p.run(audio)

    assert turns[0]["Content"] == "naive cafe — test"
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"


def test_native_subprocess_streams_stderr_logs_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from backend.providers import vibevoice as vv
    from backend.providers.vibevoice import VibeVoiceASRProvider

    audio = tmp_path / "clip.wav"
    _write_minimal_wav(audio)

    fake_py = tmp_path / "fake_python"
    fake_py.write_text("#!/bin/true\n")
    fake_py.chmod(0o755)

    class FakePopen:
        def __init__(self, cmd: list[str], **kwargs: object) -> None:
            self.stdin = _FakePipe()
            self.stdout = io.StringIO(json.dumps({"turns": [], "error": None}))
            self.stderr = io.StringIO("load complete\nprogress tick\n")
            self.returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(vv.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(vv, "_probe_audio_duration_s", lambda _: 1.25)
    caplog.set_level("INFO")

    p = VibeVoiceASRProvider(
        backend="native",
        native_venv_python=str(fake_py),
    )
    p.load()
    p.run(audio)

    assert "[vibevoice-native] load complete" in caplog.text
    assert "[vibevoice-native] progress tick" in caplog.text
