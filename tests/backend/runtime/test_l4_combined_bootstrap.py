from __future__ import annotations

import pytest


def test_build_vibevoice_start_command_uses_expected_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.runtime.l4_combined_bootstrap import build_vibevoice_start_command

    monkeypatch.delenv("CLYPT_L4_VIBEVOICE_REPO_DIR", raising=False)
    monkeypatch.delenv("CLYPT_L4_VIBEVOICE_MAX_NUM_SEQS", raising=False)
    monkeypatch.delenv("CLYPT_L4_VIBEVOICE_MAX_MODEL_LEN", raising=False)
    monkeypatch.delenv("CLYPT_L4_VIBEVOICE_GPU_MEMORY_UTILIZATION", raising=False)

    command = build_vibevoice_start_command()

    assert command == [
        "python3",
        "/app/vllm_plugin/scripts/start_server.py",
        "--skip-deps",
        "--max-num-seqs",
        "4",
        "--max-model-len",
        "16384",
        "--gpu-memory-utilization",
        "0.90",
    ]


def test_wait_for_vibevoice_health_retries_until_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.runtime.l4_combined_bootstrap import wait_for_vibevoice_health

    attempts = {"count": 0}
    slept: list[float] = []

    class _FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise OSError("connection refused")
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda seconds: slept.append(seconds))

    wait_for_vibevoice_health(
        base_url="http://127.0.0.1:8000",
        healthcheck_path="/health",
        timeout_s=30.0,
        poll_interval_s=1.0,
    )

    assert attempts["count"] == 3
    assert slept == [1.0, 1.0]
