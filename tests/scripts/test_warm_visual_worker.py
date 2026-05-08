from __future__ import annotations

import sys

from scripts.modal import warm_visual_worker


def test_normalize_visual_base_url_strips_task_path() -> None:
    assert (
        warm_visual_worker._normalize_visual_base_url(
            "https://example.modal.run/tasks/visual-extract"
        )
        == "https://example.modal.run"
    )
    assert (
        warm_visual_worker._normalize_visual_base_url("https://example.modal.run/")
        == "https://example.modal.run"
    )


def test_main_returns_early_when_ready(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def _fake_request_json(*, url, method, auth_token, payload, timeout_s):  # noqa: ARG001
        calls.append((method, url))
        if url.endswith("/ready"):
            return 200, {"status": "ready", "reason": "gpu_worker_warm"}
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(warm_visual_worker, "_request_json", _fake_request_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "warm_visual_worker.py",
            "--service-url",
            "https://example.modal.run/tasks/visual-extract",
            "--auth-token",
            "token",
        ],
    )

    assert warm_visual_worker.main() == 0
    assert calls == [("GET", "https://example.modal.run/ready")]
