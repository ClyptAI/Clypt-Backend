from __future__ import annotations

import json
from typing import Any

import pytest

from backend.providers.config import Phase1VisualServiceSettings
from backend.providers.visual_extract_client import RemoteVisualExtractClient


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_visual_extract_submit_and_poll_use_modal_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, Any]] = []

    def fake_urlopen(req, timeout):  # noqa: ARG001
        captured.append(
            {
                "url": req.full_url,
                "method": req.get_method(),
                "headers": dict(req.header_items()),
                "body": json.loads(req.data.decode("utf-8")) if req.data else None,
            }
        )
        if req.get_method() == "POST":
            return _FakeResponse(json.dumps({"call_id": "fc-visual"}).encode("utf-8"), status=202)
        return _FakeResponse(
            json.dumps(
                {
                    "status": "succeeded",
                    "phase1_visual": {"shot_changes": [], "tracks": []},
                }
            ).encode("utf-8")
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = RemoteVisualExtractClient(
        settings=Phase1VisualServiceSettings(
            service_url="https://modal.visual/tasks/visual-extract",
            auth_token="visual-token",
        ),
        max_retries=0,
    )
    future = client.submit(
        run_id="run-1",
        video_gcs_uri="gs://bucket/source.mp4",
        source_video_sha256="sha256:abc",
    )
    result = client.wait_for_result(visual_future=future)

    assert captured[0]["url"] == "https://modal.visual/tasks/visual-extract"
    assert captured[0]["method"] == "POST"
    assert captured[0]["body"]["run_id"] == "run-1"
    assert captured[0]["body"]["video_gcs_uri"] == "gs://bucket/source.mp4"
    assert {k.lower(): v for k, v in captured[0]["headers"].items()}["authorization"] == "Bearer visual-token"
    assert captured[1]["url"] == "https://modal.visual/tasks/visual-extract/result/fc-visual"
    assert future.call_id == "fc-visual"
    assert future.source_video_sha256 == "sha256:abc"
    assert result["phase1_visual"] == {"shot_changes": [], "tracks": []}
