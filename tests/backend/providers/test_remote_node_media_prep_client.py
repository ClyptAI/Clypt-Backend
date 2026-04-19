"""Unit tests for the Phase 2 remote node-media-prep client.

Phase 2 node-clip extraction runs exclusively on the remote media-prep worker.
These tests stub the HTTP layer so the client's shape,
retries, and error handling can be verified without a live endpoint.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError

import pytest

from backend.phase1_runtime.payloads import Phase1AudioAssets
from backend.providers.config import NodeMediaPrepSettings
from backend.providers.node_media_prep_client import (
    RemoteNodeMediaPrepClient,
    RemoteNodeMediaPrepError,
)


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        return None

    def read(self) -> bytes:
        return self._body


@dataclass
class _StubNode:
    node_id: str
    start_ms: int
    end_ms: int


@dataclass
class _StubPaths:
    run_id: str


@dataclass
class _StubPhase1Outputs:
    phase1_audio: Any


@dataclass
class _AttrOnlyPhase1Audio:
    video_gcs_uri: str
    audio_gcs_uri: str | None = None
    local_video_path: str | None = None


def _settings() -> NodeMediaPrepSettings:
    return NodeMediaPrepSettings(
        service_url="http://rtx-box:9100",
        auth_token="prep-token",
    )


def _phase1_outputs() -> _StubPhase1Outputs:
    return _StubPhase1Outputs(
        phase1_audio={
            "video_gcs_uri": "gs://bucket/video.mp4",
            "audio_gcs_uri": "gs://bucket/audio.wav",
            "local_video_path": "/tmp/source.mp4",
        }
    )


def test_call_posts_expected_body_and_returns_ordered_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout):  # noqa: ARG001
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["headers"] = dict(req.header_items())
        server_response = {
            "media": [
                # Intentionally out-of-order so we prove client reorders.
                {"node_id": "n_2", "file_uri": "gs://bucket/p/n_2.mp4"},
                {"node_id": "n_1", "file_uri": "gs://bucket/p/n_1.mp4"},
            ]
        }
        return _FakeResponse(json.dumps(server_response).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = RemoteNodeMediaPrepClient(settings=_settings(), max_retries=0)
    result = client(
        nodes=[
            _StubNode(node_id="n_1", start_ms=0, end_ms=1000),
            _StubNode(node_id="n_2", start_ms=1000, end_ms=2500),
        ],
        paths=_StubPaths(run_id="run-abc"),
        phase1_outputs=_phase1_outputs(),
    )

    assert captured["url"] == "http://rtx-box:9100/tasks/node-media-prep"
    assert captured["method"] == "POST"
    assert captured["body"] == {
        "run_id": "run-abc",
        "video_gcs_uri": "gs://bucket/video.mp4",
        "object_prefix": "phase14/run-abc/node_media",
        "max_concurrency": 16,
        "nodes": [
            {"node_id": "n_1", "start_ms": 0, "end_ms": 1000},
            {"node_id": "n_2", "start_ms": 1000, "end_ms": 2500},
        ],
    }
    header_dict = {k.lower(): v for k, v in captured["headers"].items()}
    assert header_dict["authorization"] == "Bearer prep-token"

    assert [item["node_id"] for item in result] == ["n_1", "n_2"]
    assert result[0]["file_uri"] == "gs://bucket/p/n_1.mp4"
    assert result[0]["mime_type"] == "video/mp4"
    assert result[0]["local_path"] == ""


def test_call_with_empty_nodes_returns_empty_list() -> None:
    client = RemoteNodeMediaPrepClient(settings=_settings())
    assert client(nodes=[], paths=_StubPaths(run_id="r"), phase1_outputs=_phase1_outputs()) == []


def test_call_requires_video_gcs_uri() -> None:
    client = RemoteNodeMediaPrepClient(settings=_settings())
    outputs = _StubPhase1Outputs(phase1_audio={"local_video_path": "/tmp/x.mp4"})
    with pytest.raises(ValueError, match="video_gcs_uri"):
        client(
            nodes=[_StubNode("n_1", 0, 100)],
            paths=_StubPaths(run_id="r"),
            phase1_outputs=outputs,
        )


def test_call_accepts_phase1_audio_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req, timeout):  # noqa: ARG001
        return _FakeResponse(
            json.dumps({"media": [{"node_id": "n_1", "file_uri": "gs://bucket/p/n_1.mp4"}]}).encode(
                "utf-8"
            )
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = RemoteNodeMediaPrepClient(settings=_settings(), max_retries=0)
    outputs = _StubPhase1Outputs(
        phase1_audio=Phase1AudioAssets(
            video_gcs_uri="gs://bucket/video.mp4",
            audio_gcs_uri="gs://bucket/audio.wav",
            local_video_path="/tmp/source.mp4",
        )
    )
    result = client(
        nodes=[_StubNode("n_1", 0, 100)],
        paths=_StubPaths(run_id="run-model"),
        phase1_outputs=outputs,
    )

    assert result[0]["file_uri"] == "gs://bucket/p/n_1.mp4"


def test_call_accepts_phase1_audio_attr_object(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req, timeout):  # noqa: ARG001
        return _FakeResponse(
            json.dumps({"media": [{"node_id": "n_1", "file_uri": "gs://bucket/p/n_1.mp4"}]}).encode(
                "utf-8"
            )
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = RemoteNodeMediaPrepClient(settings=_settings(), max_retries=0)
    outputs = _StubPhase1Outputs(
        phase1_audio=_AttrOnlyPhase1Audio(
            video_gcs_uri="gs://bucket/video.mp4",
            audio_gcs_uri="gs://bucket/audio.wav",
            local_video_path="/tmp/source.mp4",
        )
    )
    result = client(
        nodes=[_StubNode("n_1", 0, 100)],
        paths=_StubPaths(run_id="run-attr"),
        phase1_outputs=outputs,
    )

    assert result[0]["file_uri"] == "gs://bucket/p/n_1.mp4"


def test_call_raises_when_server_missing_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req, timeout):  # noqa: ARG001
        return _FakeResponse(
            json.dumps({"media": [{"node_id": "n_1", "file_uri": "gs://b/n_1.mp4"}]}).encode(
                "utf-8"
            )
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = RemoteNodeMediaPrepClient(settings=_settings(), max_retries=0)
    with pytest.raises(RemoteNodeMediaPrepError, match="missing media for nodes"):
        client(
            nodes=[_StubNode("n_1", 0, 100), _StubNode("n_2", 100, 200)],
            paths=_StubPaths(run_id="r"),
            phase1_outputs=_phase1_outputs(),
        )


def test_call_validates_gs_file_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req, timeout):  # noqa: ARG001
        return _FakeResponse(
            json.dumps({"media": [{"node_id": "n_1", "file_uri": "https://bad"}]}).encode(
                "utf-8"
            )
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = RemoteNodeMediaPrepClient(settings=_settings(), max_retries=0)
    with pytest.raises(RemoteNodeMediaPrepError, match="file_uri must be a gs://"):
        client(
            nodes=[_StubNode("n_1", 0, 100)],
            paths=_StubPaths(run_id="r"),
            phase1_outputs=_phase1_outputs(),
        )


def test_call_retries_transient_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_urlopen(req, timeout):  # noqa: ARG001
        calls["n"] += 1
        if calls["n"] == 1:
            raise HTTPError(
                url=req.full_url,
                code=502,
                msg="bad gateway",
                hdrs=None,
                fp=io.BytesIO(b""),
            )
        return _FakeResponse(
            json.dumps({"media": [{"node_id": "n_1", "file_uri": "gs://b/n_1.mp4"}]}).encode(
                "utf-8"
            )
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "backend.providers.node_media_prep_client.time.sleep", lambda _s: None
    )

    client = RemoteNodeMediaPrepClient(settings=_settings(), max_retries=2)
    result = client(
        nodes=[_StubNode("n_1", 0, 100)],
        paths=_StubPaths(run_id="r"),
        phase1_outputs=_phase1_outputs(),
    )
    assert calls["n"] == 2
    assert result[0]["file_uri"] == "gs://b/n_1.mp4"


def test_call_rejects_bad_node_shape() -> None:
    client = RemoteNodeMediaPrepClient(settings=_settings())
    with pytest.raises(ValueError, match="non-numeric"):
        client(
            nodes=[_StubNode("n_1", "abc", 100)],  # type: ignore[arg-type]
            paths=_StubPaths(run_id="r"),
            phase1_outputs=_phase1_outputs(),
        )


def test_call_rejects_inverted_timestamps() -> None:
    client = RemoteNodeMediaPrepClient(settings=_settings())
    with pytest.raises(ValueError, match="start_ms > end_ms"):
        client(
            nodes=[_StubNode("n_1", 500, 100)],
            paths=_StubPaths(run_id="r"),
            phase1_outputs=_phase1_outputs(),
        )
