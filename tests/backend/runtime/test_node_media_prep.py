"""Tests for the remote node-media-prep runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.runtime.node_media_prep import (
    NodeMediaPrepRequest,
    run_node_media_prep,
)


class _FakeStorage:
    def __init__(self) -> None:
        self.downloads: list[tuple[str, Path]] = []
        self.uploads: list[tuple[Path, str]] = []

    def download_file(self, *, gcs_uri: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"video-bytes")
        self.downloads.append((gcs_uri, local_path))
        return local_path

    def upload_file(self, *, local_path: Path, object_name: str) -> str:
        self.uploads.append((local_path, object_name))
        return f"gs://bucket/{object_name}"


def test_from_payload_rejects_non_gcs_uri() -> None:
    with pytest.raises(ValueError, match="gs://"):
        NodeMediaPrepRequest.from_payload(
            {
                "run_id": "r",
                "video_gcs_uri": "http://example.com/video.mp4",
                "nodes": [],
            }
        )


def test_from_payload_rejects_missing_run_id() -> None:
    with pytest.raises(ValueError, match="run_id"):
        NodeMediaPrepRequest.from_payload(
            {
                "run_id": "",
                "video_gcs_uri": "gs://b/v.mp4",
                "nodes": [],
            }
        )


def test_from_payload_rejects_inverted_time() -> None:
    with pytest.raises(ValueError, match="start_ms > end_ms"):
        NodeMediaPrepRequest.from_payload(
            {
                "run_id": "r",
                "video_gcs_uri": "gs://b/v.mp4",
                "nodes": [{"node_id": "n1", "start_ms": 2000, "end_ms": 1000}],
            }
        )


def test_from_payload_defaults_object_prefix() -> None:
    req = NodeMediaPrepRequest.from_payload(
        {
            "run_id": "run_123",
            "video_gcs_uri": "gs://b/v.mp4",
            "nodes": [{"node_id": "n1", "start_ms": 0, "end_ms": 1000}],
        }
    )
    assert req.object_prefix == "phase14/run_123/node_media"
    assert req.max_concurrency == 8


def test_run_node_media_prep_returns_empty_for_empty_nodes(tmp_path: Path) -> None:
    req = NodeMediaPrepRequest.from_payload(
        {
            "run_id": "run_empty",
            "video_gcs_uri": "gs://b/v.mp4",
            "nodes": [],
        }
    )
    result = run_node_media_prep(
        request=req,
        storage_client=_FakeStorage(),
        scratch_root=tmp_path,
    )
    assert result == {"run_id": "run_empty", "media": []}


def test_run_node_media_prep_invokes_ffmpeg_for_each_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def _fake_extract_node_clip(
        *,
        source_video_path: Path,
        output_path: Path,
        start_ms: int,
        end_ms: int,
    ) -> Path:
        calls.append(
            {
                "source_video_path": source_video_path,
                "output_path": output_path,
                "start_ms": start_ms,
                "end_ms": end_ms,
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"clip-bytes")
        return output_path

    monkeypatch.setattr(
        "backend.pipeline.semantics.media_embeddings.extract_node_clip",
        _fake_extract_node_clip,
    )

    storage = _FakeStorage()
    req = NodeMediaPrepRequest.from_payload(
        {
            "run_id": "run_001",
            "video_gcs_uri": "gs://bucket/source.mp4",
            "object_prefix": "phase14/run_001/node_media",
            "max_concurrency": 2,
            "nodes": [
                {"node_id": "n1", "start_ms": 0, "end_ms": 1000},
                {"node_id": "n2", "start_ms": 2000, "end_ms": 3000},
            ],
        }
    )
    result = run_node_media_prep(
        request=req,
        storage_client=storage,
        scratch_root=tmp_path,
    )
    assert result["run_id"] == "run_001"
    media = result["media"]
    assert [m["node_id"] for m in media] == ["n1", "n2"]
    assert all(m["file_uri"].startswith("gs://bucket/phase14/run_001/node_media/") for m in media)
    assert all(m["mime_type"] == "video/mp4" for m in media)
    # local_path should NOT be surfaced to the caller (clips are ephemeral on the media-prep worker).
    assert all("local_path" not in m for m in media)
    assert len(calls) == 2
    assert {c["start_ms"] for c in calls} == {0, 2000}
    assert len(storage.downloads) == 1
    assert storage.downloads[0][0] == "gs://bucket/source.mp4"
    assert len(storage.uploads) == 2
