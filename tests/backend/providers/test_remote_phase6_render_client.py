"""Unit tests for the remote Phase 6 render/export client."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.phase1_runtime.payloads import Phase1AudioAssets
from backend.providers.config import Phase6RenderSettings, StorageSettings
from backend.providers.phase6_render_client import RemotePhase6RenderClient
from backend.providers.storage import GCSStorageClient


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


class _FakeStorageClient(GCSStorageClient):
    def __init__(self) -> None:
        super().__init__(settings=StorageSettings(gcs_bucket="bucket"), storage_client=object())
        self.uploads: list[tuple[str, str]] = []

    def upload_file(self, *, local_path: Path, object_name: str) -> str:
        self.uploads.append((str(local_path), object_name))
        return f"gs://bucket/{object_name}"


@dataclass
class _StubPaths:
    run_id: str
    render_plan: Path


@dataclass
class _StubPhase1Outputs:
    phase1_audio: Any


def _settings() -> Phase6RenderSettings:
    return Phase6RenderSettings(
        service_url="http://render-box:9100",
        auth_token="render-token",
    )


def _phase1_outputs() -> _StubPhase1Outputs:
    return _StubPhase1Outputs(
        phase1_audio=Phase1AudioAssets(
            video_gcs_uri="gs://bucket/source.mp4",
            local_video_path="/tmp/source.mp4",
        )
    )


def test_render_client_uploads_artifacts_submits_and_returns_outputs(tmp_path, monkeypatch) -> None:
    render_plan = tmp_path / "render_plan.json"
    render_plan.write_text(
        json.dumps(
            {
                "run_id": "run-abc",
                "source_context_ref": "source_context.json",
                "caption_plan_ref": "caption_plan.json",
                "publish_metadata_ref": "publish_metadata.json",
                "clips": [
                    {
                        "clip_id": "clip_001",
                        "clip_start_ms": 0,
                        "clip_end_ms": 1200,
                        "caption_plan_ref": "caption_plan.json",
                        "publish_metadata_ref": "publish_metadata.json",
                        "caption_segment_ids": ["clip_001_seg_001"],
                        "caption_zone": "center_band",
                        "caption_preset_id": "karaoke_focus",
                        "review_needed": False,
                        "review_reasons": [],
                        "overlays": [],
                        "segments": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    source_context = tmp_path / "source_context.json"
    source_context.write_text('{"source_url":"https://example.com/video"}', encoding="utf-8")
    caption_plan = tmp_path / "caption_plan.json"
    caption_plan.write_text('{"run_id":"run-abc","clips":[]}', encoding="utf-8")
    publish_metadata = tmp_path / "publish_metadata.json"
    publish_metadata.write_text('{"run_id":"run-abc","clips":[]}', encoding="utf-8")
    captions_ass = tmp_path / "captions_clip_001.ass"
    captions_ass.write_text("[Script Info]\n", encoding="utf-8")

    captured: dict[str, Any] = {"requests": []}

    def fake_urlopen(req, timeout):  # noqa: ARG001
        call = {"url": req.full_url, "method": req.get_method()}
        if req.data is not None:
            call["body"] = json.loads(req.data.decode("utf-8"))
        captured["requests"].append(call)
        if req.get_method() == "POST":
            return _FakeResponse(json.dumps({"call_id": "fc-render"}).encode("utf-8"), status=202)
        return _FakeResponse(
            json.dumps(
                {
                    "status": "succeeded",
                    "outputs": [
                        {
                            "clip_id": "clip_001",
                            "video_gcs_uri": "gs://bucket/phase14/run-abc/render_outputs/clip_001.mp4",
                        }
                    ],
                }
            ).encode("utf-8")
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = RemotePhase6RenderClient(
        settings=_settings(),
        storage_client=_FakeStorageClient(),
        max_retries=0,
    )
    result = client(
        paths=_StubPaths(run_id="run-abc", render_plan=render_plan),
        phase1_outputs=_phase1_outputs(),
        artifact_paths={
            "source_context": str(source_context),
            "caption_plan": str(caption_plan),
            "publish_metadata": str(publish_metadata),
            "render_plan": str(render_plan),
            "captions_clip_001.ass": str(captions_ass),
        },
    )

    submit = captured["requests"][0]
    assert submit["url"] == "http://render-box:9100/tasks/render-video"
    assert submit["body"]["run_id"] == "run-abc"
    assert submit["body"]["source_video_gcs_uri"] == "gs://bucket/source.mp4"
    assert submit["body"]["artifact_gcs_uris"]["render_plan"] == "gs://bucket/phase14/run-abc/render_inputs/render_plan.json"
    assert submit["body"]["artifact_gcs_uris"]["captions_clip_001.ass"] == "gs://bucket/phase14/run-abc/render_inputs/captions_clip_001.ass"
    assert submit["body"]["clips"][0]["clip_id"] == "clip_001"
    assert result["outputs"][0]["clip_id"] == "clip_001"
    assert result["outputs"][0]["video_gcs_uri"].endswith("clip_001.mp4")

