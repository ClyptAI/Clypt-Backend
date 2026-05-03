from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def test_gcs_storage_client_uploads_with_expected_blob_name(tmp_path: Path):
    from backend.providers.config import StorageSettings
    from backend.providers.storage import GCSStorageClient

    uploaded: dict[str, str] = {}

    class _FakeBlob:
        def __init__(self, name: str):
            self.name = name

        def upload_from_filename(self, filename: str) -> None:
            uploaded["name"] = self.name
            uploaded["filename"] = filename

    class _FakeBucket:
        def blob(self, name: str):
            return _FakeBlob(name)

    class _FakeStorageClient:
        def bucket(self, bucket_name: str):
            uploaded["bucket"] = bucket_name
            return _FakeBucket()

    local_file = tmp_path / "video.mp4"
    local_file.write_text("video", encoding="utf-8")

    client = GCSStorageClient(
        settings=StorageSettings(gcs_bucket="clypt-bucket"),
        storage_client=_FakeStorageClient(),
    )
    uri = client.upload_file(local_path=local_file, object_name="runs/run_1/source_video.mp4")

    assert uri == "gs://clypt-bucket/runs/run_1/source_video.mp4"
    assert uploaded == {
        "bucket": "clypt-bucket",
        "name": "runs/run_1/source_video.mp4",
        "filename": str(local_file),
    }


class _FakeScribeProvider:
    def __init__(self, *, response: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.response = response or {
            "language_code": "en",
            "words": [
                {
                    "type": "word",
                    "text": "hello",
                    "start": 0.0,
                    "end": 0.3,
                    "speaker_id": "speaker_0",
                }
            ],
        }
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeVisualExtractor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def submit(self, **kwargs: Any):
        from backend.phase1_runtime.payloads import VisualFuturePayload

        self.calls.append(kwargs)
        return VisualFuturePayload(
            call_id="visual-call-1",
            service_url="https://modal.example.test",
            source_video_gcs_uri=kwargs["video_gcs_uri"],
        )


def test_run_phase1_sidecars_submits_visual_future_then_adapts_scribe(
    tmp_path: Path,
) -> None:
    from backend.phase1_runtime.extract import run_phase1_sidecars
    from backend.phase1_runtime.models import Phase1Workspace

    stage_events: list[dict[str, object]] = []
    handoffs: list[object] = []
    workspace = Phase1Workspace.create(root=tmp_path, run_id="run_001")
    workspace.video_path.write_text("video-bytes", encoding="utf-8")
    workspace.audio_path.write_text("audio-bytes", encoding="utf-8")
    scribe = _FakeScribeProvider()
    visual = _FakeVisualExtractor()

    outputs = run_phase1_sidecars(
        source_url="https://youtube.com/watch?v=demo",
        video_gcs_uri="gs://bucket/source.mp4",
        audio_gcs_uri="gs://bucket/source.wav",
        signed_audio_url="https://storage.example.test/source.wav?signed=1",
        workspace=workspace,
        scribe_provider=scribe,
        visual_extractor=visual,
        stage_event_logger=lambda **event: stage_events.append(event),
        on_audio_chain_complete=lambda payload: handoffs.append(payload),
    )

    assert outputs.phase1_audio["source_audio"] == "https://youtube.com/watch?v=demo"
    assert outputs.phase1_audio["video_gcs_uri"] == "gs://bucket/source.mp4"
    assert outputs.phase1_audio["audio_gcs_uri"] == "gs://bucket/source.wav"
    assert outputs.phase1_visual_status == "pending"
    assert outputs.phase1_visual is None
    assert outputs.visual_future is not None
    assert outputs.visual_future.call_id == "visual-call-1"
    assert outputs.diarization_payload.turns[0]["speaker_id"] == "SPEAKER_0"
    assert outputs.diarization_payload.words[0]["text"] == "hello"
    assert outputs.emotion2vec_payload.segments == []
    assert outputs.yamnet_payload.events == []

    assert visual.calls[0]["video_gcs_uri"] == "gs://bucket/source.mp4"
    assert scribe.calls[0]["source_url"].startswith("https://storage.example.test/")
    assert handoffs == [outputs]
    assert [event["stage_name"] for event in stage_events] == [
        "visual_extraction_submit",
        "scribe_transcription",
        "scribe_adapter",
    ]


def test_run_phase1_sidecars_propagates_scribe_failure_and_emits_failed_event(
    tmp_path: Path,
) -> None:
    from backend.phase1_runtime.extract import run_phase1_sidecars
    from backend.phase1_runtime.models import Phase1Workspace

    workspace = Phase1Workspace.create(root=tmp_path, run_id="run_002")
    workspace.video_path.write_text("video-bytes", encoding="utf-8")
    workspace.audio_path.write_text("audio-bytes", encoding="utf-8")
    stage_events: list[dict[str, object]] = []

    with pytest.raises(RuntimeError, match="scribe rejected"):
        run_phase1_sidecars(
            source_url="https://youtube.com/watch?v=demo",
            video_gcs_uri="gs://bucket/source.mp4",
            audio_gcs_uri="gs://bucket/source.wav",
            signed_audio_url="https://storage.example.test/source.wav?signed=1",
            workspace=workspace,
            scribe_provider=_FakeScribeProvider(error=RuntimeError("scribe rejected")),
            visual_extractor=_FakeVisualExtractor(),
            stage_event_logger=lambda **event: stage_events.append(event),
        )

    failure_events = [event for event in stage_events if event["status"] == "failed"]
    assert [event["stage_name"] for event in failure_events] == ["scribe_transcription"]


def test_run_phase1_sidecars_raises_when_required_scribe_inputs_are_missing(
    tmp_path: Path,
) -> None:
    from backend.phase1_runtime.extract import run_phase1_sidecars
    from backend.phase1_runtime.models import Phase1Workspace

    workspace = Phase1Workspace.create(root=tmp_path, run_id="run_003")
    workspace.video_path.write_text("video-bytes", encoding="utf-8")
    workspace.audio_path.write_text("audio-bytes", encoding="utf-8")

    with pytest.raises(ValueError, match="audio_gcs_uri is required"):
        run_phase1_sidecars(
            source_url="https://youtube.com/watch?v=demo",
            video_gcs_uri="gs://bucket/source.mp4",
            audio_gcs_uri=None,
            signed_audio_url="https://storage.example.test/source.wav?signed=1",
            workspace=workspace,
            scribe_provider=_FakeScribeProvider(),
            visual_extractor=_FakeVisualExtractor(),
        )

    with pytest.raises(ValueError, match="signed_audio_url is required"):
        run_phase1_sidecars(
            source_url="https://youtube.com/watch?v=demo",
            video_gcs_uri="gs://bucket/source.mp4",
            audio_gcs_uri="gs://bucket/source.wav",
            signed_audio_url=None,
            workspace=workspace,
            scribe_provider=_FakeScribeProvider(),
            visual_extractor=_FakeVisualExtractor(),
        )

    with pytest.raises(ValueError, match="Scribe provider"):
        run_phase1_sidecars(
            source_url="https://youtube.com/watch?v=demo",
            video_gcs_uri="gs://bucket/source.mp4",
            audio_gcs_uri="gs://bucket/source.wav",
            signed_audio_url="https://storage.example.test/source.wav?signed=1",
            workspace=workspace,
            scribe_provider=None,
            visual_extractor=_FakeVisualExtractor(),
        )
