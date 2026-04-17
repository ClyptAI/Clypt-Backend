from __future__ import annotations

from pathlib import Path

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


class _FakeAudioHostClient:
    """Minimal stand-in for ``RemoteAudioChainClient`` used by the H200 orchestrator.

    The real client performs an HTTP call to the RTX 6000 Ada audio host and
    re-emits stage events through ``stage_event_logger``. These tests only
    need to verify that ``run_phase1_sidecars`` passes the right kwargs and
    merges the returned payloads.
    """

    def __init__(self, *, response=None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict[str, object]] = []

    def run(
        self,
        *,
        audio_gcs_uri: str,
        source_url: str | None = None,
        video_gcs_uri: str | None = None,
        run_id: str | None = None,
        stage_event_logger=None,
    ):
        self.calls.append(
            {
                "audio_gcs_uri": audio_gcs_uri,
                "source_url": source_url,
                "video_gcs_uri": video_gcs_uri,
                "run_id": run_id,
                "stage_event_logger": stage_event_logger,
            }
        )
        if self._error is not None:
            raise self._error
        return self._response


def _default_audio_response():
    from backend.providers.audio_host_client import PhaseOneAudioResponse

    return PhaseOneAudioResponse(
        turns=[
            {
                "turn_id": "t_000001",
                "speaker_id": "SPEAKER_0",
                "start_ms": 0,
                "end_ms": 300,
                "transcript_text": "hello",
            }
        ],
        diarization_payload={
            "turns": [
                {
                    "turn_id": "t_000001",
                    "speaker_id": "SPEAKER_0",
                    "start_ms": 0,
                    "end_ms": 300,
                    "transcript_text": "hello",
                    "word_ids": ["w_000001"],
                    "identification_match": None,
                }
            ],
            "words": [
                {
                    "word_id": "w_000001",
                    "text": "hello",
                    "start_ms": 0,
                    "end_ms": 300,
                    "speaker_id": "SPEAKER_0",
                }
            ],
        },
        emotion2vec_payload={"segments": []},
        yamnet_payload={"events": []},
        stage_events=[],
    )


def test_run_phase1_sidecars_runs_visual_and_remote_audio_concurrently(tmp_path: Path):
    from backend.phase1_runtime.extract import run_phase1_sidecars
    from backend.phase1_runtime.models import Phase1Workspace

    call_order: list[str] = []
    stage_events: list[dict[str, object]] = []

    class _FakeVisualExtractor:
        def extract(self, *, video_path: Path, workspace: Phase1Workspace):
            call_order.append(f"visual:{video_path.name}")
            return {
                "video_metadata": {"fps": 10.0},
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "tracks": [],
            }

    audio_response = _default_audio_response()
    audio_host_client = _FakeAudioHostClient(response=audio_response)

    class _TrackingClient:
        def run(self, **kwargs):
            call_order.append("audio_host:run")
            return audio_host_client.run(**kwargs)

    workspace = Phase1Workspace.create(root=tmp_path, run_id="run_001")
    workspace.video_path.write_text("video-bytes", encoding="utf-8")
    workspace.audio_path.write_text("audio-bytes", encoding="utf-8")

    outputs = run_phase1_sidecars(
        source_url="https://youtube.com/watch?v=demo",
        video_gcs_uri="gs://bucket/source.mp4",
        audio_gcs_uri="gs://bucket/source.wav",
        workspace=workspace,
        audio_host_client=_TrackingClient(),
        visual_extractor=_FakeVisualExtractor(),
        stage_event_logger=lambda **event: stage_events.append(event),
    )

    assert outputs.phase1_audio["source_audio"] == "https://youtube.com/watch?v=demo"
    assert outputs.phase1_audio["video_gcs_uri"] == "gs://bucket/source.mp4"
    assert outputs.phase1_audio["audio_gcs_uri"] == "gs://bucket/source.wav"
    assert outputs.diarization_payload["turns"][0]["speaker_id"] == "SPEAKER_0"
    assert outputs.diarization_payload["words"][0]["text"] == "hello"
    assert outputs.phase1_visual["video_metadata"]["fps"] == 10.0
    assert outputs.emotion2vec_payload == {"segments": []}
    assert outputs.yamnet_payload == {"events": []}

    assert set(call_order) == {"visual:source_video.mp4", "audio_host:run"}
    assert len(audio_host_client.calls) == 1
    audio_call = audio_host_client.calls[0]
    assert audio_call["audio_gcs_uri"] == "gs://bucket/source.wav"
    assert audio_call["source_url"] == "https://youtube.com/watch?v=demo"
    assert audio_call["video_gcs_uri"] == "gs://bucket/source.mp4"
    assert audio_call["run_id"] == "run_001"
    assert callable(audio_call["stage_event_logger"])

    assert [event["stage_name"] for event in stage_events] == ["visual_extraction"]
    assert stage_events[0]["status"] == "succeeded"
    assert stage_events[0]["metadata"]["shot_change_count"] == 1
    assert stage_events[0]["metadata"]["track_count"] == 0


def test_run_phase1_sidecars_propagates_audio_host_failure_and_emits_failed_event(tmp_path: Path):
    from backend.phase1_runtime.extract import run_phase1_sidecars
    from backend.phase1_runtime.models import Phase1Workspace
    from backend.providers.audio_host_client import RemoteAudioChainError

    stage_events: list[dict[str, object]] = []

    class _FakeVisualExtractor:
        def extract(self, *, video_path: Path, workspace: Phase1Workspace):
            return {
                "video_metadata": {"fps": 10.0},
                "shot_changes": [],
                "tracks": [],
            }

    audio_host_client = _FakeAudioHostClient(
        error=RemoteAudioChainError("audio host rejected", status_code=500)
    )

    workspace = Phase1Workspace.create(root=tmp_path, run_id="run_002")
    workspace.video_path.write_text("video-bytes", encoding="utf-8")
    workspace.audio_path.write_text("audio-bytes", encoding="utf-8")

    with pytest.raises(RemoteAudioChainError, match="audio host rejected"):
        run_phase1_sidecars(
            source_url="https://youtube.com/watch?v=demo",
            video_gcs_uri="gs://bucket/source.mp4",
            audio_gcs_uri="gs://bucket/source.wav",
            workspace=workspace,
            audio_host_client=audio_host_client,
            visual_extractor=_FakeVisualExtractor(),
            stage_event_logger=lambda **event: stage_events.append(event),
        )

    failure_events = [e for e in stage_events if e["status"] == "failed"]
    assert any(e["stage_name"] == "audio_host_call" for e in failure_events), (
        f"expected audio_host_call=failed event, got stage_events={stage_events}"
    )
    audio_failure = next(
        e for e in failure_events if e["stage_name"] == "audio_host_call"
    )
    assert audio_failure["error_payload"]["code"] == "RemoteAudioChainError"
    assert "audio host rejected" in audio_failure["error_payload"]["message"]


def test_run_phase1_sidecars_raises_when_audio_gcs_uri_missing(tmp_path: Path):
    from backend.phase1_runtime.extract import run_phase1_sidecars
    from backend.phase1_runtime.models import Phase1Workspace

    class _FakeVisualExtractor:
        def extract(self, *, video_path: Path, workspace: Phase1Workspace):
            raise AssertionError("visual extractor should not run when audio_gcs_uri is missing")

    audio_host_client = _FakeAudioHostClient(response=_default_audio_response())

    workspace = Phase1Workspace.create(root=tmp_path, run_id="run_003")
    workspace.video_path.write_text("video-bytes", encoding="utf-8")
    workspace.audio_path.write_text("audio-bytes", encoding="utf-8")

    with pytest.raises(ValueError, match="audio_gcs_uri is required"):
        run_phase1_sidecars(
            source_url="https://youtube.com/watch?v=demo",
            video_gcs_uri="gs://bucket/source.mp4",
            audio_gcs_uri=None,
            workspace=workspace,
            audio_host_client=audio_host_client,
            visual_extractor=_FakeVisualExtractor(),
        )

    with pytest.raises(ValueError, match="audio_gcs_uri is required"):
        run_phase1_sidecars(
            source_url="https://youtube.com/watch?v=demo",
            video_gcs_uri="gs://bucket/source.mp4",
            audio_gcs_uri="   ",
            workspace=workspace,
            audio_host_client=audio_host_client,
            visual_extractor=_FakeVisualExtractor(),
        )

    assert audio_host_client.calls == []
