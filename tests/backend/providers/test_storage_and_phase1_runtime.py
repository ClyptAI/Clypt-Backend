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


class _FakeVibeVoiceAsrClient:
    """Minimal stand-in for ``RemoteVibeVoiceAsrClient`` used by the H200
    orchestrator.

    The real client performs an HTTP call to the RTX 6000 Ada VibeVoice ASR
    host and re-emits stage events through ``stage_event_logger``. These
    tests only need to verify that ``run_phase1_sidecars`` passes the right
    kwargs and consumes the returned ``VibeVoiceAsrResponse``.
    """

    supports_concurrent_visual = True

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


def _default_asr_response():
    from backend.providers.audio_host_client import VibeVoiceAsrResponse

    return VibeVoiceAsrResponse(
        turns=[
            {
                "Speaker": 0,
                "Start": 0.0,
                "End": 0.3,
                "Content": "hello",
            }
        ],
        stage_events=[],
    )


class _FakeForcedAligner:
    def run(self, *, audio_path: Path, turns: list[dict]) -> list[dict]:
        self.last_call = {"audio_path": audio_path, "turns": list(turns)}
        return [
            {
                "word_id": "w_000001",
                "text": "hello",
                "start_ms": 0,
                "end_ms": 300,
                "turn_id": turns[0]["turn_id"] if turns else "t_000001",
                "speaker_id": turns[0]["speaker_id"] if turns else "SPEAKER_0",
            }
        ]


class _FakeEmotionProvider:
    def run(self, *, audio_path: Path, turns: list[dict]) -> dict:
        return {"segments": []}


class _FakeYamnetProvider:
    def run(self, *, audio_path: Path) -> dict:
        return {"events": []}


def _make_providers() -> dict[str, Any]:
    return {
        "forced_aligner": _FakeForcedAligner(),
        "emotion_provider": _FakeEmotionProvider(),
        "yamnet_provider": _FakeYamnetProvider(),
    }


def test_run_phase1_sidecars_runs_visual_and_remote_asr_concurrently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from backend.phase1_runtime.extract import run_phase1_sidecars
    from backend.phase1_runtime.models import Phase1Workspace

    # Keep vibevoice_merge from demanding a specific payload shape.
    monkeypatch.setattr(
        "backend.phase1_runtime.extract.merge_vibevoice_outputs",
        lambda *, vibevoice_turns, word_alignments: {
            "turns": [
                {
                    "turn_id": "t_000001",
                    "speaker_id": "SPEAKER_0",
                    "start_ms": 0,
                    "end_ms": 300,
                    "transcript_text": "hello",
                    "word_ids": ["w_000001"],
                }
            ],
            "words": list(word_alignments),
        },
    )

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

    asr_client = _FakeVibeVoiceAsrClient(response=_default_asr_response())

    class _TrackingClient(_FakeVibeVoiceAsrClient):
        def run(self, **kwargs):
            call_order.append("vibevoice_asr:run")
            return asr_client.run(**kwargs)

    workspace = Phase1Workspace.create(root=tmp_path, run_id="run_001")
    workspace.video_path.write_text("video-bytes", encoding="utf-8")
    workspace.audio_path.write_text("audio-bytes", encoding="utf-8")

    providers = _make_providers()
    outputs = run_phase1_sidecars(
        source_url="https://youtube.com/watch?v=demo",
        video_gcs_uri="gs://bucket/source.mp4",
        audio_gcs_uri="gs://bucket/source.wav",
        workspace=workspace,
        vibevoice_asr_client=_TrackingClient(response=_default_asr_response()),
        visual_extractor=_FakeVisualExtractor(),
        stage_event_logger=lambda **event: stage_events.append(event),
        **providers,
    )

    assert outputs.phase1_audio["source_audio"] == "https://youtube.com/watch?v=demo"
    assert outputs.phase1_audio["video_gcs_uri"] == "gs://bucket/source.mp4"
    assert outputs.phase1_audio["audio_gcs_uri"] == "gs://bucket/source.wav"
    assert outputs.diarization_payload["turns"][0]["speaker_id"] == "SPEAKER_0"
    assert outputs.diarization_payload["words"][0]["text"] == "hello"
    assert outputs.phase1_visual["video_metadata"]["fps"] == 10.0
    assert outputs.emotion2vec_payload == {"segments": []}
    assert outputs.yamnet_payload == {"events": []}

    assert set(call_order) == {"visual:source_video.mp4", "vibevoice_asr:run"}

    stage_names = {event["stage_name"] for event in stage_events}
    # H200 local audio chain stage events + visual extraction.
    assert "forced_alignment" in stage_names
    assert "emotion2vec" in stage_names
    assert "yamnet" in stage_names
    assert "visual_extraction" in stage_names


def test_run_phase1_sidecars_propagates_vibevoice_asr_failure_and_emits_failed_event(
    tmp_path: Path,
):
    from backend.phase1_runtime.extract import run_phase1_sidecars
    from backend.phase1_runtime.models import Phase1Workspace
    from backend.providers.audio_host_client import RemoteVibeVoiceAsrError

    stage_events: list[dict[str, object]] = []

    class _FakeVisualExtractor:
        def extract(self, *, video_path: Path, workspace: Phase1Workspace):
            return {
                "video_metadata": {"fps": 10.0},
                "shot_changes": [],
                "tracks": [],
            }

    asr_client = _FakeVibeVoiceAsrClient(
        error=RemoteVibeVoiceAsrError("vibevoice asr rejected", status_code=500)
    )

    workspace = Phase1Workspace.create(root=tmp_path, run_id="run_002")
    workspace.video_path.write_text("video-bytes", encoding="utf-8")
    workspace.audio_path.write_text("audio-bytes", encoding="utf-8")

    providers = _make_providers()
    with pytest.raises(RemoteVibeVoiceAsrError, match="vibevoice asr rejected"):
        run_phase1_sidecars(
            source_url="https://youtube.com/watch?v=demo",
            video_gcs_uri="gs://bucket/source.mp4",
            audio_gcs_uri="gs://bucket/source.wav",
            workspace=workspace,
            vibevoice_asr_client=asr_client,
            visual_extractor=_FakeVisualExtractor(),
            stage_event_logger=lambda **event: stage_events.append(event),
            **providers,
        )

    failure_events = [e for e in stage_events if e["status"] == "failed"]
    assert any(e["stage_name"] == "vibevoice_asr" for e in failure_events), (
        f"expected vibevoice_asr=failed event, got stage_events={stage_events}"
    )


def test_run_phase1_sidecars_raises_when_audio_gcs_uri_missing(tmp_path: Path):
    from backend.phase1_runtime.extract import run_phase1_sidecars
    from backend.phase1_runtime.models import Phase1Workspace

    class _FakeVisualExtractor:
        def extract(self, *, video_path: Path, workspace: Phase1Workspace):
            raise AssertionError("visual extractor should not run when audio_gcs_uri is missing")

    asr_client = _FakeVibeVoiceAsrClient(response=_default_asr_response())

    workspace = Phase1Workspace.create(root=tmp_path, run_id="run_003")
    workspace.video_path.write_text("video-bytes", encoding="utf-8")
    workspace.audio_path.write_text("audio-bytes", encoding="utf-8")

    providers = _make_providers()
    with pytest.raises(ValueError, match="audio_gcs_uri is required"):
        run_phase1_sidecars(
            source_url="https://youtube.com/watch?v=demo",
            video_gcs_uri="gs://bucket/source.mp4",
            audio_gcs_uri=None,
            workspace=workspace,
            vibevoice_asr_client=asr_client,
            visual_extractor=_FakeVisualExtractor(),
            **providers,
        )

    with pytest.raises(ValueError, match="audio_gcs_uri is required"):
        run_phase1_sidecars(
            source_url="https://youtube.com/watch?v=demo",
            video_gcs_uri="gs://bucket/source.mp4",
            audio_gcs_uri="   ",
            workspace=workspace,
            vibevoice_asr_client=asr_client,
            visual_extractor=_FakeVisualExtractor(),
            **providers,
        )

    assert asr_client.calls == []
