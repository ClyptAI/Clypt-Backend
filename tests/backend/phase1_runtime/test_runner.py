from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _scribe_response() -> dict[str, Any]:
    return {
        "language_code": "en",
        "words": [
            {
                "type": "word",
                "text": "hello",
                "start": 0.0,
                "end": 0.2,
                "speaker_id": "speaker_0",
            },
            {
                "type": "audio_event",
                "text": "(laughter)",
                "start": 0.25,
                "end": 0.5,
                "confidence": 0.8,
            },
        ],
    }


class _FakeScribeProvider:
    def __init__(self, *, response: dict[str, Any] | None = None) -> None:
        self.response = response or _scribe_response()
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
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


class _FakeStorage:
    def __init__(self) -> None:
        self.uploaded: list[tuple[Path, str]] = []
        self.downloaded: list[tuple[str, Path]] = []
        self.signed: list[tuple[str, int]] = []

    def upload_file(self, *, local_path: Path, object_name: str) -> str:
        self.uploaded.append((Path(local_path), object_name))
        return f"gs://bucket/{object_name}"

    def download_file(self, *, gcs_uri: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        payload = "video-cache" if local_path.suffix == ".mp4" else "audio-cache"
        local_path.write_text(payload, encoding="utf-8")
        self.downloaded.append((gcs_uri, local_path))
        return local_path

    def get_https_url(self, gcs_uri: str, *, expiry_hours: int = 24) -> str:
        self.signed.append((gcs_uri, expiry_hours))
        return f"https://storage.example.test/{gcs_uri.removeprefix('gs://')}?signed=1"


class _FakeQueueClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def enqueue_phase24(self, *, run_id: str, payload: dict, worker_url: str | None = None) -> str:
        self.calls.append({"run_id": run_id, "payload": payload, "worker_url": worker_url})
        return "local-sqlite:00000000-0000-0000-0000-000000000001"


def _write_test_bank_mapping(
    tmp_path: Path,
    *,
    source_url: str,
    video_gcs_uri: str,
    audio_gcs_uri: str,
    create_files: bool = True,
) -> Path:
    mapped_video = tmp_path / "fixtures" / "mapped-video.mp4"
    mapped_audio = tmp_path / "fixtures" / "mapped-audio.wav"
    if create_files:
        mapped_video.parent.mkdir(parents=True, exist_ok=True)
        mapped_video.write_text("mapped-video", encoding="utf-8")
        mapped_audio.write_text("mapped-audio", encoding="utf-8")

    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                source_url: {
                    "local_video_path": "fixtures/mapped-video.mp4",
                    "local_audio_path": "fixtures/mapped-audio.wav",
                    "video_gcs_uri": video_gcs_uri,
                    "audio_gcs_uri": audio_gcs_uri,
                }
            }
        ),
        encoding="utf-8",
    )
    return mapping_path


@pytest.fixture(autouse=True)
def _patch_youtube_source_context(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_source_context(*, source_url: str) -> dict[str, Any]:
        return {
            "source_url": source_url,
            "youtube_video_id": source_url.rsplit("=", 1)[-1],
            "source_title": "Test Source Title",
            "channel_title": "Test Channel",
        }

    monkeypatch.setattr(
        "backend.phase1_runtime.runner.fetch_youtube_source_context",
        _fake_source_context,
    )


def test_phase1_job_runner_enqueues_phase26_after_scribe_without_waiting_for_visual(
    tmp_path: Path,
) -> None:
    from backend.phase1_runtime.input_resolver import Phase1InputResolver
    from backend.phase1_runtime.runner import Phase1JobRunner

    mapping_path = _write_test_bank_mapping(
        tmp_path,
        source_url="https://youtube.com/watch?v=queue",
        video_gcs_uri="gs://bucket/canonical/queue.mp4",
        audio_gcs_uri="gs://bucket/canonical/queue.wav",
    )
    storage = _FakeStorage()
    queue = _FakeQueueClient()
    scribe = _FakeScribeProvider()
    visual = _FakeVisualExtractor()

    runner = Phase1JobRunner(
        working_root=tmp_path,
        storage_client=storage,
        scribe_provider=scribe,
        scribe_turn_gap_ms=1200,
        visual_extractor=visual,
        phase24_task_queue_client=queue,
        phase24_query_version="graph-v2",
        input_resolver=Phase1InputResolver.from_mapping_file(mapping_path),
    )

    result = runner.run_job(
        job_id="job_001",
        source_url="https://youtube.com/watch?v=queue",
        source_path=None,
        runtime_controls={"run_phase14": True},
    )

    phase1 = result["phase1"]
    assert phase1["phase1_audio"]["video_gcs_uri"] == "gs://bucket/canonical/queue.mp4"
    assert phase1["phase1_audio"]["audio_gcs_uri"] == "gs://bucket/canonical/queue.wav"
    assert phase1["phase1_visual_status"] == "pending"
    assert phase1["phase1_visual"] is None
    assert phase1["visual_future"]["call_id"] == "visual-call-1"
    assert phase1["diarization_payload"]["turns"][0]["transcript_text"] == "hello"
    assert phase1["yamnet_payload"]["events"][0]["event_label"] == "laughter"

    assert storage.signed == [("gs://bucket/canonical/queue.wav", 3)]
    assert scribe.calls[0]["source_url"].startswith("https://storage.example.test/")
    assert visual.calls[0]["video_gcs_uri"] == "gs://bucket/canonical/queue.mp4"
    assert queue.calls[0]["payload"]["phase1_visual_status"] == "pending"
    assert queue.calls[0]["payload"]["visual_call_id"] == "visual-call-1"
    assert queue.calls[0]["payload"]["query_version"] == "graph-v2"
    assert queue.calls[0]["payload"]["phase1_outputs_gcs_uri"].endswith(
        "/phase24_inputs/phase1_outputs.json"
    )


def test_phase1_job_runner_persists_source_context_in_handoff(tmp_path: Path) -> None:
    from backend.phase1_runtime.input_resolver import Phase1InputResolver
    from backend.phase1_runtime.runner import Phase1JobRunner

    captured: dict[str, Any] = {}
    mapping_path = _write_test_bank_mapping(
        tmp_path,
        source_url="https://youtube.com/watch?v=source-context",
        video_gcs_uri="gs://bucket/canonical/source-context.mp4",
        audio_gcs_uri="gs://bucket/canonical/source-context.wav",
    )

    class _CapturingStorage(_FakeStorage):
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            if object_name.endswith("phase1_outputs.json"):
                captured["handoff"] = json.loads(local_path.read_text(encoding="utf-8"))
            return super().upload_file(local_path=local_path, object_name=object_name)

    runner = Phase1JobRunner(
        working_root=tmp_path,
        storage_client=_CapturingStorage(),
        scribe_provider=_FakeScribeProvider(),
        visual_extractor=_FakeVisualExtractor(),
        phase24_task_queue_client=_FakeQueueClient(),
        input_resolver=Phase1InputResolver.from_mapping_file(mapping_path),
    )

    result = runner.run_job(
        job_id="job_source_context",
        source_url="https://youtube.com/watch?v=source-context",
        source_path=None,
        runtime_controls={"run_phase14": True},
    )

    assert result["phase1"]["source_context"]["source_title"] == "Test Source Title"
    assert captured["handoff"]["source_context"]["channel_title"] == "Test Channel"


def test_phase1_job_runner_fails_fast_when_queue_mode_enabled_but_unconfigured(
    tmp_path: Path,
) -> None:
    from backend.phase1_runtime.runner import Phase1JobRunner

    source_video = tmp_path / "source.mp4"
    source_video.write_text("video", encoding="utf-8")

    def fake_audio_extractor(*, video_path: Path, audio_path: Path) -> None:
        audio_path.write_text("audio", encoding="utf-8")

    runner = Phase1JobRunner(
        working_root=tmp_path,
        audio_extractor=fake_audio_extractor,
        storage_client=_FakeStorage(),
        scribe_provider=_FakeScribeProvider(),
        visual_extractor=_FakeVisualExtractor(),
        phase24_task_queue_client=None,
    )

    with pytest.raises(RuntimeError, match="local queue client is unavailable"):
        runner.run_job(
            job_id="job_001",
            source_url=None,
            source_path=str(source_video),
            runtime_controls={"run_phase14": True},
        )


def test_phase1_job_runner_uses_test_bank_media_and_skips_audio_extraction(
    tmp_path: Path,
) -> None:
    from backend.phase1_runtime.input_resolver import Phase1InputResolver
    from backend.phase1_runtime.runner import Phase1JobRunner

    mapping_path = _write_test_bank_mapping(
        tmp_path,
        source_url="https://youtube.com/watch?v=test",
        video_gcs_uri="gs://bucket/canonical/test.mp4",
        audio_gcs_uri="gs://bucket/canonical/test.wav",
    )
    extractor_calls: list[str] = []

    def fake_audio_extractor(*, video_path: Path, audio_path: Path) -> None:
        extractor_calls.append(video_path.name)
        audio_path.write_text("audio", encoding="utf-8")

    runner = Phase1JobRunner(
        working_root=tmp_path,
        audio_extractor=fake_audio_extractor,
        storage_client=_FakeStorage(),
        scribe_provider=_FakeScribeProvider(),
        visual_extractor=_FakeVisualExtractor(),
        phase24_task_queue_client=None,
        input_resolver=Phase1InputResolver.from_mapping_file(mapping_path),
    )

    result = runner.run_job(
        job_id="job_003",
        source_url="https://youtube.com/watch?v=test",
        source_path=None,
        runtime_controls={},
    )

    assert extractor_calls == []
    assert result["phase1"]["phase1_audio"]["source_audio"] == "https://youtube.com/watch?v=test"
    assert result["phase1"]["phase1_audio"]["video_gcs_uri"] == "gs://bucket/canonical/test.mp4"
    assert result["phase1"]["phase1_audio"]["audio_gcs_uri"] == "gs://bucket/canonical/test.wav"


def test_phase1_job_runner_does_not_apply_input_resolver_to_source_path_inputs(
    tmp_path: Path,
) -> None:
    from backend.phase1_runtime.runner import Phase1JobRunner

    source_path = tmp_path / "source.mp4"
    source_path.write_text("source-video", encoding="utf-8")
    resolver_calls: list[str] = []

    class _RecordingResolver:
        def resolve_source_asset(self, *, source_url: str):
            resolver_calls.append(f"resolve:{source_url}")
            raise AssertionError("source_path inputs should bypass the input resolver")

    def fake_audio_extractor(*, video_path: Path, audio_path: Path) -> None:
        audio_path.write_text("audio", encoding="utf-8")

    runner = Phase1JobRunner(
        working_root=tmp_path,
        audio_extractor=fake_audio_extractor,
        storage_client=_FakeStorage(),
        scribe_provider=_FakeScribeProvider(),
        visual_extractor=_FakeVisualExtractor(),
        phase24_task_queue_client=None,
        input_resolver=_RecordingResolver(),
    )

    result = runner.run_job(
        job_id="job_004",
        source_url=None,
        source_path=str(source_path),
        runtime_controls={},
    )

    assert resolver_calls == []
    assert result["phase1"]["phase1_audio"]["audio_gcs_uri"] == "gs://bucket/phase1/job_004/canonical_audio.wav"


def test_phase1_job_runner_raises_for_unmapped_test_bank_source(tmp_path: Path) -> None:
    from backend.phase1_runtime.input_resolver import Phase1InputResolutionError, Phase1InputResolver
    from backend.phase1_runtime.runner import Phase1JobRunner

    mapping_path = _write_test_bank_mapping(
        tmp_path,
        source_url="https://youtube.com/watch?v=known",
        video_gcs_uri="gs://bucket/canonical/known.mp4",
        audio_gcs_uri="gs://bucket/canonical/known.wav",
    )

    runner = Phase1JobRunner(
        working_root=tmp_path,
        storage_client=_FakeStorage(),
        scribe_provider=_FakeScribeProvider(),
        visual_extractor=_FakeVisualExtractor(),
        input_resolver=Phase1InputResolver.from_mapping_file(mapping_path),
    )

    with pytest.raises(Phase1InputResolutionError, match="No test-bank mapping found"):
        runner.run_job(
            job_id="job_unmapped",
            source_url="https://youtube.com/watch?v=unknown",
            source_path=None,
            runtime_controls={},
        )


def test_phase1_job_runner_hydrates_canonical_assets_and_uses_mapped_gcs_uris(
    tmp_path: Path,
) -> None:
    from backend.phase1_runtime.input_resolver import Phase1InputResolver
    from backend.phase1_runtime.runner import Phase1JobRunner

    mapping_path = _write_test_bank_mapping(
        tmp_path,
        source_url="https://youtu.be/abc123?si=orig",
        video_gcs_uri="gs://bucket/canonical/abc123.mp4",
        audio_gcs_uri="gs://bucket/canonical/abc123.wav",
        create_files=False,
    )
    storage = _FakeStorage()
    visual = _FakeVisualExtractor()

    runner = Phase1JobRunner(
        working_root=tmp_path,
        storage_client=storage,
        scribe_provider=_FakeScribeProvider(),
        visual_extractor=visual,
        phase24_task_queue_client=None,
        input_resolver=Phase1InputResolver.from_mapping_file(mapping_path),
    )

    result = runner.run_job(
        job_id="job_cache_hydrate",
        source_url="https://youtu.be/abc123?si=another",
        source_path=None,
        runtime_controls={},
    )

    phase1_audio = result["phase1"]["phase1_audio"]
    assert phase1_audio["source_audio"] == "https://youtu.be/abc123?si=another"
    assert phase1_audio["video_gcs_uri"] == "gs://bucket/canonical/abc123.mp4"
    assert phase1_audio["audio_gcs_uri"] == "gs://bucket/canonical/abc123.wav"
    assert visual.calls[0]["video_gcs_uri"] == "gs://bucket/canonical/abc123.mp4"
    assert storage.downloaded == [
        ("gs://bucket/canonical/abc123.mp4", (tmp_path / "fixtures" / "mapped-video.mp4").resolve()),
        ("gs://bucket/canonical/abc123.wav", (tmp_path / "fixtures" / "mapped-audio.wav").resolve()),
    ]
    assert storage.uploaded == []
