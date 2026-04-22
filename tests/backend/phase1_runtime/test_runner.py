from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Shared fakes for the Phase 1 H200 orchestrator.
# ---------------------------------------------------------------------------


def _make_fake_vibevoice_asr_client(**overrides: Any):
    """Return a fake ``RemoteVibeVoiceAsrClient``-shaped object.

    The fake exposes a single ``.run(audio_gcs_uri, source_url, video_gcs_uri,
    run_id, stage_event_logger)`` method returning a
    :class:`backend.providers.audio_host_client.VibeVoiceAsrResponse` with
    one turn by default. Before returning, it re-emits a ``vibevoice_asr``
    ``succeeded`` stage event through ``stage_event_logger`` so downstream
    telemetry matches the real remote client.

    Supported ``overrides``:

    - ``turns`` / ``stage_events`` — replace the corresponding response fields.
    - ``raise_exc`` — if set, ``.run`` raises the given exception.
    - ``on_run`` — optional callback invoked at the start of ``.run``.
    - ``emit_stage_event`` — default True; set False to suppress the
      re-emission of the ``vibevoice_asr`` stage event.
    """
    from backend.providers.audio_host_client import VibeVoiceAsrResponse

    raise_exc: Exception | None = overrides.pop("raise_exc", None)
    on_run = overrides.pop("on_run", None)
    emit_stage_event: bool = overrides.pop("emit_stage_event", True)

    default_response_kwargs = {
        "turns": [{"Speaker": 0, "Start": 0.0, "End": 0.2, "Content": "hello"}],
        "stage_events": [],
    }
    default_response_kwargs.update(overrides)

    class _FakeVibeVoiceAsrClient:
        supports_concurrent_visual = True

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def run(
            self,
            *,
            audio_gcs_uri: str,
            source_url: str | None = None,
            video_gcs_uri: str | None = None,
            run_id: str | None = None,
            stage_event_logger: Any = None,
        ) -> VibeVoiceAsrResponse:
            call = {
                "audio_gcs_uri": audio_gcs_uri,
                "source_url": source_url,
                "video_gcs_uri": video_gcs_uri,
                "run_id": run_id,
                "stage_event_logger": stage_event_logger,
            }
            self.calls.append(call)
            if on_run is not None:
                on_run(call)
            if raise_exc is not None:
                raise raise_exc

            if emit_stage_event and stage_event_logger is not None:
                stage_event_logger(
                    stage_name="vibevoice_asr",
                    status="succeeded",
                    duration_ms=1.0,
                    metadata={},
                    error_payload=None,
                )

            return VibeVoiceAsrResponse(**default_response_kwargs)

    return _FakeVibeVoiceAsrClient()


def _make_local_audio_providers() -> dict[str, Any]:
    """Return in-process NFA / emotion2vec+ / YAMNet provider fakes for the H200."""

    class _FakeForcedAligner:
        def run(self, *, audio_path: Path, turns: list[dict]) -> list[dict]:
            return [
                {
                    "word_id": "w_000001",
                    "text": "hello",
                    "start_ms": 0,
                    "end_ms": 200,
                    "turn_id": turns[0]["turn_id"] if turns else "t_000001",
                    "speaker_id": (
                        turns[0]["speaker_id"] if turns else "SPEAKER_0"
                    ),
                }
            ]

    class _FakeEmotionProvider:
        def run(self, *, audio_path: Path, turns: list[dict]) -> dict:
            return {"segments": []}

    class _FakeYamnetProvider:
        def run(self, *, audio_path: Path) -> dict:
            return {"events": []}

    return {
        "forced_aligner": _FakeForcedAligner(),
        "emotion_provider": _FakeEmotionProvider(),
        "yamnet_provider": _FakeYamnetProvider(),
    }


@pytest.fixture(autouse=True)
def _patch_vibevoice_merge(monkeypatch: pytest.MonkeyPatch):
    """Replace the real vibevoice_merge with a deterministic stub.

    The runner tests only care about the audio-host wiring + stage-event
    emission; they don't exercise merging edge cases. Keeping merge
    deterministic avoids pulling heavy imports into this test module.
    """
    monkeypatch.setattr(
        "backend.phase1_runtime.extract.merge_vibevoice_outputs",
        lambda *, vibevoice_turns, word_alignments: {
            "turns": [
                {
                    "turn_id": "t_000001",
                    "speaker_id": "SPEAKER_0",
                    "start_ms": 0,
                    "end_ms": 200,
                    "transcript_text": "hello",
                    "word_ids": ["w_000001"],
                }
            ],
            "words": list(word_alignments),
        },
    )


@pytest.fixture(autouse=True)
def _patch_youtube_source_context(monkeypatch: pytest.MonkeyPatch):
    """Keep the runner tests offline while Phase 1 now fetches YouTube metadata."""

    def _fake_source_context(*, source_url: str) -> dict[str, Any]:
        from backend.phase1_runtime.input_resolver import _extract_youtube_video_id

        video_id = _extract_youtube_video_id(source_url) or "unknown"
        return {
            "source_url": source_url,
            "youtube_video_id": video_id,
            "source_title": "Test Source Title",
            "source_description": "Test source description.",
            "channel_id": "channel-123",
            "channel_title": "Test Channel",
            "published_at": "2026-04-22T00:00:00Z",
            "default_audio_language": "en",
            "category_id": "22",
            "tags": ["test", "phase1", "metadata"],
            "thumbnails": {"default": {"url": "https://example.test/thumb.jpg"}},
        }

    monkeypatch.setattr(
        "backend.phase1_runtime.runner.fetch_youtube_source_context",
        _fake_source_context,
    )


def _write_test_bank_mapping(
    tmp_path: Path,
    *,
    source_url: str,
    video_gcs_uri: str,
    audio_gcs_uri: str,
) -> tuple[Path, Path]:
    """Create a minimal test-bank mapping with video+audio GCS URIs."""
    mapped_video = tmp_path / "fixtures" / "mapped-video.mp4"
    mapped_video.parent.mkdir(parents=True, exist_ok=True)
    mapped_video.write_text("mapped-video", encoding="utf-8")
    mapped_audio = tmp_path / "fixtures" / "mapped-audio.wav"
    mapped_audio.write_text("mapped-audio", encoding="utf-8")

    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        f"""
{{
  "{source_url}": {{
    "local_video_path": "fixtures/mapped-video.mp4",
    "local_audio_path": "fixtures/mapped-audio.wav",
    "video_gcs_uri": "{video_gcs_uri}",
    "audio_gcs_uri": "{audio_gcs_uri}"
  }}
}}
""".strip(),
        encoding="utf-8",
    )
    return mapping_path, mapped_video


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_phase1_job_runner_enqueues_phase24_when_queue_mode_enabled(tmp_path: Path):
    from backend.phase1_runtime.input_resolver import Phase1InputResolver
    from backend.phase1_runtime.runner import Phase1JobRunner
    from backend.repository.models import Phase24JobRecord, PhaseMetricRecord, RunRecord

    calls: list[str] = []
    captured: dict[str, object] = {}

    mapping_path, _ = _write_test_bank_mapping(
        tmp_path,
        source_url="https://youtube.com/watch?v=queue",
        video_gcs_uri="gs://bucket/canonical/queue.mp4",
        audio_gcs_uri="gs://bucket/canonical/queue.wav",
    )

    def fake_audio_extractor(*, video_path: Path, audio_path: Path) -> None:
        calls.append(f"audio:{video_path.name}")
        audio_path.write_text("audio", encoding="utf-8")

    class _FakeStorage:
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            calls.append(f"upload:{object_name}")
            return f"gs://bucket/{object_name}"

    class _FakeVisual:
        def extract(self, *, video_path: Path, workspace):
            calls.append(f"visual:{video_path.name}")
            return {
                "video_metadata": {"fps": 30.0, "duration_ms": 1000},
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "tracks": [],
            }

    class _FakeQueueClient:
        def enqueue_phase24(self, *, run_id: str, payload: dict, worker_url: str | None = None) -> str:
            captured["run_id"] = run_id
            captured["payload"] = payload
            captured["worker_url"] = worker_url
            return "local-sqlite:00000000-0000-0000-0000-000000000001"

    class _FakeRepository:
        def __init__(self) -> None:
            self.run_record: RunRecord | None = None
            self.job_record: Phase24JobRecord | None = None
            self.phase_metrics: list[PhaseMetricRecord] = []

        def upsert_run(self, record: RunRecord) -> RunRecord:
            self.run_record = record
            return record

        def upsert_phase24_job(self, record: Phase24JobRecord) -> Phase24JobRecord:
            self.job_record = record
            return record

        def write_phase_metric(self, record: PhaseMetricRecord) -> PhaseMetricRecord:
            self.phase_metrics.append(record)
            return record

    repository = _FakeRepository()
    audio_host_client = _make_fake_vibevoice_asr_client()

    runner = Phase1JobRunner(
        working_root=tmp_path,
        audio_extractor=fake_audio_extractor,
        storage_client=_FakeStorage(),
        vibevoice_asr_client=audio_host_client,
        visual_extractor=_FakeVisual(),
        phase24_task_queue_client=_FakeQueueClient(),
        phase14_repository=repository,
        phase24_query_version="graph-v2",
        input_resolver=Phase1InputResolver.from_mapping_file(mapping_path),
        **_make_local_audio_providers(),
    )

    result = runner.run_job(
        job_id="job_001",
        source_url="https://youtube.com/watch?v=queue",
        source_path=None,
        runtime_controls={"run_phase14": True},
    )

    # The canonical GCS URIs come from the test-bank mapping, not a new upload.
    assert result["phase1"]["phase1_audio"]["video_gcs_uri"] == "gs://bucket/canonical/queue.mp4"
    assert result["phase1"]["phase1_audio"]["audio_gcs_uri"] == "gs://bucket/canonical/queue.wav"
    assert result["summary"] == {
        "run_id": "job_001",
        "status": "queued",
        "task_name": "local-sqlite:00000000-0000-0000-0000-000000000001",
        "artifact_paths": {},
    }
    assert captured["run_id"] == "job_001"
    assert captured["payload"]["run_id"] == "job_001"
    assert captured["payload"]["query_version"] == "graph-v2"
    assert (
        captured["payload"]["phase1_outputs_gcs_uri"]
        == "gs://bucket/phase1/job_001/phase24_inputs/phase1_outputs.json"
    )
    assert "phase1_outputs" not in captured["payload"]
    assert repository.run_record is not None
    assert repository.run_record.run_id == "job_001"
    assert repository.run_record.status == "PHASE24_QUEUED"
    assert repository.job_record is not None
    assert repository.job_record.run_id == "job_001"
    assert repository.job_record.status == "queued"
    assert repository.job_record.task_name == "local-sqlite:00000000-0000-0000-0000-000000000001"
    assert repository.job_record.metadata["query_version"] == "graph-v2"
    phase_names = [record.phase_name for record in repository.phase_metrics]
    assert phase_names == [
        "phase1_vibevoice_asr",
        "phase1_forced_alignment",
        "phase1_emotion2vec",
        "phase1_yamnet",
        "phase1_visual_extraction",
    ]
    assert all(record.status == "succeeded" for record in repository.phase_metrics)

    # The audio host client received the test-bank audio_gcs_uri.
    assert len(audio_host_client.calls) == 1
    assert audio_host_client.calls[0]["audio_gcs_uri"] == "gs://bucket/canonical/queue.wav"
    assert audio_host_client.calls[0]["video_gcs_uri"] == "gs://bucket/canonical/queue.mp4"
    assert audio_host_client.calls[0]["run_id"] == "job_001"
    assert audio_host_client.calls[0]["source_url"] == "https://youtube.com/watch?v=queue"

    # Visual extractor ran, and the phase24 handoff was uploaded.
    assert "visual:source_video.mp4" in calls
    assert "upload:phase1/job_001/phase24_inputs/phase1_outputs.json" in calls


def test_phase1_job_runner_persists_source_context_in_handoff_and_final_outputs(tmp_path: Path):
    from backend.phase1_runtime.input_resolver import Phase1InputResolver
    from backend.phase1_runtime.runner import Phase1JobRunner

    captured: dict[str, Any] = {}

    mapping_path, _ = _write_test_bank_mapping(
        tmp_path,
        source_url="https://youtube.com/watch?v=source-context",
        video_gcs_uri="gs://bucket/canonical/source-context.mp4",
        audio_gcs_uri="gs://bucket/canonical/source-context.wav",
    )

    def fake_audio_extractor(*, video_path: Path, audio_path: Path) -> None:
        audio_path.write_text("audio", encoding="utf-8")

    class _FakeStorage:
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            if object_name.endswith("phase1_outputs.json"):
                captured["handoff_payload"] = json.loads(local_path.read_text(encoding="utf-8"))
            return f"gs://bucket/{object_name}"

    class _FakeVisual:
        def extract(self, *, video_path: Path, workspace):
            return {
                "video_metadata": {"fps": 30.0, "duration_ms": 1000},
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "tracks": [],
            }

    class _FakeQueueClient:
        def enqueue_phase24(self, *, run_id: str, payload: dict, worker_url: str | None = None) -> str:
            captured["queued_payload"] = payload
            return "local-sqlite:00000000-0000-0000-0000-000000000003"

    expected_context = {
        "source_url": "https://youtube.com/watch?v=source-context",
        "youtube_video_id": "source-context",
        "source_title": "Fetched Title",
        "source_description": "Fetched Description",
        "channel_id": "channel-ctx",
        "channel_title": "Fetched Channel",
        "published_at": "2026-04-22T00:00:00Z",
        "default_audio_language": "en",
        "category_id": "22",
        "tags": ["clip", "metadata"],
        "thumbnails": {"default": {"url": "https://example.test/thumb.jpg"}},
    }

    runner = Phase1JobRunner(
        working_root=tmp_path,
        audio_extractor=fake_audio_extractor,
        storage_client=_FakeStorage(),
        vibevoice_asr_client=_make_fake_vibevoice_asr_client(),
        visual_extractor=_FakeVisual(),
        phase24_task_queue_client=_FakeQueueClient(),
        input_resolver=Phase1InputResolver.from_mapping_file(mapping_path),
        **_make_local_audio_providers(),
    )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "backend.phase1_runtime.runner.fetch_youtube_source_context",
            lambda *, source_url: expected_context,
        )
        result = runner.run_job(
            job_id="job_source_context",
            source_url="https://youtube.com/watch?v=source-context",
            source_path=None,
            runtime_controls={"run_phase14": True},
        )

    assert result["phase1"]["source_context"] == expected_context
    assert captured["handoff_payload"]["source_context"] == expected_context
    assert captured["queued_payload"]["run_id"] == "job_source_context"


def test_phase1_job_runner_queue_mode_enqueues_before_visual_completes(tmp_path: Path):
    from backend.phase1_runtime.input_resolver import Phase1InputResolver
    from backend.phase1_runtime.runner import Phase1JobRunner

    calls: list[str] = []
    queue_called = threading.Event()

    mapping_path, _ = _write_test_bank_mapping(
        tmp_path,
        source_url="https://youtube.com/watch?v=race",
        video_gcs_uri="gs://bucket/canonical/race.mp4",
        audio_gcs_uri="gs://bucket/canonical/race.wav",
    )

    def fake_audio_extractor(*, video_path: Path, audio_path: Path) -> None:
        calls.append(f"audio:{video_path.name}")
        audio_path.write_text("audio", encoding="utf-8")

    class _FakeStorage:
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            calls.append(f"upload:{object_name}")
            return f"gs://bucket/{object_name}"

    class _FakeVisual:
        def extract(self, *, video_path: Path, workspace):
            calls.append("visual:start")
            if not queue_called.wait(timeout=2.0):
                raise AssertionError("queue enqueue should happen before visual completion")
            calls.append("visual:done")
            return {
                "video_metadata": {"fps": 30.0, "duration_ms": 1000},
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "tracks": [],
            }

    class _FakeQueueClient:
        def enqueue_phase24(self, *, run_id: str, payload: dict, worker_url: str | None = None) -> str:
            calls.append(f"enqueue:{run_id}")
            queue_called.set()
            return "local-sqlite:00000000-0000-0000-0000-000000000002"

    runner = Phase1JobRunner(
        working_root=tmp_path,
        audio_extractor=fake_audio_extractor,
        storage_client=_FakeStorage(),
        vibevoice_asr_client=_make_fake_vibevoice_asr_client(),
        visual_extractor=_FakeVisual(),
        phase24_task_queue_client=_FakeQueueClient(),
        phase24_query_version="graph-v2",
        input_resolver=Phase1InputResolver.from_mapping_file(mapping_path),
        **_make_local_audio_providers(),
    )

    result = runner.run_job(
        job_id="job_002",
        source_url="https://youtube.com/watch?v=race",
        source_path=None,
        runtime_controls={"run_phase14": True},
    )

    assert result["summary"]["status"] == "queued"
    assert "upload:phase1/job_002/phase24_inputs/phase1_outputs.json" in calls
    assert queue_called.is_set()
    assert calls.index("enqueue:job_002") < calls.index("visual:done")


def test_phase1_job_runner_enforces_queue_mode_for_run_phase14(tmp_path: Path):
    from backend.phase1_runtime.input_resolver import Phase1InputResolver
    from backend.phase1_runtime.runner import Phase1JobRunner

    calls: list[str] = []

    mapping_path, _ = _write_test_bank_mapping(
        tmp_path,
        source_url="https://youtube.com/watch?v=enforce",
        video_gcs_uri="gs://bucket/canonical/enforce.mp4",
        audio_gcs_uri="gs://bucket/canonical/enforce.wav",
    )

    def fake_audio_extractor(*, video_path: Path, audio_path: Path) -> None:
        calls.append(f"audio:{video_path.name}")
        audio_path.write_text("audio", encoding="utf-8")

    class _FakeStorage:
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            calls.append(f"upload:{object_name}")
            return f"gs://bucket/{object_name}"

    class _FakeVisual:
        def extract(self, *, video_path: Path, workspace):
            calls.append(f"visual:{video_path.name}")
            return {
                "video_metadata": {"fps": 30.0, "duration_ms": 1000},
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "tracks": [],
            }

    class _FakeQueueClient:
        def enqueue_phase24(self, *, run_id: str, payload: dict, worker_url: str | None = None) -> str:
            calls.append(f"enqueue:{run_id}")
            return "local-sqlite:00000000-0000-0000-0000-000000000099"

    runner = Phase1JobRunner(
        working_root=tmp_path,
        audio_extractor=fake_audio_extractor,
        storage_client=_FakeStorage(),
        vibevoice_asr_client=_make_fake_vibevoice_asr_client(),
        visual_extractor=_FakeVisual(),
        phase24_task_queue_client=_FakeQueueClient(),
        input_resolver=Phase1InputResolver.from_mapping_file(mapping_path),
        **_make_local_audio_providers(),
    )

    result = runner.run_job(
        job_id="job_001",
        source_url="https://youtube.com/watch?v=enforce",
        source_path=None,
        runtime_controls={"run_phase14": True},
    )
    assert result["summary"]["status"] == "queued"
    assert "enqueue:job_001" in calls


def test_phase1_job_runner_fails_fast_when_queue_mode_enabled_but_unconfigured(tmp_path: Path):
    from backend.phase1_runtime.runner import Phase1JobRunner

    source_video = tmp_path / "source.mp4"
    source_video.write_text("video", encoding="utf-8")

    def fake_audio_extractor(*, video_path: Path, audio_path: Path) -> None:
        audio_path.write_text("audio", encoding="utf-8")

    class _FakeStorage:
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            return f"gs://bucket/{object_name}"

    runner = Phase1JobRunner(
        working_root=tmp_path,
        audio_extractor=fake_audio_extractor,
        storage_client=_FakeStorage(),
        vibevoice_asr_client=_make_fake_vibevoice_asr_client(),
        visual_extractor=object(),
        phase24_task_queue_client=None,
        **_make_local_audio_providers(),
    )

    with pytest.raises(RuntimeError, match="local queue client is unavailable"):
        runner.run_job(
            job_id="job_001",
            source_url=None,
            source_path=str(source_video),
            runtime_controls={"run_phase14": True},
        )


def test_phase1_job_runner_uses_test_bank_media_and_preserves_posted_source_url(tmp_path: Path):
    from backend.phase1_runtime.input_resolver import Phase1InputResolver
    from backend.phase1_runtime.runner import Phase1JobRunner

    calls: list[str] = []
    captured: dict[str, object] = {}

    mapping_path, mapped_video = _write_test_bank_mapping(
        tmp_path,
        source_url="https://youtube.com/watch?v=test",
        video_gcs_uri="gs://bucket/canonical/test.mp4",
        audio_gcs_uri="gs://bucket/canonical/test.wav",
    )

    def fake_audio_extractor(*, video_path: Path, audio_path: Path) -> None:
        calls.append(f"audio:{video_path.read_text(encoding='utf-8')}")
        audio_path.write_text("audio", encoding="utf-8")

    class _FakeStorage:
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            captured["uploaded_video"] = local_path.read_text(encoding="utf-8")
            return f"gs://bucket/{object_name}"

        def download_file(self, *, gcs_uri: str, local_path: Path) -> Path:
            # The mapped local files already exist, so this shouldn't be called.
            raise AssertionError(f"unexpected download_file call for {gcs_uri}")

    class _FakeVisual:
        def extract(self, *, video_path: Path, workspace):
            captured["visual_video_contents"] = video_path.read_text(encoding="utf-8")
            return {
                "video_metadata": {"fps": 30.0, "duration_ms": 1000},
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "tracks": [],
            }

    audio_host_client = _make_fake_vibevoice_asr_client()

    runner = Phase1JobRunner(
        working_root=tmp_path,
        audio_extractor=fake_audio_extractor,
        storage_client=_FakeStorage(),
        vibevoice_asr_client=audio_host_client,
        visual_extractor=_FakeVisual(),
        phase24_task_queue_client=None,
        input_resolver=Phase1InputResolver.from_mapping_file(mapping_path),
        **_make_local_audio_providers(),
    )

    result = runner.run_job(
        job_id="job_003",
        source_url="https://youtube.com/watch?v=test",
        source_path=None,
        runtime_controls={},
    )

    # The mapping provides canonical GCS URIs, so no new video upload is made.
    assert "uploaded_video" not in captured
    assert captured["visual_video_contents"] == "mapped-video"
    assert result["phase1"]["phase1_audio"]["source_audio"] == "https://youtube.com/watch?v=test"
    assert result["phase1"]["phase1_audio"]["video_gcs_uri"] == "gs://bucket/canonical/test.mp4"
    assert result["phase1"]["phase1_audio"]["audio_gcs_uri"] == "gs://bucket/canonical/test.wav"

    # The audio-host client received the canonical audio URI from the mapping.
    assert len(audio_host_client.calls) == 1
    assert audio_host_client.calls[0]["audio_gcs_uri"] == "gs://bucket/canonical/test.wav"
    # The mapping provides a local audio path, so the extractor should not run.
    assert calls == []


def test_phase1_job_runner_does_not_apply_input_resolver_to_source_path_inputs(tmp_path: Path):
    """``source_path`` inputs must bypass the resolver entirely.

    With the RTX audio-host refactor the runner requires an ``audio_gcs_uri``
    (which normally comes from a test-bank mapping), so a pure ``source_path``
    job now fails at the sidecars step. We still assert the input resolver was
    never consulted, which is the contract this test locks in.
    """
    from backend.phase1_runtime.runner import Phase1JobRunner

    source_path = tmp_path / "source.mp4"
    source_path.write_text("source-video", encoding="utf-8")

    resolver_calls: list[str] = []

    class _RecordingResolver:
        def resolve_source_asset(self, *, source_url: str):
            resolver_calls.append(f"resolve:{source_url}")
            raise AssertionError("source_path inputs should bypass the input resolver")

        def resolve_source_path(self, *, source_url: str) -> Path:
            resolver_calls.append(f"resolve_path:{source_url}")
            raise AssertionError("source_path inputs should bypass the input resolver")

    def fake_audio_extractor(*, video_path: Path, audio_path: Path) -> None:
        audio_path.write_text(video_path.read_text(encoding="utf-8"), encoding="utf-8")

    class _FakeStorage:
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            return f"gs://bucket/{object_name}"

    class _FakeVisual:
        def extract(self, *, video_path: Path, workspace):
            return {
                "video_metadata": {"fps": 30.0, "duration_ms": 1000},
                "shot_changes": [],
                "tracks": [],
            }

    runner = Phase1JobRunner(
        working_root=tmp_path,
        audio_extractor=fake_audio_extractor,
        storage_client=_FakeStorage(),
        vibevoice_asr_client=_make_fake_vibevoice_asr_client(),
        visual_extractor=_FakeVisual(),
        phase24_task_queue_client=None,
        input_resolver=_RecordingResolver(),
        **_make_local_audio_providers(),
    )

    # In source_path mode there is no canonical audio_gcs_uri for the RTX host,
    # so the sidecars step fails fast — but the resolver must never be called.
    with pytest.raises(ValueError, match="audio_gcs_uri is required"):
        runner.run_job(
            job_id="job_004",
            source_url=None,
            source_path=str(source_path),
            runtime_controls={},
        )

    assert resolver_calls == []


def test_phase1_job_runner_raises_for_unmapped_test_bank_source_when_strict(tmp_path: Path):
    from backend.phase1_runtime.input_resolver import Phase1InputResolutionError, Phase1InputResolver
    from backend.phase1_runtime.runner import Phase1JobRunner

    mapped_video = tmp_path / "fixtures" / "mapped-video.mp4"
    mapped_video.parent.mkdir(parents=True)
    mapped_video.write_text("mapped-video", encoding="utf-8")
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        """
{
  "https://youtube.com/watch?v=known": {
    "local_video_path": "fixtures/mapped-video.mp4"
  }
}
""".strip(),
        encoding="utf-8",
    )

    runner = Phase1JobRunner(
        working_root=tmp_path,
        audio_extractor=object(),
        storage_client=object(),
        vibevoice_asr_client=_make_fake_vibevoice_asr_client(),
        visual_extractor=object(),
        phase24_task_queue_client=None,
        input_resolver=Phase1InputResolver.from_mapping_file(mapping_path),
        input_resolver_strict=True,
        **_make_local_audio_providers(),
    )

    with pytest.raises(Phase1InputResolutionError, match="No test-bank mapping found"):
        runner.run_job(
            job_id="job_unmapped_strict",
            source_url="https://youtube.com/watch?v=unknown",
            source_path=None,
            runtime_controls={},
        )


def test_phase1_job_runner_raises_for_unmapped_test_bank_source_when_non_strict(tmp_path: Path):
    from backend.phase1_runtime.input_resolver import Phase1InputResolutionError, Phase1InputResolver
    from backend.phase1_runtime.runner import Phase1JobRunner

    mapped_video = tmp_path / "fixtures" / "mapped-video.mp4"
    mapped_video.parent.mkdir(parents=True)
    mapped_video.write_text("mapped-video", encoding="utf-8")
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        """
{
  "https://youtube.com/watch?v=known": {
    "local_video_path": "fixtures/mapped-video.mp4"
  }
}
""".strip(),
        encoding="utf-8",
    )

    runner = Phase1JobRunner(
        working_root=tmp_path,
        audio_extractor=object(),
        storage_client=object(),
        vibevoice_asr_client=_make_fake_vibevoice_asr_client(),
        visual_extractor=object(),
        phase24_task_queue_client=None,
        input_resolver=Phase1InputResolver.from_mapping_file(mapping_path),
        input_resolver_strict=False,
        **_make_local_audio_providers(),
    )

    with pytest.raises(Phase1InputResolutionError, match="No test-bank mapping found"):
        runner.run_job(
            job_id="job_unmapped_nonstrict",
            source_url="https://youtube.com/watch?v=unknown",
            source_path=None,
            runtime_controls={},
        )


def test_phase1_job_runner_hydrates_canonical_assets_and_uses_mapped_gcs_uris(tmp_path: Path):
    from backend.phase1_runtime.input_resolver import Phase1InputResolver
    from backend.phase1_runtime.runner import Phase1JobRunner

    captured: dict[str, object] = {}
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        """
{
  "https://youtu.be/abc123?si=orig": {
    "local_video_path": "cache/abc123.mp4",
    "local_audio_path": "cache/abc123.wav",
    "video_gcs_uri": "gs://bucket/canonical/abc123.mp4",
    "audio_gcs_uri": "gs://bucket/canonical/abc123.wav"
  }
}
""".strip(),
        encoding="utf-8",
    )

    class _FakeStorage:
        def __init__(self):
            self.downloaded: list[tuple[str, Path]] = []
            self.uploaded: list[tuple[Path, str]] = []

        def download_file(self, *, gcs_uri: str, local_path: Path) -> Path:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            payload = "video-cache" if local_path.suffix == ".mp4" else "audio-cache"
            local_path.write_text(payload, encoding="utf-8")
            self.downloaded.append((gcs_uri, local_path))
            return local_path

        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            self.uploaded.append((local_path, object_name))
            return f"gs://bucket/{object_name}"

    class _FakeVisual:
        def extract(self, *, video_path: Path, workspace):
            captured["visual_video_contents"] = video_path.read_text(encoding="utf-8")
            return {
                "video_metadata": {"fps": 30.0, "duration_ms": 1000},
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "tracks": [],
            }

    storage = _FakeStorage()
    audio_host_client = _make_fake_vibevoice_asr_client()

    runner = Phase1JobRunner(
        working_root=tmp_path,
        audio_extractor=object(),
        storage_client=storage,
        vibevoice_asr_client=audio_host_client,
        visual_extractor=_FakeVisual(),
        phase24_task_queue_client=None,
        input_resolver=Phase1InputResolver.from_mapping_file(mapping_path),
        **_make_local_audio_providers(),
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
    assert captured["visual_video_contents"] == "video-cache"

    # The audio-chain path is now driven by GCS, so we verify the client saw
    # the canonical audio URI rather than inspecting any local audio file.
    assert len(audio_host_client.calls) == 1
    assert audio_host_client.calls[0]["audio_gcs_uri"] == "gs://bucket/canonical/abc123.wav"
    assert audio_host_client.calls[0]["video_gcs_uri"] == "gs://bucket/canonical/abc123.mp4"

    assert storage.downloaded == [
        ("gs://bucket/canonical/abc123.mp4", (tmp_path / "cache" / "abc123.mp4").resolve()),
        ("gs://bucket/canonical/abc123.wav", (tmp_path / "cache" / "abc123.wav").resolve()),
    ]
    assert storage.uploaded == []
