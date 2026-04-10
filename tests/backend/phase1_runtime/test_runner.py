from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import threading

import pytest


def test_phase1_job_runner_enqueues_phase24_when_queue_mode_enabled(tmp_path: Path):
    from backend.phase1_runtime.runner import Phase1JobRunner
    from backend.repository.models import Phase24JobRecord, RunRecord

    calls: list[str] = []
    captured: dict[str, object] = {}

    class _FakeDownloader:
        def download(self, *, source_url: str, output_path: Path) -> Path:
            calls.append(f"download:{source_url}")
            output_path.write_text("video", encoding="utf-8")
            return output_path

    def fake_audio_extractor(*, video_path: Path, audio_path: Path) -> None:
        calls.append(f"audio:{video_path.name}")
        audio_path.write_text("audio", encoding="utf-8")

    class _FakeStorage:
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            calls.append(f"upload:{object_name}")
            return f"gs://bucket/{object_name}"

    class _FakeVibeVoice:
        supports_concurrent_visual = True

        def run(self, *, audio_path: Path, context_info=None):
            calls.append(f"vibevoice:{audio_path.name}")
            return [{"Start": 0.0, "End": 0.2, "Speaker": 0, "Content": "hello"}]

    class _FakeForcedAligner:
        def run(self, *, audio_path: Path, turns: list[dict]):
            calls.append(f"forced_aligner:{len(turns)}")
            return [{"word_id": "w_000001", "text": "hello", "start_ms": 0, "end_ms": 200, "speaker_id": "SPEAKER_0"}]

    class _FakeVisual:
        def extract(self, *, video_path: Path, workspace):
            calls.append(f"visual:{video_path.name}")
            return {
                "video_metadata": {"fps": 30.0, "duration_ms": 1000},
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "tracks": [],
            }

    class _FakeEmotion:
        def run(self, *, audio_path: Path, turns: list[dict]):
            calls.append(f"emotion:{len(turns)}")
            return {"segments": []}

    class _FakeYamnet:
        def run(self, *, audio_path: Path):
            calls.append("yamnet")
            return {"events": []}

    class _FakeQueueClient:
        def enqueue_phase24(self, *, run_id: str, payload: dict, worker_url: str | None = None) -> str:
            captured["run_id"] = run_id
            captured["payload"] = payload
            captured["worker_url"] = worker_url
            return "projects/test/locations/us-central1/queues/phase24/tasks/run-001"

    class _FakeRepository:
        def __init__(self) -> None:
            self.run_record: RunRecord | None = None
            self.job_record: Phase24JobRecord | None = None

        def upsert_run(self, record: RunRecord) -> RunRecord:
            self.run_record = record
            return record

        def upsert_phase24_job(self, record: Phase24JobRecord) -> Phase24JobRecord:
            self.job_record = record
            return record

    class _FakePhase14Runner:
        def run(self, *, run_id: str, source_url: str, phase1_outputs, **kwargs):
            raise AssertionError("inline phase14 runner should not be called when queue mode is enabled")

    repository = _FakeRepository()

    runner = Phase1JobRunner(
        working_root=tmp_path,
        downloader=_FakeDownloader(),
        audio_extractor=fake_audio_extractor,
        storage_client=_FakeStorage(),
        vibevoice_provider=_FakeVibeVoice(),
        forced_aligner=_FakeForcedAligner(),
        visual_extractor=_FakeVisual(),
        emotion_provider=_FakeEmotion(),
        yamnet_provider=_FakeYamnet(),
        phase14_runner=_FakePhase14Runner(),
        phase24_task_queue_client=_FakeQueueClient(),
        phase14_repository=repository,
        phase24_worker_url="https://phase24-worker.example.com/tasks/phase24",
        phase24_query_version="graph-v2",
    )

    result = runner.run_job(
        job_id="job_001",
        source_url="https://youtube.com/watch?v=test",
        source_path=None,
        runtime_controls={"run_phase14": True},
    )

    assert result["phase1"]["phase1_audio"]["video_gcs_uri"] == "gs://bucket/phase1/job_001/source_video.mp4"
    assert result["summary"] == {
        "run_id": "job_001",
        "status": "queued",
        "task_name": "projects/test/locations/us-central1/queues/phase24/tasks/run-001",
        "artifact_paths": {},
    }
    assert captured["run_id"] == "job_001"
    assert captured["worker_url"] == "https://phase24-worker.example.com/tasks/phase24"
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
    assert repository.job_record.task_name == "projects/test/locations/us-central1/queues/phase24/tasks/run-001"
    assert repository.job_record.metadata["query_version"] == "graph-v2"
    assert calls == [
        "download:https://youtube.com/watch?v=test",
        "audio:source_video.mp4",
        "upload:phase1/job_001/source_video.mp4",
        "visual:source_video.mp4",
        "vibevoice:source_audio.wav",
        "forced_aligner:1",
        "emotion:1",
        "yamnet",
        "upload:phase1/job_001/phase24_inputs/phase1_outputs.json",
    ]


def test_phase1_job_runner_queue_mode_enqueues_before_visual_completes(tmp_path: Path):
    from backend.phase1_runtime.runner import Phase1JobRunner

    calls: list[str] = []
    queue_called = threading.Event()

    class _FakeDownloader:
        def download(self, *, source_url: str, output_path: Path) -> Path:
            calls.append(f"download:{source_url}")
            output_path.write_text("video", encoding="utf-8")
            return output_path

    def fake_audio_extractor(*, video_path: Path, audio_path: Path) -> None:
        calls.append(f"audio:{video_path.name}")
        audio_path.write_text("audio", encoding="utf-8")

    class _FakeStorage:
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            calls.append(f"upload:{object_name}")
            return f"gs://bucket/{object_name}"

    class _FakeVibeVoice:
        supports_concurrent_visual = True

        def run(self, *, audio_path: Path, context_info=None):
            calls.append(f"vibevoice:{audio_path.name}")
            return [{"Start": 0.0, "End": 0.2, "Speaker": 0, "Content": "hello"}]

    class _FakeForcedAligner:
        def run(self, *, audio_path: Path, turns: list[dict]):
            calls.append(f"forced_aligner:{len(turns)}")
            return [{"word_id": "w_000001", "text": "hello", "start_ms": 0, "end_ms": 200, "speaker_id": "SPEAKER_0"}]

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

    class _FakeEmotion:
        def run(self, *, audio_path: Path, turns: list[dict]):
            calls.append(f"emotion:{len(turns)}")
            return {"segments": []}

    class _FakeYamnet:
        def run(self, *, audio_path: Path):
            calls.append("yamnet")
            return {"events": []}

    class _FakeQueueClient:
        def enqueue_phase24(self, *, run_id: str, payload: dict, worker_url: str | None = None) -> str:
            calls.append(f"enqueue:{run_id}")
            queue_called.set()
            return "projects/test/locations/us-central1/queues/phase24/tasks/run-002"

    runner = Phase1JobRunner(
        working_root=tmp_path,
        downloader=_FakeDownloader(),
        audio_extractor=fake_audio_extractor,
        storage_client=_FakeStorage(),
        vibevoice_provider=_FakeVibeVoice(),
        forced_aligner=_FakeForcedAligner(),
        visual_extractor=_FakeVisual(),
        emotion_provider=_FakeEmotion(),
        yamnet_provider=_FakeYamnet(),
        phase14_runner=object(),
        phase24_task_queue_client=_FakeQueueClient(),
        phase24_worker_url="https://phase24-worker.example.com/tasks/phase24",
        phase24_query_version="graph-v2",
    )

    result = runner.run_job(
        job_id="job_002",
        source_url="https://youtube.com/watch?v=test",
        source_path=None,
        runtime_controls={"run_phase14": True},
    )

    assert result["summary"]["status"] == "queued"
    assert "upload:phase1/job_002/phase24_inputs/phase1_outputs.json" in calls
    assert queue_called.is_set()
    assert calls.index("enqueue:job_002") < calls.index("visual:done")


def test_phase1_job_runner_uses_inline_phase14_when_queue_mode_disabled(tmp_path: Path):
    from backend.phase1_runtime.runner import Phase1JobRunner

    calls: list[str] = []

    class _FakeDownloader:
        def download(self, *, source_url: str, output_path: Path) -> Path:
            calls.append(f"download:{source_url}")
            output_path.write_text("video", encoding="utf-8")
            return output_path

    def fake_audio_extractor(*, video_path: Path, audio_path: Path) -> None:
        calls.append(f"audio:{video_path.name}")
        audio_path.write_text("audio", encoding="utf-8")

    class _FakeStorage:
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            calls.append(f"upload:{object_name}")
            return f"gs://bucket/{object_name}"

    class _FakeVibeVoice:
        supports_concurrent_visual = True

        def run(self, *, audio_path: Path, context_info=None):
            calls.append(f"vibevoice:{audio_path.name}")
            return [{"Start": 0.0, "End": 0.2, "Speaker": 0, "Content": "hello"}]

    class _FakeForcedAligner:
        def run(self, *, audio_path: Path, turns: list[dict]):
            calls.append(f"forced_aligner:{len(turns)}")
            return [{"word_id": "w_000001", "text": "hello", "start_ms": 0, "end_ms": 200, "speaker_id": "SPEAKER_0"}]

    class _FakeVisual:
        def extract(self, *, video_path: Path, workspace):
            calls.append(f"visual:{video_path.name}")
            return {
                "video_metadata": {"fps": 30.0, "duration_ms": 1000},
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "tracks": [],
            }

    class _FakeEmotion:
        def run(self, *, audio_path: Path, turns: list[dict]):
            calls.append(f"emotion:{len(turns)}")
            return {"segments": []}

    class _FakeYamnet:
        def run(self, *, audio_path: Path):
            calls.append("yamnet")
            return {"events": []}

    class _FakeQueueClient:
        def enqueue_phase24(self, *, run_id: str, payload: dict, worker_url: str | None = None) -> str:
            raise AssertionError("queue client should not be used when queue mode is disabled")

    class _FakePhase14Runner:
        config = type("Config", (), {"output_root": tmp_path})()

        def run(self, *, run_id: str, source_url: str, phase1_outputs, **kwargs):
            calls.append(f"phase14:{run_id}")
            return type(
                "Summary",
                (),
                {
                    "model_dump": lambda self, mode="json": {
                        "run_id": run_id,
                        "artifact_paths": {"clip_candidates": "x.json"},
                    }
                },
            )()

    runner = Phase1JobRunner(
        working_root=tmp_path,
        downloader=_FakeDownloader(),
        audio_extractor=fake_audio_extractor,
        storage_client=_FakeStorage(),
        vibevoice_provider=_FakeVibeVoice(),
        forced_aligner=_FakeForcedAligner(),
        visual_extractor=_FakeVisual(),
        emotion_provider=_FakeEmotion(),
        yamnet_provider=_FakeYamnet(),
        phase14_runner=_FakePhase14Runner(),
        phase24_task_queue_client=_FakeQueueClient(),
    )

    result = runner.run_job(
        job_id="job_001",
        source_url="https://youtube.com/watch?v=test",
        source_path=None,
        runtime_controls={"run_phase14": True, "phase24_queue_enabled": False},
    )

    assert result["summary"]["artifact_paths"]["clip_candidates"] == "x.json"
    assert calls == [
        "download:https://youtube.com/watch?v=test",
        "audio:source_video.mp4",
        "upload:phase1/job_001/source_video.mp4",
        "visual:source_video.mp4",
        "vibevoice:source_audio.wav",
        "forced_aligner:1",
        "emotion:1",
        "yamnet",
        "phase14:job_001",
    ]


def test_phase1_job_runner_fails_fast_when_queue_mode_enabled_but_unconfigured(tmp_path: Path):
    from backend.phase1_runtime.runner import Phase1JobRunner

    class _FakeDownloader:
        def download(self, *, source_url: str, output_path: Path) -> Path:
            output_path.write_text("video", encoding="utf-8")
            return output_path

    def fake_audio_extractor(*, video_path: Path, audio_path: Path) -> None:
        audio_path.write_text("audio", encoding="utf-8")

    class _FakeStorage:
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            return f"gs://bucket/{object_name}"

    class _FakePhase14Runner:
        def run(self, *, run_id: str, source_url: str, phase1_outputs, **kwargs):
            raise AssertionError("inline phase14 runner should not be used in queue mode")

    runner = Phase1JobRunner(
        working_root=tmp_path,
        downloader=_FakeDownloader(),
        audio_extractor=fake_audio_extractor,
        storage_client=_FakeStorage(),
        vibevoice_provider=object(),
        forced_aligner=object(),
        visual_extractor=object(),
        emotion_provider=object(),
        yamnet_provider=object(),
        phase14_runner=_FakePhase14Runner(),
        phase24_task_queue_client=None,
    )

    with pytest.raises(RuntimeError, match="Cloud Tasks client is unavailable"):
        runner.run_job(
            job_id="job_001",
            source_url="https://youtube.com/watch?v=test",
            source_path=None,
            runtime_controls={"run_phase14": True, "phase24_queue_enabled": True},
        )


def test_phase1_job_runner_fails_inline_mode_when_phase14_runner_missing(tmp_path: Path):
    from backend.phase1_runtime.runner import Phase1JobRunner

    class _FakeDownloader:
        def download(self, *, source_url: str, output_path: Path) -> Path:
            output_path.write_text("video", encoding="utf-8")
            return output_path

    def fake_audio_extractor(*, video_path: Path, audio_path: Path) -> None:
        audio_path.write_text("audio", encoding="utf-8")

    class _FakeStorage:
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            return f"gs://bucket/{object_name}"

    runner = Phase1JobRunner(
        working_root=tmp_path,
        downloader=_FakeDownloader(),
        audio_extractor=fake_audio_extractor,
        storage_client=_FakeStorage(),
        vibevoice_provider=object(),
        forced_aligner=object(),
        visual_extractor=object(),
        emotion_provider=object(),
        yamnet_provider=object(),
        phase14_runner=None,
        phase24_task_queue_client=None,
    )

    with pytest.raises(RuntimeError, match="phase14_runner is unavailable"):
        runner.run_job(
            job_id="job_001",
            source_url="https://youtube.com/watch?v=test",
            source_path=None,
        runtime_controls={"run_phase14": True, "phase24_queue_enabled": False},
    )


def test_phase1_job_runner_uses_test_bank_media_and_preserves_posted_source_url(tmp_path: Path):
    from backend.phase1_runtime.input_resolver import Phase1InputResolver
    from backend.phase1_runtime.runner import Phase1JobRunner

    calls: list[str] = []
    captured: dict[str, object] = {}

    mapped_video = tmp_path / "fixtures" / "mapped-video.mp4"
    mapped_video.parent.mkdir(parents=True)
    mapped_video.write_text("mapped-video", encoding="utf-8")
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        """
{
  "https://youtube.com/watch?v=test": {
    "local_video_path": "fixtures/mapped-video.mp4"
  }
}
""".strip(),
        encoding="utf-8",
    )

    class _FailingDownloader:
        def download(self, *, source_url: str, output_path: Path) -> Path:
            raise AssertionError("downloader should not be called when test-bank mapping is enabled")

    def fake_audio_extractor(*, video_path: Path, audio_path: Path) -> None:
        calls.append(f"audio:{video_path.read_text(encoding='utf-8')}")
        audio_path.write_text("audio", encoding="utf-8")

    class _FakeStorage:
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            captured["uploaded_video"] = local_path.read_text(encoding="utf-8")
            return f"gs://bucket/{object_name}"

    class _FakeVibeVoice:
        supports_concurrent_visual = True

        def run(self, *, audio_path: Path, context_info=None):
            return [{"Start": 0.0, "End": 0.2, "Speaker": 0, "Content": "hello"}]

    class _FakeForcedAligner:
        def run(self, *, audio_path: Path, turns: list[dict]):
            return [{"word_id": "w_000001", "text": "hello", "start_ms": 0, "end_ms": 200, "speaker_id": "SPEAKER_0"}]

    class _FakeVisual:
        def extract(self, *, video_path: Path, workspace):
            captured["visual_video_contents"] = video_path.read_text(encoding="utf-8")
            return {
                "video_metadata": {"fps": 30.0, "duration_ms": 1000},
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "tracks": [],
            }

    class _FakeEmotion:
        def run(self, *, audio_path: Path, turns: list[dict]):
            return {"segments": []}

    class _FakeYamnet:
        def run(self, *, audio_path: Path):
            return {"events": []}

    class _FakePhase14Runner:
        config = type("Config", (), {"output_root": tmp_path})()

        def run(self, *, run_id: str, source_url: str, phase1_outputs, **kwargs):
            captured["phase14_source_url"] = source_url
            return type(
                "Summary",
                (),
                {
                    "model_dump": lambda self, mode="json": {
                        "run_id": run_id,
                        "artifact_paths": {},
                    }
                },
            )()

    runner = Phase1JobRunner(
        working_root=tmp_path,
        downloader=_FailingDownloader(),
        audio_extractor=fake_audio_extractor,
        storage_client=_FakeStorage(),
        vibevoice_provider=_FakeVibeVoice(),
        forced_aligner=_FakeForcedAligner(),
        visual_extractor=_FakeVisual(),
        emotion_provider=_FakeEmotion(),
        yamnet_provider=_FakeYamnet(),
        phase14_runner=None,
        phase24_task_queue_client=None,
        input_resolver=Phase1InputResolver.from_mapping_file(mapping_path),
    )

    result = runner.run_job(
        job_id="job_003",
        source_url="https://youtube.com/watch?v=test",
        source_path=None,
        runtime_controls={},
    )

    assert captured["uploaded_video"] == "mapped-video"
    assert captured["visual_video_contents"] == "mapped-video"
    assert result["phase1"]["phase1_audio"]["source_audio"] == "https://youtube.com/watch?v=test"
    assert calls == ["audio:mapped-video"]


def test_phase1_job_runner_does_not_apply_input_resolver_to_source_path_inputs(tmp_path: Path):
    from backend.phase1_runtime.runner import Phase1JobRunner

    source_path = tmp_path / "source.mp4"
    source_path.write_text("source-video", encoding="utf-8")
    calls: list[str] = []
    captured: dict[str, object] = {}

    class _RecordingResolver:
        def resolve_source_path(self, *, source_url: str) -> Path:
            calls.append(f"resolve:{source_url}")
            raise AssertionError("source_path inputs should bypass the input resolver")

    class _FakeDownloader:
        def download(self, *, source_url: str, output_path: Path) -> Path:
            raise AssertionError("downloader should not be called for source_path inputs")

    def fake_audio_extractor(*, video_path: Path, audio_path: Path) -> None:
        audio_path.write_text(video_path.read_text(encoding="utf-8"), encoding="utf-8")

    class _FakeStorage:
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            calls.append(f"upload:{local_path.read_text(encoding='utf-8')}")
            return f"gs://bucket/{object_name}"

    class _FakeVibeVoice:
        supports_concurrent_visual = True

        def run(self, *, audio_path: Path, context_info=None):
            return [{"Start": 0.0, "End": 0.2, "Speaker": 0, "Content": "hello"}]

    class _FakeForcedAligner:
        def run(self, *, audio_path: Path, turns: list[dict]):
            return [{"word_id": "w_000001", "text": "hello", "start_ms": 0, "end_ms": 200, "speaker_id": "SPEAKER_0"}]

    class _FakeVisual:
        def extract(self, *, video_path: Path, workspace):
            captured["visual_video_contents"] = video_path.read_text(encoding="utf-8")
            return {
                "video_metadata": {"fps": 30.0, "duration_ms": 1000},
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "tracks": [],
            }

    class _FakeEmotion:
        def run(self, *, audio_path: Path, turns: list[dict]):
            return {"segments": []}

    class _FakeYamnet:
        def run(self, *, audio_path: Path):
            return {"events": []}

    class _FakePhase14Runner:
        config = type("Config", (), {"output_root": tmp_path})()

        def run(self, *, run_id: str, source_url: str, phase1_outputs, **kwargs):
            captured["phase14_source_url"] = source_url
            return type(
                "Summary",
                (),
                {
                    "model_dump": lambda self, mode="json": {
                        "run_id": run_id,
                        "artifact_paths": {},
                    }
                },
            )()

    runner = Phase1JobRunner(
        working_root=tmp_path,
        downloader=_FakeDownloader(),
        audio_extractor=fake_audio_extractor,
        storage_client=_FakeStorage(),
        vibevoice_provider=_FakeVibeVoice(),
        forced_aligner=_FakeForcedAligner(),
        visual_extractor=_FakeVisual(),
        emotion_provider=_FakeEmotion(),
        yamnet_provider=_FakeYamnet(),
        phase14_runner=_FakePhase14Runner(),
        phase24_task_queue_client=None,
        input_resolver=_RecordingResolver(),
    )

    result = runner.run_job(
        job_id="job_004",
        source_url=None,
        source_path=str(source_path),
        runtime_controls={"run_phase14": True, "phase24_queue_enabled": False},
    )

    assert calls == ["upload:source-video"]
    assert "resolve:" not in "".join(calls)
    assert captured["visual_video_contents"] == "source-video"
    assert captured["phase14_source_url"] == str(source_path)
    assert result["phase1"]["phase1_audio"]["source_audio"] == str(source_path)


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

    class _FailingDownloader:
        def download(self, *, source_url: str, output_path: Path) -> Path:
            raise AssertionError("downloader should not be reached when strict unmapped resolution fails")

    runner = Phase1JobRunner(
        working_root=tmp_path,
        downloader=_FailingDownloader(),
        audio_extractor=object(),
        storage_client=object(),
        vibevoice_provider=object(),
        forced_aligner=object(),
        visual_extractor=object(),
        emotion_provider=object(),
        yamnet_provider=object(),
        phase14_runner=None,
        phase24_task_queue_client=None,
        input_resolver=Phase1InputResolver.from_mapping_file(mapping_path),
        input_resolver_strict=True,
    )

    with pytest.raises(Phase1InputResolutionError, match="No test-bank mapping found"):
        runner.run_job(
            job_id="job_unmapped_strict",
            source_url="https://youtube.com/watch?v=unknown",
            source_path=None,
            runtime_controls={},
        )


def test_phase1_job_runner_falls_back_to_direct_source_when_unmapped_and_not_strict(tmp_path: Path):
    from backend.phase1_runtime.input_resolver import Phase1InputResolver
    from backend.phase1_runtime.runner import Phase1JobRunner

    calls: list[str] = []
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

    class _Downloader:
        def download(self, *, source_url: str, output_path: Path) -> Path:
            calls.append(f"download:{source_url}")
            output_path.write_text("downloaded-video", encoding="utf-8")
            return output_path

    def fake_audio_extractor(*, video_path: Path, audio_path: Path) -> None:
        audio_path.write_text("audio", encoding="utf-8")

    class _FakeStorage:
        def upload_file(self, *, local_path: Path, object_name: str) -> str:
            return f"gs://bucket/{object_name}"

    class _FakeVibeVoice:
        supports_concurrent_visual = True

        def run(self, *, audio_path: Path, context_info=None):
            return [{"Start": 0.0, "End": 0.2, "Speaker": 0, "Content": "hello"}]

    class _FakeForcedAligner:
        def run(self, *, audio_path: Path, turns: list[dict]):
            return [{"word_id": "w_000001", "text": "hello", "start_ms": 0, "end_ms": 200, "speaker_id": "SPEAKER_0"}]

    class _FakeVisual:
        def extract(self, *, video_path: Path, workspace):
            return {
                "video_metadata": {"fps": 30.0, "duration_ms": 1000},
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "tracks": [],
            }

    class _FakeEmotion:
        def run(self, *, audio_path: Path, turns: list[dict]):
            return {"segments": []}

    class _FakeYamnet:
        def run(self, *, audio_path: Path):
            return {"events": []}

    runner = Phase1JobRunner(
        working_root=tmp_path,
        downloader=_Downloader(),
        audio_extractor=fake_audio_extractor,
        storage_client=_FakeStorage(),
        vibevoice_provider=_FakeVibeVoice(),
        forced_aligner=_FakeForcedAligner(),
        visual_extractor=_FakeVisual(),
        emotion_provider=_FakeEmotion(),
        yamnet_provider=_FakeYamnet(),
        phase14_runner=None,
        phase24_task_queue_client=None,
        input_resolver=Phase1InputResolver.from_mapping_file(mapping_path),
        input_resolver_strict=False,
    )

    result = runner.run_job(
        job_id="job_unmapped_nonstrict",
        source_url="https://youtube.com/watch?v=unknown",
        source_path=None,
        runtime_controls={},
    )

    assert calls == ["download:https://youtube.com/watch?v=unknown"]
    assert result["phase1"]["phase1_audio"]["source_audio"] == "https://youtube.com/watch?v=unknown"
