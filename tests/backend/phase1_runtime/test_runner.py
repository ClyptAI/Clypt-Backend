from __future__ import annotations

from pathlib import Path


def test_phase1_job_runner_prepares_media_runs_sidecars_and_optional_pipeline(tmp_path: Path):
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

    class _FakePyannote:
        def run_diarize(self, *, media_url: str):
            calls.append(f"pyannote:{media_url}")
            return {
                "wordLevelTranscription": [{"word": "hello", "start": 0.0, "end": 0.2, "speaker": "S1"}],
                "diarization": [{"speaker": "S1", "start": 0.0, "end": 0.2}],
            }

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

    class _FakePhase14Runner:
        def run(self, *, run_id: str, source_url: str, phase1_outputs, **kwargs):
            calls.append(f"phase14:{run_id}")
            return type("Summary", (), {"model_dump": lambda self, mode="json": {"run_id": run_id, "artifact_paths": {"clip_candidates": "x.json"}}})()

    runner = Phase1JobRunner(
        working_root=tmp_path,
        downloader=_FakeDownloader(),
        audio_extractor=fake_audio_extractor,
        storage_client=_FakeStorage(),
        pyannote_client=_FakePyannote(),
        visual_extractor=_FakeVisual(),
        emotion_provider=_FakeEmotion(),
        yamnet_provider=_FakeYamnet(),
        phase14_runner=_FakePhase14Runner(),
    )

    result = runner.run_job(
        job_id="job_001",
        source_url="https://youtube.com/watch?v=test",
        source_path=None,
        runtime_controls={"run_phase14": True},
    )

    assert result["phase1"]["phase1_audio"]["video_gcs_uri"] == "gs://bucket/phase1/job_001/source_video.mp4"
    assert result["summary"]["artifact_paths"]["clip_candidates"] == "x.json"
    assert calls == [
        "download:https://youtube.com/watch?v=test",
        "audio:source_video.mp4",
        "upload:phase1/job_001/source_video.mp4",
        "pyannote:gs://bucket/phase1/job_001/source_video.mp4",
        "visual:source_video.mp4",
        "emotion:1",
        "yamnet",
        "phase14:job_001",
    ]
