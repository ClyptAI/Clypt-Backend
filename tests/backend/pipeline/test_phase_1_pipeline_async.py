from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from backend.pipeline.phase1_contract import Phase1Manifest



def _manifest_payload(source_url: str = "https://youtube.com/watch?v=x") -> dict:
    return {
        "contract_version": "v2",
        "job_id": "job_123",
        "status": "succeeded",
        "source_video": {"source_url": source_url},
        "canonical_video_gcs_uri": "gs://bucket/phase_1/jobs/job_123/source_video.mp4",
        "artifacts": {
            "transcript": {
                "uri": "gs://bucket/phase_1/jobs/job_123/phase_1_audio.json",
                "source_audio": source_url,
                "video_gcs_uri": "gs://bucket/phase_1/jobs/job_123/source_video.mp4",
                "words": [
                    {
                        "word": "hello",
                        "start_time_ms": 0,
                        "end_time_ms": 500,
                        "speaker_track_id": "Global_Person_0",
                        "speaker_tag": "speaker_0",
                    }
                ],
                "speaker_bindings": [
                    {
                        "track_id": "Global_Person_0",
                        "start_time_ms": 0,
                        "end_time_ms": 500,
                        "word_count": 1,
                    }
                ],
            },
            "visual_tracking": {
                "uri": "gs://bucket/phase_1/jobs/job_123/phase_1_visual.json",
                "source_video": source_url,
                "video_gcs_uri": "gs://bucket/phase_1/jobs/job_123/source_video.mp4",
                "schema_version": "2.0.0",
                "task_type": "person_tracking",
                "coordinate_space": "absolute_original_frame_xyxy",
                "geometry_type": "aabb",
                "class_taxonomy": {"0": "person"},
                "tracking_metrics": {"schema_pass_rate": 1.0},
                "tracks": [
                    {
                        "frame_idx": 0,
                        "local_frame_idx": 0,
                        "chunk_idx": 0,
                        "track_id": "Global_Person_0",
                        "local_track_id": 0,
                        "class_id": 0,
                        "label": "person",
                        "confidence": 0.99,
                        "x1": 100.0,
                        "y1": 120.0,
                        "x2": 220.0,
                        "y2": 340.0,
                        "x_center": 160.0,
                        "y_center": 230.0,
                        "width": 120.0,
                        "height": 220.0,
                        "source": "do_phase1",
                        "geometry_type": "aabb",
                        "bbox_norm_xywh": {
                            "x_center": 0.5,
                            "y_center": 0.5,
                            "width": 0.2,
                            "height": 0.3,
                        },
                    }
                ],
                "face_detections": [],
                "person_detections": [],
                "label_detections": [],
                "object_tracking": [],
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 500}],
                "video_metadata": {
                    "width": 1920,
                    "height": 1080,
                    "fps": 30.0,
                    "duration_ms": 500,
                },
            },
            "events": None,
        },
        "metadata": {
            "runtime": {"provider": "digitalocean", "worker_id": "worker-1", "region": None},
            "timings": {"ingest_ms": 1, "processing_ms": 2, "upload_ms": 3},
            "quality_metrics": {"schema_pass_rate": 1.0, "transcript_coverage": 1.0, "tracking_confidence": 1.0},
            "retry": None,
            "failure": None,
        },
    }


class FakePhase1Client:
    def __init__(self, *, statuses: list[str], manifest: Phase1Manifest):
        self.statuses = list(statuses)
        self.manifest = manifest
        self.submit_calls: list[str] = []
        self.get_job_calls: list[str] = []
        self.get_result_calls: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def submit_job(self, source_url: str):
        self.submit_calls.append(source_url)
        return SimpleNamespace(job_id=self.manifest.job_id, status="queued")

    async def get_job(self, job_id: str):
        self.get_job_calls.append(job_id)
        status = self.statuses.pop(0) if self.statuses else "succeeded"
        return SimpleNamespace(job_id=job_id, status=status, manifest=None, failure=None)

    async def get_result(self, job_id: str):
        self.get_result_calls.append(job_id)
        return self.manifest


class DummyYDL:
    last_instance = None
    prepare_filename_path = "/tmp/video.mp4"

    def __init__(self, opts):
        self.opts = opts
        self.download_calls = 0
        DummyYDL.last_instance = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def extract_info(self, url, download=True):
        self.download_calls += 1
        if self.opts["format"] == "strict-h264":
            raise RuntimeError("requested format not available")
        return {"ext": "mp4", "format_id": "18"}

    def prepare_filename(self, info):
        return self.prepare_filename_path


@pytest.fixture
def phase1_subject():
    import backend.pipeline.phase_1_modal_pipeline as subject

    return subject


@pytest.fixture
def configured_phase1(tmp_path: Path, monkeypatch, phase1_subject):
    output_dir = tmp_path / "outputs"
    download_dir = tmp_path / "downloads"
    output_dir.mkdir()
    download_dir.mkdir()
    video_path = download_dir / "video.mp4"
    audio_path = download_dir / "audio_16k.wav"
    video_path.write_bytes(b"video")
    audio_path.write_bytes(b"audio")

    monkeypatch.setattr(phase1_subject, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(phase1_subject, "DOWNLOAD_DIR", download_dir)
    monkeypatch.setattr(phase1_subject, "PHASE1_NDJSON_PATH", output_dir / "phase_1_visual.ndjson")
    monkeypatch.setattr(phase1_subject, "DETACHED_STATE_PATH", output_dir / "phase_1_detached_state.json")
    monkeypatch.setattr(phase1_subject, "download_media", lambda url: (str(video_path), str(audio_path)))
    monkeypatch.setenv("DO_PHASE1_BASE_URL", "https://do.example")
    monkeypatch.setenv("DO_PHASE1_POLL_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("DO_PHASE1_TIMEOUT_SECONDS", "30")

    return phase1_subject, output_dir, video_path, audio_path


def test_phase_1_main_waits_for_manifest_and_writes_compat_outputs(configured_phase1, monkeypatch, caplog):
    caplog.set_level("INFO")
    subject, output_dir, video_path, _audio_path = configured_phase1
    manifest = Phase1Manifest.model_validate(_manifest_payload())
    fake_client = FakePhase1Client(statuses=["running", "succeeded"], manifest=manifest)

    monkeypatch.setattr(subject, "build_phase1_client", lambda: fake_client)

    result = asyncio.run(subject.main("https://youtube.com/watch?v=x"))

    assert result.job_id == "job_123"
    assert fake_client.submit_calls == ["https://youtube.com/watch?v=x"]
    assert fake_client.get_result_calls == ["job_123"]

    visual_payload = json.loads((output_dir / "phase_1_visual.json").read_text())
    audio_payload = json.loads((output_dir / "phase_1_audio.json").read_text())
    ndjson_rows = (output_dir / "phase_1_visual.ndjson").read_text().strip().splitlines()

    assert visual_payload["uri"] == manifest.artifacts.visual_tracking.uri
    assert visual_payload["video_gcs_uri"] == manifest.canonical_video_gcs_uri
    assert audio_payload["uri"] == manifest.artifacts.transcript.uri
    assert audio_payload["video_gcs_uri"] == manifest.canonical_video_gcs_uri
    assert len(ndjson_rows) == 2
    assert not subject.DETACHED_STATE_PATH.exists()
    assert video_path.exists()
    assert "queued" in caplog.text or "running" in caplog.text


def test_phase_1_main_resumes_existing_job_without_resubmitting(configured_phase1, monkeypatch, caplog):
    caplog.set_level("INFO")
    subject, output_dir, _video_path, _audio_path = configured_phase1
    manifest = Phase1Manifest.model_validate(_manifest_payload())
    fake_client = FakePhase1Client(statuses=["running", "succeeded"], manifest=manifest)

    subject.DETACHED_STATE_PATH.write_text(
        json.dumps(
            {
                "provider": "digitalocean",
                "job_id": "job_123",
                "source_url": "https://youtube.com/watch?v=x",
                "status": "running",
            }
        )
    )
    monkeypatch.setattr(subject, "build_phase1_client", lambda: fake_client)

    result = asyncio.run(subject.main("https://youtube.com/watch?v=x"))

    assert result.job_id == "job_123"
    assert fake_client.submit_calls == []
    assert fake_client.get_job_calls == ["job_123", "job_123"]
    assert "Resuming" in caplog.text
    assert not subject.DETACHED_STATE_PATH.exists()
    assert (output_dir / "phase_1_audio.json").exists()


def test_phase_1_main_persists_resume_state_and_logs_timeout(configured_phase1, monkeypatch, caplog):
    caplog.set_level("INFO")
    subject, _output_dir, _video_path, _audio_path = configured_phase1
    manifest = Phase1Manifest.model_validate(_manifest_payload())
    fake_client = FakePhase1Client(statuses=["running", "running", "running"], manifest=manifest)

    monkeypatch.setattr(subject, "build_phase1_client", lambda: fake_client)
    monkeypatch.setenv("DO_PHASE1_TIMEOUT_SECONDS", "0")

    with pytest.raises(TimeoutError):
        asyncio.run(subject.main("https://youtube.com/watch?v=x"))

    saved_state = json.loads(subject.DETACHED_STATE_PATH.read_text())
    assert saved_state["provider"] == "digitalocean"
    assert saved_state["job_id"] == "job_123"
    assert saved_state["source_url"] == "https://youtube.com/watch?v=x"
    assert "Timeout" in caplog.text or "resume" in caplog.text


def test_run_pipeline_consumes_async_phase_1_manifest(monkeypatch, caplog):
    caplog.set_level("INFO")
    import backend.pipeline.run_pipeline as subject

    manifest = Phase1Manifest.model_validate(_manifest_payload())
    calls: list[str] = []

    async def fake_phase1_main(*, youtube_url: str | None = None):
        calls.append(f"phase1:{youtube_url}")
        return manifest

    def _stub(label: str):
        return lambda: calls.append(label)

    phase1_module = ModuleType("pipeline.phase_1_modal_pipeline")
    phase1_module.main = fake_phase1_main

    phase2a_module = ModuleType("pipeline.phase_2a_make_nodes")
    phase2a_module.main = _stub("phase2a")

    phase2b_module = ModuleType("pipeline.phase_2b_draw_edges")
    phase2b_module.main = _stub("phase2b")

    phase3_module = ModuleType("pipeline.phase_3_multimodal_embeddings")
    phase3_module.main = _stub("phase3")

    phase4_module = ModuleType("pipeline.phase_4_store_graph")
    phase4_module.main = _stub("phase4")

    phase5_module = ModuleType("pipeline.phase_5_auto_curate")
    phase5_module.main = _stub("phase5")

    monkeypatch.setitem(sys.modules, "pipeline.phase_1_modal_pipeline", phase1_module)
    monkeypatch.setitem(sys.modules, "pipeline.phase_2a_make_nodes", phase2a_module)
    monkeypatch.setitem(sys.modules, "pipeline.phase_2b_draw_edges", phase2b_module)
    monkeypatch.setitem(sys.modules, "pipeline.phase_3_multimodal_embeddings", phase3_module)
    monkeypatch.setitem(sys.modules, "pipeline.phase_4_store_graph", phase4_module)
    monkeypatch.setitem(sys.modules, "pipeline.phase_5_auto_curate", phase5_module)
    monkeypatch.setattr("builtins.input", lambda _prompt='': "https://youtube.com/watch?v=x")
    monkeypatch.setattr(subject, "reencode_video", lambda: calls.append("reencode"))
    monkeypatch.setattr(subject, "setup_render_engine", lambda: calls.append("render-setup"))
    monkeypatch.setattr(subject, "run_fetch_tracking", lambda: calls.append("fetch-tracking"))
    monkeypatch.setattr(subject, "run_remotion_render", lambda: calls.append("render"))

    subject.main()

    assert calls == [
        "phase1:https://youtube.com/watch?v=x",
        "reencode",
        "phase2a",
        "phase2b",
        "phase3",
        "phase4",
        "phase5",
        "render-setup",
        "fetch-tracking",
        "render",
    ]
    assert manifest.job_id in caplog.text


def test_download_media_tries_h264_first_then_falls_back(monkeypatch, tmp_path, phase1_subject):
    subject = phase1_subject
    monkeypatch.setattr(subject, "DOWNLOAD_DIR", tmp_path / "downloads")
    monkeypatch.setattr(subject, "OUTPUT_DIR", tmp_path / "outputs")
    subject.DOWNLOAD_DIR.mkdir()
    subject.OUTPUT_DIR.mkdir()
    monkeypatch.setattr(subject, "ensure_h264_local", lambda path: path)
    monkeypatch.setattr(subject, "probe_video_stream", lambda path: (1920, 1080, "30/1"))
    monkeypatch.setattr(subject, "probe_duration_seconds", lambda path: 10.0)
    def fake_run(cmd, *args, **kwargs):
        audio_out = subject.DOWNLOAD_DIR / "audio_16k.wav"
        audio_out.write_bytes(b"audio")
        return None

    monkeypatch.setattr(subject, "subprocess", SimpleNamespace(run=fake_run, DEVNULL=-1))
    monkeypatch.setattr(subject.yt_dlp, "YoutubeDL", DummyYDL)
    (tmp_path / "video.mp4").write_bytes(b"video")
    monkeypatch.setattr(subject.log, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(subject.log, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(subject, "YTDLP_VIDEO_FORMAT", "fallback-any")
    monkeypatch.setattr(subject, "YTDLP_H264_PREFERRED_FORMAT", "strict-h264")
    DummyYDL.prepare_filename_path = str(tmp_path / "video.mp4")

    video_path, audio_path = subject.download_media("https://youtube.com/watch?v=x")

    assert video_path == str(tmp_path / "video.mp4")
    assert audio_path == str(subject.DOWNLOAD_DIR / "audio_16k.wav")
    assert DummyYDL.last_instance is not None
    assert DummyYDL.last_instance.opts["format"] == "fallback-any"
