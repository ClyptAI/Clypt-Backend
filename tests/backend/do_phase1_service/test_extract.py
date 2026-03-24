from contextlib import contextmanager
from pathlib import Path
import sys
import types
import threading
import time

import numpy as np
import pytest

from backend.do_phase1_service.extract import host_extraction_lock, run_extraction_job
from backend.pipeline import phase_1_do_pipeline as phase1_pipeline
from backend.do_phase1_worker import ClyptWorker


class FakeStorage:
    def __init__(self):
        self.uploaded = []

    def upload_file(self, source_path, object_name: str) -> str:
        self.uploaded.append((str(source_path), object_name))
        return f"gs://bucket/{object_name}"

    def upload_bytes(self, data: bytes, object_name: str) -> str:
        self.uploaded.append((len(data), object_name))
        return f"gs://bucket/{object_name}"


def test_extract_job_produces_manifest_and_artifacts(tmp_path: Path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")

    monkeypatch.setattr(
        "backend.do_phase1_service.extract.download_media",
        lambda url: (str(video_path), str(audio_path)),
    )
    monkeypatch.setattr(
        "backend.do_phase1_service.extract.execute_local_extraction",
        lambda video_path, audio_path, youtube_url: {
            "status": "success",
            "phase_1_visual": {
                "source_video": youtube_url,
                "schema_version": "2.0.0",
                "task_type": "person_tracking",
                "coordinate_space": "absolute_original_frame_xyxy",
                "geometry_type": "aabb",
                "class_taxonomy": {"0": "person"},
                "tracking_metrics": {"schema_pass_rate": 1.0},
                "tracks": [],
                "face_detections": [],
                "person_detections": [],
                "label_detections": [],
                "object_tracking": [],
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "video_metadata": {"width": 1920, "height": 1080, "fps": 30.0, "duration_ms": 1000},
            },
            "phase_1_audio": {
                "source_audio": youtube_url,
                "words": [],
                "speaker_bindings": [],
            },
        },
    )
    monkeypatch.setattr("backend.do_phase1_service.extract.enrich_visual_ledger_for_downstream", lambda phase_1_visual, phase_1_audio, video_path: phase_1_visual)
    monkeypatch.setattr("backend.do_phase1_service.extract.validate_phase_handoff", lambda visual_ledger, audio_ledger: None)

    result = run_extraction_job(
        source_url="https://youtube.com/watch?v=x",
        job_id="job_123",
        output_dir=tmp_path,
        storage=FakeStorage(),
        host_lock_path=tmp_path / "extract.lock",
    )

    assert result.status == "succeeded"
    assert result.artifacts.transcript.uri
    assert result.artifacts.visual_tracking.uri
    assert result.canonical_video_gcs_uri == "gs://bucket/phase_1/jobs/job_123/source_video.mp4"
    assert result.metadata.retry is None


def test_extract_job_records_stage_timings(tmp_path: Path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")

    monkeypatch.setattr(
        "backend.do_phase1_service.extract.download_media",
        lambda url: (str(video_path), str(audio_path)),
    )
    monkeypatch.setattr(
        "backend.do_phase1_service.extract.execute_local_extraction",
        lambda video_path, audio_path, youtube_url: {
            "status": "success",
            "phase_1_visual": {
                "source_video": youtube_url,
                "schema_version": "2.0.0",
                "task_type": "person_tracking",
                "coordinate_space": "absolute_original_frame_xyxy",
                "geometry_type": "aabb",
                "class_taxonomy": {"0": "person"},
                "tracking_metrics": {"schema_pass_rate": 1.0},
                "tracks": [],
                "face_detections": [],
                "person_detections": [],
                "label_detections": [],
                "object_tracking": [],
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "video_metadata": {"width": 1920, "height": 1080, "fps": 30.0, "duration_ms": 1000},
            },
            "phase_1_audio": {
                "source_audio": youtube_url,
                "words": [],
                "speaker_bindings": [],
            },
        },
    )
    monkeypatch.setattr("backend.do_phase1_service.extract.enrich_visual_ledger_for_downstream", lambda phase_1_visual, phase_1_audio, video_path: phase_1_visual)
    monkeypatch.setattr("backend.do_phase1_service.extract.validate_phase_handoff", lambda visual_ledger, audio_ledger: None)

    perf_counter_values = iter([10.0, 12.5, 20.0, 21.0])
    monkeypatch.setattr("backend.do_phase1_service.extract.time.perf_counter", lambda: next(perf_counter_values))

    result = run_extraction_job(
        source_url="https://youtube.com/watch?v=timings",
        job_id="job_timing",
        output_dir=tmp_path,
        storage=FakeStorage(),
        host_lock_path=tmp_path / "extract.lock",
    )

    assert result.metadata.timings.model_dump() == {
        "ingest_ms": 2500,
        "processing_ms": 7500,
        "upload_ms": 1000,
    }


def test_extract_job_writes_job_log_and_forwards_progress(tmp_path: Path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    progress_events = []
    log_path = tmp_path / "logs" / "job_123.log"

    monkeypatch.setattr(
        "backend.do_phase1_service.extract.download_media",
        lambda url: (str(video_path), str(audio_path)),
    )

    def fake_execute_local_extraction(*, video_path, audio_path, youtube_url):
        print("[Phase 1] Received video (10.0 MB) + audio (1.0 MB)")
        print("[Phase 1] Step 1+2/4: Running Parakeet ASR + YOLO26 tracking concurrently...")
        print("Tracking complete: 0 boxes across 10 frames, 20.0 effective fps")
        print("[Phase 1] Step 3/4: Clustering tracklets into global IDs...")
        print("[Phase 1] Step 4/4: Running speaker binding...")
        print("[Phase 1] Complete — 0 words, 0 tracks")
        return {
            "status": "success",
            "phase_1_visual": {
                "source_video": youtube_url,
                "schema_version": "2.0.0",
                "task_type": "person_tracking",
                "coordinate_space": "absolute_original_frame_xyxy",
                "geometry_type": "aabb",
                "class_taxonomy": {"0": "person"},
                "tracking_metrics": {"schema_pass_rate": 1.0},
                "tracks": [],
                "face_detections": [],
                "person_detections": [],
                "label_detections": [],
                "object_tracking": [],
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "video_metadata": {"width": 1920, "height": 1080, "fps": 30.0, "duration_ms": 1000},
            },
            "phase_1_audio": {
                "source_audio": youtube_url,
                "words": [],
                "speaker_bindings": [],
            },
        }

    monkeypatch.setattr("backend.do_phase1_service.extract.execute_local_extraction", fake_execute_local_extraction)
    monkeypatch.setattr(
        "backend.do_phase1_service.extract.enrich_visual_ledger_for_downstream",
        lambda phase_1_visual, phase_1_audio, video_path: phase_1_visual,
    )
    monkeypatch.setattr("backend.do_phase1_service.extract.validate_phase_handoff", lambda visual_ledger, audio_ledger: None)

    run_extraction_job(
        source_url="https://youtube.com/watch?v=logs",
        job_id="job_123",
        output_dir=tmp_path,
        storage=FakeStorage(),
        host_lock_path=tmp_path / "extract.lock",
        log_path=log_path,
        progress_callback=lambda step, message=None, pct=None: progress_events.append((step, message, pct)),
    )

    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "[Phase 1] Step 1+2/4" in log_text
    assert "[Phase 1] Complete" in log_text
    assert any(step == "asr_tracking" for step, _, _ in progress_events)
    assert any(step == "speaker_binding" for step, _, _ in progress_events)
    assert any(step == "phase1_complete" and pct == 0.9 for step, _, pct in progress_events)
    assert progress_events.index(next(event for event in progress_events if event[0] == "phase1_complete")) < progress_events.index(
        next(event for event in progress_events if event[0] == "uploading_source")
    )
    assert progress_events[-2][0] == "persisting_manifest"
    assert progress_events[-1][0] == "complete"
    assert progress_events[-1][2] == 1.0


def test_extract_job_applies_runtime_controls_to_local_extraction_env(tmp_path: Path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    observed = {}

    monkeypatch.setattr(
        "backend.do_phase1_service.extract.download_media",
        lambda url: (str(video_path), str(audio_path)),
    )

    def fake_execute_local_extraction(*, video_path, audio_path, youtube_url):
        import os

        observed["speaker_binding_mode"] = os.getenv("CLYPT_SPEAKER_BINDING_MODE")
        observed["tracking_mode"] = os.getenv("CLYPT_TRACKING_MODE")
        observed["eval_profile"] = os.getenv("CLYPT_PHASE1_EVAL_PROFILE")
        return {
            "status": "success",
            "phase_1_visual": {
                "source_video": youtube_url,
                "schema_version": "2.0.0",
                "task_type": "person_tracking",
                "coordinate_space": "absolute_original_frame_xyxy",
                "geometry_type": "aabb",
                "class_taxonomy": {"0": "person"},
                "tracking_metrics": {"schema_pass_rate": 1.0},
                "tracks": [],
                "face_detections": [],
                "person_detections": [],
                "label_detections": [],
                "object_tracking": [],
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "video_metadata": {"width": 1920, "height": 1080, "fps": 30.0, "duration_ms": 1000},
            },
            "phase_1_audio": {
                "source_audio": youtube_url,
                "words": [],
                "speaker_bindings": [],
            },
        }

    monkeypatch.setattr("backend.do_phase1_service.extract.execute_local_extraction", fake_execute_local_extraction)
    monkeypatch.setattr(
        "backend.do_phase1_service.extract.enrich_visual_ledger_for_downstream",
        lambda phase_1_visual, phase_1_audio, video_path: phase_1_visual,
    )
    monkeypatch.setattr("backend.do_phase1_service.extract.validate_phase_handoff", lambda visual_ledger, audio_ledger: None)

    run_extraction_job(
        source_url="https://youtube.com/watch?v=controls",
        job_id="job_controls",
        runtime_controls={
            "profile_name": "podcast_eval",
            "evaluation_mode": True,
            "speaker_binding_mode": "lrasd",
            "tracking_mode": "direct",
        },
        output_dir=tmp_path,
        storage=FakeStorage(),
        host_lock_path=tmp_path / "extract.lock",
    )

    assert observed == {
        "speaker_binding_mode": "lrasd",
        "tracking_mode": "direct",
        "eval_profile": "podcast_eval",
    }


def test_extract_job_isolates_phase1_pipeline_workspace_per_job(tmp_path: Path, monkeypatch):
    original_download_dir = phase1_pipeline.DOWNLOAD_DIR
    original_output_dir = phase1_pipeline.OUTPUT_DIR
    observed = {}

    video_path = tmp_path / "isolated_video.mp4"
    video_path.write_bytes(b"video")
    audio_path = tmp_path / "isolated_audio.wav"
    audio_path.write_bytes(b"audio")

    def fake_download_media(url):
        observed["download_dir"] = phase1_pipeline.DOWNLOAD_DIR
        observed["output_dir"] = phase1_pipeline.OUTPUT_DIR
        observed["runtime_controls_path"] = phase1_pipeline.PHASE1_RUNTIME_CONTROLS_PATH
        observed["detached_state_path"] = phase1_pipeline.DETACHED_STATE_PATH
        return str(video_path), str(audio_path)

    monkeypatch.setattr("backend.do_phase1_service.extract.download_media", fake_download_media)
    monkeypatch.setattr(
        "backend.do_phase1_service.extract.execute_local_extraction",
        lambda video_path, audio_path, youtube_url: {
            "status": "success",
            "phase_1_visual": {
                "source_video": youtube_url,
                "schema_version": "2.0.0",
                "task_type": "person_tracking",
                "coordinate_space": "absolute_original_frame_xyxy",
                "geometry_type": "aabb",
                "class_taxonomy": {"0": "person"},
                "tracking_metrics": {"schema_pass_rate": 1.0},
                "tracks": [],
                "face_detections": [],
                "person_detections": [],
                "label_detections": [],
                "object_tracking": [],
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "video_metadata": {"width": 1920, "height": 1080, "fps": 30.0, "duration_ms": 1000},
            },
            "phase_1_audio": {
                "source_audio": youtube_url,
                "words": [],
                "speaker_bindings": [],
            },
        },
    )
    monkeypatch.setattr(
        "backend.do_phase1_service.extract.enrich_visual_ledger_for_downstream",
        lambda phase_1_visual, phase_1_audio, video_path: phase_1_visual,
    )
    monkeypatch.setattr("backend.do_phase1_service.extract.validate_phase_handoff", lambda visual_ledger, audio_ledger: None)

    run_extraction_job(
        source_url="https://youtube.com/watch?v=isolated",
        job_id="job_isolated",
        output_dir=tmp_path,
        storage=FakeStorage(),
        host_lock_path=tmp_path / "extract.lock",
    )

    expected_workspace = tmp_path / "job_isolated" / "_workspace"
    assert observed["download_dir"] == expected_workspace / "downloads"
    assert observed["output_dir"] == expected_workspace / "outputs"
    assert observed["runtime_controls_path"] == expected_workspace / "outputs" / "phase_1_runtime_controls.json"
    assert observed["detached_state_path"] == expected_workspace / "outputs" / "phase_1_detached_state.json"
    assert phase1_pipeline.DOWNLOAD_DIR == original_download_dir
    assert phase1_pipeline.OUTPUT_DIR == original_output_dir


def test_speaker_binding_proxy_reencodes_large_video(tmp_path: Path, monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    video_path = tmp_path / "video_h264.mp4"
    video_path.write_bytes(b"video")
    proxy_path = tmp_path / "video_h264_speaker_proxy.mp4"
    ffmpeg_invocations = []

    monkeypatch.setenv("CLYPT_SPEAKER_BINDING_PROXY_ENABLE", "1")
    monkeypatch.setenv("CLYPT_SPEAKER_BINDING_PROXY_MAX_LONG_EDGE", "1280")

    meta_by_path = {
        str(video_path): {"width": 3840, "height": 2160, "fps": 24.0, "total_frames": 10, "duration_s": 1.0},
        str(proxy_path): {"width": 1280, "height": 720, "fps": 24.0, "total_frames": 10, "duration_s": 1.0},
    }
    monkeypatch.setattr(worker, "_probe_video_meta", lambda path: meta_by_path[str(path)])

    def fake_subprocess_run(cmd, check, stdout, stderr):
        ffmpeg_invocations.append(cmd)
        proxy_path.write_bytes(b"proxy")
        return None

    monkeypatch.setattr("backend.do_phase1_worker.subprocess.run", fake_subprocess_run)

    chosen_path, scale_x, scale_y = worker._prepare_speaker_binding_video(str(video_path))

    assert chosen_path == str(proxy_path)
    assert scale_x == 1280 / 3840
    assert scale_y == 720 / 2160
    assert ffmpeg_invocations
    assert ffmpeg_invocations[0][0] == "ffmpeg"


def test_scale_detection_geometry_scales_spatial_fields_only():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    det = {
        "frame_idx": 42,
        "track_id": "track-1",
        "x_center": 120.0,
        "y_center": 240.0,
        "width": 60.0,
        "height": 80.0,
        "confidence": 0.9,
    }

    scaled = worker._scale_detection_geometry(det, scale_x=0.5, scale_y=0.25)

    assert scaled["x_center"] == 60.0
    assert scaled["y_center"] == 60.0
    assert scaled["width"] == 30.0
    assert scaled["height"] == 20.0
    assert scaled["frame_idx"] == 42
    assert scaled["track_id"] == "track-1"
    assert scaled["confidence"] == 0.9


def test_speaker_binding_mode_forces_heuristic(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    calls = []

    monkeypatch.setenv("CLYPT_SPEAKER_BINDING_MODE", "heuristic")
    monkeypatch.setattr(worker, "_run_lrasd_binding", lambda **kwargs: calls.append("lrasd"))
    monkeypatch.setattr(
        worker,
        "_run_speaker_binding_heuristic",
        lambda *args, **kwargs: calls.append("heuristic") or [{"track_id": "t1"}],
    )

    result = worker._run_speaker_binding(
        video_path="video.mp4",
        audio_wav_path="audio.wav",
        tracks=[{"track_id": "t1"}],
        words=[{"start_time_ms": 0, "end_time_ms": 100}],
    )

    assert result == [{"track_id": "t1"}]
    assert calls == ["heuristic"]


def test_speaker_binding_mode_auto_prefers_heuristic_for_large_long_video(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    calls = []

    monkeypatch.delenv("CLYPT_SPEAKER_BINDING_MODE", raising=False)
    monkeypatch.setattr(
        worker,
        "_probe_video_meta",
        lambda path: {"width": 3840, "height": 2160, "duration_s": 392.0},
    )
    monkeypatch.setattr(worker, "_run_lrasd_binding", lambda **kwargs: calls.append("lrasd"))
    monkeypatch.setattr(
        worker,
        "_run_speaker_binding_heuristic",
        lambda *args, **kwargs: calls.append("heuristic") or [{"track_id": "t1"}],
    )

    result = worker._run_speaker_binding(
        video_path="video.mp4",
        audio_wav_path="audio.wav",
        tracks=[{"track_id": "t1"}],
        words=[{"start_time_ms": 0, "end_time_ms": 100}],
    )

    assert result == [{"track_id": "t1"}]
    assert calls == ["heuristic"]


def test_speaker_binding_mode_auto_prefers_lrasd_for_small_short_video(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    calls = []

    monkeypatch.delenv("CLYPT_SPEAKER_BINDING_MODE", raising=False)
    monkeypatch.setattr(
        worker,
        "_probe_video_meta",
        lambda path: {"width": 1280, "height": 720, "duration_s": 45.0},
    )
    monkeypatch.setattr(
        worker,
        "_run_lrasd_binding",
        lambda **kwargs: calls.append("lrasd") or [{"track_id": "t1"}],
    )
    monkeypatch.setattr(
        worker,
        "_run_speaker_binding_heuristic",
        lambda *args, **kwargs: calls.append("heuristic") or [{"track_id": "t2"}],
    )

    result = worker._run_speaker_binding(
        video_path="video.mp4",
        audio_wav_path="audio.wav",
        tracks=[{"track_id": "t1"}],
        words=[{"start_time_ms": 0, "end_time_ms": 100}],
    )

    assert result == [{"track_id": "t1"}]
    assert calls == ["lrasd"]


def test_speaker_binding_mode_eval_profile_forces_lrasd_for_large_long_video(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    calls = []

    monkeypatch.delenv("CLYPT_SPEAKER_BINDING_MODE", raising=False)
    monkeypatch.setenv("CLYPT_PHASE1_EVAL_PROFILE", "podcast")
    monkeypatch.setattr(
        worker,
        "_probe_video_meta",
        lambda path: {"width": 3840, "height": 2160, "duration_s": 392.0},
    )
    monkeypatch.setattr(
        worker,
        "_run_lrasd_binding",
        lambda **kwargs: calls.append("lrasd") or [{"track_id": "t1"}],
    )
    monkeypatch.setattr(
        worker,
        "_run_speaker_binding_heuristic",
        lambda *args, **kwargs: calls.append("heuristic") or [{"track_id": "t2"}],
    )

    result = worker._run_speaker_binding(
        video_path="video.mp4",
        audio_wav_path="audio.wav",
        tracks=[{"track_id": "t1"}],
        words=[{"start_time_ms": 0, "end_time_ms": 100}],
    )

    assert result == [{"track_id": "t1"}]
    assert calls == ["lrasd"]


def test_speaker_binding_mode_lrasd_still_falls_back_when_primary_returns_none(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    calls = []

    monkeypatch.setenv("CLYPT_SPEAKER_BINDING_MODE", "lrasd")
    monkeypatch.setattr(worker, "_run_lrasd_binding", lambda **kwargs: calls.append("lrasd") or None)
    monkeypatch.setattr(
        worker,
        "_run_speaker_binding_heuristic",
        lambda *args, **kwargs: calls.append("heuristic") or [{"track_id": "fallback"}],
    )

    result = worker._run_speaker_binding(
        video_path="video.mp4",
        audio_wav_path="audio.wav",
        tracks=[{"track_id": "t1"}],
        words=[{"start_time_ms": 0, "end_time_ms": 100}],
    )

    assert result == [{"track_id": "fallback"}]
    assert calls == ["lrasd", "heuristic"]


def test_build_visual_detection_ledgers_uses_canonical_face_tracks(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracks = [
        {
            "frame_idx": 0,
            "track_id": "track-1",
            "x1": 10.0,
            "y1": 20.0,
            "x2": 110.0,
            "y2": 220.0,
            "x_center": 60.0,
            "y_center": 120.0,
            "width": 100.0,
            "height": 200.0,
            "confidence": 0.9,
        },
        {
            "frame_idx": 30,
            "track_id": "track-1",
            "x1": 20.0,
            "y1": 30.0,
            "x2": 120.0,
            "y2": 230.0,
            "x_center": 70.0,
            "y_center": 130.0,
            "width": 100.0,
            "height": 200.0,
            "confidence": 0.85,
        },
    ]
    frame_to_dets = {0: [tracks[0]], 30: [tracks[1]]}
    track_to_dets = {"track-1": tracks}

    monkeypatch.setattr(
        worker,
        "_probe_video_meta",
        lambda path: {"fps": 30.0, "width": 1920, "height": 1080, "duration_s": 1.0},
    )
    face_detections, person_detections, metrics = worker._build_visual_detection_ledgers(
        video_path="video.mp4",
        tracks=tracks,
        frame_to_dets=frame_to_dets,
        track_to_dets=track_to_dets,
        track_identity_features={
            "track-1": {
                "face_observations": [
                    {
                        "frame_idx": 0,
                        "confidence": 0.95,
                        "bounding_box": {
                            "left": 0.11,
                            "top": 0.07,
                            "right": 0.19,
                            "bottom": 0.18,
                        },
                        "source": "face_detector",
                        "provenance": "scrfd_fullframe",
                    },
                    {
                        "frame_idx": 30,
                        "confidence": 0.90,
                        "bounding_box": {
                            "left": 0.12,
                            "top": 0.08,
                            "right": 0.20,
                            "bottom": 0.19,
                        },
                        "source": "face_detector",
                        "provenance": "scrfd_fullframe",
                    },
                ]
            }
        },
    )

    assert len(person_detections) == 1
    assert len(face_detections) == 1
    assert face_detections[0]["source"] == "face_detector"
    assert face_detections[0]["provenance"] == "scrfd_fullframe"
    assert face_detections[0]["track_id"] == "track-1"
    assert face_detections[0]["timestamped_objects"][0]["bounding_box"]["left"] > 0.0
    assert face_detections[0]["timestamped_objects"][0]["bounding_box"]["right"] < 1.0
    assert metrics["face_detection_frame_samples"] == 2
    assert metrics["face_detection_track_count"] == 1


def test_build_visual_detection_ledgers_prefers_precomputed_face_observations(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracks = [
        {
            "frame_idx": 0,
            "track_id": "track-1",
            "x1": 10.0,
            "y1": 20.0,
            "x2": 110.0,
            "y2": 220.0,
            "x_center": 60.0,
            "y_center": 120.0,
            "width": 100.0,
            "height": 200.0,
            "confidence": 0.9,
        }
    ]
    frame_to_dets = {0: [tracks[0]]}
    track_to_dets = {"track-1": tracks}

    monkeypatch.setattr(
        worker,
        "_probe_video_meta",
        lambda path: {"fps": 30.0, "width": 1920, "height": 1080, "duration_s": 1.0},
    )
    face_detections, person_detections, metrics = worker._build_visual_detection_ledgers(
        video_path="video.mp4",
        tracks=tracks,
        frame_to_dets=frame_to_dets,
        track_to_dets=track_to_dets,
        track_identity_features={
            "track-1": {
                "face_observations": [
                    {
                        "frame_idx": 0,
                        "confidence": 0.91,
                        "quality": 1.0,
                        "bounding_box": {
                            "left": 0.15,
                            "top": 0.08,
                            "right": 0.24,
                            "bottom": 0.18,
                        },
                        "source": "face_detector",
                        "provenance": "scrfd_fullframe",
                    },
                    {
                        "frame_idx": 30,
                        "confidence": 0.88,
                        "quality": 0.9,
                        "bounding_box": {
                            "left": 0.16,
                            "top": 0.09,
                            "right": 0.25,
                            "bottom": 0.19,
                        },
                        "source": "face_detector",
                        "provenance": "scrfd_fullframe",
                    }
                ]
            }
        },
    )

    assert len(person_detections) == 1
    assert len(face_detections) == 1
    assert face_detections[0]["track_id"] == "track-1"
    assert [obj["time_ms"] for obj in face_detections[0]["timestamped_objects"]] == [0, 1000]
    assert face_detections[0]["timestamped_objects"][0]["bounding_box"]["left"] == 0.15
    assert face_detections[0]["timestamped_objects"][1]["bounding_box"]["left"] == 0.16
    assert all(obj["source"] == "face_detector" for obj in face_detections[0]["timestamped_objects"])
    assert metrics["face_detection_track_count"] == 1


def test_associate_faces_to_person_dets_prefers_head_aligned_match():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    dets = [
        {"track_id": "left", "x1": 100.0, "y1": 100.0, "x2": 300.0, "y2": 500.0},
        {"track_id": "right", "x1": 500.0, "y1": 110.0, "x2": 700.0, "y2": 520.0},
    ]
    faces = [
        {"bbox_xyxy": (150.0, 120.0, 240.0, 220.0)},
        {"bbox_xyxy": (555.0, 130.0, 645.0, 235.0)},
    ]

    assert worker._associate_faces_to_person_dets(faces, dets) == ["left", "right"]


def test_shared_analysis_proxy_can_drive_tracking_and_lrasd_selection(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.setenv("CLYPT_SHARED_ANALYSIS_PROXY", "1")
    monkeypatch.setenv("CLYPT_TRACKING_MODE", "auto")
    monkeypatch.setenv("CLYPT_SPEAKER_BINDING_MODE", "auto")
    monkeypatch.setattr(
        worker,
        "_probe_video_meta",
        lambda path: (_ for _ in ()).throw(AssertionError("video metadata should not be queried for shared-analysis proxy selection")),
    )

    assert worker._select_tracking_mode() == "shared_analysis_proxy"
    assert worker._select_speaker_binding_mode(
        video_path="video.mp4",
        tracks=[{"track_id": "track-1"}],
        words=[{"start_time_ms": 0, "end_time_ms": 100}],
    ) == "shared_analysis_proxy"


def test_run_lrasd_binding_uses_precomputed_feature_cache(monkeypatch, tmp_path: Path):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    worker.lrasd_model = object()
    worker.lrasd_loss_av = object()
    worker.gpu_device = "cpu"
    worker.model_debug = False
    worker.model_debug_every = 0

    video_path = tmp_path / "video.mp4"
    audio_path = tmp_path / "audio.wav"
    cached_features_path = tmp_path / "video_lrasd_features.npz"
    video_path.write_bytes(b"video")
    audio_path.write_bytes(b"audio")
    np.savez_compressed(
        cached_features_path,
        audio_features=np.full((400, 13), 0.25, dtype=np.float32),
    )

    class _Batch:
        def __init__(self, frames):
            self._frames = frames

        def asnumpy(self):
            return np.stack(self._frames, axis=0)

    class _VideoReader:
        def __init__(self, path, ctx=None):
            self._frames = [
                np.full((112, 112, 3), fill_value=i, dtype=np.uint8)
                for i in range(40)
            ]

        def __len__(self):
            return len(self._frames)

        def get_avg_fps(self):
            return 30.0

        def get_batch(self, indices):
            return _Batch([self._frames[i] for i in indices])

    fake_decord = types.ModuleType("decord")
    fake_decord.VideoReader = _VideoReader
    fake_decord.cpu = lambda _index: object()
    monkeypatch.setitem(__import__("sys").modules, "decord", fake_decord)

    fake_psf = types.ModuleType("python_speech_features")

    def _mfcc(*args, **kwargs):
        raise AssertionError("MFCC should be loaded from the cached feature path, not recomputed per subchunk")

    fake_psf.mfcc = _mfcc
    monkeypatch.setitem(__import__("sys").modules, "python_speech_features", fake_psf)

    import scipy.io.wavfile as wavfile_module

    monkeypatch.setattr(wavfile_module, "read", lambda path: (16000, np.zeros(48000, dtype=np.int16)))
    monkeypatch.setenv("CLYPT_ASD_PRECOMPUTED_FACE", "0")
    monkeypatch.setattr(
        worker,
        "_build_track_indexes",
        lambda tracks: (
            {i: [tracks[0]] for i in range(40)},
            {"track-1": [dict(tracks[0], frame_idx=i) for i in range(40)]},
        ),
    )
    monkeypatch.setattr(
        worker,
        "_lrasd_forward_scores",
        lambda audio_t, visual_t: type(
            "ScoreTensor",
            (),
            {
                "detach": lambda self: self,
                "float": lambda self: self,
                "cpu": lambda self: self,
                "numpy": lambda self: np.full((audio_t.shape[0], audio_t.shape[1]), 0.9, dtype=np.float32),
            },
        )(),
    )
    tracks = [
        {
            "frame_idx": i,
            "track_id": "track-1",
            "x_center": 60.0,
            "y_center": 60.0,
            "width": 48.0,
            "height": 72.0,
            "confidence": 0.9,
        }
        for i in range(40)
    ]
    words = [{"start_time_ms": 0, "end_time_ms": 1000}]

    worker._run_lrasd_binding(
        video_path=str(video_path),
        audio_wav_path=str(audio_path),
        tracks=tracks,
        words=words,
        track_identity_features={
            "track-1": {
                "face_observations": [
                    {
                        "frame_idx": i,
                        "bounding_box": {"left": 0.2, "top": 0.1, "right": 0.4, "bottom": 0.35},
                        "confidence": 0.9,
                        "quality": 0.9,
                        "source": "face_detector",
                        "provenance": "scrfd_fullframe",
                    }
                    for i in range(40)
                ]
            }
        },
    )


def test_finalize_includes_visual_ledgers_and_stage_metrics(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracks = [
        {
            "frame_idx": 0,
            "track_id": "track-1",
            "x1": 10.0,
            "y1": 20.0,
            "x2": 110.0,
            "y2": 220.0,
            "x_center": 60.0,
            "y_center": 120.0,
            "width": 100.0,
            "height": 200.0,
            "confidence": 0.9,
        }
    ]
    words = [
        {"text": "hello", "start_time_ms": 0, "end_time_ms": 100, "speaker_track_id": "track-1"},
        {"text": "world", "start_time_ms": 120, "end_time_ms": 220, "speaker_track_id": None},
    ]

    monkeypatch.setattr(worker, "_tracking_contract_pass_rate", lambda tracks: 1.0)
    monkeypatch.setattr(worker, "_validate_tracking_contract", lambda tracks: None)
    monkeypatch.setattr(worker, "_enforce_rollout_gates", lambda metrics: None)
    monkeypatch.setattr(
        worker,
        "_build_track_indexes",
        lambda tracks: ({0: tracks}, {"track-1": tracks}),
    )
    def fake_cluster_tracklets(video_path, tracks, track_to_dets=None, track_identity_features=None, face_track_features=None):
        worker._last_clustering_metrics = {
            "cluster_visible_people_estimate": 2,
            "overfragmentation_proxy": 0.5,
            "accidental_merge_proxy": 0.0,
        }
        return tracks

    monkeypatch.setattr(worker, "_cluster_tracklets", fake_cluster_tracklets)
    monkeypatch.setattr(
        worker,
        "_build_visual_detection_ledgers",
        lambda video_path, tracks, frame_to_dets=None, track_to_dets=None, track_identity_features=None: (
                [
                    {
                        "track_id": "track-1",
                        "source": "face_detector",
                        "provenance": "scrfd_fullframe",
                        "timestamped_objects": [],
                    }
                ],
            [
                {
                    "track_id": "track-1",
                    "timestamped_objects": [],
                }
            ],
            {
                "face_detection_wallclock_s": 0.25,
                "face_detection_frame_samples": 1,
                "face_detection_track_count": 1,
            },
        ),
    )
    monkeypatch.setattr(worker, "_run_speaker_binding", lambda *args, **kwargs: [{"track_id": "track-1"}])

    result = worker._finalize_from_words_tracks(
        video_path="video.mp4",
        audio_path="audio.wav",
        youtube_url="https://youtube.com/watch?v=example",
        words=words,
        tracks=tracks,
        tracking_metrics={},
    )

    visual = result["phase_1_visual"]
    metrics = visual["tracking_metrics"]

    assert visual["face_detections"][0]["source"] == "face_detector"
    assert visual["person_detections"][0]["track_id"] == "track-1"
    assert metrics["face_detection_wallclock_s"] == 0.25
    assert metrics["speaker_binding_assignment_ratio"] == 0.5
    assert metrics["identity_track_count_before_clustering"] == 1
    assert metrics["identity_track_count_after_clustering"] == 1
    assert metrics["cluster_visible_people_estimate"] == 2
    assert metrics["overfragmentation_proxy"] == 0.5


def test_finalize_passes_track_identity_features_to_clustering_and_ledgers(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracks = [
        {
            "frame_idx": 0,
            "track_id": "track-1",
            "x1": 10.0,
            "y1": 20.0,
            "x2": 110.0,
            "y2": 220.0,
            "x_center": 60.0,
            "y_center": 120.0,
            "width": 100.0,
            "height": 200.0,
            "confidence": 0.9,
        }
    ]
    words = [{"text": "hello", "start_time_ms": 0, "end_time_ms": 100}]
    passed = {}

    monkeypatch.setattr(worker, "_tracking_contract_pass_rate", lambda tracks: 1.0)
    monkeypatch.setattr(worker, "_validate_tracking_contract", lambda tracks: None)
    monkeypatch.setattr(worker, "_enforce_rollout_gates", lambda metrics: None)
    monkeypatch.setattr(
        worker,
        "_build_track_indexes",
        lambda tracks: ({0: tracks}, {"track-1": tracks}),
    )

    def fake_cluster_tracklets(video_path, tracks, track_to_dets=None, track_identity_features=None, face_track_features=None):
        passed["cluster"] = track_identity_features
        passed["face_tracks"] = face_track_features
        worker._last_clustering_metrics = {}
        return tracks

    def fake_build_visual_detection_ledgers(
        video_path,
        tracks,
        frame_to_dets=None,
        track_to_dets=None,
        track_identity_features=None,
    ):
        passed["ledgers"] = track_identity_features
        return [], [], {"face_detection_track_count": 0}

    monkeypatch.setattr(worker, "_cluster_tracklets", fake_cluster_tracklets)
    monkeypatch.setattr(worker, "_build_visual_detection_ledgers", fake_build_visual_detection_ledgers)
    monkeypatch.setattr(worker, "_run_speaker_binding", lambda *args, **kwargs: [])

    identity_features = {
        "track-1": {
            "face_observations": [
                {
                    "frame_idx": 0,
                    "bounding_box": {"left": 0.1, "top": 0.1, "right": 0.2, "bottom": 0.2},
                    "confidence": 0.9,
                    "quality": 1.0,
                }
            ]
        }
    }

    worker._finalize_from_words_tracks(
        video_path="video.mp4",
        audio_path="audio.wav",
        youtube_url="https://youtube.com/watch?v=example",
        words=words,
        tracks=tracks,
        tracking_metrics={"track_identity_features": identity_features},
    )

    assert passed["cluster"] == identity_features
    assert passed["ledgers"] == identity_features
    assert passed["face_tracks"] is None


def test_clusters_conflict_by_visibility_detects_far_apart_covisible_people():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracklets = {
        "left": [
            {
                "frame_idx": 10,
                "x_center": 200.0,
                "y_center": 300.0,
                "width": 140.0,
                "height": 260.0,
                "confidence": 0.95,
            }
        ],
        "right": [
            {
                "frame_idx": 10,
                "x_center": 1620.0,
                "y_center": 320.0,
                "width": 150.0,
                "height": 255.0,
                "confidence": 0.9,
            }
        ],
    }

    assert worker._clusters_conflict_by_visibility(tracklets, ["left"], ["right"]) is True


def test_clusters_reject_incompatible_seat_signatures():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracklets = {
        "host_left": [
            {"frame_idx": 0, "x_center": 220.0, "y_center": 320.0, "width": 180.0, "height": 280.0},
            {"frame_idx": 10, "x_center": 230.0, "y_center": 318.0, "width": 182.0, "height": 282.0},
        ],
        "guest_right": [
            {"frame_idx": 200, "x_center": 1680.0, "y_center": 330.0, "width": 176.0, "height": 278.0},
            {"frame_idx": 210, "x_center": 1695.0, "y_center": 332.0, "width": 178.0, "height": 280.0},
        ],
    }

    assert worker._clusters_have_compatible_seat_signature(tracklets, ["host_left"], ["guest_right"]) is False


def test_repair_covisible_cluster_merges_splits_invalid_global_identity():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracklets = {
        "host_frag_a": [
            {"frame_idx": 0, "x_center": 220.0, "y_center": 320.0, "width": 180.0, "height": 280.0, "confidence": 0.92},
            {"frame_idx": 1, "x_center": 225.0, "y_center": 322.0, "width": 182.0, "height": 282.0, "confidence": 0.93},
        ],
        "host_frag_b": [
            {"frame_idx": 10, "x_center": 228.0, "y_center": 321.0, "width": 181.0, "height": 281.0, "confidence": 0.91},
            {"frame_idx": 11, "x_center": 232.0, "y_center": 319.0, "width": 183.0, "height": 283.0, "confidence": 0.92},
        ],
        "guest_frag": [
            {"frame_idx": 1, "x_center": 1620.0, "y_center": 330.0, "width": 176.0, "height": 278.0, "confidence": 0.89},
            {"frame_idx": 2, "x_center": 1625.0, "y_center": 332.0, "width": 178.0, "height": 280.0, "confidence": 0.9},
        ],
    }
    label_by_tid = {
        "host_frag_a": 0,
        "host_frag_b": 0,
        "guest_frag": 0,
    }

    repaired, metrics = worker._repair_covisible_cluster_merges(tracklets, label_by_tid)

    assert repaired["host_frag_a"] == repaired["host_frag_b"]
    assert repaired["guest_frag"] != repaired["host_frag_a"]
    assert metrics["repaired_cluster_count"] == 1
    assert metrics["repaired_tracklet_count"] == 3
    assert metrics["repaired_conflict_pair_count"] >= 1


def test_cluster_tracklets_keeps_histogram_fragment_separate_when_attachment_is_not_plausible(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    class _FakeDBSCAN:
        def __init__(self, eps, min_samples, metric):
            self.labels_ = None

        def fit(self, X):
            self.labels_ = np.arange(len(X), dtype=int)
            return self

    sklearn_cluster = types.ModuleType("sklearn.cluster")
    sklearn_cluster.DBSCAN = _FakeDBSCAN
    sklearn_pkg = types.ModuleType("sklearn")
    sklearn_pkg.cluster = sklearn_cluster
    monkeypatch.setitem(sys.modules, "sklearn", sklearn_pkg)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", sklearn_cluster)

    tracks = [
        {
            "frame_idx": 0,
            "track_id": "face_left",
            "x_center": 220.0,
            "y_center": 320.0,
            "width": 180.0,
            "height": 280.0,
            "confidence": 0.95,
        },
        {
            "frame_idx": 1,
            "track_id": "face_left",
            "x_center": 225.0,
            "y_center": 322.0,
            "width": 182.0,
            "height": 282.0,
            "confidence": 0.94,
        },
        {
            "frame_idx": 50,
            "track_id": "face_right",
            "x_center": 1620.0,
            "y_center": 330.0,
            "width": 176.0,
            "height": 278.0,
            "confidence": 0.93,
        },
        {
            "frame_idx": 51,
            "track_id": "face_right",
            "x_center": 1625.0,
            "y_center": 332.0,
            "width": 178.0,
            "height": 280.0,
            "confidence": 0.92,
        },
        {
            "frame_idx": 90,
            "track_id": "hist_far",
            "x_center": 3050.0,
            "y_center": 340.0,
            "width": 150.0,
            "height": 250.0,
            "confidence": 0.8,
        },
        {
            "frame_idx": 91,
            "track_id": "hist_far",
            "x_center": 3040.0,
            "y_center": 338.0,
            "width": 152.0,
            "height": 252.0,
            "confidence": 0.79,
        },
    ]
    tracklets = {
        "face_left": [tracks[0], tracks[1]],
        "face_right": [tracks[2], tracks[3]],
        "hist_far": [tracks[4], tracks[5]],
    }
    identity_features = {
        "face_left": {
            "embedding": [1.0, 0.0, 0.0],
            "embedding_source": "face",
            "embedding_count": 2,
            "face_observations": [],
        },
        "face_right": {
            "embedding": [0.0, 1.0, 0.0],
            "embedding_source": "face",
            "embedding_count": 2,
            "face_observations": [],
        },
        "hist_far": {
            "embedding": [0.0, 0.0, 1.0],
            "embedding_source": "histogram",
            "embedding_count": 1,
            "face_observations": [],
        },
    }

    clustered = worker._cluster_tracklets(
        video_path="video.mp4",
        tracks=[dict(track) for track in tracks],
        track_to_dets=tracklets,
        track_identity_features=identity_features,
    )

    mapped = {}
    for det in clustered:
        mapped.setdefault(det["track_id"], set()).add(det["frame_idx"])

    assert len(mapped) == 3
    metrics = worker._last_clustering_metrics
    assert metrics["histogram_attach_rejections"] >= 1


def test_face_pipeline_workers_cap_for_gpu_runtime(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    worker.face_detector = object()
    worker.face_recognizer = object()
    worker._face_runtime_ctx_id = 0

    monkeypatch.setenv("CLYPT_FACE_PIPELINE_WORKERS", "24")
    monkeypatch.delenv("CLYPT_FACE_PIPELINE_GPU_WORKERS", raising=False)

    assert worker._face_pipeline_workers() == 1

    monkeypatch.setenv("CLYPT_FACE_PIPELINE_GPU_WORKERS", "2")
    assert worker._face_pipeline_workers() == 2


def test_tracking_chunk_workers_default_to_one(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.delenv("CLYPT_TRACK_CHUNK_WORKERS", raising=False)

    assert worker._tracking_chunk_workers() == 1


def test_tracking_chunk_workers_honor_env_and_clamp(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.setenv("CLYPT_TRACK_CHUNK_WORKERS", "3")
    assert worker._tracking_chunk_workers() == 3

    monkeypatch.setenv("CLYPT_TRACK_CHUNK_WORKERS", "0")
    assert worker._tracking_chunk_workers() == 1


def test_get_tracking_model_reuses_loaded_yolo_model():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    sentinel = object()
    worker.yolo_model = sentinel

    assert worker._get_tracking_model() is sentinel


def test_tracking_mode_auto_prefers_direct_with_single_worker(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.delenv("CLYPT_TRACKING_MODE", raising=False)
    monkeypatch.setattr(worker, "_tracking_chunk_workers", lambda: 1)

    assert worker._select_tracking_mode() == "direct"


def test_tracking_mode_auto_prefers_chunked_with_multiple_workers(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.delenv("CLYPT_TRACKING_MODE", raising=False)
    monkeypatch.setattr(worker, "_tracking_chunk_workers", lambda: 2)

    assert worker._select_tracking_mode() == "chunked"


def test_run_tracking_uses_direct_mode(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    calls = []

    monkeypatch.setattr(worker, "_select_tracking_mode", lambda: "direct")
    monkeypatch.setattr(
        worker,
        "_run_tracking_direct",
        lambda video_path: calls.append(("direct", video_path)) or ([{"track_id": "t1"}], {"tracking_mode": "direct"}),
    )
    monkeypatch.setattr(
        worker,
        "_run_tracking_chunked",
        lambda video_path: calls.append(("chunked", video_path)) or ([{"track_id": "t2"}], {"tracking_mode": "chunked"}),
    )

    tracks, metrics = worker._run_tracking("video.mp4")

    assert tracks == [{"track_id": "t1"}]
    assert metrics == {"tracking_mode": "direct"}
    assert calls == [("direct", "video.mp4")]


def test_run_tracking_direct_emits_contract_compatible_tracks(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    track_calls = []

    class _Boxes:
        xyxy = type("TensorLike", (), {"cpu": lambda self: self, "numpy": lambda self: __import__("numpy").array([[10, 20, 30, 40]])})()
        id = type("TensorLike", (), {"cpu": lambda self: self, "numpy": lambda self: __import__("numpy").array([7])})()
        conf = type("TensorLike", (), {"cpu": lambda self: self, "numpy": lambda self: __import__("numpy").array([0.9])})()

    class _Result:
        boxes = _Boxes()
        obb = None

    class _Model:
        def track(self, **kwargs):
            track_calls.append(kwargs)
            return [_Result()]

    monkeypatch.setattr(worker, "_ensure_h264", lambda path: path)
    monkeypatch.setattr(worker, "_probe_video_meta", lambda path: {"fps": 24.0, "total_frames": 1, "width": 1920, "height": 1080})
    monkeypatch.setattr(worker, "_ensure_botsort_reid_yaml", lambda: "tracker.yaml")
    monkeypatch.setattr(worker, "_get_tracking_model", lambda: _Model())
    monkeypatch.setattr(worker, "_compute_letterbox_meta", lambda *args: {})
    monkeypatch.setattr(worker, "_xyxy_abs_to_xywhn", lambda *args: (0.1, 0.2, 0.3, 0.4))
    monkeypatch.setattr(worker, "_xywhn_to_xyxy_abs", lambda *args: (10.0, 20.0, 30.0, 40.0))
    monkeypatch.setattr(worker, "_forward_letterbox_xyxy", lambda *args: (10.0, 20.0, 30.0, 40.0))
    monkeypatch.setattr(worker, "_inverse_letterbox_xyxy", lambda *args: (10.0, 20.0, 30.0, 40.0))
    monkeypatch.setattr(worker, "_xyxy_to_xywh", lambda *args: (20.0, 30.0, 20.0, 20.0))

    tracks, metrics = worker._run_tracking_direct("video.mp4")

    assert tracks[0]["frame_idx"] == 0
    assert tracks[0]["local_frame_idx"] == 0
    assert tracks[0]["chunk_idx"] == 0
    assert "tracking_mode" not in metrics
    assert track_calls[0]["device"] == "cpu"




def test_derive_track_identity_features_from_face_tracks_propagates_multi_track_associations():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    features = worker._derive_track_identity_features_from_face_tracks(
        {
            "face_0_0": {
                "face_track_id": "face_0_0",
                "embedding": [0.1, 0.2, 0.3],
                "embedding_count": 2,
                "face_observations": [
                    {"frame_idx": 0, "confidence": 0.95, "associated_track_id": "track_a"},
                    {"frame_idx": 1, "confidence": 0.94, "associated_track_id": "track_a"},
                    {"frame_idx": 2, "confidence": 0.93, "associated_track_id": "track_b"},
                    {"frame_idx": 3, "confidence": 0.92, "associated_track_id": "track_b"},
                    {"frame_idx": 4, "confidence": 0.50, "associated_track_id": "track_c"},
                ],
                "associated_track_counts": {
                    "track_a": 2,
                    "track_b": 2,
                    "track_c": 1,
                },
            }
        }
    )

    assert set(features.keys()) == {"track_a", "track_b"}
    assert features["track_a"]["embedding_source"] == "face"
    assert features["track_b"]["embedding_source"] == "face"
    assert np.allclose(features["track_a"]["embedding"], [0.1, 0.2, 0.3])
    assert np.allclose(features["track_b"]["embedding"], [0.1, 0.2, 0.3])
    assert [obs["frame_idx"] for obs in features["track_a"]["face_observations"]] == [0, 1]
    assert [obs["frame_idx"] for obs in features["track_b"]["face_observations"]] == [2, 3]

def test_run_tracking_direct_emits_track_identity_features(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    class _Boxes:
        xyxy = type("TensorLike", (), {"cpu": lambda self: self, "numpy": lambda self: __import__("numpy").array([[10, 20, 30, 40]])})()
        id = type("TensorLike", (), {"cpu": lambda self: self, "numpy": lambda self: __import__("numpy").array([7])})()
        conf = type("TensorLike", (), {"cpu": lambda self: self, "numpy": lambda self: __import__("numpy").array([0.9])})()

    class _Result:
        boxes = _Boxes()
        obb = None

    class _Model:
        def track(self, **kwargs):
            return [_Result()]

    monkeypatch.setattr(worker, "_ensure_h264", lambda path: path)
    monkeypatch.setattr(worker, "_probe_video_meta", lambda path: {"fps": 24.0, "total_frames": 1, "width": 1920, "height": 1080})
    monkeypatch.setattr(worker, "_ensure_botsort_reid_yaml", lambda: "tracker.yaml")
    monkeypatch.setattr(worker, "_get_tracking_model", lambda: _Model())
    monkeypatch.setattr(worker, "_compute_letterbox_meta", lambda *args: {})
    monkeypatch.setattr(worker, "_xyxy_abs_to_xywhn", lambda *args: (0.1, 0.2, 0.3, 0.4))
    monkeypatch.setattr(worker, "_xywhn_to_xyxy_abs", lambda *args: (10.0, 20.0, 30.0, 40.0))
    monkeypatch.setattr(worker, "_forward_letterbox_xyxy", lambda *args: (10.0, 20.0, 30.0, 40.0))
    monkeypatch.setattr(worker, "_inverse_letterbox_xyxy", lambda *args: (10.0, 20.0, 30.0, 40.0))
    monkeypatch.setattr(worker, "_xyxy_to_xywh", lambda *args: (20.0, 30.0, 20.0, 20.0))
    monkeypatch.setattr(worker, "_face_pipeline_segment_frames", lambda: 120)
    monkeypatch.setattr(worker, "_face_pipeline_workers", lambda: 8)
    monkeypatch.setattr(
        worker,
        "_extract_face_track_features_from_segments",
        lambda **kwargs: (
            {
                "track_7": {
                    "embedding": [0.1, 0.2, 0.3],
                    "face_observations": [{"frame_idx": 0, "confidence": 0.9, "quality": 1.0}],
                }
            },
            {"face_0_0": {"embedding": [0.1, 0.2, 0.3], "associated_track_counts": {"track_7": 1}}},
            {
                "face_pipeline_mode": "staggered",
                "face_pipeline_segments_processed": 1,
            },
        ),
    )

    _, metrics = worker._run_tracking_direct("video.mp4")

    assert metrics["track_identity_features"]["track_7"]["embedding"] == [0.1, 0.2, 0.3]
    assert metrics["face_pipeline_mode"] == "staggered"
    assert metrics["face_pipeline_segments_processed"] == 1


def test_run_tracking_direct_logs_face_pipeline_progress(monkeypatch, capsys):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    class _Boxes:
        def __init__(self):
            self.xyxy = type("TensorLike", (), {"cpu": lambda self: self, "numpy": lambda self: __import__("numpy").array([[10, 20, 30, 40]])})()
            self.id = type("TensorLike", (), {"cpu": lambda self: self, "numpy": lambda self: __import__("numpy").array([7])})()
            self.conf = type("TensorLike", (), {"cpu": lambda self: self, "numpy": lambda self: __import__("numpy").array([0.9])})()

    class _Result:
        def __init__(self):
            self.boxes = _Boxes()
            self.obb = None

    class _Model:
        def track(self, **kwargs):
            return [_Result(), _Result()]

    monkeypatch.setattr(worker, "_ensure_h264", lambda path: path)
    monkeypatch.setattr(worker, "_probe_video_meta", lambda path: {"fps": 24.0, "total_frames": 2, "width": 1920, "height": 1080})
    monkeypatch.setattr(worker, "_ensure_botsort_reid_yaml", lambda: "tracker.yaml")
    monkeypatch.setattr(worker, "_get_tracking_model", lambda: _Model())
    monkeypatch.setattr(worker, "_compute_letterbox_meta", lambda *args: {})
    monkeypatch.setattr(worker, "_xyxy_abs_to_xywhn", lambda *args: (0.1, 0.2, 0.3, 0.4))
    monkeypatch.setattr(worker, "_xywhn_to_xyxy_abs", lambda *args: (10.0, 20.0, 30.0, 40.0))
    monkeypatch.setattr(worker, "_forward_letterbox_xyxy", lambda *args: (10.0, 20.0, 30.0, 40.0))
    monkeypatch.setattr(worker, "_inverse_letterbox_xyxy", lambda *args: (10.0, 20.0, 30.0, 40.0))
    monkeypatch.setattr(worker, "_xyxy_to_xywh", lambda *args: (20.0, 30.0, 20.0, 20.0))
    monkeypatch.setattr(worker, "_face_pipeline_segment_frames", lambda: 1)
    monkeypatch.setattr(worker, "_face_pipeline_workers", lambda: 2)
    monkeypatch.setattr(
        worker,
        "_extract_face_track_features_for_frame_segment",
        lambda **kwargs: {
            "face_track_features": {
                "face_0_0": {
                    "embedding": [0.1, 0.2, 0.3],
                    "embedding_count": 1,
                    "face_observations": [{"frame_idx": 0, "confidence": 0.9, "quality": 1.0}],
                    "associated_track_counts": {"track_7": 1},
                    "dominant_track_id": "track_7",
                }
            }
        },
    )

    worker._run_tracking_direct("video.mp4")

    captured = capsys.readouterr().out
    assert "Face pipeline queued:" in captured
    assert "Face pipeline complete:" in captured


def test_run_tracking_direct_delays_first_face_submission_until_start_frame(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    submit_calls = []

    class _Boxes:
        def __init__(self):
            self.xyxy = type("TensorLike", (), {"cpu": lambda self: self, "numpy": lambda self: __import__("numpy").array([[10, 20, 30, 40]])})()
            self.id = type("TensorLike", (), {"cpu": lambda self: self, "numpy": lambda self: __import__("numpy").array([7])})()
            self.conf = type("TensorLike", (), {"cpu": lambda self: self, "numpy": lambda self: __import__("numpy").array([0.9])})()

    class _Result:
        def __init__(self):
            self.boxes = _Boxes()
            self.obb = None

    class _Model:
        def track(self, **kwargs):
            return [_Result(), _Result()]

    monkeypatch.setattr(worker, "_ensure_h264", lambda path: path)
    monkeypatch.setattr(worker, "_probe_video_meta", lambda path: {"fps": 24.0, "total_frames": 2, "width": 1920, "height": 1080})
    monkeypatch.setattr(worker, "_ensure_botsort_reid_yaml", lambda: "tracker.yaml")
    monkeypatch.setattr(worker, "_get_tracking_model", lambda: _Model())
    monkeypatch.setattr(worker, "_compute_letterbox_meta", lambda *args: {})
    monkeypatch.setattr(worker, "_xyxy_abs_to_xywhn", lambda *args: (0.1, 0.2, 0.3, 0.4))
    monkeypatch.setattr(worker, "_xywhn_to_xyxy_abs", lambda *args: (10.0, 20.0, 30.0, 40.0))
    monkeypatch.setattr(worker, "_forward_letterbox_xyxy", lambda *args: (10.0, 20.0, 30.0, 40.0))
    monkeypatch.setattr(worker, "_inverse_letterbox_xyxy", lambda *args: (10.0, 20.0, 30.0, 40.0))
    monkeypatch.setattr(worker, "_xyxy_to_xywh", lambda *args: (20.0, 30.0, 20.0, 20.0))
    monkeypatch.setattr(worker, "_face_pipeline_segment_frames", lambda: 1)
    monkeypatch.setattr(worker, "_face_pipeline_workers", lambda: 2)
    monkeypatch.setattr(worker, "_requested_face_pipeline_workers", lambda: 24)
    monkeypatch.setattr(worker, "_face_pipeline_start_frame", lambda: 1)
    monkeypatch.setattr(
        worker,
        "_extract_face_track_features_for_frame_segment",
        lambda **kwargs: submit_calls.append(list(kwargs["frame_items"])) or {
            "face_track_features": {
                "face_0_0": {
                    "embedding": [0.1, 0.2, 0.3],
                    "embedding_count": 1,
                    "face_observations": [{"frame_idx": 0, "confidence": 0.9, "quality": 1.0}],
                    "associated_track_counts": {"track_7": 1},
                    "dominant_track_id": "track_7",
                }
            }
        },
    )

    worker._run_tracking_direct("video.mp4")

    assert len(submit_calls) >= 1
    assert len(submit_calls[0]) == 1


def test_run_tracking_chunk_passes_explicit_yolo_device(monkeypatch, tmp_path):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    track_calls = []

    class _Boxes:
        xyxy = type("TensorLike", (), {"cpu": lambda self: self, "numpy": lambda self: __import__("numpy").array([[10, 20, 30, 40]])})()
        id = type("TensorLike", (), {"cpu": lambda self: self, "numpy": lambda self: __import__("numpy").array([7])})()
        conf = type("TensorLike", (), {"cpu": lambda self: self, "numpy": lambda self: __import__("numpy").array([0.9])})()

    class _Result:
        boxes = _Boxes()
        obb = None

    class _Model:
        def track(self, **kwargs):
            track_calls.append(kwargs)
            return [_Result()]

    monkeypatch.setattr(worker, "_compute_letterbox_meta", lambda *args: {})
    monkeypatch.setattr(worker, "_xyxy_abs_to_xywhn", lambda *args: (0.1, 0.2, 0.3, 0.4))
    monkeypatch.setattr(worker, "_xywhn_to_xyxy_abs", lambda *args: (10.0, 20.0, 30.0, 40.0))
    monkeypatch.setattr(worker, "_forward_letterbox_xyxy", lambda *args: (10.0, 20.0, 30.0, 40.0))
    monkeypatch.setattr(worker, "_inverse_letterbox_xyxy", lambda *args: (10.0, 20.0, 30.0, 40.0))
    monkeypatch.setattr(worker, "_xyxy_to_xywh", lambda *args: (20.0, 30.0, 20.0, 20.0))
    monkeypatch.setattr(
        worker,
        "_extract_face_track_features_from_segments",
        lambda **kwargs: ({}, {}, {"face_pipeline_mode": "staggered", "face_pipeline_segments_processed": 0}),
    )
    monkeypatch.setattr(
        "backend.do_phase1_worker.subprocess.run",
        lambda *args, **kwargs: type("Completed", (), {"returncode": 0, "stderr": b""})(),
    )

    chunk_dir = tmp_path / "chunk0"
    chunk_dir.mkdir()
    result = worker._track_single_chunk(
        video_path="video.mp4",
        meta={"fps": 24.0, "width": 1920, "height": 1080},
        chunk={"start_frame": 0, "end_frame": 1, "overlap_frames": 0, "chunk_idx": 0},
        tracker_cfg="tracker.yaml",
        chunk_dir=str(chunk_dir),
        output_meta={"width": 1920, "height": 1080},
        coord_scale_x=1.0,
        coord_scale_y=1.0,
        model=_Model(),
    )

    assert result["processed_frames"] == 1
    assert track_calls[0]["device"] == "cpu"


def test_host_lock_serializes_second_extraction_attempt(tmp_path: Path):
    lock_path = tmp_path / "extract.lock"
    events = []

    def second_waiter():
        events.append("second-start")
        with host_extraction_lock(lock_path):
            events.append("second-acquired")

    with host_extraction_lock(lock_path):
        worker = threading.Thread(target=second_waiter)
        worker.start()
        time.sleep(0.1)
        events.append("first-held")
        assert events == ["second-start", "first-held"]
    worker.join(timeout=1.0)

    assert events == ["second-start", "first-held", "second-acquired"]


def test_gpu_slot_serializes_only_extraction_and_allows_parallel_download(tmp_path: Path, monkeypatch):
    lock_path = tmp_path / "extract.lock"
    order = []
    first_extract_may_finish = threading.Event()
    uploaded_bytes = {}
    enriched_bytes = {}

    def fake_download_media(url):
        order.append(f"download:{url}")
        video_path = tmp_path / f"{url.rsplit('=', 1)[-1]}.mp4"
        audio_path = tmp_path / f"{url.rsplit('=', 1)[-1]}.wav"
        video_path.write_bytes(url.encode("utf-8"))
        audio_path.write_bytes(b"audio")
        return str(video_path), str(audio_path)

    def fake_execute_local_extraction(**kwargs):
        order.append(f"extract:{kwargs['youtube_url']}")
        if kwargs["youtube_url"].endswith("first"):
            first_extract_may_finish.wait(timeout=2.0)
        return {
            "status": "success",
            "phase_1_visual": {
                "source_video": kwargs["youtube_url"],
                "schema_version": "2.0.0",
                "task_type": "person_tracking",
                "coordinate_space": "absolute_original_frame_xyxy",
                "geometry_type": "aabb",
                "class_taxonomy": {"0": "person"},
                "tracking_metrics": {"schema_pass_rate": 1.0},
                "tracks": [],
                "face_detections": [],
                "person_detections": [],
                "label_detections": [],
                "object_tracking": [],
                "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
                "video_metadata": {"width": 1920, "height": 1080, "fps": 30.0, "duration_ms": 1000},
            },
            "phase_1_audio": {
                "source_audio": kwargs["youtube_url"],
                "words": [],
                "speaker_bindings": [],
            },
        }

    monkeypatch.setattr("backend.do_phase1_service.extract.download_media", fake_download_media)
    monkeypatch.setattr("backend.do_phase1_service.extract.execute_local_extraction", fake_execute_local_extraction)
    class InspectingStorage(FakeStorage):
        def upload_file(self, source_path, object_name: str) -> str:
            uploaded_bytes[object_name] = Path(source_path).read_bytes()
            return super().upload_file(source_path, object_name)

    def fake_enrich_visual_ledger_for_downstream(phase_1_visual, phase_1_audio, video_path):
        enriched_bytes[str(video_path)] = Path(video_path).read_bytes()
        return phase_1_visual

    monkeypatch.setattr("backend.do_phase1_service.extract.enrich_visual_ledger_for_downstream", fake_enrich_visual_ledger_for_downstream)
    monkeypatch.setattr("backend.do_phase1_service.extract.validate_phase_handoff", lambda visual_ledger, audio_ledger: None)
    monkeypatch.setenv("DO_PHASE1_GPU_SLOTS", "1")

    @contextmanager
    def fake_capture_job_logs(*args, **kwargs):
        yield tmp_path / "job.log"

    monkeypatch.setattr("backend.do_phase1_service.extract._capture_job_logs", fake_capture_job_logs)

    first_done = []

    def run_first():
        result = run_extraction_job(
            source_url="https://youtube.com/watch?v=first",
            job_id="job_first",
            output_dir=tmp_path,
            storage=InspectingStorage(),
            host_lock_path=lock_path,
        )
        first_done.append(result.job_id)

    worker = threading.Thread(target=run_first)
    worker.start()
    time.sleep(0.1)

    second = threading.Thread(
        target=lambda: run_extraction_job(
            source_url="https://youtube.com/watch?v=second",
            job_id="job_second",
            output_dir=tmp_path,
            storage=InspectingStorage(),
            host_lock_path=lock_path,
        )
    )
    second.start()
    time.sleep(0.1)
    assert order[:2] == [
        "download:https://youtube.com/watch?v=first",
        "extract:https://youtube.com/watch?v=first",
    ]
    assert "download:https://youtube.com/watch?v=second" in order
    assert "extract:https://youtube.com/watch?v=second" not in order

    first_extract_may_finish.set()
    worker.join(timeout=2.0)
    second.join(timeout=2.0)

    assert first_done == ["job_first"]
    assert order.index("download:https://youtube.com/watch?v=second") < order.index("extract:https://youtube.com/watch?v=second")
    assert order.index("extract:https://youtube.com/watch?v=first") < order.index("extract:https://youtube.com/watch?v=second")
    assert uploaded_bytes["phase_1/jobs/job_first/source_video.mp4"] == b"https://youtube.com/watch?v=first"
    assert uploaded_bytes["phase_1/jobs/job_second/source_video.mp4"] == b"https://youtube.com/watch?v=second"
    assert enriched_bytes[str(tmp_path / "first.mp4")] == b"https://youtube.com/watch?v=first"
    assert enriched_bytes[str(tmp_path / "second.mp4")] == b"https://youtube.com/watch?v=second"


def test_extract_serializes_asr_before_gpu_tracking(monkeypatch, tmp_path: Path):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    video_bytes = b"video"
    audio_bytes = b"audio"
    asr_started = threading.Event()
    active = {"count": 0}
    observed = {}

    def fake_run_asr(audio_path):
        active["count"] += 1
        asr_started.set()
        time.sleep(0.1)
        active["count"] -= 1
        return [{"word": "hello"}]

    def fake_run_tracking(video_path):
        asr_started.wait(timeout=1.0)
        observed["active_asr_during_tracking"] = active["count"]
        return ([{"track_id": "t1"}], {"tracking_mode": "direct"})

    def fake_finalize_from_words_tracks(**kwargs):
        return {
            "status": "success",
            "phase_1_visual": {"tracks": kwargs["tracks"]},
            "phase_1_audio": {"words": kwargs["words"]},
        }

    monkeypatch.setattr(worker, "_run_asr", fake_run_asr)
    monkeypatch.setattr(worker, "_run_tracking", fake_run_tracking)
    monkeypatch.setattr(worker, "_finalize_from_words_tracks", fake_finalize_from_words_tracks)

    result = worker.extract(video_bytes=video_bytes, audio_wav_bytes=audio_bytes, youtube_url="https://youtube.com/watch?v=x")

    assert observed["active_asr_during_tracking"] == 0
    assert result["phase_1_audio"]["words"] == [{"word": "hello"}]
    assert result["phase_1_visual"]["tracks"] == [{"track_id": "t1"}]
