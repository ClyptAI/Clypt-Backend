from contextlib import contextmanager
from pathlib import Path
import sys
import types
import threading
import time

import numpy as np
import pytest

from backend.do_phase1_service.extract import host_extraction_lock, run_extraction_job
from backend.do_phase1_service.extract import _forward_progress_line
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
        observed["tracker_backend"] = os.getenv("CLYPT_TRACKER_BACKEND")
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
            "tracker_backend": "bytetrack",
        },
        output_dir=tmp_path,
        storage=FakeStorage(),
        host_lock_path=tmp_path / "extract.lock",
    )

    assert observed == {
        "speaker_binding_mode": "lrasd",
        "tracking_mode": "direct",
        "tracker_backend": "bytetrack",
        "eval_profile": "podcast_eval",
    }


def test_forward_progress_line_parses_live_tracking_and_binding_updates():
    events = []

    def record(step, message=None, pct=None):
        events.append((step, message, pct))

    _forward_progress_line(
        "  YOLO progress: 5400/9419 frames (57.3%), 18594 boxes, 14.8 fps",
        record,
    )
    _forward_progress_line(
        "  Face pipeline queued: 24 segments, workers=1",
        record,
    )
    _forward_progress_line(
        "  LR-ASD progress: prepared=96, inferred=64, track_windows=120/322, scored_tracks=11",
        record,
    )

    assert events == [
        ("tracking", "YOLO progress: 5400/9419 frames (57.3%)", 0.5),
        ("tracking", "Face pipeline queued: 24 segments", 0.55),
        ("speaker_binding", "LR-ASD progress: inferred=64, windows=120/322", 0.86),
    ]


def test_pyannote_config_defaults_and_env_overrides(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()

    monkeypatch.delenv("CLYPT_AUDIO_DIARIZATION_ENABLE", raising=False)
    monkeypatch.delenv("CLYPT_AUDIO_DIARIZATION_MODEL", raising=False)
    monkeypatch.delenv("CLYPT_AUDIO_DIARIZATION_MIN_SEGMENT_MS", raising=False)

    defaults = worker_cls._audio_diarization_config()
    assert defaults["enabled"] is False
    assert defaults["model_name"] == "pyannote/speaker-diarization-3.1"
    assert defaults["min_segment_ms"] == 400
    assert defaults["min_segment_s"] == 0.4

    monkeypatch.setenv("CLYPT_AUDIO_DIARIZATION_ENABLE", "1")
    monkeypatch.setenv("CLYPT_AUDIO_DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
    monkeypatch.setenv("CLYPT_AUDIO_DIARIZATION_MIN_SEGMENT_MS", "900")

    overrides = worker_cls._audio_diarization_config()
    assert overrides["enabled"] is True
    assert overrides["model_name"] == "pyannote/speaker-diarization-3.1"
    assert overrides["min_segment_ms"] == 900
    assert overrides["min_segment_s"] == 0.9


def test_audio_diarization_serializes_raw_turn_fields(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.setenv("CLYPT_AUDIO_DIARIZATION_ENABLE", "1")

    class FakeSegment:
        def __init__(self, start: float, end: float):
            self.start = start
            self.end = end

    class FakeDiarization:
        def itertracks(self, yield_label: bool = False):
            assert yield_label is True
            return iter(
                [
                    (FakeSegment(0.0, 1.5), None, "SPEAKER_00"),
                    (FakeSegment(1.5, 3.0), None, "SPEAKER_01"),
                ]
            )

    turns = worker._serialize_audio_speaker_turns(FakeDiarization())

    assert turns == [
        {"speaker_id": "SPEAKER_00", "start_time_ms": 0, "end_time_ms": 1500},
        {"speaker_id": "SPEAKER_01", "start_time_ms": 1500, "end_time_ms": 3000},
    ]


def test_audio_diarization_serializes_optional_turn_metadata(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.setenv("CLYPT_AUDIO_DIARIZATION_ENABLE", "1")

    class FakeSegment:
        def __init__(self, start: float, end: float):
            self.start = start
            self.end = end

    class FakeTurn:
        def __init__(self):
            self.speaker_id = "SPEAKER_09"
            self.segment = FakeSegment(4.0, 6.25)
            self.exclusive = "true"
            self.overlap = "0"
            self.score = "0.82"

    turns = worker._serialize_audio_speaker_turns([FakeTurn()])

    assert turns == [
        {
            "speaker_id": "SPEAKER_09",
            "start_time_ms": 4000,
            "end_time_ms": 6250,
            "exclusive": True,
            "overlap": False,
            "confidence": 0.82,
        }
    ]


def test_finalize_from_words_tracks_persists_audio_speaker_turns(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    words = [
        {
            "word": "hello",
            "start_time_ms": 0,
            "end_time_ms": 900,
            "speaker_track_id": None,
            "speaker_tag": "unknown",
            "speaker_local_track_id": None,
            "speaker_local_tag": "unknown",
        }
    ]
    tracks = [
        {
            "frame_idx": 0,
            "local_frame_idx": 0,
            "chunk_idx": 0,
            "track_id": "track_1",
            "local_track_id": 1,
            "class_id": 0,
            "label": "person",
            "confidence": 0.99,
            "x1": 10.0,
            "y1": 20.0,
            "x2": 30.0,
            "y2": 40.0,
            "x_center": 20.0,
            "y_center": 30.0,
            "width": 20.0,
            "height": 20.0,
            "source": "test",
            "geometry_type": "aabb",
            "bbox_norm_xywh": {
                "x_center": 0.1,
                "y_center": 0.2,
                "width": 0.3,
                "height": 0.4,
            },
        }
    ]
    diarization_turns = [
        {
            "speaker_id": "SPEAKER_00",
            "start_time_ms": 0,
            "end_time_ms": 900,
            "exclusive": True,
            "confidence": 0.91,
        }
    ]

    monkeypatch.setattr(worker, "_validate_tracking_contract", lambda tracks: None)
    monkeypatch.setattr(worker, "_enforce_rollout_gates", lambda tracking_metrics: None)
    monkeypatch.setattr(worker, "_tracking_contract_pass_rate", lambda tracks: 1.0)
    monkeypatch.setattr(worker, "_cluster_tracklets", lambda *args, **kwargs: tracks)
    monkeypatch.setattr(
        worker,
        "_build_visual_detection_ledgers",
        lambda **kwargs: ([], [], {"schema_pass_rate": 1.0}),
    )
    monkeypatch.setattr(
        worker,
        "_run_speaker_binding",
        lambda *args, **kwargs: [
            {"track_id": "track_1", "start_time_ms": 0, "end_time_ms": 900, "word_count": 1}
        ],
    )
    monkeypatch.setattr(worker, "_build_speaker_follow_bindings", lambda bindings: list(bindings))
    monkeypatch.setattr(worker, "_speaker_remap_collision_metrics", lambda words: {})
    monkeypatch.setattr(worker, "_run_audio_diarization", lambda audio_path: diarization_turns)
    monkeypatch.setattr(worker, "_local_clip_bindings_enabled", lambda: False)

    result = worker._finalize_from_words_tracks(
        video_path="video.mp4",
        audio_path="audio.wav",
        youtube_url="https://youtube.com/watch?v=finalize",
        words=words,
        tracks=tracks,
        tracking_metrics={"schema_pass_rate": 1.0},
    )

    assert result["phase_1_audio"]["audio_speaker_turns"] == diarization_turns
    assert result["phase_1a_audio"]["audio_speaker_turns"] == diarization_turns


def test_audio_diarization_disabled_returns_no_turns(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.delenv("CLYPT_AUDIO_DIARIZATION_ENABLE", raising=False)
    loaded = False

    def fake_load_audio_diarization_pipeline():
        nonlocal loaded
        loaded = True
        return object()

    monkeypatch.setattr(worker, "_load_audio_diarization_pipeline", fake_load_audio_diarization_pipeline)

    turns = worker._run_audio_diarization("audio.wav")

    assert turns == []
    assert loaded is False


def test_audio_diarization_disabled_records_fallback_status(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.delenv("CLYPT_AUDIO_DIARIZATION_ENABLE", raising=False)

    turns = worker._run_audio_diarization("audio.wav")

    assert turns == []
    assert worker._last_audio_diarization_metrics == {
        "audio_diarization_enabled": False,
        "audio_diarization_fallback": True,
        "audio_diarization_status": "disabled",
        "audio_diarization_turn_count": 0,
    }


def test_audio_turn_binds_to_best_visible_local_track():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    turns = [
        {
            "speaker_id": "SPEAKER_00",
            "start_time_ms": 1000,
            "end_time_ms": 2400,
            "exclusive": True,
        }
    ]
    local_candidate_evidence = [
        {
            "start_time_ms": 1000,
            "end_time_ms": 1200,
            "candidates": [
                {"local_track_id": "track_8", "score": 0.92},
                {"local_track_id": "track_3", "score": 0.20},
            ],
        },
        {
            "start_time_ms": 1200,
            "end_time_ms": 1800,
            "candidates": [
                {"local_track_id": "track_3", "score": 0.85},
                {"local_track_id": "track_8", "score": 0.30},
            ],
        },
        {
            "start_time_ms": 1800,
            "end_time_ms": 2400,
            "candidates": [
                {"local_track_id": "track_3", "score": 0.80},
                {"local_track_id": "track_8", "score": 0.25},
            ],
        },
    ]

    bindings = worker._bind_audio_turns_to_local_tracks(turns, local_candidate_evidence)

    assert bindings == [
        {
            "speaker_id": "SPEAKER_00",
            "start_time_ms": 1000,
            "end_time_ms": 2400,
            "local_track_id": "track_3",
            "ambiguous": False,
            "winning_score": pytest.approx(1030.0 / 1400.0, abs=1e-6),
            "winning_margin": pytest.approx((1030.0 - 514.0) / 1400.0, abs=1e-6),
            "support_ratio": pytest.approx(1.0, abs=1e-6),
        }
    ]


def test_audio_turn_stays_unknown_when_candidates_are_near_tied():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    turns = [
        {
            "speaker_id": "SPEAKER_01",
            "start_time_ms": 0,
            "end_time_ms": 1000,
            "exclusive": True,
        }
    ]
    local_candidate_evidence = [
        {
            "start_time_ms": 0,
            "end_time_ms": 500,
            "candidates": [
                {"local_track_id": "track_1", "score": 0.82},
                {"local_track_id": "track_2", "score": 0.79},
            ],
        },
        {
            "start_time_ms": 500,
            "end_time_ms": 1000,
            "candidates": [
                {"local_track_id": "track_1", "score": 0.80},
                {"local_track_id": "track_2", "score": 0.81},
            ],
        },
    ]

    bindings = worker._bind_audio_turns_to_local_tracks(
        turns,
        local_candidate_evidence,
        ambiguity_margin=0.02,
    )

    assert bindings == [
        {
            "speaker_id": "SPEAKER_01",
            "start_time_ms": 0,
            "end_time_ms": 1000,
            "local_track_id": None,
            "ambiguous": True,
            "winning_score": pytest.approx(0.81, abs=1e-6),
            "winning_margin": pytest.approx(0.01, abs=1e-6),
            "support_ratio": pytest.approx(1.0, abs=1e-6),
        }
    ]


def test_audio_turn_breaks_score_tie_with_stronger_support():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    turns = [
        {
            "speaker_id": "SPEAKER_02",
            "start_time_ms": 0,
            "end_time_ms": 1000,
            "exclusive": True,
        }
    ]
    local_candidate_evidence = [
        {
            "start_time_ms": 0,
            "end_time_ms": 1000,
            "candidates": [
                {"local_track_id": "track_full", "score": 0.80},
            ],
        },
        {
            "start_time_ms": 0,
            "end_time_ms": 800,
            "candidates": [
                {"local_track_id": "track_partial", "score": 1.00},
            ],
        },
    ]

    bindings = worker._bind_audio_turns_to_local_tracks(turns, local_candidate_evidence)

    assert bindings == [
        {
            "speaker_id": "SPEAKER_02",
            "start_time_ms": 0,
            "end_time_ms": 1000,
            "local_track_id": "track_full",
            "ambiguous": False,
            "winning_score": pytest.approx(0.80, abs=1e-6),
            "winning_margin": pytest.approx(0.0, abs=1e-6),
            "support_ratio": pytest.approx(1.0, abs=1e-6),
        }
    ]


def test_audio_turn_overlap_stays_ambiguous_instead_of_committing():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    turns = [
        {
            "speaker_id": "SPEAKER_02",
            "start_time_ms": 0,
            "end_time_ms": 1000,
            "exclusive": False,
            "overlap": True,
        }
    ]
    local_candidate_evidence = [
        {
            "start_time_ms": 0,
            "end_time_ms": 1000,
            "candidates": [
                {"local_track_id": "track_full", "score": 0.90},
            ],
        },
    ]

    bindings = worker._bind_audio_turns_to_local_tracks(turns, local_candidate_evidence)

    assert bindings == [
        {
            "speaker_id": "SPEAKER_02",
            "start_time_ms": 0,
            "end_time_ms": 1000,
            "local_track_id": None,
            "ambiguous": True,
            "winning_score": pytest.approx(0.45, abs=1e-6),
            "winning_margin": pytest.approx(0.45, abs=1e-6),
            "support_ratio": pytest.approx(1.0, abs=1e-6),
        }
    ]


def test_audio_turn_overlap_without_visible_evidence_stays_explicitly_ambiguous():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    turns = [
        {
            "speaker_id": "SPEAKER_04",
            "start_time_ms": 0,
            "end_time_ms": 1000,
            "exclusive": False,
            "overlap": True,
        }
    ]

    bindings = worker._bind_audio_turns_to_local_tracks(turns, [])

    assert bindings == [
        {
            "speaker_id": "SPEAKER_04",
            "start_time_ms": 0,
            "end_time_ms": 1000,
            "local_track_id": None,
            "ambiguous": True,
            "winning_score": None,
            "winning_margin": None,
            "support_ratio": 0.0,
        }
    ]


def test_audio_turn_overlapping_evidence_does_not_exceed_turn_bounds():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    turns = [
        {
            "speaker_id": "SPEAKER_03",
            "start_time_ms": 0,
            "end_time_ms": 1000,
            "exclusive": True,
        }
    ]
    local_candidate_evidence = [
        {
            "start_time_ms": 0,
            "end_time_ms": 700,
            "candidates": [
                {"local_track_id": "track_1", "score": 1.0},
            ],
        },
        {
            "start_time_ms": 300,
            "end_time_ms": 1000,
            "candidates": [
                {"local_track_id": "track_1", "score": 1.0},
            ],
        },
        {
            "start_time_ms": 0,
            "end_time_ms": 1000,
            "candidates": [
                {"local_track_id": "track_2", "score": 0.4},
            ],
        },
    ]

    bindings = worker._bind_audio_turns_to_local_tracks(turns, local_candidate_evidence)

    assert bindings == [
        {
            "speaker_id": "SPEAKER_03",
            "start_time_ms": 0,
            "end_time_ms": 1000,
            "local_track_id": "track_1",
            "ambiguous": False,
            "winning_score": pytest.approx(1.0, abs=1e-6),
            "winning_margin": pytest.approx(0.6, abs=1e-6),
            "support_ratio": pytest.approx(1.0, abs=1e-6),
        }
    ]


def test_audio_speaker_local_track_map_keeps_only_clear_stable_support():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    mapping = worker._build_audio_speaker_local_track_map(
        [
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 900,
                "local_track_id": "track_3",
                "ambiguous": False,
                "winning_score": 0.93,
                "winning_margin": 0.28,
                "support_ratio": 0.96,
            },
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 1100,
                "end_time_ms": 2100,
                "local_track_id": "track_3",
                "ambiguous": False,
                "winning_score": 0.89,
                "winning_margin": 0.24,
                "support_ratio": 0.91,
            },
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 2200,
                "end_time_ms": 3000,
                "local_track_id": "track_9",
                "ambiguous": True,
                "winning_score": 0.84,
                "winning_margin": 0.02,
                "support_ratio": 0.74,
            },
            {
                "speaker_id": "SPEAKER_01",
                "start_time_ms": 0,
                "end_time_ms": 700,
                "local_track_id": "track_4",
                "ambiguous": False,
                "winning_score": 0.66,
                "winning_margin": 0.09,
                "support_ratio": 0.58,
            },
        ]
    )

    assert mapping == [
        {
            "speaker_id": "SPEAKER_00",
            "local_track_id": "track_3",
            "support_segments": 2,
            "support_ms": 1900,
            "confidence": pytest.approx(0.87, abs=0.05),
        }
    ]


def test_audio_speaker_local_track_map_drops_unclear_shared_local_track_ownership():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    mapping = worker._build_audio_speaker_local_track_map(
        [
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 900,
                "local_track_id": "track_shared",
                "ambiguous": False,
                "winning_score": 0.92,
                "winning_margin": 0.27,
                "support_ratio": 0.95,
            },
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 1100,
                "end_time_ms": 2100,
                "local_track_id": "track_shared",
                "ambiguous": False,
                "winning_score": 0.88,
                "winning_margin": 0.23,
                "support_ratio": 0.90,
            },
            {
                "speaker_id": "SPEAKER_01",
                "start_time_ms": 2500,
                "end_time_ms": 3400,
                "local_track_id": "track_shared",
                "ambiguous": False,
                "winning_score": 0.91,
                "winning_margin": 0.26,
                "support_ratio": 0.94,
            },
            {
                "speaker_id": "SPEAKER_01",
                "start_time_ms": 3600,
                "end_time_ms": 4550,
                "local_track_id": "track_shared",
                "ambiguous": False,
                "winning_score": 0.87,
                "winning_margin": 0.22,
                "support_ratio": 0.89,
            },
        ]
    )

    assert mapping == []


def test_audio_speaker_local_track_map_ignores_crowded_three_candidate_windows():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    mapping = worker._build_audio_speaker_local_track_map(
        [
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 900,
                "local_track_id": "track_3",
                "ambiguous": False,
                "winning_score": 0.93,
                "winning_margin": 0.28,
                "support_ratio": 0.96,
                "max_visible_candidates": 3,
            },
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 1100,
                "end_time_ms": 2100,
                "local_track_id": "track_3",
                "ambiguous": False,
                "winning_score": 0.89,
                "winning_margin": 0.24,
                "support_ratio": 0.91,
                "max_visible_candidates": 3,
            },
        ]
    )

    assert mapping == []


def test_audio_speaker_local_track_map_keeps_mostly_clean_turn_with_brief_crowded_slice():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    turns = [
        {
            "speaker_id": "SPEAKER_00",
            "start_time_ms": 0,
            "end_time_ms": 2200,
            "exclusive": True,
        },
        {
            "speaker_id": "SPEAKER_00",
            "start_time_ms": 2600,
            "end_time_ms": 3600,
            "exclusive": True,
        },
    ]
    local_candidate_evidence = [
        {
            "start_time_ms": 0,
            "end_time_ms": 800,
            "candidates": [
                {"local_track_id": "track_3", "score": 0.93},
                {"local_track_id": "track_8", "score": 0.24},
            ],
        },
        {
            "start_time_ms": 800,
            "end_time_ms": 1200,
            "candidates": [
                {"local_track_id": "track_3", "score": 0.87},
                {"local_track_id": "track_8", "score": 0.38},
                {"local_track_id": "track_9", "score": 0.33},
            ],
        },
        {
            "start_time_ms": 1200,
            "end_time_ms": 2200,
            "candidates": [
                {"local_track_id": "track_3", "score": 0.91},
                {"local_track_id": "track_8", "score": 0.21},
            ],
        },
        {
            "start_time_ms": 2600,
            "end_time_ms": 3600,
            "candidates": [
                {"local_track_id": "track_3", "score": 0.89},
                {"local_track_id": "track_8", "score": 0.22},
            ],
        },
    ]

    bindings = worker._bind_audio_turns_to_local_tracks(turns, local_candidate_evidence)
    mapping = worker._build_audio_speaker_local_track_map(bindings)

    assert bindings[0]["local_track_id"] == "track_3"
    assert bindings[0]["max_visible_candidates"] == 3
    assert bindings[0]["clean_support_ms"] == 1800
    assert bindings[0]["clean_support_ratio"] == pytest.approx(1800.0 / 2200.0, abs=1e-6)
    assert mapping == [
        {
            "speaker_id": "SPEAKER_00",
            "local_track_id": "track_3",
            "support_segments": 2,
            "support_ms": 2800,
            "confidence": pytest.approx(0.88, abs=0.06),
        }
    ]


def test_audio_speaker_local_track_map_uses_clean_winner_when_full_turn_winner_differs():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    turns = [
        {
            "speaker_id": "SPEAKER_00",
            "start_time_ms": 0,
            "end_time_ms": 2200,
            "exclusive": True,
        },
        {
            "speaker_id": "SPEAKER_00",
            "start_time_ms": 2600,
            "end_time_ms": 3600,
            "exclusive": True,
        },
    ]
    local_candidate_evidence = [
        {
            "start_time_ms": 0,
            "end_time_ms": 800,
            "candidates": [
                {"local_track_id": "track_3", "score": 0.92},
                {"local_track_id": "track_8", "score": 0.18},
            ],
        },
        {
            "start_time_ms": 800,
            "end_time_ms": 1200,
            "candidates": [
                {"local_track_id": "track_8", "score": 4.9},
                {"local_track_id": "track_3", "score": 0.40},
                {"local_track_id": "track_9", "score": 0.39},
            ],
        },
        {
            "start_time_ms": 1200,
            "end_time_ms": 2200,
            "candidates": [
                {"local_track_id": "track_3", "score": 0.90},
                {"local_track_id": "track_8", "score": 0.20},
            ],
        },
        {
            "start_time_ms": 2600,
            "end_time_ms": 3600,
            "candidates": [
                {"local_track_id": "track_3", "score": 0.89},
                {"local_track_id": "track_8", "score": 0.22},
            ],
        },
    ]

    bindings = worker._bind_audio_turns_to_local_tracks(turns, local_candidate_evidence)
    mapping = worker._build_audio_speaker_local_track_map(bindings)

    assert bindings[0]["local_track_id"] == "track_8"
    assert bindings[0]["clean_local_track_id"] == "track_3"
    assert bindings[0]["clean_support_ms"] == 1800
    assert mapping == [
        {
            "speaker_id": "SPEAKER_00",
            "local_track_id": "track_3",
            "support_segments": 2,
            "support_ms": 2800,
            "confidence": pytest.approx(0.88, abs=0.06),
        }
    ]


def test_audio_speaker_local_track_map_salvages_ambiguous_full_turn_from_clean_slices():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    turns = [
        {
            "speaker_id": "SPEAKER_00",
            "start_time_ms": 0,
            "end_time_ms": 2200,
            "exclusive": True,
        },
        {
            "speaker_id": "SPEAKER_00",
            "start_time_ms": 2600,
            "end_time_ms": 3600,
            "exclusive": True,
        },
    ]
    local_candidate_evidence = [
        {
            "start_time_ms": 0,
            "end_time_ms": 800,
            "candidates": [
                {"local_track_id": "track_3", "score": 0.92},
                {"local_track_id": "track_8", "score": 0.18},
            ],
        },
        {
            "start_time_ms": 800,
            "end_time_ms": 1200,
            "candidates": [
                {"local_track_id": "track_8", "score": 3.45},
                {"local_track_id": "track_3", "score": 0.40},
                {"local_track_id": "track_9", "score": 0.39},
            ],
        },
        {
            "start_time_ms": 1200,
            "end_time_ms": 2200,
            "candidates": [
                {"local_track_id": "track_3", "score": 0.90},
                {"local_track_id": "track_8", "score": 0.20},
            ],
        },
        {
            "start_time_ms": 2600,
            "end_time_ms": 3600,
            "candidates": [
                {"local_track_id": "track_3", "score": 0.89},
                {"local_track_id": "track_8", "score": 0.22},
            ],
        },
    ]

    bindings = worker._bind_audio_turns_to_local_tracks(turns, local_candidate_evidence)
    mapping = worker._build_audio_speaker_local_track_map(bindings)

    assert bindings[0]["ambiguous"] is True
    assert bindings[0]["local_track_id"] is None
    assert bindings[0]["clean_local_track_id"] == "track_3"
    assert bindings[0]["clean_support_ms"] == 1800
    assert mapping == [
        {
            "speaker_id": "SPEAKER_00",
            "local_track_id": "track_3",
            "support_segments": 2,
            "support_ms": 2800,
            "confidence": pytest.approx(0.88, abs=0.06),
        }
    ]


def test_face_detector_input_size_defaults_to_960_and_honors_env(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()

    monkeypatch.delenv("CLYPT_FACE_DETECTOR_INPUT_SIZE", raising=False)
    monkeypatch.delenv("CLYPT_FACE_DETECTOR_INPUT_LONG_EDGE", raising=False)
    assert worker_cls._face_detector_input_size() == (960, 960)

    monkeypatch.setenv("CLYPT_FACE_DETECTOR_INPUT_SIZE", "960")
    assert worker_cls._face_detector_input_size() == (960, 960)

    monkeypatch.delenv("CLYPT_FACE_DETECTOR_INPUT_SIZE", raising=False)
    monkeypatch.setenv("CLYPT_FACE_DETECTOR_INPUT_LONG_EDGE", "768")
    assert worker_cls._face_detector_input_size() == (768, 768)


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
    monkeypatch.setenv("CLYPT_SHARED_ANALYSIS_PROXY", "0")
    monkeypatch.setenv("CLYPT_ANALYSIS_PROXY_ENABLE", "0")

    meta_by_path = {
        str(video_path): {"width": 3840, "height": 2160, "fps": 24.0, "total_frames": 10, "duration_s": 1.0},
        str(proxy_path): {"width": 1280, "height": 720, "fps": 24.0, "total_frames": 10, "duration_s": 1.0},
    }
    monkeypatch.setattr(worker, "_probe_video_meta", lambda path: meta_by_path[str(path)])
    monkeypatch.setattr(worker, "_ensure_h264", lambda path: str(path))

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


def test_speaker_binding_mode_forces_pyannote_visual(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    calls = []

    monkeypatch.setenv("CLYPT_SPEAKER_BINDING_MODE", "pyannote_visual")
    monkeypatch.setattr(
        worker,
        "_run_pyannote_visual_binding",
        lambda **kwargs: calls.append("pyannote_visual") or [],
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

    assert result == []
    assert calls == ["pyannote_visual"]


def test_speaker_binding_mode_forces_heuristic_clears_stale_candidate_debug(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    worker._last_speaker_candidate_debug = [{"word": "stale"}]

    monkeypatch.setenv("CLYPT_SPEAKER_BINDING_MODE", "heuristic")
    monkeypatch.setattr(
        worker,
        "_run_speaker_binding_heuristic",
        lambda *args, **kwargs: [{"track_id": "t1"}],
    )

    worker._run_speaker_binding(
        video_path="video.mp4",
        audio_wav_path="audio.wav",
        tracks=[{"track_id": "t1"}],
        words=[{"start_time_ms": 0, "end_time_ms": 100}],
    )

    assert worker._last_speaker_candidate_debug == []


def test_speaker_binding_mode_auto_prefers_heuristic_for_large_long_video(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    calls = []

    monkeypatch.delenv("CLYPT_SPEAKER_BINDING_MODE", raising=False)
    monkeypatch.setenv("CLYPT_SHARED_ANALYSIS_PROXY", "0")
    monkeypatch.setenv("CLYPT_ANALYSIS_PROXY_ENABLE", "0")
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


def test_run_lrasd_binding_clears_stale_candidate_debug_on_early_return(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    worker._last_speaker_candidate_debug = [{"word": "stale"}]
    worker.lrasd_model = None
    worker.lrasd_loss_av = None

    result = worker._run_lrasd_binding(
        video_path="video.mp4",
        audio_wav_path="audio.wav",
        tracks=[{"track_id": "t1"}],
        words=[{"start_time_ms": 0, "end_time_ms": 100}],
    )

    assert result is None
    assert worker._last_speaker_candidate_debug == []


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


def test_associate_faces_to_person_dets_uses_one_to_one_assignment_for_adjacent_people():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    dets = [
        {"track_id": "a", "x1": 100.0, "y1": 100.0, "x2": 420.0, "y2": 600.0},
        {"track_id": "b", "x1": 280.0, "y1": 100.0, "x2": 600.0, "y2": 600.0},
    ]
    faces = [
        {"bbox_xyxy": (160.0, 120.0, 240.0, 220.0)},
        {"bbox_xyxy": (300.0, 120.0, 380.0, 220.0)},
    ]

    assert worker._associate_faces_to_person_dets(faces, dets) == ["a", "b"]


def test_prewarm_face_runtime_in_pool_initializes_worker_thread_runtime(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    worker._face_runtime_main_thread_ident = threading.get_ident()
    worker._face_runtime_local = threading.local()
    worker._face_detector_input_size = (640, 640)

    call_thread_ids = []
    detect_calls = []
    recognize_calls = []

    class FakeDetector:
        def detect(self, frame, input_size=None, max_num=0):
            detect_calls.append((threading.get_ident(), tuple(frame.shape), tuple(input_size or ()), int(max_num)))
            return np.zeros((0, 5), dtype=np.float32), None

    class FakeRecognizer:
        def get(self, frame, face):
            recognize_calls.append((threading.get_ident(), tuple(frame.shape), tuple(np.asarray(face.bbox).tolist())))
            return np.ones((512,), dtype=np.float32)

    fake_common = types.ModuleType("insightface.app.common")

    class FakeFace:
        def __init__(self, bbox, kps, det_score):
            self.bbox = bbox
            self.kps = kps
            self.det_score = det_score

    fake_common.Face = FakeFace
    monkeypatch.setitem(sys.modules, "insightface.app.common", fake_common)

    def fake_get_thread_face_runtime():
        call_thread_ids.append(threading.get_ident())
        return FakeDetector(), FakeRecognizer()

    worker._get_thread_face_runtime = fake_get_thread_face_runtime

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker._prewarm_face_runtime_in_pool(pool, 1)

    assert call_thread_ids
    assert any(thread_id != threading.get_ident() for thread_id in call_thread_ids)
    assert detect_calls and recognize_calls
    assert all(thread_id != threading.get_ident() for thread_id, *_ in detect_calls)
    assert all(thread_id != threading.get_ident() for thread_id, *_ in recognize_calls)


def test_eligible_associated_track_ids_defaults_to_single_dominant_track(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.delenv("CLYPT_FACE_TRACK_ALLOW_MULTI_ASSOC", raising=False)

    assert worker._eligible_associated_track_ids({"host": 8, "neighbor": 6}) == {"host"}


def test_shared_analysis_proxy_defaults_to_enabled_with_1920_long_edge(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.delenv("CLYPT_SHARED_ANALYSIS_PROXY", raising=False)
    monkeypatch.delenv("CLYPT_ANALYSIS_PROXY_ENABLE", raising=False)
    monkeypatch.delenv("CLYPT_ANALYSIS_PROXY_MAX_LONG_EDGE", raising=False)
    monkeypatch.delenv("CLYPT_SPEAKER_BINDING_PROXY_MAX_LONG_EDGE", raising=False)

    assert worker._shared_analysis_proxy_enabled() is True
    assert worker._analysis_proxy_max_long_edge() == 1920


def test_shared_analysis_proxy_explicit_disable_overrides_default(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.setenv("CLYPT_SHARED_ANALYSIS_PROXY", "0")
    monkeypatch.delenv("CLYPT_ANALYSIS_PROXY_ENABLE", raising=False)

    assert worker._shared_analysis_proxy_enabled() is False


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


def test_select_speaker_binding_mode_supports_pyannote_visual(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.setenv("CLYPT_SPEAKER_BINDING_MODE", "pyannote_visual")

    assert worker._select_speaker_binding_mode(
        video_path="video.mp4",
        tracks=[{"track_id": "track-1"}],
        words=[{"start_time_ms": 0, "end_time_ms": 100}],
    ) == "pyannote_visual"


def test_make_lrasd_frame_provider_reads_and_caches(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    class _Batch:
        def __init__(self, frames):
            self._frames = frames

        def asnumpy(self):
            return np.stack(self._frames, axis=0)

    class _VideoReader:
        def __init__(self):
            self.calls = []
            self.frames = [
                np.full((8, 8, 3), fill_value=i, dtype=np.uint8)
                for i in range(4)
            ]

        def __len__(self):
            return len(self.frames)

        def get_batch(self, indices):
            self.calls.append(tuple(indices))
            return _Batch([self.frames[i] for i in indices])

    vr = _VideoReader()
    get_frame = worker._make_lrasd_frame_provider(vr)

    frame_1 = get_frame(1)
    frame_1_again = get_frame(1)

    assert frame_1.shape == (8, 8, 3)
    assert np.array_equal(frame_1, frame_1_again)
    assert vr.calls == [(1, 2, 3)]


def test_prepare_lrasd_visual_crop_cpu_fallback_outputs_grayscale_112(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    worker.gpu_device = "cpu"

    monkeypatch.setenv("CLYPT_LRASD_GPU_PREPROCESS", "1")

    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[10:30, 10:30, 0] = 255
    crop = worker._prepare_lrasd_visual_crop(
        frame,
        x1=10,
        y1=10,
        x2=30,
        y2=30,
    )

    assert crop is not None
    assert crop.shape == (112, 112)
    assert crop.dtype == np.uint8
    assert float(crop.mean()) > 0.0


def test_prepare_lrasd_visual_crop_uses_torch_path_when_enabled(monkeypatch):
    import cv2

    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    worker.gpu_device = "cpu"

    monkeypatch.setenv("CLYPT_LRASD_GPU_PREPROCESS", "1")
    monkeypatch.setattr(
        cv2,
        "resize",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cv2.resize should not run")),
    )
    monkeypatch.setattr(
        cv2,
        "cvtColor",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cv2.cvtColor should not run")),
    )

    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[10:30, 10:30, 1] = 255
    crop = worker._prepare_lrasd_visual_crop(
        frame,
        x1=10,
        y1=10,
        x2=30,
        y2=30,
    )

    assert crop is not None
    assert crop.shape == (112, 112)
    assert crop.dtype == np.uint8
    assert float(crop.mean()) > 0.0


def test_prepare_lrasd_visual_batch_uses_torch_path_and_returns_tensor(monkeypatch):
    import cv2
    import torch

    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    worker.gpu_device = "cpu"

    monkeypatch.setenv("CLYPT_LRASD_GPU_PREPROCESS", "1")
    monkeypatch.setattr(
        cv2,
        "resize",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cv2.resize should not run")),
    )
    monkeypatch.setattr(
        cv2,
        "cvtColor",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cv2.cvtColor should not run")),
    )

    crops = [
        np.full((18, 20, 3), 64, dtype=np.uint8),
        np.full((22, 16, 3), 192, dtype=np.uint8),
    ]
    batch = worker._prepare_lrasd_visual_batch(crops, bucket_length=4)

    assert isinstance(batch, torch.Tensor)
    assert tuple(batch.shape) == (4, 112, 112)
    assert str(batch.device) == "cpu"
    assert float(batch[0].mean()) > 0.0
    assert torch.equal(batch[2], batch[1])
    assert torch.equal(batch[3], batch[1])


def test_open_lrasd_video_reader_prefers_gpu_when_available(monkeypatch, tmp_path: Path):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    calls = []

    class _VideoReader:
        def __init__(self, path, ctx=None):
            calls.append((str(path), ctx))
            self.path = path
            self.ctx = ctx

    fake_decord = types.ModuleType("decord")
    fake_decord.VideoReader = _VideoReader
    fake_decord.cpu = lambda index: f"cpu:{index}"
    fake_decord.gpu = lambda index: f"gpu:{index}"
    fake_decord.bridge = types.SimpleNamespace(set_bridge=lambda _name: None)
    monkeypatch.setitem(__import__("sys").modules, "decord", fake_decord)
    monkeypatch.setenv("CLYPT_LRASD_GPU_DECODE", "1")

    vr, backend = worker._open_lrasd_video_reader(str(video_path))

    assert backend == "gpu"
    assert calls == [
        (str(video_path), "gpu:0"),
    ]
    assert vr.ctx == "gpu:0"


def test_open_lrasd_video_reader_raises_when_gpu_unavailable(monkeypatch, tmp_path: Path):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    calls = []

    class _VideoReader:
        def __init__(self, path, ctx=None):
            calls.append((str(path), ctx))
            if ctx == "gpu:0":
                raise RuntimeError("gpu unavailable")
            self.path = path
            self.ctx = ctx

    fake_decord = types.ModuleType("decord")
    fake_decord.VideoReader = _VideoReader
    fake_decord.cpu = lambda index: f"cpu:{index}"
    fake_decord.gpu = lambda index: f"gpu:{index}"
    fake_decord.bridge = types.SimpleNamespace(set_bridge=lambda _name: None)
    monkeypatch.setitem(__import__("sys").modules, "decord", fake_decord)
    monkeypatch.setenv("CLYPT_LRASD_GPU_DECODE", "1")

    with pytest.raises(RuntimeError, match="gpu unavailable"):
        worker._open_lrasd_video_reader(str(video_path))

    assert calls == [
        (str(video_path), "gpu:0"),
    ]


def test_make_video_reader_delegates_to_open_lrasd_video_reader(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    sentinel = object()
    calls = []

    def _fake_open(path):
        calls.append(path)
        return sentinel, "gpu"

    monkeypatch.setattr(worker, "_open_lrasd_video_reader", _fake_open)

    vr, backend = worker._make_video_reader("video.mp4")

    assert vr is sentinel
    assert backend == "gpu"
    assert calls == ["video.mp4"]


def test_run_lrasd_binding_strict_gpu_decode_error_includes_underlying_exception(monkeypatch, tmp_path: Path):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    worker.lrasd_model = object()
    worker.lrasd_loss_av = object()

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")

    monkeypatch.setenv("CLYPT_LRASD_GPU_DECODE", "1")
    monkeypatch.setattr(
        worker,
        "_open_lrasd_video_reader",
        lambda path: (_ for _ in ()).throw(RuntimeError("nvdec init failed")),
    )

    with pytest.raises(
        RuntimeError,
        match="LR-ASD GPU decode was requested but binding video initialization failed.*nvdec init failed",
    ):
        worker._run_lrasd_binding(
            video_path=str(video_path),
            audio_wav_path=str(audio_path),
            tracks=[{"track_id": "speaker", "frame_idx": 0}],
            words=[{"text": "hello", "start_time_ms": 0, "end_time_ms": 1000}],
            analysis_context={
                "analysis_video_path": str(video_path),
                "analysis_meta": {"width": 1920, "height": 1080, "fps": 30.0},
                "scale_x": 1.0,
                "scale_y": 1.0,
            },
        )


def test_make_lrasd_frame_provider_recovers_from_gpu_batch_failure(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    class _Batch:
        def __init__(self, frames):
            self._frames = frames

        def asnumpy(self):
            return np.stack(self._frames, axis=0)

    class _VideoReader:
        def __init__(self, frames, fail_once=False):
            self.frames = frames
            self.fail_once = fail_once
            self.calls = []

        def __len__(self):
            return len(self.frames)

        def get_batch(self, indices):
            self.calls.append(tuple(indices))
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("gpu batch failed")
            return _Batch([self.frames[i] for i in indices])

    frames = [np.full((8, 8, 3), fill_value=i, dtype=np.uint8) for i in range(4)]
    gpu_reader = _VideoReader(frames, fail_once=True)
    cpu_reader = _VideoReader(frames, fail_once=False)
    fallback_calls = []

    def _fallback_factory():
        fallback_calls.append("cpu")
        return cpu_reader

    get_frame = worker._make_lrasd_frame_provider(gpu_reader, fallback_factory=_fallback_factory)
    frame = get_frame(1)

    assert frame.shape == (8, 8, 3)
    assert int(frame[0, 0, 0]) == 1
    assert fallback_calls == ["cpu"]
    assert gpu_reader.calls == [(1, 2, 3)]
    assert cpu_reader.calls == [(1, 2, 3)]


def test_make_lrasd_frame_provider_strict_mode_raises_on_gpu_batch_failure(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    class _VideoReader:
        def __init__(self, frames, fail_once=False):
            self.frames = frames
            self.fail_once = fail_once
            self.calls = []

        def __len__(self):
            return len(self.frames)

        def get_batch(self, indices):
            self.calls.append(tuple(indices))
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("gpu batch failed")
            raise AssertionError("fallback reader should never be used in strict mode")

    frames = [np.full((8, 8, 3), fill_value=i, dtype=np.uint8) for i in range(4)]
    gpu_reader = _VideoReader(frames, fail_once=True)
    fallback_calls = []

    def _fallback_factory():
        fallback_calls.append("cpu")
        return _VideoReader(frames, fail_once=False)

    monkeypatch.setenv("CLYPT_LRASD_GPU_DECODE_STRICT", "1")
    get_frame = worker._make_lrasd_frame_provider(gpu_reader, fallback_factory=_fallback_factory)

    with pytest.raises(RuntimeError, match="gpu batch failed"):
        get_frame(1)

    assert fallback_calls == []
    assert gpu_reader.calls == [(1, 2, 3)]


def test_lrasd_batch_length_bucket_maps_raw_lengths_into_small_bucket_set():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    assert [worker._lrasd_batch_length_bucket(length) for length in (20, 24, 25, 32, 33, 48, 49, 64, 65, 80, 81, 96, 97, 120)] == [
        24,
        24,
        32,
        32,
        48,
        48,
        64,
        64,
        80,
        80,
        96,
        96,
        120,
        120,
    ]


def test_lrasd_should_flush_bucket_when_underfilled_but_stale_or_scan_gap_exceeded():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    assert worker._lrasd_should_flush_bucket(
        pending_count=2,
        batch_size=4,
        oldest_age_s=0.40,
        scan_gap=1,
    ) is True
    assert worker._lrasd_should_flush_bucket(
        pending_count=2,
        batch_size=4,
        oldest_age_s=0.05,
        scan_gap=16,
    ) is True
    assert worker._lrasd_should_flush_bucket(
        pending_count=2,
        batch_size=4,
        oldest_age_s=0.05,
        scan_gap=3,
    ) is False


def test_lrasd_align_score_row_ignores_padded_tail_using_original_valid_length():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    row = np.concatenate(
        [
            np.full((12,), 0.25, dtype=np.float32),
            np.full((4,), 0.95, dtype=np.float32),
        ]
    )

    aligned = worker._lrasd_align_score_row(
        row,
        valid_length=6,
        bucket_length=8,
    )

    assert aligned.shape == (6,)
    assert np.allclose(aligned, 0.25)


def test_lrasd_build_pending_subchunk_pads_visual_and_audio_to_bucket(monkeypatch):
    import torch

    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    worker.gpu_device = "cpu"

    monkeypatch.setenv("CLYPT_LRASD_GPU_PREPROCESS", "1")

    frames = list(range(10, 35))
    crops = [np.full((12 + (i % 3), 14 + (i % 5), 3), fill_value=10 + i, dtype=np.uint8) for i in range(len(frames))]
    audio_features = np.ones((512, 13), dtype=np.float32)

    pending_entry = worker._lrasd_build_pending_subchunk(
        tid="track_1",
        frames=frames,
        crops=crops,
        full_audio_features=audio_features,
        min_chunk_frames=20,
    )

    assert pending_entry is not None
    bucket_length, (tid, valid_frames, valid_length, visual_batch, audio_np) = pending_entry
    assert bucket_length == 32
    assert tid == "track_1"
    assert valid_frames == frames
    assert valid_length == 25
    assert isinstance(visual_batch, torch.Tensor)
    assert tuple(visual_batch.shape) == (32, 112, 112)
    assert audio_np.shape == (128, 13)


def test_lrasd_prep_worker_and_queue_settings_are_bounded(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.delenv("CLYPT_LRASD_PREP_WORKERS", raising=False)
    monkeypatch.delenv("CLYPT_LRASD_PREP_QUEUE", raising=False)
    assert worker._lrasd_prep_workers() == 4
    assert worker._lrasd_prep_queue_limit(4) == 128

    monkeypatch.setenv("CLYPT_LRASD_PREP_WORKERS", "99")
    monkeypatch.setenv("CLYPT_LRASD_PREP_QUEUE", "1")
    assert worker._lrasd_prep_workers() == 16
    assert worker._lrasd_prep_queue_limit(16) == 32


def test_prepare_lrasd_chunk_pending_entries_splits_on_real_face_gap(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.setattr(worker, "_scale_detection_geometry", lambda det, **kwargs: det)

    pending_calls = []

    def fake_build_pending_subchunk(**kwargs):
        pending_calls.append(list(kwargs["frames"]))
        frames = list(kwargs["frames"])
        return 24, (
            kwargs["tid"],
            frames,
            len(frames),
            np.zeros((24, 112, 112), dtype=np.uint8),
            np.zeros((96, 13), dtype=np.float32),
        )

    monkeypatch.setattr(worker, "_lrasd_build_pending_subchunk", fake_build_pending_subchunk)
    monkeypatch.setattr(
        worker,
        "_extract_lrasd_visual_crop_region",
        lambda frame, **kwargs: np.ones((12, 12, 3), dtype=np.uint8),
    )

    chunk_frames = list(range(47))
    best_by_frame = {
        fi: {
            "x_center": 20.0,
            "y_center": 20.0,
            "width": 10.0,
            "height": 10.0,
        }
        for fi in list(range(20)) + list(range(27, 47))
    }

    payload = worker._prepare_lrasd_chunk_pending_entries(
        tid="track_1",
        chunk_frames=chunk_frames,
        best_by_frame=best_by_frame,
        get_frame=lambda fi: np.ones((32, 32, 3), dtype=np.uint8),
        face_cache={},
        canonical_face_boxes={},
        full_audio_features=np.ones((512, 13), dtype=np.float32),
        min_chunk_frames=20,
        binding_scale_x=1.0,
        binding_scale_y=1.0,
    )

    assert payload["face_hits"] == 40
    assert payload["face_misses"] == 7
    assert [len(entry[1][1]) for entry in payload["pending_entries"]] == [20, 20]
    assert pending_calls == [list(range(20)), list(range(27, 47))]


def test_lrasd_should_report_drain_progress_on_progress_or_stall():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    assert worker._lrasd_should_report_drain_progress(
        prepared_chunks=10,
        scored_chunks=8,
        last_prepared_chunks=9,
        last_scored_chunks=8,
        prep_queue_depth=4,
        inflight_batches=1,
        pending_bucket_count=2,
        now_s=10.0,
        last_report_s=9.5,
    )

    assert worker._lrasd_should_report_drain_progress(
        prepared_chunks=10,
        scored_chunks=8,
        last_prepared_chunks=10,
        last_scored_chunks=8,
        prep_queue_depth=4,
        inflight_batches=0,
        pending_bucket_count=0,
        now_s=15.1,
        last_report_s=10.0,
    )

    assert not worker._lrasd_should_report_drain_progress(
        prepared_chunks=10,
        scored_chunks=8,
        last_prepared_chunks=10,
        last_scored_chunks=8,
        prep_queue_depth=4,
        inflight_batches=0,
        pending_bucket_count=0,
        now_s=12.0,
        last_report_s=10.0,
    )


def test_extract_lrasd_visual_crop_region_supports_torch_tensor():
    import torch

    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    frame = torch.arange(6 * 8 * 3, dtype=torch.float32).reshape(6, 8, 3)
    crop = worker._extract_lrasd_visual_crop_region(
        frame,
        x1=1,
        y1=2,
        x2=5,
        y2=6,
    )

    assert isinstance(crop, torch.Tensor)
    assert tuple(crop.shape) == (4, 4, 3)
    assert crop.is_contiguous()


def test_prepare_lrasd_visual_batch_supports_torch_crops():
    import torch

    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    worker.gpu_device = "cpu"

    crop_a = torch.full((12, 10, 3), 32.0)
    crop_b = torch.full((8, 6, 3), 96.0)
    batch = worker._prepare_lrasd_visual_batch(
        [crop_a, crop_b],
        bucket_length=4,
        output_size=(16, 16),
    )

    assert isinstance(batch, torch.Tensor)
    assert tuple(batch.shape) == (4, 16, 16)
    assert str(batch.dtype) == "torch.float32"


def test_open_lrasd_video_reader_requires_gpu_when_enabled(monkeypatch):
    fake_decord = types.ModuleType("decord")

    class _VideoReader:
        def __init__(self, path, ctx=None):
            raise RuntimeError(f"ctx={ctx}")

    fake_decord.VideoReader = _VideoReader
    fake_decord.cpu = lambda _index: "cpu"
    fake_decord.gpu = lambda _index: "gpu"
    fake_decord.bridge = types.SimpleNamespace(set_bridge=lambda _name: None)
    monkeypatch.setitem(__import__("sys").modules, "decord", fake_decord)
    monkeypatch.setenv("CLYPT_LRASD_GPU_DECODE", "1")
    monkeypatch.delenv("CLYPT_LRASD_GPU_DECODE_STRICT", raising=False)

    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    with pytest.raises(RuntimeError, match="ctx=gpu"):
        worker._open_lrasd_video_reader("/tmp/demo.mp4")


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


def _run_fake_lrasd_binding_case(
    monkeypatch,
    tmp_path: Path,
    *,
    track_specs: dict[str, dict],
    score_by_track: dict[str, float],
    words: list[dict] | None = None,
    track_id_remap: dict[str, str] | None = None,
    audio_speaker_turns: list[dict] | None = None,
    audio_turn_bindings: list[dict] | None = None,
):
    def _frame_is_visible(spec: dict, frame_idx: int, default_end: int) -> bool:
        visible_frames = spec.get("visible_frames")
        if visible_frames is not None:
            return frame_idx in {int(fi) for fi in visible_frames}
        hidden_frames = spec.get("hidden_frames")
        if hidden_frames is not None and frame_idx in {int(fi) for fi in hidden_frames}:
            return False
        visible_start = int(spec.get("visible_start_frame", 0))
        visible_end = int(spec.get("visible_end_frame", default_end))
        return visible_start <= frame_idx <= visible_end

    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    worker.lrasd_model = object()
    worker.lrasd_loss_av = object()
    worker.gpu_device = "cpu"
    worker.model_debug = False
    worker.model_debug_every = 0
    worker._last_speaker_binding_metrics = {}

    video_path = tmp_path / "video.mp4"
    audio_path = tmp_path / "audio.wav"
    cached_features_path = tmp_path / "video_lrasd_features.npz"
    video_path.write_bytes(b"video")
    audio_path.write_bytes(b"audio")
    np.savez_compressed(
        cached_features_path,
        audio_features=np.full((400, 13), 0.25, dtype=np.float32),
    )

    frame_h = 200
    frame_w = 300
    frame_count = 40
    frames = []
    for frame_idx in range(frame_count):
        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        for spec in track_specs.values():
            if not _frame_is_visible(spec, frame_idx, frame_count - 1):
                continue
            x1 = max(0, int(round(float(spec["x_center"]) - (0.5 * float(spec["width"])))))
            y1 = max(0, int(round(float(spec["y_center"]) - (0.5 * float(spec["height"])))))
            x2 = min(frame_w, int(round(float(spec["x_center"]) + (0.5 * float(spec["width"])))))
            y2 = min(frame_h, int(round(float(spec["y_center"]) + (0.5 * float(spec["height"])))))
            if x2 > x1 and y2 > y1:
                frame[y1:y2, x1:x2] = int(spec["intensity"])
        frames.append(frame)

    class _Batch:
        def __init__(self, batch_frames):
            self._frames = batch_frames

        def asnumpy(self):
            return np.stack(self._frames, axis=0)

    class _VideoReader:
        def __init__(self, path, ctx=None):
            self._frames = frames

        def __len__(self):
            return len(self._frames)

        def get_avg_fps(self):
            return 30.0

        def get_batch(self, indices):
            return _Batch([self._frames[i] for i in indices])

    fake_decord = types.ModuleType("decord")
    fake_decord.VideoReader = _VideoReader
    fake_decord.cpu = lambda _index: object()
    monkeypatch.setitem(sys.modules, "decord", fake_decord)

    fake_psf = types.ModuleType("python_speech_features")
    fake_psf.mfcc = lambda *args, **kwargs: np.zeros((10, 13), dtype=np.float32)
    monkeypatch.setitem(sys.modules, "python_speech_features", fake_psf)

    import scipy.io.wavfile as wavfile_module

    monkeypatch.setattr(wavfile_module, "read", lambda path: (16000, np.zeros(48000, dtype=np.int16)))
    monkeypatch.setattr(worker, "_build_canonical_face_bbox_lookup", lambda **kwargs: {})
    if audio_turn_bindings is not None:
        monkeypatch.setattr(
            worker,
            "_bind_audio_turns_to_local_tracks",
            lambda turns, local_candidate_evidence, **kwargs: audio_turn_bindings,
        )

    tracks = []
    for frame_idx in range(frame_count):
        for tid, spec in track_specs.items():
            if not _frame_is_visible(spec, frame_idx, frame_count - 1):
                continue
            width = float(spec["width"])
            height = float(spec["height"])
            x_center = float(spec["x_center"])
            y_center = float(spec["y_center"])
            tracks.append(
                {
                    "frame_idx": frame_idx,
                    "track_id": tid,
                    "class_id": 0,
                    "label": "person",
                    "geometry_type": "aabb",
                    "x1": x_center - (0.5 * width),
                    "y1": y_center - (0.5 * height),
                    "x2": x_center + (0.5 * width),
                    "y2": y_center + (0.5 * height),
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "confidence": float(spec.get("confidence", 0.9)),
                }
            )

    intensity_to_tid = {
        int(spec["intensity"]): tid
        for tid, spec in track_specs.items()
    }
    scored_local_track_ids = []

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
                "numpy": lambda self: np.stack(
                    [
                        np.full(
                            (audio_t.shape[1],),
                            float(
                                score_by_track[
                                    intensity_to_tid[
                                        min(
                                            intensity_to_tid.keys(),
                                            key=lambda intensity: abs(
                                                intensity - int(round(float(row.mean())))
                                            ),
                                        )
                                    ]
                                ]
                            ),
                            dtype=np.float32,
                        )
                        for row in _record_scored_rows(
                            visual_t.detach().cpu().numpy().mean(axis=(1, 2, 3)),
                            intensity_to_tid,
                            scored_local_track_ids,
                        )
                    ],
                    axis=0,
                ),
            },
        )(),
    )

    if words is None:
        words = [
            {"text": "hello", "start_time_ms": 0, "end_time_ms": 1000},
        ]

    bindings = worker._run_lrasd_binding(
        video_path=str(video_path),
        audio_wav_path=str(audio_path),
        tracks=tracks,
        words=words,
        frame_to_dets=None,
        track_to_dets=None,
        track_identity_features=None,
        analysis_context={
            "analysis_video_path": str(video_path),
            "scale_x": 1.0,
            "scale_y": 1.0,
            "analysis_meta": {"width": frame_w, "height": frame_h, "fps": 30.0},
            "audio_speaker_turns": audio_speaker_turns or [],
        },
        track_id_remap=track_id_remap,
    )
    worker._test_lrasd_scored_local_track_ids = scored_local_track_ids
    return worker, words, bindings


def _record_scored_rows(rows, intensity_to_tid, scored_local_track_ids: list[str]):
    for row in rows:
        scored_local_track_ids.append(
            intensity_to_tid[
                min(
                    intensity_to_tid.keys(),
                    key=lambda intensity: abs(
                        intensity - int(round(float(row.mean())))
                    ),
                )
            ]
        )
        yield row


def test_run_lrasd_binding_prefers_clean_body_track_over_sink_like_candidate(monkeypatch, tmp_path: Path):
    worker, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "sink": {
                "x_center": 228.0,
                "y_center": 105.0,
                "width": 210.0,
                "height": 178.0,
                "confidence": 0.98,
                "intensity": 80,
            },
            "speaker": {
                "x_center": 78.0,
                "y_center": 102.0,
                "width": 74.0,
                "height": 156.0,
                "confidence": 0.92,
                "intensity": 210,
            },
        },
        score_by_track={
            "sink": 0.84,
            "speaker": 0.78,
        },
    )

    assert words[0]["speaker_track_id"] == "speaker"
    assert bindings[0]["track_id"] == "speaker"
    assert worker._last_speaker_binding_metrics["canonical_face_stream_coverage"] == 0.0


def test_run_lrasd_binding_rejects_tiny_edge_fragment_candidate(monkeypatch, tmp_path: Path):
    _, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "fragment": {
                "x_center": 12.0,
                "y_center": 58.0,
                "width": 18.0,
                "height": 28.0,
                "confidence": 0.97,
                "intensity": 60,
            },
            "speaker": {
                "x_center": 156.0,
                "y_center": 100.0,
                "width": 76.0,
                "height": 150.0,
                "confidence": 0.88,
                "intensity": 200,
            },
        },
        score_by_track={
            "fragment": 0.88,
            "speaker": 0.76,
        },
    )

    assert words[0]["speaker_track_id"] == "speaker"
    assert bindings[0]["track_id"] == "speaker"


def test_run_lrasd_binding_prunes_sink_like_track_before_chunk_generation(monkeypatch, tmp_path: Path):
    worker, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "sink": {
                "x_center": 228.0,
                "y_center": 105.0,
                "width": 210.0,
                "height": 178.0,
                "confidence": 0.98,
                "intensity": 80,
            },
            "speaker": {
                "x_center": 78.0,
                "y_center": 102.0,
                "width": 74.0,
                "height": 156.0,
                "confidence": 0.92,
                "intensity": 210,
            },
        },
        score_by_track={
            "sink": 0.84,
            "speaker": 0.78,
        },
    )

    assert set(worker._test_lrasd_scored_local_track_ids) == {"speaker"}


def test_run_lrasd_binding_prunes_tiny_fragment_before_chunk_generation(monkeypatch, tmp_path: Path):
    worker, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "fragment": {
                "x_center": 12.0,
                "y_center": 58.0,
                "width": 18.0,
                "height": 28.0,
                "confidence": 0.97,
                "intensity": 60,
            },
            "speaker": {
                "x_center": 156.0,
                "y_center": 100.0,
                "width": 76.0,
                "height": 150.0,
                "confidence": 0.88,
                "intensity": 200,
            },
        },
        score_by_track={
            "fragment": 0.88,
            "speaker": 0.76,
        },
    )

    assert set(worker._test_lrasd_scored_local_track_ids) == {"speaker"}


def test_rank_lrasd_turn_candidates_keeps_stable_body_track_with_sparse_face_coverage():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    def _track_dets(track_id: str, *, x_center: float, y_center: float, width: float, height: float, confidence: float):
        rows = []
        for frame_idx in range(30):
            rows.append(
                {
                    "track_id": track_id,
                    "frame_idx": frame_idx,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "x1": x_center - (0.5 * width),
                    "y1": y_center - (0.5 * height),
                    "x2": x_center + (0.5 * width),
                    "y2": y_center + (0.5 * height),
                    "confidence": confidence,
                }
            )
        return rows

    track_to_dets = {
        "speaker": _track_dets("speaker", x_center=90.0, y_center=100.0, width=82.0, height=150.0, confidence=0.92),
        "tiny_face": _track_dets("tiny_face", x_center=270.0, y_center=38.0, width=18.0, height=28.0, confidence=0.97),
    }
    frame_to_dets = {
        frame_idx: [track_to_dets["speaker"][frame_idx], track_to_dets["tiny_face"][frame_idx]]
        for frame_idx in range(30)
    }
    track_quality_by_tid = {
        "speaker": {"track_quality": 0.93, "median_area_norm": 0.205},
        "tiny_face": {"track_quality": 0.54, "median_area_norm": 0.008},
    }
    canonical_face_boxes = {
        ("speaker", 0): (55.0, 36.0, 125.0, 88.0, 0.9, "speaker"),
        ("speaker", 15): (55.0, 36.0, 125.0, 88.0, 0.9, "speaker"),
    }
    canonical_face_boxes.update(
        {
            ("tiny_face", frame_idx): (262.0, 22.0, 278.0, 34.0, 0.95, "tiny_face")
            for frame_idx in range(30)
        }
    )

    ranked = worker._rank_lrasd_turn_candidates(
        turn={"speaker_id": "SPEAKER_00", "start_time_ms": 0, "end_time_ms": 1000},
        eligible_track_ids={"speaker", "tiny_face"},
        track_to_dets=track_to_dets,
        frame_to_dets=frame_to_dets,
        track_quality_by_tid=track_quality_by_tid,
        canonical_face_boxes=canonical_face_boxes,
        frame_width=300,
        frame_height=200,
        fps=30.0,
    )

    assert [candidate["local_track_id"] for candidate in ranked[:2]] == ["speaker", "tiny_face"]
    assert ranked[0]["face_coverage"] < ranked[1]["face_coverage"]
    assert ranked[0]["speech_overlap_ratio"] > 0.95
    assert ranked[0]["rank_score"] > ranked[1]["rank_score"]


def test_rank_lrasd_turn_candidates_penalizes_tiny_face_heavy_fragment():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    def _track_dets(track_id: str, *, x_center: float, y_center: float, width: float, height: float, confidence: float):
        rows = []
        for frame_idx in range(24):
            rows.append(
                {
                    "track_id": track_id,
                    "frame_idx": frame_idx,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "x1": x_center - (0.5 * width),
                    "y1": y_center - (0.5 * height),
                    "x2": x_center + (0.5 * width),
                    "y2": y_center + (0.5 * height),
                    "confidence": confidence,
                }
            )
        return rows

    track_to_dets = {
        "speaker": _track_dets("speaker", x_center=120.0, y_center=102.0, width=74.0, height=148.0, confidence=0.89),
        "fragment": _track_dets("fragment", x_center=10.0, y_center=46.0, width=16.0, height=24.0, confidence=0.99),
    }
    frame_to_dets = {
        frame_idx: [track_to_dets["speaker"][frame_idx], track_to_dets["fragment"][frame_idx]]
        for frame_idx in range(24)
    }
    track_quality_by_tid = {
        "speaker": {"track_quality": 0.86, "median_area_norm": 0.182},
        "fragment": {"track_quality": 0.76, "median_area_norm": 0.006},
    }
    canonical_face_boxes = {
        ("fragment", frame_idx): (4.0, 15.0, 16.0, 24.0, 0.96, "fragment")
        for frame_idx in range(24)
    }

    ranked = worker._rank_lrasd_turn_candidates(
        turn={"speaker_id": "SPEAKER_00", "start_time_ms": 0, "end_time_ms": 800},
        eligible_track_ids={"speaker", "fragment"},
        track_to_dets=track_to_dets,
        frame_to_dets=frame_to_dets,
        track_quality_by_tid=track_quality_by_tid,
        canonical_face_boxes=canonical_face_boxes,
        frame_width=300,
        frame_height=200,
        fps=30.0,
    )

    assert [candidate["local_track_id"] for candidate in ranked[:2]] == ["speaker", "fragment"]
    assert ranked[1]["face_coverage"] == pytest.approx(1.0, abs=1e-6)
    assert ranked[1]["median_area_norm"] < 0.01
    assert ranked[0]["rank_score"] > ranked[1]["rank_score"]


def test_run_lrasd_binding_scores_all_globally_eligible_candidates_when_turn_topk_enabled(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CLYPT_LRASD_TOPK_PER_TURN", "1")
    worker, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "speaker": {
                "x_center": 82.0,
                "y_center": 100.0,
                "width": 78.0,
                "height": 152.0,
                "confidence": 0.91,
                "intensity": 210,
            },
            "listener": {
                "x_center": 222.0,
                "y_center": 102.0,
                "width": 64.0,
                "height": 144.0,
                "confidence": 0.88,
                "intensity": 120,
            },
        },
        score_by_track={
            "speaker": 0.77,
            "listener": 0.79,
        },
        audio_speaker_turns=[
            {"speaker_id": "SPEAKER_00", "start_time_ms": 0, "end_time_ms": 1000},
        ],
    )

    assert set(worker._test_lrasd_scored_local_track_ids) == {"speaker", "listener"}


def test_lrasd_topk_candidates_per_turn_disabled_when_unset_or_zero(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()

    monkeypatch.delenv("CLYPT_LRASD_TOPK_PER_TURN", raising=False)
    assert worker_cls._lrasd_topk_candidates_per_turn() == 0

    monkeypatch.setenv("CLYPT_LRASD_TOPK_PER_TURN", "0")
    assert worker_cls._lrasd_topk_candidates_per_turn() == 0


def test_run_lrasd_binding_turn_topk_zero_disables_pruning(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CLYPT_LRASD_TOPK_PER_TURN", "0")
    worker, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "speaker": {
                "x_center": 82.0,
                "y_center": 100.0,
                "width": 78.0,
                "height": 152.0,
                "confidence": 0.91,
                "intensity": 210,
            },
            "listener": {
                "x_center": 222.0,
                "y_center": 102.0,
                "width": 64.0,
                "height": 144.0,
                "confidence": 0.88,
                "intensity": 120,
            },
        },
        score_by_track={
            "speaker": 0.77,
            "listener": 0.79,
        },
        audio_speaker_turns=[
            {"speaker_id": "SPEAKER_00", "start_time_ms": 0, "end_time_ms": 1000},
        ],
    )

    assert set(worker._test_lrasd_scored_local_track_ids) == {"speaker", "listener"}


def test_run_lrasd_binding_does_not_reduce_candidates_using_turn_subselection(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CLYPT_LRASD_TOPK_PER_TURN", "1")
    worker, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "listener": {
                "x_center": 222.0,
                "y_center": 102.0,
                "width": 68.0,
                "height": 144.0,
                "confidence": 0.9,
                "intensity": 120,
            },
            "speaker": {
                "x_center": 82.0,
                "y_center": 100.0,
                "width": 78.0,
                "height": 152.0,
                "confidence": 0.91,
                "intensity": 210,
                "visible_start_frame": 20,
                "visible_end_frame": 39,
            },
        },
        score_by_track={
            "listener": 0.79,
            "speaker": 0.77,
        },
        words=[
            {"text": "late", "start_time_ms": 800, "end_time_ms": 1000},
        ],
        audio_speaker_turns=[
            {"speaker_id": "SPEAKER_00", "start_time_ms": 0, "end_time_ms": 1000},
        ],
    )

    assert set(worker._test_lrasd_scored_local_track_ids) == {"speaker", "listener"}


def test_run_lrasd_binding_uses_runtime_relevant_candidate_count_for_fallback_gate(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CLYPT_LRASD_TOPK_PER_TURN", "1")
    worker, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "listener": {
                "x_center": 222.0,
                "y_center": 102.0,
                "width": 68.0,
                "height": 144.0,
                "confidence": 0.9,
                "intensity": 120,
                "visible_start_frame": 0,
                "visible_end_frame": 19,
            },
            "speaker": {
                "x_center": 82.0,
                "y_center": 100.0,
                "width": 78.0,
                "height": 152.0,
                "confidence": 0.91,
                "intensity": 210,
                "visible_start_frame": 20,
                "visible_end_frame": 39,
            },
        },
        score_by_track={
            "listener": 0.79,
            "speaker": 0.77,
        },
        words=[
            {"text": "late", "start_time_ms": 800, "end_time_ms": 1000},
        ],
        audio_speaker_turns=[
            {"speaker_id": "SPEAKER_00", "start_time_ms": 0, "end_time_ms": 1000},
        ],
    )

    assert set(worker._test_lrasd_scored_local_track_ids) == {"speaker"}
    assert worker._last_speaker_binding_metrics["lrasd_eligible_track_count"] == 2
    assert worker._last_speaker_binding_metrics["lrasd_turn_selected_track_count"] == 1
    assert bindings == [
        {"track_id": "speaker", "start_time_ms": 800, "end_time_ms": 1000, "word_count": 1}
    ]


def test_run_lrasd_binding_counts_occluded_runtime_candidate_for_fallback_gate(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CLYPT_LRASD_TOPK_PER_TURN", "1")
    worker, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "listener": {
                "x_center": 222.0,
                "y_center": 102.0,
                "width": 68.0,
                "height": 144.0,
                "confidence": 0.9,
                "intensity": 120,
                "visible_frames": list(range(0, 19)) + [25, 27, 28, 29],
            },
            "speaker": {
                "x_center": 82.0,
                "y_center": 100.0,
                "width": 78.0,
                "height": 152.0,
                "confidence": 0.91,
                "intensity": 210,
            },
        },
        score_by_track={
            "listener": 0.79,
            "speaker": 0.77,
        },
        words=[
            {"text": "late", "start_time_ms": 760, "end_time_ms": 980},
        ],
        audio_speaker_turns=[
            {"speaker_id": "SPEAKER_00", "start_time_ms": 0, "end_time_ms": 1000},
        ],
    )

    assert set(worker._test_lrasd_scored_local_track_ids) == {"speaker"}
    assert worker._last_speaker_binding_metrics["lrasd_eligible_track_count"] == 2
    assert bindings is None


def test_run_lrasd_binding_allows_scoring_in_diarization_gap(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CLYPT_LRASD_TOPK_PER_TURN", "1")
    worker, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "speaker": {
                "x_center": 82.0,
                "y_center": 100.0,
                "width": 78.0,
                "height": 152.0,
                "confidence": 0.91,
                "intensity": 210,
            },
            "listener": {
                "x_center": 222.0,
                "y_center": 102.0,
                "width": 64.0,
                "height": 144.0,
                "confidence": 0.88,
                "intensity": 120,
            },
        },
        score_by_track={
            "speaker": 0.77,
            "listener": 0.79,
        },
        words=[
            {"text": "gapword", "start_time_ms": 700, "end_time_ms": 1000},
        ],
        audio_speaker_turns=[
            {"speaker_id": "SPEAKER_00", "start_time_ms": 0, "end_time_ms": 200},
        ],
    )

    assert set(worker._test_lrasd_scored_local_track_ids) == {"speaker", "listener"}


def test_run_lrasd_binding_preserves_local_track_id_when_global_remap_applies(monkeypatch, tmp_path: Path):
    _, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "lawyer_local": {
                "x_center": 78.0,
                "y_center": 102.0,
                "width": 74.0,
                "height": 156.0,
                "confidence": 0.92,
                "intensity": 210,
            },
            "listener_local": {
                "x_center": 228.0,
                "y_center": 105.0,
                "width": 80.0,
                "height": 156.0,
                "confidence": 0.88,
                "intensity": 80,
            },
        },
        score_by_track={
            "lawyer_local": 0.82,
            "listener_local": 0.21,
        },
        track_id_remap={
            "lawyer_local": "Global_Person_0",
            "listener_local": "Global_Person_0",
        },
    )

    assert words[0]["speaker_track_id"] == "Global_Person_0"
    assert words[0]["speaker_local_track_id"] == "lawyer_local"
    assert words[0]["speaker_local_tag"] == "lawyer_local"
    assert bindings[0]["track_id"] == "Global_Person_0"


def test_run_lrasd_binding_uses_audio_speaker_prior_when_visual_margin_is_small(monkeypatch, tmp_path: Path):
    _, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "speaker_local": {
                "x_center": 78.0,
                "y_center": 102.0,
                "width": 72.0,
                "height": 148.0,
                "confidence": 0.88,
                "intensity": 210,
            },
            "listener_local": {
                "x_center": 222.0,
                "y_center": 104.0,
                "width": 70.0,
                "height": 146.0,
                "confidence": 0.87,
                "intensity": 80,
            },
        },
        score_by_track={
            "speaker_local": 0.180,
            "listener_local": 0.172,
        },
        words=[
            {"text": "hello", "start_time_ms": 0, "end_time_ms": 1000},
        ],
        audio_speaker_turns=[
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "exclusive": True,
            }
        ],
        audio_turn_bindings=[
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "local_track_id": "listener_local",
                "ambiguous": False,
                "winning_score": 0.92,
                "winning_margin": 0.32,
                "support_ratio": 1.0,
            }
        ],
    )

    assert words[0]["speaker_local_track_id"] == "listener_local"
    assert bindings[0]["track_id"] == "listener_local"


def test_run_lrasd_binding_uses_audio_prior_when_local_tracks_share_global_id(monkeypatch, tmp_path: Path):
    _, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "speaker_local_a": {
                "x_center": 78.0,
                "y_center": 102.0,
                "width": 72.0,
                "height": 148.0,
                "confidence": 0.88,
                "intensity": 210,
            },
            "speaker_local_b": {
                "x_center": 222.0,
                "y_center": 104.0,
                "width": 70.0,
                "height": 146.0,
                "confidence": 0.87,
                "intensity": 80,
            },
        },
        score_by_track={
            "speaker_local_a": 0.180,
            "speaker_local_b": 0.172,
        },
        words=[
            {"text": "hello", "start_time_ms": 0, "end_time_ms": 1000},
        ],
        track_id_remap={
            "speaker_local_a": "Global_Person_0",
            "speaker_local_b": "Global_Person_0",
        },
        audio_speaker_turns=[
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "exclusive": True,
            }
        ],
        audio_turn_bindings=[
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "local_track_id": "speaker_local_b",
                "ambiguous": False,
                "winning_score": 0.92,
                "winning_margin": 0.32,
                "support_ratio": 1.0,
            }
        ],
    )

    assert words[0]["speaker_track_id"] == "Global_Person_0"
    assert words[0]["speaker_local_track_id"] == "speaker_local_b"
    assert bindings[0]["track_id"] == "Global_Person_0"


def test_run_lrasd_binding_reuses_strong_turn_owner_without_strong_contrary_visual_event(monkeypatch, tmp_path: Path):
    _, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "speaker_local": {
                "x_center": 78.0,
                "y_center": 102.0,
                "width": 72.0,
                "height": 148.0,
                "confidence": 0.88,
                "intensity": 210,
            },
            "listener_local": {
                "x_center": 222.0,
                "y_center": 104.0,
                "width": 70.0,
                "height": 146.0,
                "confidence": 0.87,
                "intensity": 80,
            },
        },
        score_by_track={
            "speaker_local": 0.18,
            "listener_local": 0.28,
        },
        words=[
            {"text": "hello", "start_time_ms": 0, "end_time_ms": 1000},
        ],
        audio_speaker_turns=[
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "exclusive": True,
            }
        ],
        audio_turn_bindings=[
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "local_track_id": "speaker_local",
                "ambiguous": False,
                "winning_score": 0.94,
                "winning_margin": 0.31,
                "support_ratio": 0.96,
            }
        ],
    )

    assert words[0]["speaker_local_track_id"] == "speaker_local"
    assert bindings[0]["track_id"] == "speaker_local"


def test_run_lrasd_binding_persists_speaker_candidate_debug(monkeypatch, tmp_path: Path):
    worker, words, _bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "speaker_local": {
                "x_center": 78.0,
                "y_center": 102.0,
                "width": 72.0,
                "height": 148.0,
                "confidence": 0.88,
                "intensity": 210,
            },
            "listener_local": {
                "x_center": 222.0,
                "y_center": 104.0,
                "width": 70.0,
                "height": 146.0,
                "confidence": 0.87,
                "intensity": 80,
            },
        },
        score_by_track={
            "speaker_local": 0.180,
            "listener_local": 0.172,
        },
        words=[
            {"text": "hello", "start_time_ms": 0, "end_time_ms": 1000},
        ],
        audio_speaker_turns=[
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "exclusive": True,
            }
        ],
        audio_turn_bindings=[
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "local_track_id": "listener_local",
                "ambiguous": False,
                "winning_score": 0.92,
                "winning_margin": 0.32,
                "support_ratio": 1.0,
            }
        ],
    )

    assert words[0]["speaker_local_track_id"] == "listener_local"
    assert worker._last_speaker_candidate_debug == [
        {
            "word": "hello",
            "start_time_ms": 0,
            "end_time_ms": 1000,
            "active_audio_speaker_id": "SPEAKER_00",
            "active_audio_local_track_id": "listener_local",
            "chosen_track_id": "listener_local",
            "chosen_local_track_id": "listener_local",
            "decision_source": "audio_boosted_visual",
            "ambiguous": False,
            "top_1_top_2_margin": pytest.approx(0.0, abs=0.1),
            "candidates": [
                {
                    "local_track_id": "listener_local",
                    "track_id": "listener_local",
                    "blended_score": pytest.approx(0.0, abs=1.0),
                    "asd_probability": pytest.approx(0.172, abs=0.001),
                    "body_prior": pytest.approx(0.0, abs=1.0),
                    "detection_confidence": pytest.approx(0.87, abs=0.001),
                },
                {
                    "local_track_id": "speaker_local",
                    "track_id": "speaker_local",
                    "blended_score": pytest.approx(0.0, abs=1.0),
                    "asd_probability": pytest.approx(0.18, abs=0.001),
                    "body_prior": pytest.approx(0.0, abs=1.0),
                    "detection_confidence": pytest.approx(0.88, abs=0.001),
                },
            ],
        }
    ]


def test_run_lrasd_binding_persists_candidate_debug_for_off_screen_audio_abstention(monkeypatch, tmp_path: Path):
    worker, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "listener_local": {
                "x_center": 220.0,
                "y_center": 104.0,
                "width": 68.0,
                "height": 142.0,
                "confidence": 0.84,
                "intensity": 80,
            },
        },
        score_by_track={
            "listener_local": 0.19,
        },
        words=[
            {"text": "hello", "start_time_ms": 0, "end_time_ms": 1000},
        ],
        audio_speaker_turns=[
            {
                "speaker_id": "SPEAKER_02",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "exclusive": True,
            }
        ],
        audio_turn_bindings=[
            {
                "speaker_id": "SPEAKER_02",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "local_track_id": "offscreen_local",
                "ambiguous": False,
                "winning_score": 0.91,
                "winning_margin": 0.35,
                "support_ratio": 1.0,
            }
        ],
    )

    assert words[0]["speaker_track_id"] is None
    assert bindings == []
    assert worker._last_speaker_candidate_debug == [
        {
            "word": "hello",
            "start_time_ms": 0,
            "end_time_ms": 1000,
            "active_audio_speaker_id": "SPEAKER_02",
            "active_audio_local_track_id": "offscreen_local",
            "chosen_track_id": None,
            "chosen_local_track_id": None,
            "decision_source": "unknown",
            "ambiguous": False,
            "top_1_top_2_margin": pytest.approx(0.0, abs=1.0),
            "candidates": [
                {
                    "local_track_id": "listener_local",
                    "track_id": "listener_local",
                    "blended_score": pytest.approx(0.0, abs=1.0),
                    "asd_probability": pytest.approx(0.19, abs=0.001),
                    "body_prior": pytest.approx(0.0, abs=1.0),
                    "detection_confidence": pytest.approx(0.84, abs=0.001),
                }
            ],
        }
    ]


def test_run_lrasd_binding_persists_ambiguous_audio_turn_context_in_debug(monkeypatch, tmp_path: Path):
    worker, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "speaker_local": {
                "x_center": 78.0,
                "y_center": 102.0,
                "width": 76.0,
                "height": 152.0,
                "confidence": 0.90,
                "intensity": 210,
            },
            "listener_local": {
                "x_center": 222.0,
                "y_center": 104.0,
                "width": 72.0,
                "height": 148.0,
                "confidence": 0.88,
                "intensity": 80,
            },
        },
        score_by_track={
            "speaker_local": 0.82,
            "listener_local": 0.26,
        },
        words=[
            {"text": "hello", "start_time_ms": 0, "end_time_ms": 1000},
        ],
        audio_speaker_turns=[
            {
                "speaker_id": "SPEAKER_01",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "exclusive": True,
            }
        ],
        audio_turn_bindings=[
            {
                "speaker_id": "SPEAKER_01",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "local_track_id": "listener_local",
                "ambiguous": True,
                "winning_score": 0.52,
                "winning_margin": 0.01,
                "support_ratio": 1.0,
            }
        ],
    )

    assert words[0]["speaker_local_track_id"] == "speaker_local"
    assert bindings == [{"track_id": "speaker_local", "start_time_ms": 0, "end_time_ms": 1000, "word_count": 1}]
    assert worker._last_speaker_candidate_debug == [
        {
            "word": "hello",
            "start_time_ms": 0,
            "end_time_ms": 1000,
            "active_audio_speaker_id": "SPEAKER_01",
            "active_audio_local_track_id": "listener_local",
            "chosen_track_id": "speaker_local",
            "chosen_local_track_id": "speaker_local",
            "decision_source": "visual",
            "ambiguous": True,
            "top_1_top_2_margin": pytest.approx(0.0, abs=1.0),
            "candidates": [
                {
                    "local_track_id": "speaker_local",
                    "track_id": "speaker_local",
                    "blended_score": pytest.approx(0.0, abs=1.0),
                    "asd_probability": pytest.approx(0.82, abs=0.001),
                    "body_prior": pytest.approx(0.0, abs=1.0),
                    "detection_confidence": pytest.approx(0.90, abs=0.001),
                },
                {
                    "local_track_id": "listener_local",
                    "track_id": "listener_local",
                    "blended_score": pytest.approx(0.0, abs=1.0),
                    "asd_probability": pytest.approx(0.26, abs=0.001),
                    "body_prior": pytest.approx(0.0, abs=1.0),
                    "detection_confidence": pytest.approx(0.88, abs=0.001),
                },
            ],
        }
    ]


def test_run_lrasd_binding_uses_clean_local_track_id_for_ambiguous_crowded_turn_debug(monkeypatch, tmp_path: Path):
    worker, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "speaker_local": {
                "x_center": 78.0,
                "y_center": 102.0,
                "width": 76.0,
                "height": 152.0,
                "confidence": 0.90,
                "intensity": 210,
            },
            "listener_local": {
                "x_center": 222.0,
                "y_center": 104.0,
                "width": 72.0,
                "height": 148.0,
                "confidence": 0.88,
                "intensity": 80,
            },
        },
        score_by_track={
            "speaker_local": 0.82,
            "listener_local": 0.26,
        },
        words=[
            {"text": "hello", "start_time_ms": 0, "end_time_ms": 1000},
        ],
        audio_speaker_turns=[
            {
                "speaker_id": "SPEAKER_01",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "exclusive": True,
            }
        ],
        audio_turn_bindings=[
            {
                "speaker_id": "SPEAKER_01",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "local_track_id": None,
                "clean_local_track_id": "listener_local",
                "ambiguous": True,
                "winning_score": 0.52,
                "winning_margin": 0.01,
                "support_ratio": 1.0,
                "max_visible_candidates": 3,
                "clean_support_ratio": 0.95,
                "clean_winning_score": 0.88,
                "clean_winning_margin": 0.21,
            }
        ],
    )

    assert words[0]["speaker_local_track_id"] == "speaker_local"
    assert bindings == [{"track_id": "speaker_local", "start_time_ms": 0, "end_time_ms": 1000, "word_count": 1}]
    assert worker._last_speaker_candidate_debug[0]["active_audio_local_track_id"] == "listener_local"
    assert worker._last_speaker_candidate_debug[0]["ambiguous"] is True


def test_run_lrasd_binding_does_not_override_clear_visual_winner_with_audio_prior(monkeypatch, tmp_path: Path):
    _, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "speaker_local": {
                "x_center": 78.0,
                "y_center": 102.0,
                "width": 76.0,
                "height": 152.0,
                "confidence": 0.90,
                "intensity": 210,
            },
            "listener_local": {
                "x_center": 222.0,
                "y_center": 104.0,
                "width": 72.0,
                "height": 148.0,
                "confidence": 0.88,
                "intensity": 80,
            },
        },
        score_by_track={
            "speaker_local": 0.82,
            "listener_local": 0.26,
        },
        words=[
            {"text": "hello", "start_time_ms": 0, "end_time_ms": 1000},
        ],
        audio_speaker_turns=[
            {
                "speaker_id": "SPEAKER_01",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "exclusive": True,
            }
        ],
        audio_turn_bindings=[
            {
                "speaker_id": "SPEAKER_01",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "local_track_id": "listener_local",
                "ambiguous": False,
                "winning_score": 0.94,
                "winning_margin": 0.41,
                "support_ratio": 1.0,
            }
        ],
    )

    assert words[0]["speaker_local_track_id"] == "speaker_local"
    assert bindings[0]["track_id"] == "speaker_local"


def test_run_lrasd_binding_keeps_unknown_for_off_screen_audio_turn(monkeypatch, tmp_path: Path):
    _, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "listener_local": {
                "x_center": 220.0,
                "y_center": 104.0,
                "width": 68.0,
                "height": 142.0,
                "confidence": 0.84,
                "intensity": 80,
            },
        },
        score_by_track={
            "listener_local": 0.19,
        },
        words=[
            {"text": "hello", "start_time_ms": 0, "end_time_ms": 1000},
        ],
        audio_speaker_turns=[
            {
                "speaker_id": "SPEAKER_02",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "exclusive": True,
            }
        ],
        audio_turn_bindings=[
            {
                "speaker_id": "SPEAKER_02",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "local_track_id": "offscreen_local",
                "ambiguous": False,
                "winning_score": 0.91,
                "winning_margin": 0.35,
                "support_ratio": 1.0,
            }
        ],
    )

    assert words[0]["speaker_track_id"] is None
    assert words[0]["speaker_local_track_id"] is None
    assert bindings == []


def test_run_lrasd_binding_keeps_unknown_when_audio_turn_has_no_visible_match(monkeypatch, tmp_path: Path):
    worker, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "listener_local": {
                "x_center": 220.0,
                "y_center": 104.0,
                "width": 68.0,
                "height": 142.0,
                "confidence": 0.84,
                "intensity": 80,
            },
        },
        score_by_track={
            "listener_local": 0.82,
        },
        words=[
            {"text": "hello", "start_time_ms": 0, "end_time_ms": 1000},
        ],
        audio_speaker_turns=[
            {
                "speaker_id": "SPEAKER_02",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "exclusive": True,
            }
        ],
        audio_turn_bindings=[
            {
                "speaker_id": "SPEAKER_02",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "local_track_id": None,
                "ambiguous": False,
                "winning_score": 0.91,
                "winning_margin": 0.35,
                "support_ratio": 1.0,
            }
        ],
    )

    assert words[0]["speaker_track_id"] is None
    assert words[0]["speaker_local_track_id"] is None
    assert bindings == []
    assert worker._last_speaker_candidate_debug[0]["active_audio_local_track_id"] is None
    assert worker._last_speaker_candidate_debug[0]["decision_source"] == "unknown"


def test_run_lrasd_binding_preserves_overlap_turn_as_ambiguous_unknown(monkeypatch, tmp_path: Path):
    worker, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "listener_local": {
                "x_center": 220.0,
                "y_center": 104.0,
                "width": 68.0,
                "height": 142.0,
                "confidence": 0.84,
                "intensity": 80,
            },
        },
        score_by_track={
            "listener_local": 0.82,
        },
        words=[
            {"text": "hello", "start_time_ms": 0, "end_time_ms": 1000},
        ],
        audio_speaker_turns=[
            {
                "speaker_id": "SPEAKER_03",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "exclusive": False,
                "overlap": True,
            }
        ],
    )

    assert words[0]["speaker_track_id"] is None
    assert words[0]["speaker_local_track_id"] is None
    assert bindings == []
    assert worker._last_audio_turn_bindings == [
        {
            "speaker_id": "SPEAKER_03",
            "start_time_ms": 0,
            "end_time_ms": 1000,
            "local_track_id": None,
            "ambiguous": True,
            "winning_score": pytest.approx(0.39416, abs=1e-4),
            "winning_margin": pytest.approx(0.39416, abs=1e-4),
            "support_ratio": pytest.approx(1.0, abs=1e-6),
        }
    ]
    assert worker._last_speaker_candidate_debug[0]["active_audio_speaker_id"] == "SPEAKER_03"
    assert worker._last_speaker_candidate_debug[0]["active_audio_local_track_id"] is None
    assert worker._last_speaker_candidate_debug[0]["ambiguous"] is True
    assert worker._last_speaker_candidate_debug[0]["decision_source"] == "unknown"


def test_run_lrasd_binding_preserves_overlap_turn_unknown_with_clean_local_track_hint(monkeypatch, tmp_path: Path):
    worker, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "speaker_local": {
                "x_center": 78.0,
                "y_center": 102.0,
                "width": 76.0,
                "height": 152.0,
                "confidence": 0.90,
                "intensity": 210,
            },
            "listener_local": {
                "x_center": 222.0,
                "y_center": 104.0,
                "width": 72.0,
                "height": 148.0,
                "confidence": 0.88,
                "intensity": 80,
            },
        },
        score_by_track={
            "speaker_local": 0.82,
            "listener_local": 0.26,
        },
        words=[
            {"text": "hello", "start_time_ms": 0, "end_time_ms": 1000},
        ],
        audio_speaker_turns=[
            {
                "speaker_id": "SPEAKER_03",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "exclusive": False,
                "overlap": True,
            }
        ],
        audio_turn_bindings=[
            {
                "speaker_id": "SPEAKER_03",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "local_track_id": None,
                "clean_local_track_id": "listener_local",
                "ambiguous": True,
                "winning_score": 0.52,
                "winning_margin": 0.01,
                "support_ratio": 1.0,
                "max_visible_candidates": 3,
                "clean_support_ratio": 0.95,
                "clean_winning_score": 0.88,
                "clean_winning_margin": 0.21,
            }
        ],
    )

    assert words[0]["speaker_track_id"] is None
    assert words[0]["speaker_local_track_id"] is None
    assert bindings == []
    assert worker._last_speaker_candidate_debug[0]["active_audio_speaker_id"] == "SPEAKER_03"
    assert worker._last_speaker_candidate_debug[0]["active_audio_local_track_id"] == "listener_local"
    assert worker._last_speaker_candidate_debug[0]["ambiguous"] is True
    assert worker._last_speaker_candidate_debug[0]["decision_source"] == "unknown"


def test_run_lrasd_binding_preserves_off_screen_abstention_through_smoothing(monkeypatch, tmp_path: Path):
    _, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "speaker_local": {
                "x_center": 78.0,
                "y_center": 102.0,
                "width": 76.0,
                "height": 152.0,
                "confidence": 0.90,
                "intensity": 210,
            },
            "listener_local": {
                "x_center": 222.0,
                "y_center": 104.0,
                "width": 72.0,
                "height": 148.0,
                "confidence": 0.88,
                "intensity": 80,
            },
        },
        score_by_track={
            "speaker_local": 0.180,
            "listener_local": 0.172,
        },
        words=[
            {"text": "one", "start_time_ms": 0, "end_time_ms": 120},
            {"text": "two", "start_time_ms": 180, "end_time_ms": 300},
            {"text": "three", "start_time_ms": 360, "end_time_ms": 480},
        ],
        audio_speaker_turns=[
            {
                "speaker_id": "SPEAKER_10",
                "start_time_ms": 180,
                "end_time_ms": 300,
                "exclusive": True,
            }
        ],
        audio_turn_bindings=[
            {
                "speaker_id": "SPEAKER_10",
                "start_time_ms": 180,
                "end_time_ms": 300,
                "local_track_id": "offscreen_local",
                "ambiguous": False,
                "winning_score": 0.95,
                "winning_margin": 0.40,
                "support_ratio": 1.0,
            }
        ],
    )

    assert words[0]["speaker_local_track_id"] == "speaker_local"
    assert words[1]["speaker_track_id"] is None
    assert words[1]["speaker_local_track_id"] is None
    assert words[2]["speaker_local_track_id"] == "speaker_local"
    assert all("_speaker_binding_protected_unknown" not in word for word in words)
    assert bindings == [
        {
            "track_id": "speaker_local",
            "start_time_ms": 0,
            "end_time_ms": 480,
            "word_count": 2,
        }
    ]


def test_run_speaker_binding_fallback_preserves_off_screen_abstention_in_mixed_segment(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    words = [
        {"text": "visible", "start_time_ms": 0, "end_time_ms": 100},
        {"text": "offscreen", "start_time_ms": 200, "end_time_ms": 300},
    ]

    monkeypatch.setattr(worker, "_select_speaker_binding_mode", lambda *args, **kwargs: "lrasd")

    def fake_lrasd_binding(**kwargs):
        kwargs["words"][0]["speaker_track_id"] = "track-1"
        kwargs["words"][0]["speaker_tag"] = "track-1"
        kwargs["words"][0]["speaker_local_track_id"] = "track-1-local"
        kwargs["words"][0]["speaker_local_tag"] = "track-1-local"
        kwargs["words"][1]["speaker_track_id"] = None
        kwargs["words"][1]["speaker_tag"] = "unknown"
        kwargs["words"][1]["speaker_local_track_id"] = None
        kwargs["words"][1]["speaker_local_tag"] = "unknown"
        kwargs["words"][1]["_speaker_binding_protected_unknown"] = True
        return None

    def fake_heuristic_binding(*args, **kwargs):
        for word in words:
            word["speaker_track_id"] = "track-1"
            word["speaker_tag"] = "track-1"
            word["speaker_local_track_id"] = "track-1-local"
            word["speaker_local_tag"] = "track-1-local"
        return [
            {"track_id": "track-1", "start_time_ms": 0, "end_time_ms": 300, "word_count": 2}
        ]

    monkeypatch.setattr(worker, "_run_lrasd_binding", fake_lrasd_binding)
    monkeypatch.setattr(worker, "_run_speaker_binding_heuristic", fake_heuristic_binding)

    bindings = worker._run_speaker_binding(
        video_path="video.mp4",
        audio_wav_path="audio.wav",
        tracks=[{"track_id": "track-1"}],
        words=words,
        analysis_context={},
    )

    assert words[0]["speaker_track_id"] == "track-1"
    assert words[1]["speaker_track_id"] is None
    assert words[1]["speaker_local_track_id"] is None
    assert all("_speaker_binding_protected_unknown" not in word for word in words)
    assert bindings == [
        {"track_id": "track-1", "start_time_ms": 0, "end_time_ms": 100, "word_count": 1}
    ]


def test_speaker_remap_collision_metrics_counts_multiple_locals_per_global():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    metrics = worker._speaker_remap_collision_metrics(
        [
            {"speaker_track_id": "Global_Person_0", "speaker_local_track_id": "local-lawyer"},
            {"speaker_track_id": "Global_Person_0", "speaker_local_track_id": "local-andrew"},
            {"speaker_track_id": "Global_Person_0", "speaker_local_track_id": "local-lawyer"},
            {"speaker_track_id": "Global_Person_1", "speaker_local_track_id": "local-akaash"},
        ]
    )

    assert metrics["speaker_binding_globals_with_multiple_local_ids"] == 1
    assert metrics["speaker_binding_max_local_ids_per_global"] == 2


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
    monkeypatch.setattr(worker, "_run_audio_diarization", lambda audio_path: [])

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


def test_finalize_records_audio_diarization_fallback_metrics_when_pyannote_unavailable(monkeypatch):
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
    ]

    monkeypatch.setenv("CLYPT_AUDIO_DIARIZATION_ENABLE", "1")
    monkeypatch.setattr(worker, "_load_audio_diarization_pipeline", lambda: None)
    monkeypatch.setattr(worker, "_tracking_contract_pass_rate", lambda tracks: 1.0)
    monkeypatch.setattr(worker, "_validate_tracking_contract", lambda tracks: None)
    monkeypatch.setattr(worker, "_enforce_rollout_gates", lambda metrics: None)
    monkeypatch.setattr(
        worker,
        "_build_track_indexes",
        lambda tracks: ({0: tracks}, {"track-1": tracks}),
    )
    monkeypatch.setattr(worker, "_cluster_tracklets", lambda *args, **kwargs: tracks)
    monkeypatch.setattr(
        worker,
        "_build_visual_detection_ledgers",
        lambda **kwargs: ([], [], {}),
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

    assert result["phase_1_audio"]["audio_speaker_turns"] == []
    assert result["phase_1_visual"]["tracking_metrics"]["audio_diarization_enabled"] is True
    assert result["phase_1_visual"]["tracking_metrics"]["audio_diarization_fallback"] is True
    assert result["phase_1_visual"]["tracking_metrics"]["audio_diarization_status"] == "unavailable"
    assert result["phase_1_visual"]["tracking_metrics"]["audio_diarization_turn_count"] == 0


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
    monkeypatch.setattr(worker, "_run_audio_diarization", lambda audio_path: [])

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


def test_finalize_emits_visual_identities_from_clustered_features(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracks = [
        {
            "frame_idx": 0,
            "track_id": "local-1",
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

    monkeypatch.setattr(worker, "_tracking_contract_pass_rate", lambda tracks: 1.0)
    monkeypatch.setattr(worker, "_validate_tracking_contract", lambda tracks: None)
    monkeypatch.setattr(worker, "_enforce_rollout_gates", lambda metrics: None)
    monkeypatch.setattr(
        worker,
        "_build_track_indexes",
        lambda tracks: ({0: tracks}, {str(tracks[0]["track_id"]): tracks}),
    )
    monkeypatch.setattr(worker, "_build_visual_detection_ledgers", lambda *args, **kwargs: ([], [], {}))
    monkeypatch.setattr(worker, "_run_speaker_binding", lambda *args, **kwargs: [])
    monkeypatch.setattr(worker, "_run_audio_diarization", lambda audio_path: [])

    def fake_cluster_tracklets(video_path, tracks, track_to_dets=None, track_identity_features=None, face_track_features=None):
        worker._last_cluster_id_map = {"local-1": "Global_Person_0"}
        worker._last_track_identity_features_after_clustering = {
            "Global_Person_0": {
                "confidence": 0.88,
                "embedding": [0.1, 0.2, 0.3],
                "embedding_source": "face",
            }
        }
        for track in tracks:
            track["track_id"] = "Global_Person_0"
        worker._last_clustering_metrics = {}
        return tracks

    monkeypatch.setattr(worker, "_cluster_tracklets", fake_cluster_tracklets)

    result = worker._finalize_from_words_tracks(
        video_path="video.mp4",
        audio_path="audio.wav",
        youtube_url="https://youtube.com/watch?v=example",
        words=words,
        tracks=tracks,
        tracking_metrics={
            "track_identity_features": {
                "local-1": {
                    "confidence": 0.6,
                    "embedding": [0.1, 0.2, 0.3],
                    "embedding_source": "face",
                }
            },
            "face_track_features": {
                "face-1": {
                    "embedding": [0.1, 0.2, 0.3],
                    "dominant_track_id": "local-1",
                    "associated_track_counts": {"local-1": 3},
                }
            },
        },
    )

    assert result["phase_1_visual"]["visual_identities"] == [
        {
            "identity_id": "Global_Person_0",
            "confidence": 0.9,
            "track_ids": ("Global_Person_0",),
            "face_track_ids": ("face-1",),
            "person_track_ids": ("Global_Person_0",),
            "evidence_edge_ids": (),
            "metadata": {
                "track_count": 1,
                "person_track_count": 1,
                "face_track_count": 1,
                "source_counts": {
                    "tracks": 1,
                    "track_identity_features": 1,
                    "face_track_features": 1,
                },
                "sources": ("face_track_features", "track_identity_features", "tracks"),
            },
        }
    ]
    assert result["phase_1_visual"]["tracking_metrics"]["visual_identity_count"] == 1


def test_finalize_emits_audio_visual_mappings_from_clean_turn_bindings(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracks = [
        {
            "frame_idx": 0,
            "track_id": "local-1",
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
    words = [{"text": "hello", "start_time_ms": 0, "end_time_ms": 300}]

    monkeypatch.setattr(worker, "_tracking_contract_pass_rate", lambda tracks: 1.0)
    monkeypatch.setattr(worker, "_validate_tracking_contract", lambda tracks: None)
    monkeypatch.setattr(worker, "_enforce_rollout_gates", lambda metrics: None)
    monkeypatch.setattr(
        worker,
        "_build_track_indexes",
        lambda tracks: ({0: tracks}, {str(tracks[0]["track_id"]): tracks}),
    )
    monkeypatch.setattr(worker, "_build_visual_detection_ledgers", lambda *args, **kwargs: ([], [], {}))

    def fake_cluster_tracklets(video_path, tracks, track_to_dets=None, track_identity_features=None, face_track_features=None):
        worker._last_cluster_id_map = {"local-1": "Global_Person_0"}
        worker._last_track_identity_features_after_clustering = {
            "Global_Person_0": {
                "confidence": 0.88,
                "embedding": [0.1, 0.2, 0.3],
                "embedding_source": "face",
            }
        }
        for track in tracks:
            track["track_id"] = "Global_Person_0"
        worker._last_clustering_metrics = {}
        return tracks

    monkeypatch.setattr(worker, "_cluster_tracklets", fake_cluster_tracklets)
    monkeypatch.setattr(
        worker,
        "_run_audio_diarization",
        lambda audio_path: [
            {
                "speaker_id": "speaker-0",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "overlap": False,
            }
        ],
    )

    def fake_run_speaker_binding(*args, **kwargs):
        worker._last_audio_turn_bindings = [
            {
                "speaker_id": "speaker-0",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "local_track_id": "local-1",
                "ambiguous": False,
                "winning_score": 0.9,
                "winning_margin": 0.25,
                "support_ratio": 0.85,
            }
        ]
        return []

    monkeypatch.setattr(worker, "_run_speaker_binding", fake_run_speaker_binding)

    result = worker._finalize_from_words_tracks(
        video_path="video.mp4",
        audio_path="audio.wav",
        youtube_url="https://youtube.com/watch?v=example",
        words=words,
        tracks=tracks,
        tracking_metrics={},
    )

    assert result["phase_1_audio"]["audio_visual_mappings"] == [
        {
            "audio_speaker_id": "speaker-0",
            "matched_visual_identity_id": "Global_Person_0",
            "confidence": 1.0,
            "candidate_visual_identity_ids": ("Global_Person_0",),
            "evidence_edges": (
                {
                    "audio_speaker_id": "speaker-0",
                    "visual_identity_id": "Global_Person_0",
                    "confidence": 0.9,
                    "support_track_ids": ("Global_Person_0",),
                    "evidence_kind": "clean_span",
                    "metadata": {"start_time_ms": 0, "end_time_ms": 1000},
                },
            ),
            "supporting_track_ids": ("Global_Person_0",),
            "mapping_strategy": "clean-span-aggregation",
            "ambiguous": False,
            "metadata": {
                "candidate_stats": (
                    {
                        "visual_identity_id": "Global_Person_0",
                        "support_count": 1,
                        "confidence_sum": 0.9,
                        "average_confidence": 0.9,
                    },
                ),
                "clean_evidence_count": 1,
                "ignored_evidence_count": 0,
                "top_score_margin": 0.9,
            },
        }
    ]
    assert result["phase_1_visual"]["tracking_metrics"]["audio_visual_mapping_count"] == 1


def test_finalize_emits_mapping_first_span_assignments(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracks = [
        {
            "frame_idx": 0,
            "track_id": "local-1",
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
    words = [{"text": "hello", "start_time_ms": 0, "end_time_ms": 300}]

    monkeypatch.setattr(worker, "_tracking_contract_pass_rate", lambda tracks: 1.0)
    monkeypatch.setattr(worker, "_validate_tracking_contract", lambda tracks: None)
    monkeypatch.setattr(worker, "_enforce_rollout_gates", lambda metrics: None)
    monkeypatch.setattr(
        worker,
        "_build_track_indexes",
        lambda tracks: ({0: tracks}, {str(tracks[0]["track_id"]): tracks}),
    )
    monkeypatch.setattr(worker, "_build_visual_detection_ledgers", lambda *args, **kwargs: ([], [], {}))

    def fake_cluster_tracklets(video_path, tracks, track_to_dets=None, track_identity_features=None, face_track_features=None):
        worker._last_cluster_id_map = {"local-1": "Global_Person_0"}
        worker._last_track_identity_features_after_clustering = {
            "Global_Person_0": {
                "confidence": 0.88,
                "embedding": [0.1, 0.2, 0.3],
                "embedding_source": "face",
            }
        }
        for track in tracks:
            track["track_id"] = "Global_Person_0"
        worker._last_clustering_metrics = {}
        return tracks

    monkeypatch.setattr(worker, "_cluster_tracklets", fake_cluster_tracklets)
    monkeypatch.setattr(
        worker,
        "_run_audio_diarization",
        lambda audio_path: [
            {
                "speaker_id": "speaker-0",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "overlap": False,
            }
        ],
    )

    def fake_run_speaker_binding(*args, **kwargs):
        worker._last_audio_turn_bindings = [
            {
                "speaker_id": "speaker-0",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "local_track_id": "local-1",
                "ambiguous": False,
                "winning_score": 0.9,
                "winning_margin": 0.25,
                "support_ratio": 0.85,
            }
        ]
        return []

    monkeypatch.setattr(worker, "_run_speaker_binding", fake_run_speaker_binding)

    result = worker._finalize_from_words_tracks(
        video_path="video.mp4",
        audio_path="audio.wav",
        youtube_url="https://youtube.com/watch?v=example",
        words=words,
        tracks=tracks,
        tracking_metrics={},
    )

    assert result["phase_1_audio"]["span_assignments"] == [
        {
            "start_time_ms": 0,
            "end_time_ms": 1080,
            "audio_speaker_ids": ("speaker-0",),
            "assigned_visual_identity_ids": ("Global_Person_0",),
            "dominant_visual_identity_id": "Global_Person_0",
            "offscreen_audio_speaker_ids": (),
            "unresolved_audio_speaker_ids": (),
            "require_hard_disambiguation": False,
            "decision_source": "mapping",
        }
    ]
    assert result["phase_1_visual"]["tracking_metrics"]["span_assignment_count"] == 1


def test_finalize_projects_hard_span_visual_disambiguation_to_words(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracks = [
        {
            "frame_idx": 0,
            "track_id": "local-1",
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
            "frame_idx": 0,
            "track_id": "local-2",
            "x1": 210.0,
            "y1": 20.0,
            "x2": 310.0,
            "y2": 220.0,
            "x_center": 260.0,
            "y_center": 120.0,
            "width": 100.0,
            "height": 200.0,
            "confidence": 0.88,
        },
    ]
    words = [{"text": "hello", "start_time_ms": 0, "end_time_ms": 300}]

    monkeypatch.setattr(worker, "_tracking_contract_pass_rate", lambda tracks: 1.0)
    monkeypatch.setattr(worker, "_validate_tracking_contract", lambda tracks: None)
    monkeypatch.setattr(worker, "_enforce_rollout_gates", lambda metrics: None)
    monkeypatch.setattr(
        worker,
        "_build_track_indexes",
        lambda tracks: (
            {0: tracks},
            {str(track["track_id"]): [track] for track in tracks},
        ),
    )
    monkeypatch.setattr(worker, "_build_visual_detection_ledgers", lambda *args, **kwargs: ([], [], {}))

    def fake_cluster_tracklets(video_path, tracks, track_to_dets=None, track_identity_features=None, face_track_features=None):
        worker._last_cluster_id_map = {
            "local-1": "Global_Person_0",
            "local-2": "Global_Person_1",
        }
        worker._last_track_identity_features_after_clustering = {}
        for track in tracks:
            track["track_id"] = worker._last_cluster_id_map[track["track_id"]]
        worker._last_clustering_metrics = {}
        return tracks

    monkeypatch.setattr(worker, "_cluster_tracklets", fake_cluster_tracklets)
    monkeypatch.setattr(
        worker,
        "_run_audio_diarization",
        lambda audio_path: [
            {
                "speaker_id": "speaker-0",
                "start_time_ms": 0,
                "end_time_ms": 300,
                "overlap": False,
            }
        ],
    )

    def fake_run_speaker_binding(*args, **kwargs):
        worker._last_audio_turn_bindings = []
        return []

    monkeypatch.setattr(worker, "_run_speaker_binding", fake_run_speaker_binding)
    monkeypatch.setattr(worker, "_build_audio_visual_mappings", lambda **kwargs: [])
    monkeypatch.setattr(
        worker,
        "_build_active_speakers_local",
        lambda **kwargs: [
            {
                "start_time_ms": 0,
                "end_time_ms": 300,
                "audio_speaker_ids": ["speaker-0"],
                "visible_local_track_ids": ["local-1", "local-2"],
                "visible_track_ids": ["Global_Person_0", "Global_Person_1"],
                "offscreen_audio_speaker_ids": [],
                "overlap": False,
                "decision_source": "visual_candidates",
            }
        ],
    )
    monkeypatch.setattr(
        worker,
        "_enrich_spans_with_hard_candidates",
        lambda **kwargs: [
            {
                "start_time_ms": 0,
                "end_time_ms": 300,
                "audio_speaker_ids": ["speaker-0"],
                "visible_local_track_ids": ["local-1", "local-2"],
                "visible_track_ids": ["Global_Person_0", "Global_Person_1"],
                "offscreen_audio_speaker_ids": [],
                "overlap": False,
                "decision_source": "visual_candidates",
                "hard_span_candidates": [
                    {
                        "visual_identity_id": "Global_Person_0",
                        "mouth_motion_score": 0.92,
                        "pose_visibility_score": 0.78,
                        "face_visibility_score": 0.83,
                        "mapping_confidence": 0.2,
                    },
                    {
                        "visual_identity_id": "Global_Person_1",
                        "mouth_motion_score": 0.31,
                        "pose_visibility_score": 0.79,
                        "face_visibility_score": 0.8,
                        "mapping_confidence": 0.2,
                    },
                ],
            }
        ],
    )
    monkeypatch.setattr(worker, "_run_overlap_follow_postpass", lambda **kwargs: [])

    result = worker._finalize_from_words_tracks(
        video_path="video.mp4",
        audio_path="audio.wav",
        youtube_url="https://youtube.com/watch?v=example",
        words=words,
        tracks=tracks,
        tracking_metrics={},
    )

    assert result["phase_1_audio"]["span_assignments"] == [
        {
            "start_time_ms": 0,
            "end_time_ms": 300,
            "audio_speaker_ids": ("speaker-0",),
            "assigned_visual_identity_ids": ("Global_Person_0",),
            "dominant_visual_identity_id": "Global_Person_0",
            "offscreen_audio_speaker_ids": (),
            "unresolved_audio_speaker_ids": (),
            "require_hard_disambiguation": False,
            "decision_source": "hard_visual_disambiguation",
        }
    ]
    assert result["phase_1_audio"]["words"][0]["speaker_track_id"] == "Global_Person_0"
    assert result["phase_1_audio"]["words"][0]["speaker_track_ids"] == ["Global_Person_0"]
    assert result["phase_1_audio"]["words"][0]["speaker_assignment_source"] == "hard_visual_disambiguation"
    assert "hard_span_candidates" not in result["phase_1_audio"]["active_speakers_local"][0]


def test_finalize_projects_overlap_visible_and_offscreen_truth_without_fake_single_winner(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracks = [
        {
            "frame_idx": 0,
            "track_id": "local-1",
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
    words = [{"text": "hello", "start_time_ms": 0, "end_time_ms": 300}]

    monkeypatch.setattr(worker, "_tracking_contract_pass_rate", lambda tracks: 1.0)
    monkeypatch.setattr(worker, "_validate_tracking_contract", lambda tracks: None)
    monkeypatch.setattr(worker, "_enforce_rollout_gates", lambda metrics: None)
    monkeypatch.setattr(
        worker,
        "_build_track_indexes",
        lambda tracks: (
            {0: tracks},
            {str(track["track_id"]): [track] for track in tracks},
        ),
    )
    monkeypatch.setattr(worker, "_build_visual_detection_ledgers", lambda *args, **kwargs: ([], [], {}))

    def fake_cluster_tracklets(video_path, tracks, track_to_dets=None, track_identity_features=None, face_track_features=None):
        worker._last_cluster_id_map = {"local-1": "Global_Person_0"}
        worker._last_track_identity_features_after_clustering = {}
        for track in tracks:
            track["track_id"] = "Global_Person_0"
        worker._last_clustering_metrics = {}
        return tracks

    monkeypatch.setattr(worker, "_cluster_tracklets", fake_cluster_tracklets)
    monkeypatch.setattr(
        worker,
        "_run_audio_diarization",
        lambda audio_path: [
            {"speaker_id": "speaker-0", "start_time_ms": 0, "end_time_ms": 300, "overlap": True},
            {"speaker_id": "speaker-1", "start_time_ms": 0, "end_time_ms": 300, "overlap": True},
        ],
    )
    monkeypatch.setattr(worker, "_run_speaker_binding", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        worker,
        "_build_audio_visual_mappings",
        lambda **kwargs: [
            {
                "audio_speaker_id": "speaker-0",
                "matched_visual_identity_id": "Global_Person_0",
                "confidence": 0.91,
                "candidate_visual_identity_ids": ("Global_Person_0",),
                "ambiguous": False,
            },
            {
                "audio_speaker_id": "speaker-1",
                "matched_visual_identity_id": "Global_Person_9",
                "confidence": 0.88,
                "candidate_visual_identity_ids": ("Global_Person_9",),
                "ambiguous": False,
            },
        ],
    )
    monkeypatch.setattr(
        worker,
        "_build_active_speakers_local",
        lambda **kwargs: [
            {
                "start_time_ms": 0,
                "end_time_ms": 300,
                "audio_speaker_ids": ["speaker-0", "speaker-1"],
                "visible_local_track_ids": ["local-1"],
                "visible_track_ids": ["Global_Person_0"],
                "offscreen_audio_speaker_ids": [],
                "overlap": True,
                "decision_source": "pyannote_overlap",
            }
        ],
    )
    monkeypatch.setattr(worker, "_enrich_spans_with_hard_candidates", lambda spans, **kwargs: spans)
    monkeypatch.setattr(worker, "_run_overlap_follow_postpass", lambda **kwargs: [])

    result = worker._finalize_from_words_tracks(
        video_path="video.mp4",
        audio_path="audio.wav",
        youtube_url="https://youtube.com/watch?v=example",
        words=words,
        tracks=tracks,
        tracking_metrics={},
    )

    assert result["phase_1_audio"]["span_assignments"] == [
        {
            "start_time_ms": 0,
            "end_time_ms": 300,
            "audio_speaker_ids": ("speaker-0", "speaker-1"),
            "assigned_visual_identity_ids": ("Global_Person_0",),
            "dominant_visual_identity_id": None,
            "offscreen_audio_speaker_ids": ("speaker-1",),
            "unresolved_audio_speaker_ids": (),
            "require_hard_disambiguation": False,
            "decision_source": "overlap_mapping",
        }
    ]
    assert result["phase_1_audio"]["words"][0]["speaker_track_id"] is None
    assert result["phase_1_audio"]["words"][0]["speaker_track_ids"] == ["Global_Person_0"]
    assert result["phase_1_audio"]["words"][0]["offscreen_audio_speaker_ids"] == ["speaker-1"]
    assert result["phase_1_audio"]["words"][0]["speaker_assignment_source"] == "overlap_mapping"


def test_build_speaker_follow_bindings_absorbs_brief_midstream_blip():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    bindings = [
        {"track_id": "A", "start_time_ms": 0, "end_time_ms": 3200, "word_count": 10},
        {"track_id": "B", "start_time_ms": 3200, "end_time_ms": 3550, "word_count": 1},
        {"track_id": "A", "start_time_ms": 3550, "end_time_ms": 7200, "word_count": 11},
    ]

    follow = worker._build_speaker_follow_bindings(bindings)

    assert follow == [
        {"track_id": "A", "start_time_ms": 0, "end_time_ms": 7200, "word_count": 22}
    ]


def test_build_speaker_follow_bindings_keeps_sustained_turn_change():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    bindings = [
        {"track_id": "A", "start_time_ms": 0, "end_time_ms": 3000, "word_count": 10},
        {"track_id": "B", "start_time_ms": 3000, "end_time_ms": 5200, "word_count": 8},
        {"track_id": "A", "start_time_ms": 5200, "end_time_ms": 8400, "word_count": 9},
    ]

    follow = worker._build_speaker_follow_bindings(bindings)

    assert [segment["track_id"] for segment in follow] == ["A", "B", "A"]
    assert follow[1]["start_time_ms"] == 3000
    assert follow[1]["end_time_ms"] == 5200


def test_finalize_emits_speaker_follow_bindings(monkeypatch):
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

    monkeypatch.setattr(worker, "_tracking_contract_pass_rate", lambda tracks: 1.0)
    monkeypatch.setattr(worker, "_validate_tracking_contract", lambda tracks: None)
    monkeypatch.setattr(worker, "_enforce_rollout_gates", lambda metrics: None)
    monkeypatch.setattr(
        worker,
        "_build_track_indexes",
        lambda tracks: ({0: tracks}, {"track-1": tracks}),
    )
    monkeypatch.setattr(worker, "_cluster_tracklets", lambda *args, **kwargs: tracks)
    monkeypatch.setattr(worker, "_build_visual_detection_ledgers", lambda *args, **kwargs: ([], [], {}))
    monkeypatch.setattr(
        worker,
        "_run_speaker_binding",
        lambda *args, **kwargs: [
            {"track_id": "track-1", "start_time_ms": 0, "end_time_ms": 3000, "word_count": 10},
            {"track_id": "track-2", "start_time_ms": 3000, "end_time_ms": 3300, "word_count": 1},
            {"track_id": "track-1", "start_time_ms": 3300, "end_time_ms": 6000, "word_count": 11},
        ],
    )
    monkeypatch.setattr(worker, "_run_audio_diarization", lambda audio_path: [])

    result = worker._finalize_from_words_tracks(
        video_path="video.mp4",
        audio_path="audio.wav",
        youtube_url="https://youtube.com/watch?v=example",
        words=words,
        tracks=tracks,
        tracking_metrics={},
    )

    assert result["phase_1_audio"]["speaker_bindings"][1]["track_id"] == "track-2"
    assert result["phase_1_audio"]["speaker_follow_bindings"] == [
        {"track_id": "track-1", "start_time_ms": 0, "end_time_ms": 6000, "word_count": 22}
    ]


def test_finalize_runs_speaker_binding_on_precluster_tracks_and_maps_to_globals(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracks = [
        {
            "frame_idx": 0,
            "track_id": "local-1",
            "class_id": 0,
            "label": "person",
            "geometry_type": "aabb",
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

    monkeypatch.setattr(worker, "_tracking_contract_pass_rate", lambda tracks: 1.0)
    monkeypatch.setattr(worker, "_validate_tracking_contract", lambda tracks: None)
    monkeypatch.setattr(worker, "_enforce_rollout_gates", lambda metrics: None)
    monkeypatch.setattr(worker, "_build_visual_detection_ledgers", lambda *args, **kwargs: ([], [], {}))

    def _fake_cluster(video_path, tracks, **kwargs):
        worker._last_cluster_id_map = {"local-1": "Global_Person_0"}
        worker._last_track_identity_features_after_clustering = None
        for track in tracks:
            track["track_id"] = "Global_Person_0"
        return tracks

    monkeypatch.setattr(worker, "_cluster_tracklets", _fake_cluster)
    monkeypatch.setattr(worker, "_run_audio_diarization", lambda audio_path: [])

    def _fake_run_speaker_binding(
        video_path,
        audio_path,
        tracks,
        words,
        frame_to_dets=None,
        track_to_dets=None,
        track_identity_features=None,
        analysis_context=None,
        track_id_remap=None,
    ):
        assert {track["track_id"] for track in tracks} == {"local-1"}
        assert track_id_remap == {"local-1": "Global_Person_0"}
        for word in words:
            word["speaker_track_id"] = "Global_Person_0"
            word["speaker_tag"] = "Global_Person_0"
        return [
            {
                "track_id": "Global_Person_0",
                "start_time_ms": 0,
                "end_time_ms": 100,
                "word_count": 1,
            }
        ]

    monkeypatch.setattr(worker, "_run_speaker_binding", _fake_run_speaker_binding)

    result = worker._finalize_from_words_tracks(
        video_path="video.mp4",
        audio_path="audio.wav",
        youtube_url="https://youtube.com/watch?v=example",
        words=words,
        tracks=tracks,
        tracking_metrics={},
    )

    assert result["phase_1_audio"]["speaker_bindings"] == [
        {"track_id": "Global_Person_0", "start_time_ms": 0, "end_time_ms": 100, "word_count": 1}
    ]
    assert result["phase_1_audio"]["words"][0]["speaker_track_id"] == "Global_Person_0"


def test_finalize_emits_local_speaker_bindings_when_experiment_enabled(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracks = [
        {
            "frame_idx": 0,
            "track_id": "local-1",
            "class_id": 0,
            "label": "person",
            "geometry_type": "aabb",
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

    monkeypatch.setenv("CLYPT_EXPERIMENT_LOCAL_CLIP_BINDINGS", "1")
    monkeypatch.setattr(worker, "_tracking_contract_pass_rate", lambda tracks: 1.0)
    monkeypatch.setattr(worker, "_validate_tracking_contract", lambda tracks: None)
    monkeypatch.setattr(worker, "_enforce_rollout_gates", lambda metrics: None)
    monkeypatch.setattr(worker, "_build_visual_detection_ledgers", lambda *args, **kwargs: ([], [], {}))

    def _fake_cluster(video_path, tracks, **kwargs):
        worker._last_cluster_id_map = {"local-1": "Global_Person_0"}
        worker._last_track_identity_features_after_clustering = None
        for track in tracks:
            track["track_id"] = "Global_Person_0"
        return tracks

    monkeypatch.setattr(worker, "_cluster_tracklets", _fake_cluster)
    monkeypatch.setattr(worker, "_run_audio_diarization", lambda audio_path: [])

    def _fake_run_speaker_binding(
        video_path,
        audio_path,
        tracks,
        words,
        frame_to_dets=None,
        track_to_dets=None,
        track_identity_features=None,
        analysis_context=None,
        track_id_remap=None,
    ):
        assert {track["track_id"] for track in tracks} == {"local-1"}
        assert track_id_remap == {"local-1": "Global_Person_0"}
        for word in words:
            word["speaker_track_id"] = "Global_Person_0"
            word["speaker_tag"] = "Global_Person_0"
        return [
            {
                "track_id": "Global_Person_0",
                "start_time_ms": 0,
                "end_time_ms": 100,
                "word_count": 1,
            }
        ]

    monkeypatch.setattr(worker, "_run_speaker_binding", _fake_run_speaker_binding)

    result = worker._finalize_from_words_tracks(
        video_path="video.mp4",
        audio_path="audio.wav",
        youtube_url="https://youtube.com/watch?v=example",
        words=words,
        tracks=tracks,
        tracking_metrics={},
    )

    assert result["phase_1_audio"]["speaker_bindings_local"] == [
        {"track_id": "local-1", "start_time_ms": 0, "end_time_ms": 100, "word_count": 1}
    ]
    assert result["phase_1_audio"]["speaker_follow_bindings_local"] == [
        {"track_id": "local-1", "start_time_ms": 0, "end_time_ms": 100, "word_count": 1}
    ]
    assert result["phase_1_visual"]["tracks_local"][0]["track_id"] == "local-1"


def test_finalize_persists_audio_speaker_local_track_map_when_experiment_enabled(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracks = [
        {
            "frame_idx": 0,
            "track_id": "local-1",
            "class_id": 0,
            "label": "person",
            "geometry_type": "aabb",
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
        {
            "text": "hello",
            "start_time_ms": 0,
            "end_time_ms": 100,
            "speaker_track_id": "Global_Person_0",
            "speaker_tag": "Global_Person_0",
            "speaker_local_track_id": "local-1",
            "speaker_local_tag": "local-1",
        }
    ]

    monkeypatch.setenv("CLYPT_EXPERIMENT_LOCAL_CLIP_BINDINGS", "1")
    monkeypatch.setattr(worker, "_tracking_contract_pass_rate", lambda tracks: 1.0)
    monkeypatch.setattr(worker, "_validate_tracking_contract", lambda tracks: None)
    monkeypatch.setattr(worker, "_enforce_rollout_gates", lambda metrics: None)
    monkeypatch.setattr(
        worker,
        "_build_track_indexes",
        lambda tracks: ({0: tracks}, {"local-1": tracks}),
    )
    monkeypatch.setattr(worker, "_build_visual_detection_ledgers", lambda *args, **kwargs: ([], [], {}))
    monkeypatch.setattr(worker, "_run_audio_diarization", lambda audio_path: [])

    def _fake_cluster(video_path, tracks, **kwargs):
        worker._last_cluster_id_map = {"local-1": "Global_Person_0"}
        worker._last_track_identity_features_after_clustering = None
        return [
            {
                **track,
                "track_id": "Global_Person_0",
            }
            for track in tracks
        ]

    def _fake_run_speaker_binding(*args, **kwargs):
        worker._last_audio_turn_bindings = [
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 900,
                "local_track_id": "local-1",
                "ambiguous": False,
                "winning_score": 0.93,
                "winning_margin": 0.28,
                "support_ratio": 0.96,
            },
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 1100,
                "end_time_ms": 2100,
                "local_track_id": "local-1",
                "ambiguous": False,
                "winning_score": 0.89,
                "winning_margin": 0.24,
                "support_ratio": 0.91,
            },
        ]
        return [
            {
                "track_id": "Global_Person_0",
                "start_time_ms": 0,
                "end_time_ms": 100,
                "word_count": 1,
            }
        ]

    monkeypatch.setattr(worker, "_cluster_tracklets", _fake_cluster)
    monkeypatch.setattr(worker, "_run_speaker_binding", _fake_run_speaker_binding)

    result = worker._finalize_from_words_tracks(
        video_path="video.mp4",
        audio_path="audio.wav",
        youtube_url="https://youtube.com/watch?v=example",
        words=words,
        tracks=tracks,
        tracking_metrics={},
    )

    assert result["phase_1_audio"]["audio_speaker_local_track_map"] == [
        {
            "speaker_id": "SPEAKER_00",
            "local_track_id": "local-1",
            "support_segments": 2,
            "support_ms": 1900,
            "confidence": pytest.approx(0.87, abs=0.05),
        }
    ]


def test_build_active_speakers_local_splits_overlap_and_marks_offscreen():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    spans = worker._build_active_speakers_local(
        audio_speaker_turns=[
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "exclusive": True,
            },
            {
                "speaker_id": "SPEAKER_01",
                "start_time_ms": 500,
                "end_time_ms": 1500,
                "exclusive": False,
                "overlap": True,
            },
        ],
        audio_turn_bindings=[
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "local_track_id": "local-1",
                "ambiguous": False,
                "winning_score": 0.92,
            },
            {
                "speaker_id": "SPEAKER_01",
                "start_time_ms": 500,
                "end_time_ms": 1500,
                "local_track_id": None,
                "ambiguous": True,
                "winning_score": 0.41,
            },
        ],
        local_to_global_track_id={"local-1": "Global_Person_0"},
    )

    assert spans == [
        {
            "start_time_ms": 0,
            "end_time_ms": 420,
            "audio_speaker_ids": ["SPEAKER_00"],
            "visible_local_track_ids": ["local-1"],
            "visible_track_ids": ["Global_Person_0"],
            "offscreen_audio_speaker_ids": [],
            "overlap": False,
            "confidence": pytest.approx(0.92, abs=1e-3),
            "decision_source": "turn_binding",
        },
        {
            "start_time_ms": 420,
            "end_time_ms": 1080,
            "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
            "visible_local_track_ids": ["local-1"],
            "visible_track_ids": ["Global_Person_0"],
            "offscreen_audio_speaker_ids": ["SPEAKER_01"],
            "overlap": True,
            "confidence": pytest.approx(0.665, abs=1e-3),
            "decision_source": "turn_binding",
        },
        {
            "start_time_ms": 1080,
            "end_time_ms": 1580,
            "audio_speaker_ids": ["SPEAKER_01"],
            "visible_local_track_ids": [],
            "visible_track_ids": [],
            "offscreen_audio_speaker_ids": ["SPEAKER_01"],
            "overlap": True,
            "confidence": pytest.approx(0.41, abs=1e-3),
            "decision_source": "audio_only",
        },
    ]


def test_normalize_audio_speaker_turns_merges_tiny_same_speaker_gap_for_binding_context():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    normalized_turns = worker._normalize_audio_speaker_turns_for_binding_context(
        [
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 100,
                "end_time_ms": 500,
                "exclusive": True,
            },
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 560,
                "end_time_ms": 900,
                "exclusive": True,
            },
        ]
    )

    assert normalized_turns == [
        {
            "speaker_id": "SPEAKER_00",
            "start_time_ms": 20,
            "end_time_ms": 980,
            "exclusive": True,
            "source_start_time_ms": 100,
            "source_end_time_ms": 900,
            "source_turn_count": 2,
        }
    ]


def test_build_active_speakers_local_pads_turn_boundaries_for_adjacent_context():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    spans = worker._build_active_speakers_local(
        audio_speaker_turns=[
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "exclusive": True,
            },
            {
                "speaker_id": "SPEAKER_01",
                "start_time_ms": 1000,
                "end_time_ms": 2000,
                "exclusive": True,
            },
        ],
        audio_turn_bindings=[
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "local_track_id": "local-0",
                "ambiguous": False,
                "winning_score": 0.91,
            },
            {
                "speaker_id": "SPEAKER_01",
                "start_time_ms": 1000,
                "end_time_ms": 2000,
                "local_track_id": "local-1",
                "ambiguous": False,
                "winning_score": 0.89,
            },
        ],
        local_to_global_track_id={
            "local-0": "Global_Person_0",
            "local-1": "Global_Person_1",
        },
    )

    assert any(
        span["audio_speaker_ids"] == ["SPEAKER_00", "SPEAKER_01"]
        and span["visible_local_track_ids"] == ["local-0", "local-1"]
        and span["overlap"] is True
        and span["start_time_ms"] < 1000 < span["end_time_ms"]
        for span in spans
    )


def test_run_lrasd_binding_debug_surfaces_multiple_active_speakers_near_turn_boundary(monkeypatch, tmp_path: Path):
    worker, words, bindings = _run_fake_lrasd_binding_case(
        monkeypatch,
        tmp_path,
        track_specs={
            "speaker_local": {
                "x_center": 78.0,
                "y_center": 102.0,
                "width": 72.0,
                "height": 148.0,
                "confidence": 0.88,
                "intensity": 210,
            },
            "listener_local": {
                "x_center": 222.0,
                "y_center": 104.0,
                "width": 70.0,
                "height": 146.0,
                "confidence": 0.87,
                "intensity": 80,
            },
        },
        score_by_track={
            "speaker_local": 0.175,
            "listener_local": 0.173,
        },
        words=[
            {"text": "handoff", "start_time_ms": 960, "end_time_ms": 1040},
        ],
        audio_speaker_turns=[
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "exclusive": True,
            },
            {
                "speaker_id": "SPEAKER_01",
                "start_time_ms": 1000,
                "end_time_ms": 2000,
                "exclusive": True,
            },
        ],
        audio_turn_bindings=[
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "local_track_id": "speaker_local",
                "ambiguous": False,
                "winning_score": 0.92,
                "winning_margin": 0.30,
                "support_ratio": 1.0,
            },
            {
                "speaker_id": "SPEAKER_01",
                "start_time_ms": 1000,
                "end_time_ms": 2000,
                "local_track_id": "listener_local",
                "ambiguous": False,
                "winning_score": 0.91,
                "winning_margin": 0.28,
                "support_ratio": 1.0,
            },
        ],
    )

    assert bindings == [{"track_id": "speaker_local", "start_time_ms": 960, "end_time_ms": 1040, "word_count": 1}]
    assert worker._last_speaker_candidate_debug[0]["active_audio_speaker_ids"] == ["SPEAKER_00", "SPEAKER_01"]
    assert worker._last_speaker_candidate_debug[0]["active_audio_local_track_ids"] == ["speaker_local", "listener_local"]


def test_finalize_persists_overlap_artifacts_when_experiment_enabled(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracks = [
        {
            "frame_idx": 0,
            "track_id": "local-1",
            "class_id": 0,
            "label": "person",
            "geometry_type": "aabb",
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
        {
            "text": "hello",
            "start_time_ms": 0,
            "end_time_ms": 100,
            "speaker_track_id": "Global_Person_0",
            "speaker_tag": "Global_Person_0",
            "speaker_local_track_id": "local-1",
            "speaker_local_tag": "local-1",
        }
    ]
    audio_turns = [
        {
            "speaker_id": "SPEAKER_00",
            "start_time_ms": 0,
            "end_time_ms": 600,
            "exclusive": True,
        },
        {
            "speaker_id": "SPEAKER_01",
            "start_time_ms": 300,
            "end_time_ms": 800,
            "exclusive": False,
            "overlap": True,
        },
    ]
    overlap_decisions = [
        {
            "start_time_ms": 300,
            "end_time_ms": 600,
            "camera_target_local_track_id": "local-1",
            "camera_target_track_id": "Global_Person_0",
            "stay_wide": False,
            "visible_local_track_ids": ["local-1"],
            "offscreen_audio_speaker_ids": ["SPEAKER_01"],
            "decision_model": "gemini-3-flash-preview",
            "decision_source": "gemini",
            "confidence": 0.81,
        }
    ]

    monkeypatch.setenv("CLYPT_EXPERIMENT_LOCAL_CLIP_BINDINGS", "1")
    monkeypatch.setattr(worker, "_tracking_contract_pass_rate", lambda tracks: 1.0)
    monkeypatch.setattr(worker, "_validate_tracking_contract", lambda tracks: None)
    monkeypatch.setattr(worker, "_enforce_rollout_gates", lambda metrics: None)
    monkeypatch.setattr(worker, "_build_track_indexes", lambda tracks: ({0: tracks}, {"local-1": tracks}))
    monkeypatch.setattr(worker, "_build_visual_detection_ledgers", lambda *args, **kwargs: ([], [], {}))
    monkeypatch.setattr(worker, "_run_audio_diarization", lambda audio_path: audio_turns)
    monkeypatch.setattr(
        worker,
        "_run_overlap_follow_postpass",
        lambda **kwargs: overlap_decisions,
    )

    def _fake_cluster(video_path, tracks, **kwargs):
        worker._last_cluster_id_map = {"local-1": "Global_Person_0"}
        worker._last_track_identity_features_after_clustering = None
        return [{**track, "track_id": "Global_Person_0"} for track in tracks]

    def _fake_run_speaker_binding(*args, **kwargs):
        worker._last_audio_turn_bindings = [
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 600,
                "local_track_id": "local-1",
                "ambiguous": False,
                "winning_score": 0.93,
                "winning_margin": 0.28,
                "support_ratio": 0.96,
            },
            {
                "speaker_id": "SPEAKER_01",
                "start_time_ms": 300,
                "end_time_ms": 800,
                "local_track_id": None,
                "ambiguous": True,
                "winning_score": 0.41,
                "winning_margin": 0.02,
                "support_ratio": 0.32,
            },
        ]
        return [
            {
                "track_id": "Global_Person_0",
                "start_time_ms": 0,
                "end_time_ms": 100,
                "word_count": 1,
            }
        ]

    monkeypatch.setattr(worker, "_cluster_tracklets", _fake_cluster)
    monkeypatch.setattr(worker, "_run_speaker_binding", _fake_run_speaker_binding)

    result = worker._finalize_from_words_tracks(
        video_path="video.mp4",
        audio_path="audio.wav",
        youtube_url="https://youtube.com/watch?v=example",
        words=words,
        tracks=tracks,
        tracking_metrics={},
    )

    assert result["phase_1_audio"]["active_speakers_local"] == [
        {
            "start_time_ms": 0,
            "end_time_ms": 220,
            "audio_speaker_ids": ["SPEAKER_00"],
            "visible_local_track_ids": ["local-1"],
            "visible_track_ids": ["Global_Person_0"],
            "offscreen_audio_speaker_ids": [],
            "overlap": False,
            "confidence": pytest.approx(0.93, abs=1e-3),
            "decision_source": "turn_binding",
        },
        {
            "start_time_ms": 220,
            "end_time_ms": 680,
            "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
            "visible_local_track_ids": ["local-1"],
            "visible_track_ids": ["Global_Person_0"],
            "offscreen_audio_speaker_ids": ["SPEAKER_01"],
            "overlap": True,
            "confidence": pytest.approx(0.67, abs=0.01),
            "decision_source": "turn_binding",
        },
        {
            "start_time_ms": 680,
            "end_time_ms": 880,
            "audio_speaker_ids": ["SPEAKER_01"],
            "visible_local_track_ids": [],
            "visible_track_ids": [],
            "offscreen_audio_speaker_ids": ["SPEAKER_01"],
            "overlap": True,
            "confidence": pytest.approx(0.41, abs=1e-3),
            "decision_source": "audio_only",
        },
    ]
    assert result["phase_1_audio"]["overlap_follow_decisions"] == overlap_decisions


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


def test_clusters_conflict_by_visibility_ignores_single_frame_overlapping_duplicate():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracklets = {
        "dup_a": [
            {
                "frame_idx": 10,
                "x_center": 500.0,
                "y_center": 300.0,
                "width": 120.0,
                "height": 240.0,
                "confidence": 0.95,
            }
        ],
        "dup_b": [
            {
                "frame_idx": 10,
                "x_center": 590.0,
                "y_center": 302.0,
                "width": 120.0,
                "height": 238.0,
                "confidence": 0.92,
            }
        ],
    }

    assert worker._clusters_conflict_by_visibility(tracklets, ["dup_a"], ["dup_b"]) is False


def test_clusters_conflict_by_visibility_ignores_repeated_overlapping_duplicates():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracklets = {
        "dup_a": [
            {"frame_idx": 10, "x_center": 500.0, "y_center": 300.0, "width": 120.0, "height": 240.0, "confidence": 0.95},
            {"frame_idx": 11, "x_center": 504.0, "y_center": 302.0, "width": 121.0, "height": 241.0, "confidence": 0.94},
        ],
        "dup_b": [
            {"frame_idx": 10, "x_center": 560.0, "y_center": 302.0, "width": 119.0, "height": 239.0, "confidence": 0.92},
            {"frame_idx": 11, "x_center": 566.0, "y_center": 303.0, "width": 120.0, "height": 240.0, "confidence": 0.91},
        ],
    }

    assert worker._clusters_conflict_by_visibility(tracklets, ["dup_a"], ["dup_b"]) is False


def test_clusters_conflict_by_visibility_can_be_disabled(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.setenv("CLYPT_CLUSTER_DISABLE_COVISIBILITY", "1")

    tracklets = {
        "left": [
            {"frame_idx": 10, "x_center": 120.0, "y_center": 300.0, "width": 90.0, "height": 220.0, "confidence": 0.94},
            {"frame_idx": 11, "x_center": 122.0, "y_center": 302.0, "width": 92.0, "height": 222.0, "confidence": 0.93},
        ],
        "right": [
            {"frame_idx": 10, "x_center": 1620.0, "y_center": 300.0, "width": 92.0, "height": 222.0, "confidence": 0.95},
            {"frame_idx": 11, "x_center": 1622.0, "y_center": 302.0, "width": 94.0, "height": 224.0, "confidence": 0.94},
        ],
    }

    assert worker._clusters_conflict_by_visibility(tracklets, ["left"], ["right"]) is False


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


def test_repair_covisible_cluster_merges_keeps_single_frame_duplicate_together():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracklets = {
        "same_person_a": [
            {"frame_idx": 10, "x_center": 500.0, "y_center": 300.0, "width": 120.0, "height": 240.0, "confidence": 0.95},
            {"frame_idx": 11, "x_center": 506.0, "y_center": 302.0, "width": 122.0, "height": 242.0, "confidence": 0.94},
        ],
        "same_person_b": [
            {"frame_idx": 10, "x_center": 590.0, "y_center": 302.0, "width": 120.0, "height": 238.0, "confidence": 0.92},
        ],
    }
    label_by_tid = {"same_person_a": 0, "same_person_b": 0}

    repaired, metrics = worker._repair_covisible_cluster_merges(tracklets, label_by_tid)

    assert repaired["same_person_a"] == repaired["same_person_b"]
    assert metrics["repaired_cluster_count"] == 0


def test_repair_covisible_cluster_merges_preserves_face_anchor_core_and_ejects_conflicting_tail():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracklets = {
        "host_face_a": [
            {"frame_idx": 0, "x_center": 220.0, "y_center": 320.0, "width": 180.0, "height": 280.0, "confidence": 0.95},
            {"frame_idx": 1, "x_center": 224.0, "y_center": 322.0, "width": 182.0, "height": 282.0, "confidence": 0.94},
        ],
        "host_face_b": [
            {"frame_idx": 12, "x_center": 226.0, "y_center": 321.0, "width": 181.0, "height": 281.0, "confidence": 0.93},
            {"frame_idx": 13, "x_center": 230.0, "y_center": 323.0, "width": 183.0, "height": 283.0, "confidence": 0.92},
        ],
        "tail_bad": [
            {"frame_idx": 1, "x_center": 1620.0, "y_center": 330.0, "width": 176.0, "height": 278.0, "confidence": 0.91},
            {"frame_idx": 2, "x_center": 1624.0, "y_center": 332.0, "width": 178.0, "height": 280.0, "confidence": 0.90},
        ],
    }
    label_by_tid = {"host_face_a": 0, "host_face_b": 0, "tail_bad": 0}

    repaired, metrics = worker._repair_covisible_cluster_merges(
        tracklets,
        label_by_tid,
        anchored_tids={"host_face_a", "host_face_b"},
    )

    assert repaired["host_face_a"] == repaired["host_face_b"]
    assert repaired["tail_bad"] != repaired["host_face_a"]
    assert metrics["repaired_cluster_count"] == 1


def test_should_skip_cluster_repair_when_face_core_is_already_near_target():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    assert worker._should_skip_cluster_repair(
        face_cluster_count=6,
        clusters_before_repair=7,
        visible_people_est=6,
        anchored_track_count=12,
    ) is True

    assert worker._should_skip_cluster_repair(
        face_cluster_count=6,
        clusters_before_repair=12,
        visible_people_est=6,
        anchored_track_count=12,
    ) is False


def test_same_identity_frame_collision_metrics_detect_distinct_covisible_instances():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracklets = {
        "left_a": [
            {"frame_idx": 10, "x_center": 220.0, "y_center": 320.0, "width": 180.0, "height": 280.0, "confidence": 0.95},
            {"frame_idx": 11, "x_center": 224.0, "y_center": 322.0, "width": 182.0, "height": 282.0, "confidence": 0.94},
        ],
        "right_b": [
            {"frame_idx": 10, "x_center": 1620.0, "y_center": 330.0, "width": 176.0, "height": 278.0, "confidence": 0.91},
            {"frame_idx": 11, "x_center": 1624.0, "y_center": 332.0, "width": 178.0, "height": 280.0, "confidence": 0.90},
        ],
    }
    label_by_tid = {"left_a": 0, "right_b": 0}

    metrics = worker._same_identity_frame_collision_metrics(tracklets, label_by_tid)

    assert metrics["same_identity_frame_collision_pairs"] >= 1
    assert metrics["same_identity_frame_collision_frames"] >= 2


def test_same_identity_frame_collision_metrics_ignore_duplicate_like_overlap():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracklets = {
        "dup_a": [
            {"frame_idx": 10, "x_center": 500.0, "y_center": 300.0, "width": 120.0, "height": 240.0, "confidence": 0.95},
            {"frame_idx": 11, "x_center": 504.0, "y_center": 302.0, "width": 121.0, "height": 241.0, "confidence": 0.94},
        ],
        "dup_b": [
            {"frame_idx": 10, "x_center": 560.0, "y_center": 302.0, "width": 119.0, "height": 239.0, "confidence": 0.92},
            {"frame_idx": 11, "x_center": 566.0, "y_center": 303.0, "width": 120.0, "height": 240.0, "confidence": 0.91},
        ],
    }
    label_by_tid = {"dup_a": 0, "dup_b": 0}

    metrics = worker._same_identity_frame_collision_metrics(tracklets, label_by_tid)

    assert metrics["same_identity_frame_collision_pairs"] == 0
    assert metrics["same_identity_frame_collision_frames"] == 0




def test_cluster_tracklets_propagates_face_seed_across_short_gap(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    class _FakeDBSCAN:
        def __init__(self, eps, min_samples, metric):
            self.labels_ = None

        def fit(self, X):
            if len(X) <= 1:
                self.labels_ = np.zeros(len(X), dtype=int)
            else:
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
            "track_id": "host_a",
            "x_center": 240.0,
            "y_center": 320.0,
            "width": 180.0,
            "height": 280.0,
            "confidence": 0.95,
        },
        {
            "frame_idx": 1,
            "track_id": "host_a",
            "x_center": 242.0,
            "y_center": 322.0,
            "width": 182.0,
            "height": 282.0,
            "confidence": 0.94,
        },
        {
            "frame_idx": 4,
            "track_id": "host_b",
            "x_center": 245.0,
            "y_center": 324.0,
            "width": 181.0,
            "height": 281.0,
            "confidence": 0.92,
        },
        {
            "frame_idx": 5,
            "track_id": "host_b",
            "x_center": 247.0,
            "y_center": 326.0,
            "width": 183.0,
            "height": 283.0,
            "confidence": 0.91,
        },
    ]
    tracklets = {
        "host_a": [tracks[0], tracks[1]],
        "host_b": [tracks[2], tracks[3]],
    }
    identity_features = {
        "host_a": {
            "embedding": [1.0, 0.0, 0.0],
            "embedding_source": "face",
            "embedding_count": 2,
            "face_observations": [
                {"frame_idx": 0, "confidence": 0.95, "associated_track_id": "host_a"},
                {"frame_idx": 1, "confidence": 0.94, "associated_track_id": "host_a"},
            ],
        },
    }
    face_track_features = {
        "face_0_0": {
            "face_track_id": "face_0_0",
            "embedding": [1.0, 0.0, 0.0],
            "embedding_source": "face",
            "embedding_count": 2,
            "face_observations": [
                {"frame_idx": 0, "confidence": 0.95, "associated_track_id": "host_a"},
                {"frame_idx": 1, "confidence": 0.94, "associated_track_id": "host_a"},
            ],
            "associated_track_counts": {"host_a": 2},
            "dominant_track_id": "host_a",
        }
    }

    clustered = worker._cluster_tracklets(
        video_path="video.mp4",
        tracks=[dict(track) for track in tracks],
        track_to_dets=tracklets,
        track_identity_features=identity_features,
        face_track_features=face_track_features,
    )

    mapped = {}
    for det in clustered:
        mapped.setdefault(det["track_id"], set()).add(det["frame_idx"])

    assert len(mapped) == 1
    assert mapped[next(iter(mapped.keys()))] == {0, 1, 4, 5}
    metrics = worker._last_clustering_metrics
    assert metrics["face_track_gap_propagated_tracklets"] >= 1

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


def test_cluster_tracklets_refines_seeded_face_clusters_with_later_face_merge(monkeypatch):
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
            "track_id": "host_frag_a",
            "x_center": 240.0,
            "y_center": 320.0,
            "width": 180.0,
            "height": 280.0,
            "confidence": 0.95,
        },
        {
            "frame_idx": 1,
            "track_id": "host_frag_a",
            "x_center": 244.0,
            "y_center": 322.0,
            "width": 182.0,
            "height": 282.0,
            "confidence": 0.94,
        },
        {
            "frame_idx": 12,
            "track_id": "host_frag_b",
            "x_center": 248.0,
            "y_center": 324.0,
            "width": 181.0,
            "height": 281.0,
            "confidence": 0.93,
        },
        {
            "frame_idx": 13,
            "track_id": "host_frag_b",
            "x_center": 252.0,
            "y_center": 326.0,
            "width": 183.0,
            "height": 283.0,
            "confidence": 0.92,
        },
    ]
    tracklets = {
        "host_frag_a": [tracks[0], tracks[1]],
        "host_frag_b": [tracks[2], tracks[3]],
    }
    face_track_features = {
        "face_0_0": {
            "face_track_id": "face_0_0",
            "embedding": [1.0, 0.0, 0.0],
            "embedding_source": "face",
            "embedding_count": 2,
            "face_observations": [],
            "associated_track_counts": {"host_frag_a": 2},
            "dominant_track_id": "host_frag_a",
        },
        "face_12_1": {
            "face_track_id": "face_12_1",
            "embedding": [0.999, 0.001, 0.0],
            "embedding_source": "face",
            "embedding_count": 2,
            "face_observations": [],
            "associated_track_counts": {"host_frag_b": 2},
            "dominant_track_id": "host_frag_b",
        },
    }

    clustered = worker._cluster_tracklets(
        video_path="video.mp4",
        tracks=[dict(track) for track in tracks],
        track_to_dets=tracklets,
        face_track_features=face_track_features,
    )

    mapped = {}
    for det in clustered:
        mapped.setdefault(det["track_id"], set()).add(det["frame_idx"])

    assert len(mapped) == 1
    assert mapped[next(iter(mapped.keys()))] == {0, 1, 12, 13}


def test_cluster_tracklets_face_track_seeding_requires_strong_association(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    class _FakeDBSCAN:
        def __init__(self, eps, min_samples, metric):
            self.labels_ = None

        def fit(self, X):
            self.labels_ = np.zeros(len(X), dtype=int)
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
            "track_id": "host_strong",
            "x_center": 240.0,
            "y_center": 320.0,
            "width": 180.0,
            "height": 280.0,
            "confidence": 0.95,
        },
        {
            "frame_idx": 1,
            "track_id": "host_strong",
            "x_center": 244.0,
            "y_center": 322.0,
            "width": 182.0,
            "height": 282.0,
            "confidence": 0.94,
        },
        {
            "frame_idx": 120,
            "track_id": "neighbor_weak",
            "x_center": 880.0,
            "y_center": 318.0,
            "width": 176.0,
            "height": 278.0,
            "confidence": 0.91,
        },
    ]
    tracklets = {
        "host_strong": [tracks[0], tracks[1]],
        "neighbor_weak": [tracks[2]],
    }
    face_track_features = {
        "face_host": {
            "face_track_id": "face_host",
            "embedding": [1.0, 0.0, 0.0],
            "embedding_source": "face",
            "embedding_count": 4,
            "face_observations": [],
            "associated_track_counts": {"host_strong": 10, "neighbor_weak": 1},
            "dominant_track_id": "host_strong",
        },
    }

    worker._cluster_tracklets(
        video_path="video.mp4",
        tracks=[dict(track) for track in tracks],
        track_to_dets=tracklets,
        face_track_features=face_track_features,
    )

    metrics = worker._last_clustering_metrics
    assert metrics["face_track_seeded_tracklets"] == 1


def test_cluster_seed_track_ids_for_face_track_allows_safe_non_covisible_secondary_track():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracklets = {
        "host_frag_a": [
            {"frame_idx": 0, "x_center": 240.0, "y_center": 320.0, "width": 180.0, "height": 280.0, "confidence": 0.95},
            {"frame_idx": 1, "x_center": 244.0, "y_center": 322.0, "width": 182.0, "height": 282.0, "confidence": 0.94},
        ],
        "host_frag_b": [
            {"frame_idx": 80, "x_center": 242.0, "y_center": 321.0, "width": 181.0, "height": 281.0, "confidence": 0.93},
            {"frame_idx": 81, "x_center": 246.0, "y_center": 323.0, "width": 183.0, "height": 283.0, "confidence": 0.92},
        ],
    }

    selected = worker._cluster_seed_track_ids_for_face_track(
        {"host_frag_a": 8, "host_frag_b": 6},
        tracklets=tracklets,
    )

    assert selected == {"host_frag_a", "host_frag_b"}


def test_choose_signature_attachment_label_prefers_temporally_adjacent_cluster():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracklets = {
        "hist_frag": [
            {"frame_idx": 42, "x_center": 242.0, "y_center": 322.0, "width": 182.0, "height": 282.0, "confidence": 0.91},
        ],
        "cluster_a_tid": [
            {"frame_idx": 40, "x_center": 240.0, "y_center": 320.0, "width": 180.0, "height": 280.0, "confidence": 0.95},
        ],
        "cluster_b_tid": [
            {"frame_idx": 400, "x_center": 244.0, "y_center": 323.0, "width": 181.0, "height": 281.0, "confidence": 0.94},
        ],
    }

    label = worker._choose_signature_attachment_label(
        tid="hist_frag",
        tracklets=tracklets,
        face_label_by_tid={"cluster_a_tid": 0, "cluster_b_tid": 1},
        histogram_attach_max_sig=1.15,
    )

    assert label == 0


def test_choose_signature_attachment_label_abstains_on_ambiguous_candidates(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.setenv("CLYPT_CLUSTER_ATTACH_MAX_GAP_FRAMES", "180")
    monkeypatch.setenv("CLYPT_CLUSTER_ATTACH_AMBIGUITY_MARGIN", "0.10")

    tracklets = {
        "hist_frag": [
            {"frame_idx": 100, "x_center": 242.0, "y_center": 322.0, "width": 182.0, "height": 282.0, "confidence": 0.91},
        ],
        "cluster_a_tid": [
            {"frame_idx": 92, "x_center": 240.0, "y_center": 320.0, "width": 180.0, "height": 280.0, "confidence": 0.95},
        ],
        "cluster_b_tid": [
            {"frame_idx": 94, "x_center": 244.0, "y_center": 321.0, "width": 180.0, "height": 280.0, "confidence": 0.94},
        ],
    }

    label = worker._choose_signature_attachment_label(
        tid="hist_frag",
        tracklets=tracklets,
        face_label_by_tid={"cluster_a_tid": 0, "cluster_b_tid": 1},
        histogram_attach_max_sig=1.15,
    )

    assert label is None


def test_cluster_signature_only_tracklets_groups_same_seat_fragments():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracklets = {
        "a1": [
            {"frame_idx": 0, "x_center": 100.0, "y_center": 100.0, "width": 80.0, "height": 160.0},
            {"frame_idx": 10, "x_center": 101.0, "y_center": 101.0, "width": 80.0, "height": 160.0},
        ],
        "a2": [
            {"frame_idx": 200, "x_center": 102.0, "y_center": 100.0, "width": 82.0, "height": 161.0},
            {"frame_idx": 210, "x_center": 103.0, "y_center": 101.0, "width": 82.0, "height": 161.0},
        ],
        "b1": [
            {"frame_idx": 0, "x_center": 240.0, "y_center": 100.0, "width": 84.0, "height": 160.0},
            {"frame_idx": 10, "x_center": 241.0, "y_center": 100.0, "width": 84.0, "height": 160.0},
        ],
    }

    groups = worker._cluster_signature_only_tracklets(
        track_ids=["a1", "a2", "b1"],
        tracklets=tracklets,
        base_max_sig=0.18,
    )

    normalized = {frozenset(group) for group in groups}
    assert frozenset({"a1", "a2"}) in normalized
    assert frozenset({"b1"}) in normalized


def test_choose_signature_attachment_label_for_group_prefers_matching_face_anchor():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracklets = {
        "hist_a": [
            {"frame_idx": 42, "x_center": 242.0, "y_center": 322.0, "width": 182.0, "height": 282.0, "confidence": 0.91},
        ],
        "hist_b": [
            {"frame_idx": 60, "x_center": 244.0, "y_center": 321.0, "width": 181.0, "height": 281.0, "confidence": 0.90},
        ],
        "cluster_a_tid": [
            {"frame_idx": 40, "x_center": 240.0, "y_center": 320.0, "width": 180.0, "height": 280.0, "confidence": 0.95},
        ],
        "cluster_b_tid": [
            {"frame_idx": 400, "x_center": 1440.0, "y_center": 330.0, "width": 178.0, "height": 280.0, "confidence": 0.94},
        ],
    }

    label = worker._choose_signature_attachment_label_for_group(
        tids=["hist_a", "hist_b"],
        tracklets=tracklets,
        face_label_by_tid={"cluster_a_tid": 0, "cluster_b_tid": 1},
        histogram_attach_max_sig=1.15,
    )

    assert label == 0


def test_choose_signature_attachment_label_for_group_requires_majority_support(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.setenv("CLYPT_CLUSTER_GROUP_ATTACH_MIN_SUPPORT_SHARE", "0.6")

    tracklets = {
        "hist_a": [
            {"frame_idx": 42, "x_center": 242.0, "y_center": 322.0, "width": 182.0, "height": 282.0, "confidence": 0.91},
        ],
        "hist_b": [
            {"frame_idx": 60, "x_center": 1442.0, "y_center": 330.0, "width": 181.0, "height": 281.0, "confidence": 0.90},
        ],
        "cluster_a_tid": [
            {"frame_idx": 40, "x_center": 240.0, "y_center": 320.0, "width": 180.0, "height": 280.0, "confidence": 0.95},
        ],
    }

    label = worker._choose_signature_attachment_label_for_group(
        tids=["hist_a", "hist_b"],
        tracklets=tracklets,
        face_label_by_tid={"cluster_a_tid": 0},
        histogram_attach_max_sig=1.15,
    )

    assert label is None


def test_merge_face_track_feature_sets_stitches_adjacent_segments_for_same_person():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    feature_maps = [
        {
            "face_0_0": {
                "face_track_id": "face_0_0",
                "embedding": [1.0, 0.0, 0.0],
                "embedding_count": 2,
                "face_observations": [
                    {
                        "frame_idx": 10,
                        "confidence": 0.9,
                        "bounding_box": {"x": 100.0, "y": 80.0, "width": 60.0, "height": 70.0},
                    },
                    {
                        "frame_idx": 20,
                        "confidence": 0.9,
                        "bounding_box": {"x": 102.0, "y": 80.0, "width": 60.0, "height": 70.0},
                    },
                ],
                "associated_track_counts": {"host_a": 6},
            }
        },
        {
            "face_240_3": {
                "face_track_id": "face_240_3",
                "embedding": [0.999, 0.001, 0.0],
                "embedding_count": 2,
                "face_observations": [
                    {
                        "frame_idx": 55,
                        "confidence": 0.88,
                        "bounding_box": {"x": 104.0, "y": 82.0, "width": 60.0, "height": 70.0},
                    },
                    {
                        "frame_idx": 68,
                        "confidence": 0.88,
                        "bounding_box": {"x": 105.0, "y": 82.0, "width": 60.0, "height": 70.0},
                    },
                ],
                "associated_track_counts": {"host_a": 5},
            }
        },
    ]

    merged = worker._merge_face_track_feature_sets(feature_maps)

    assert len(merged) == 1
    feature = next(iter(merged.values()))
    assert feature["embedding_count"] == 4
    assert feature["associated_track_counts"] == {"host_a": 11}
    assert feature["face_observation_count"] == 4


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
    monkeypatch.setenv("CLYPT_SHARED_ANALYSIS_PROXY", "0")
    monkeypatch.setenv("CLYPT_ANALYSIS_PROXY_ENABLE", "0")
    monkeypatch.setattr(worker, "_tracking_chunk_workers", lambda: 1)

    assert worker._select_tracking_mode() == "direct"


def test_tracking_mode_auto_prefers_chunked_with_multiple_workers(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.delenv("CLYPT_TRACKING_MODE", raising=False)
    monkeypatch.setenv("CLYPT_SHARED_ANALYSIS_PROXY", "0")
    monkeypatch.setenv("CLYPT_ANALYSIS_PROXY_ENABLE", "0")
    monkeypatch.setattr(worker, "_tracking_chunk_workers", lambda: 2)

    assert worker._select_tracking_mode() == "chunked"


def test_tracking_backend_defaults_to_botsort_reid(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.delenv("CLYPT_TRACKER_BACKEND", raising=False)

    assert worker._select_tracker_backend() == "botsort_reid"


def test_tracking_backend_accepts_bytetrack(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.setenv("CLYPT_TRACKER_BACKEND", "bytetrack")

    assert worker._select_tracker_backend() == "bytetrack"


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


def test_run_tracking_direct_mode_bypasses_shared_analysis_proxy(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    seen = {}

    def fake_run_tracking_direct(video_path, analysis_context=None):
        seen["analysis_context"] = analysis_context
        return [{"track_id": "t1"}], {}

    monkeypatch.setattr(worker, "_select_tracking_mode", lambda: "direct")
    monkeypatch.setattr(
        worker,
        "_prepare_direct_analysis_context",
        lambda video_path: {"analysis_video_path": "video.mp4", "prepared_video_path": "video.mp4"},
    )
    monkeypatch.setattr(
        worker,
        "_prepare_analysis_video",
        lambda video_path: {"analysis_video_path": "video_proxy.mp4", "prepared_video_path": "video.mp4"},
    )
    monkeypatch.setattr(
        worker,
        "_run_tracking_direct",
        fake_run_tracking_direct,
    )
    monkeypatch.setattr(worker, "_run_tracking_chunked", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("chunked should not run")))

    worker._run_tracking("video.mp4")

    assert seen["analysis_context"]["analysis_video_path"] == "video.mp4"


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


def test_run_tracking_direct_uses_bytetrack_when_requested(monkeypatch):
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

    monkeypatch.setenv("CLYPT_TRACKER_BACKEND", "bytetrack")
    monkeypatch.setattr(worker, "_ensure_h264", lambda path: path)
    monkeypatch.setattr(worker, "_probe_video_meta", lambda path: {"fps": 24.0, "total_frames": 1, "width": 1920, "height": 1080})
    monkeypatch.setattr(worker, "_ensure_bytetrack_yaml", lambda: "bytetrack.yaml")
    monkeypatch.setattr(worker, "_get_tracking_model", lambda: _Model())
    monkeypatch.setattr(worker, "_compute_letterbox_meta", lambda *args: {})
    monkeypatch.setattr(worker, "_xyxy_abs_to_xywhn", lambda *args: (0.1, 0.2, 0.3, 0.4))
    monkeypatch.setattr(worker, "_xywhn_to_xyxy_abs", lambda *args: (10.0, 20.0, 30.0, 40.0))
    monkeypatch.setattr(worker, "_forward_letterbox_xyxy", lambda *args: (10.0, 20.0, 30.0, 40.0))
    monkeypatch.setattr(worker, "_inverse_letterbox_xyxy", lambda *args: (10.0, 20.0, 30.0, 40.0))
    monkeypatch.setattr(worker, "_xyxy_to_xywh", lambda *args: (20.0, 30.0, 20.0, 20.0))

    worker._run_tracking_direct("video.mp4")

    assert track_calls[0]["tracker"] == "bytetrack.yaml"




def test_derive_track_identity_features_from_face_tracks_defaults_to_single_dominant_association(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.delenv("CLYPT_FACE_TRACK_ALLOW_MULTI_ASSOC", raising=False)

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

    assert set(features.keys()) == {"track_a"}
    assert features["track_a"]["embedding_source"] == "face"
    assert np.allclose(features["track_a"]["embedding"], [0.1, 0.2, 0.3])
    assert [obs["frame_idx"] for obs in features["track_a"]["face_observations"]] == [0, 1]


def test_derive_track_identity_features_from_face_tracks_can_opt_into_multi_associations(monkeypatch):
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    monkeypatch.setenv("CLYPT_FACE_TRACK_ALLOW_MULTI_ASSOC", "1")

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
                ],
                "associated_track_counts": {
                    "track_a": 2,
                    "track_b": 2,
                },
            }
        }
    )

    assert set(features.keys()) == {"track_a", "track_b"}
    assert np.allclose(features["track_a"]["embedding"], [0.1, 0.2, 0.3])
    assert np.allclose(features["track_b"]["embedding"], [0.1, 0.2, 0.3])
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


def test_build_stable_follow_bindings_absorbs_brief_mid_clip_turn():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    bindings = [
        {"track_id": "A", "start_time_ms": 0, "end_time_ms": 3000, "word_count": 30},
        {"track_id": "B", "start_time_ms": 3000, "end_time_ms": 3500, "word_count": 4},
        {"track_id": "A", "start_time_ms": 3500, "end_time_ms": 7000, "word_count": 32},
    ]

    stabilized = worker._build_stable_follow_bindings(
        bindings=bindings,
        track_to_dets={},
        track_identity_features=None,
    )

    assert stabilized == [
        {"track_id": "A", "start_time_ms": 0, "end_time_ms": 7000, "word_count": 66}
    ]


def test_build_stable_follow_bindings_preserves_sustained_turn_change():
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    bindings = [
        {"track_id": "A", "start_time_ms": 0, "end_time_ms": 3000, "word_count": 30},
        {"track_id": "B", "start_time_ms": 3000, "end_time_ms": 6200, "word_count": 28},
        {"track_id": "A", "start_time_ms": 6200, "end_time_ms": 9000, "word_count": 24},
    ]

    stabilized = worker._build_stable_follow_bindings(
        bindings=bindings,
        track_to_dets={},
        track_identity_features=None,
    )

    assert [segment["track_id"] for segment in stabilized] == ["A", "B", "A"]
