"""Tests for Phase 1 benchmark scorecard (pure metrics from synthetic ledgers)."""

from __future__ import annotations

import pytest

from backend.pipeline.phase1.metrics_scorecard import compute_phase1_scorecard


def _minimal_visual(tracking_metrics: dict | None = None) -> dict:
    return {
        "source_video": "https://youtube.com/watch?v=x",
        "schema_version": "2.0.0",
        "task_type": "person_tracking",
        "coordinate_space": "absolute_original_frame_xyxy",
        "geometry_type": "aabb",
        "class_taxonomy": {"0": "person"},
        "tracking_metrics": tracking_metrics or {"schema_pass_rate": 1.0},
        "tracks": [],
        "face_detections": [],
        "person_detections": [],
        "label_detections": [],
        "object_tracking": [],
        "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
        "video_metadata": {"width": 1920, "height": 1080, "fps": 30.0, "duration_ms": 1000},
    }


def test_assignment_coverage_and_unknown_rate():
    audio = {
        "words": [
            {"speaker_track_id": "G0", "word": "a"},
            {"speaker_track_id": "G0", "word": "b"},
            {"speaker_track_id": None, "word": "c"},
            {"speaker_track_id": "", "word": "d"},
        ],
        "speaker_bindings": [],
    }
    card = compute_phase1_scorecard(audio, _minimal_visual())
    assert card["assignment_coverage"] == 0.5
    assert card["unknown_rate"] == 0.5
    assert card["counts"]["word_count"] == 4
    assert card["counts"]["assigned_word_count"] == 2
    assert card["counts"]["unknown_word_count"] == 2


def test_empty_words_yields_null_assignment_rates():
    card = compute_phase1_scorecard({"words": [], "speaker_bindings": []}, _minimal_visual())
    assert card["assignment_coverage"] is None
    assert card["unknown_rate"] is None


def test_with_scored_candidate_ratio():
    audio = {
        "words": [{"speaker_track_id": "G0", "word": "x"}],
        "speaker_candidate_debug": [
            {"candidates": [{"blended_score": 0.1}]},
            {"candidates": []},
            {"candidates": [{"blended_score": 0.2}, {"blended_score": 0.1}]},
        ],
    }
    card = compute_phase1_scorecard(audio, _minimal_visual())
    assert card["with_scored_candidate_ratio"] == pytest.approx(2 / 3)


def test_with_scored_candidate_ratio_none_without_debug():
    audio = {"words": [{"speaker_track_id": None, "word": "x"}], "speaker_candidate_debug": []}
    card = compute_phase1_scorecard(audio, _minimal_visual())
    assert card["with_scored_candidate_ratio"] is None


def test_high_confidence_misassignment_proxy():
    audio = {
        "words": [],
        "speaker_candidate_debug": [
            {
                "calibrated_confidence": 0.9,
                "chosen_local_track_id": "t1",
                "active_audio_local_track_id": "t2",
                "ambiguous": False,
                "top_1_top_2_margin": 0.2,
            },
            {
                "calibrated_confidence": 0.9,
                "chosen_local_track_id": "t1",
                "active_audio_local_track_id": "t1",
                "ambiguous": False,
                "top_1_top_2_margin": 0.2,
            },
            {
                "calibrated_confidence": 0.9,
                "chosen_local_track_id": "t1",
                "active_audio_local_track_id": "t1",
                "ambiguous": True,
                "top_1_top_2_margin": 0.2,
            },
        ],
    }
    card = compute_phase1_scorecard(audio, _minimal_visual())
    # 3 eligible (high conf + chosen local); flagged: mismatch + ambiguous = 2
    assert card["high_confidence_misassignment_proxy_ratio"] == pytest.approx(2 / 3)


def test_overlap_camera_consistency():
    audio = {
        "words": [],
        "overlap_follow_decisions": [
            {
                "stay_wide": False,
                "visible_local_track_ids": ["a"],
                "camera_target_local_track_id": "a",
            },
            {
                "stay_wide": False,
                "visible_local_track_ids": ["a"],
                "camera_target_local_track_id": "b",
            },
            {"stay_wide": True, "visible_local_track_ids": ["a", "b"], "camera_target_local_track_id": None},
        ],
    }
    card = compute_phase1_scorecard(audio, _minimal_visual())
    assert card["overlap_camera_consistency_ratio"] == pytest.approx(2 / 3)


def test_wallclock_and_stage_timings_from_tracking_metrics():
    audio = {"words": [], "speaker_bindings": []}
    visual = _minimal_visual(
        {
            "schema_pass_rate": 0.99,
            "tracking_wallclock_s": 12.5,
            "speaker_binding_wallclock_s": 3.25,
            "track_identity_features": {"should_not_appear": {}},
        }
    )
    card = compute_phase1_scorecard(
        audio,
        visual,
        job_timings_ms={"ingest_ms": 100, "processing_ms": 200, "upload_ms": 50},
    )
    assert card["wallclock_ms"]["total_ms"] == 350
    assert card["stage_wallclock_s"] == {
        "speaker_binding_wallclock_s": 3.25,
        "tracking_wallclock_s": 12.5,
    }
    assert "track_identity_features" not in card["tracking_metrics_summary"]
    assert card["tracking_metrics_summary"]["schema_pass_rate"] == 0.99


def test_persist_manifest_includes_benchmark_scorecard(tmp_path):
    from backend.do_phase1_service.storage import LocalGCSStorage, persist_phase1_outputs

    storage = LocalGCSStorage(bucket="b", root_dir=tmp_path / "gcs")
    uri = storage.upload_bytes(b"v", "phase_1/jobs/j1/source_video.mp4")
    m = persist_phase1_outputs(
        storage=storage,
        output_dir=tmp_path,
        job_id="j1",
        source_url="https://youtube.com/watch?v=x",
        canonical_video_uri=uri,
        phase_1_audio={"source_audio": "https://youtube.com/watch?v=x", "words": [], "speaker_bindings": []},
        phase_1_visual=_minimal_visual(),
        timings={"ingest_ms": 1, "processing_ms": 2, "upload_ms": 3},
    )
    assert m.metadata.benchmark_scorecard is not None
    assert m.metadata.benchmark_scorecard["wallclock_ms"]["total_ms"] == 6
    assert m.metadata.benchmark_scorecard["version"] == 1
