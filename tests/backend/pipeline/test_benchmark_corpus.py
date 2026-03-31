"""Tests for Phase 1 benchmark corpus loader / report aggregation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.pipeline.phase1.benchmark_corpus import (
    BENCHMARK_REPORT_VERSION,
    aggregate_scorecard_summary,
    build_benchmark_report,
    extract_ledgers_from_payload,
    scorecard_for_payload,
)
from backend.pipeline.phase1_contract import JobState, Phase1Manifest


def _minimal_visual(tracking_metrics: dict | None = None) -> dict:
    return {
        "uri": "gs://b/v.json",
        "source_video": "https://youtube.com/watch?v=x",
        "video_gcs_uri": "gs://b/v.mp4",
            "schema_version": "3.0.0",
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


def _bundle(
    *,
    job_id: str,
    words: list[dict],
    tracking_metrics: dict | None = None,
    timings: dict | None = None,
    extra_audio: dict | None = None,
) -> dict:
    audio: dict = {
        "source_audio": "https://youtube.com/watch?v=x",
        "words": words,
        "speaker_bindings": [],
    }
    if extra_audio:
        audio.update(extra_audio)
    out: dict = {
        "job_id": job_id,
        "phase_1_audio": audio,
        "phase_1_visual": _minimal_visual(tracking_metrics),
        "metadata": {"timings": timings or {}},
    }
    return out


def test_extract_ledgers_bundle_and_manifest_roundtrip_fields():
    bundle = _bundle(
        job_id="j1",
        words=[{"speaker_track_id": "G0", "word": "a", "start_time_ms": 0, "end_time_ms": 100, "speaker_tag": "t"}],
        timings={"ingest_ms": 10, "processing_ms": 20, "upload_ms": 5},
    )
    a, v, jid, tms = extract_ledgers_from_payload(bundle)
    assert jid == "j1"
    assert tms == {"ingest_ms": 10, "processing_ms": 20, "upload_ms": 5}
    assert a["words"][0]["speaker_track_id"] == "G0"
    assert v["tracking_metrics"]["schema_pass_rate"] == 1.0


def test_extract_ledgers_rejects_unknown_shape():
    with pytest.raises(ValueError, match="expected Phase 1 manifest"):
        extract_ledgers_from_payload({"foo": 1})


def test_scorecard_for_payload_matches_direct_metrics():
    bundle = _bundle(
        job_id="j2",
        words=[
            {"speaker_track_id": "G0", "word": "a", "start_time_ms": 0, "end_time_ms": 50, "speaker_tag": "t"},
            {"speaker_track_id": None, "word": "b", "start_time_ms": 50, "end_time_ms": 100, "speaker_tag": "u"},
        ],
        tracking_metrics={
            "schema_pass_rate": 1.0,
            "tracking_wallclock_s": 1.0,
            "custom_stage_wallclock_s": 2.5,
        },
        timings={"ingest_ms": 100, "processing_ms": 0, "upload_ms": 0},
    )
    card = scorecard_for_payload(bundle)
    assert card["assignment_coverage"] == 0.5
    assert card["unknown_rate"] == 0.5
    assert card["wallclock_ms"]["total_ms"] == 100
    assert card["stage_wallclock_s"]["custom_stage_wallclock_s"] == 2.5
    assert card["stage_wallclock_s"]["tracking_wallclock_s"] == 1.0


def test_aggregate_scorecard_summary_two_clips():
    c1 = scorecard_for_payload(
        _bundle(
            job_id="a",
            words=[
                {"speaker_track_id": "G0", "word": "x", "start_time_ms": 0, "end_time_ms": 10, "speaker_tag": "t"}
            ],
            tracking_metrics={
                "tracking_wallclock_s": 10.0,
                "canonical_face_stream_coverage": 0.6,
                "identity_track_count_before_clustering": 10,
                "identity_track_count_after_reid_merge": 8,
                "identity_track_count_after_clustering": 8,
                "decode_prepare_wallclock_ms": 50,
                "decode_source_video_mb": 100.0,
                "decode_analysis_video_mb": 90.0,
                "lrasd_stage_gpu_utilization_pct_sampled_mean": 70.0,
            },
            timings={"processing_ms": 200},
        )
    )
    c2 = scorecard_for_payload(
        _bundle(
            job_id="b",
            words=[
                {"speaker_track_id": None, "word": "y", "start_time_ms": 0, "end_time_ms": 10, "speaker_tag": "t"},
                {"speaker_track_id": "G1", "word": "z", "start_time_ms": 10, "end_time_ms": 20, "speaker_tag": "t"},
            ],
            tracking_metrics={
                "tracking_wallclock_s": 20.0,
                "canonical_face_stream_coverage": 0.8,
                "identity_track_count_before_clustering": 12,
                "identity_track_count_after_reid_merge": 6,
                "identity_track_count_after_clustering": 6,
                "decode_prepare_wallclock_ms": 120,
                "decode_source_video_mb": 100.0,
                "decode_analysis_video_mb": 70.0,
                "lrasd_stage_gpu_utilization_pct_sampled_mean": 80.0,
            },
            timings={"processing_ms": 400},
        )
    )
    s = aggregate_scorecard_summary([c1, c2])
    assert s["assignment_coverage"]["mean"] == pytest.approx(0.75)
    assert s["assignment_coverage"]["n"] == 2
    assert s["unknown_rate"]["mean"] == pytest.approx(0.25)
    assert s["stage_wallclock_s"]["tracking_wallclock_s"]["mean"] == pytest.approx(15.0)
    assert s["stage_wallclock_s"]["tracking_wallclock_s"]["n"] == 2
    assert s["canonical_face_stream_coverage"]["mean"] == pytest.approx(0.7)
    assert s["identity_fragmentation_reduction_ratio"]["mean"] == pytest.approx(0.35)
    assert s["decode_overhead_ratio"]["mean"] == pytest.approx(0.275)
    assert s["decode_before_after_size_ratio"]["mean"] == pytest.approx(0.8)
    assert s["gpu_utilization_pct"]["lrasd_stage"]["mean"] == pytest.approx(75.0)
    assert s["aggregation_inputs"]["input_clip_count"] == 2
    assert s["aggregation_inputs"]["eligible_clip_count"] == 2
    assert s["aggregation_inputs"]["excluded_missing_wallclock_count"] == 0


def test_aggregate_scorecard_summary_excludes_missing_wallclock_rows():
    with_wallclock = {
        "assignment_coverage": 1.0,
        "with_scored_candidate_ratio": 1.0,
        "unknown_rate": 0.0,
        "overlap_camera_consistency_ratio": 1.0,
        "canonical_face_stream_coverage": 0.8,
        "identity_fragmentation_reduction_ratio": 0.3,
        "decode_overhead_ratio": 0.2,
        "decode_before_after_size_ratio": 0.9,
        "wallclock_ms": {"ingest_ms": 1, "processing_ms": 2, "upload_ms": 3, "total_ms": 6},
        "decode_overhead_ms": {"prepare_ms": 100},
        "gpu_utilization_pct": {"lrasd_stage": 50.0, "face_stage": 40.0},
        "stage_wallclock_s": {},
    }
    missing_wallclock = {
        **with_wallclock,
        "assignment_coverage": 0.0,
    }
    missing_wallclock.pop("wallclock_ms", None)
    s = aggregate_scorecard_summary([with_wallclock, missing_wallclock])
    # Ratio aggregates include all scorecards; wallclock aggregates use eligible subset.
    assert s["assignment_coverage"]["mean"] == pytest.approx(0.5)
    assert s["aggregation_inputs"]["input_clip_count"] == 2
    assert s["aggregation_inputs"]["eligible_clip_count"] == 1
    assert s["aggregation_inputs"]["excluded_missing_wallclock_count"] == 1


def test_build_benchmark_report_from_files(tmp_path: Path):
    p1 = tmp_path / "m1.json"
    p2 = tmp_path / "m2.json"
    fixed = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    p1.write_text(
        json.dumps(
            _bundle(
                job_id="clip_one",
                words=[
                    {
                        "speaker_track_id": "G0",
                        "word": "a",
                        "start_time_ms": 0,
                        "end_time_ms": 10,
                        "speaker_tag": "t",
                    }
                ],
                tracking_metrics={"speaker_binding_wallclock_s": 4.0},
                timings={"ingest_ms": 0, "processing_ms": 100, "upload_ms": 0},
            )
        ),
        encoding="utf-8",
    )
    p2.write_text(
        json.dumps(
            _bundle(
                job_id="clip_two",
                words=[
                    {
                        "speaker_track_id": "G0",
                        "word": "b",
                        "start_time_ms": 0,
                        "end_time_ms": 10,
                        "speaker_tag": "t",
                    }
                ],
                tracking_metrics={"speaker_binding_wallclock_s": 8.0},
                timings={"ingest_ms": 0, "processing_ms": 300, "upload_ms": 0},
            )
        ),
        encoding="utf-8",
    )

    report = build_benchmark_report([p1, p2], now_utc=fixed)
    assert report["version"] == BENCHMARK_REPORT_VERSION
    assert report["generated_at"] == "2026-01-15T12:00:00Z"
    assert len(report["clips"]) == 2
    assert report["clips"][0]["error"] is None
    assert report["clips"][0]["scorecard"]["wallclock_ms"]["processing_ms"] == 100
    assert report["summary"]["successful_clip_count"] == 2
    assert report["summary"]["failed_clip_count"] == 0
    assert report["summary"]["wallclock_ms"]["processing_ms"]["mean"] == pytest.approx(200.0)
    assert report["summary"]["stage_wallclock_s"]["speaker_binding_wallclock_s"]["mean"] == pytest.approx(6.0)


def test_build_benchmark_report_manifest_shape(tmp_path: Path):
    payload = {
        "contract_version": "v3",
        "job_id": "manifest_job",
        "status": JobState.SUCCEEDED,
        "source_video": {"source_url": "https://youtube.com/watch?v=x"},
        "canonical_video_gcs_uri": "gs://b/v.mp4",
        "artifacts": {
            "transcript": {
                "uri": "gs://b/t.json",
                "source_audio": "https://youtube.com/watch?v=x",
                "video_gcs_uri": "gs://b/v.mp4",
                "words": [
                    {
                        "word": "hi",
                        "start_time_ms": 0,
                        "end_time_ms": 100,
                        "speaker_track_id": "G0",
                        "speaker_tag": "Global_Person_0",
                    }
                ],
                "speaker_bindings": [],
            },
            "visual_tracking": _minimal_visual({"cluster_tracklets_wallclock_s": 1.25}),
        },
        "metadata": {
            "runtime": {"provider": "test"},
            "timings": {"ingest_ms": 1, "processing_ms": 2, "upload_ms": 3},
        },
    }
    Phase1Manifest.model_validate(payload)
    path = tmp_path / "full.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = build_benchmark_report([path])
    assert report["clips"][0]["job_id"] == "manifest_job"
    card = report["clips"][0]["scorecard"]
    assert card is not None
    assert card["assignment_coverage"] == 1.0
    assert card["stage_wallclock_s"]["cluster_tracklets_wallclock_s"] == 1.25
    assert card["wallclock_ms"]["total_ms"] == 6
    assert report["summary"]["overlap_camera_consistency_ratio"]["n"] == 0


def test_build_benchmark_report_records_load_error(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    report = build_benchmark_report([bad])
    assert report["summary"]["successful_clip_count"] == 0
    assert report["summary"]["failed_clip_count"] == 1
    assert report["clips"][0]["error"] is not None
    assert "JSON" in report["clips"][0]["error"] or "Expecting" in report["clips"][0]["error"]
