"""Tests for benchmark report comparison and CLI baseline handling."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from backend.pipeline.phase1.benchmark_corpus import (
    BENCHMARK_COMPARISON_VERSION,
    compare_benchmark_reports,
    load_benchmark_report,
    build_benchmark_report,
)


def _bundle(*, job_id: str, words: list[dict], processing_ms: int = 100, tracking_s: float = 1.0) -> dict:
    return {
        "job_id": job_id,
        "phase_1_audio": {
            "source_audio": "https://example.com/x",
            "words": words,
            "speaker_bindings": [],
        },
        "phase_1_visual": {
            "uri": "gs://b/v.json",
            "source_video": "https://example.com/x",
            "video_gcs_uri": "gs://b/v.mp4",
            "schema_version": "2.0.0",
            "task_type": "person_tracking",
            "coordinate_space": "absolute_original_frame_xyxy",
            "geometry_type": "aabb",
            "class_taxonomy": {"0": "person"},
            "tracking_metrics": {"schema_pass_rate": 1.0, "tracking_wallclock_s": tracking_s},
            "tracks": [],
            "face_detections": [],
            "person_detections": [],
            "label_detections": [],
            "object_tracking": [],
            "shot_changes": [{"start_time_ms": 0, "end_time_ms": 1000}],
            "video_metadata": {"width": 1920, "height": 1080, "fps": 30.0, "duration_ms": 1000},
        },
        "metadata": {"timings": {"ingest_ms": 0, "processing_ms": processing_ms, "upload_ms": 0}},
    }


def test_compare_identical_reports_zero_aggregate_deltas(tmp_path: Path):
    p = tmp_path / "a.json"
    p.write_text(
        json.dumps(
            _bundle(
                job_id="j1",
                words=[
                    {
                        "speaker_track_id": "G0",
                        "word": "a",
                        "start_time_ms": 0,
                        "end_time_ms": 10,
                        "speaker_tag": "t",
                    }
                ],
            )
        ),
        encoding="utf-8",
    )
    r1 = build_benchmark_report([p])
    r2 = build_benchmark_report([p])
    diff = compare_benchmark_reports(r2, r1)
    assert diff["version"] == BENCHMARK_COMPARISON_VERSION
    agg = diff["aggregate"]
    assert agg["summary_ratio_means"]["assignment_coverage"]["delta"] == pytest.approx(0.0)
    assert agg["wallclock_ms_means"]["processing_ms"]["delta"] == pytest.approx(0.0)
    assert agg["stage_wallclock_s_means"]["tracking_wallclock_s"]["delta"] == pytest.approx(0.0)
    assert len(diff["per_clip"]) == 1
    assert diff["per_clip"][0]["match_basis"] == "job_id"
    m = diff["per_clip"][0]["metrics"]["assignment_coverage"]
    assert m["delta"] == pytest.approx(0.0)


def test_compare_per_clip_metric_delta(tmp_path: Path):
    p1 = tmp_path / "c1.json"
    p2 = tmp_path / "c2.json"
    words_half = [
        {"speaker_track_id": "G0", "word": "a", "start_time_ms": 0, "end_time_ms": 10, "speaker_tag": "t"},
        {"speaker_track_id": None, "word": "b", "start_time_ms": 10, "end_time_ms": 20, "speaker_tag": "t"},
    ]
    p1.write_text(json.dumps(_bundle(job_id="x", words=words_half, processing_ms=50)), encoding="utf-8")
    p2.write_text(json.dumps(_bundle(job_id="x", words=words_half, processing_ms=150)), encoding="utf-8")
    base = build_benchmark_report([p1])
    cur = build_benchmark_report([p2])
    diff = compare_benchmark_reports(cur, base)
    proc = diff["per_clip"][0]["metrics"]["wallclock_ms.processing_ms"]
    assert proc["baseline"] == pytest.approx(50.0)
    assert proc["current"] == pytest.approx(150.0)
    assert proc["delta"] == pytest.approx(100.0)
    assign = diff["aggregate"]["summary_ratio_means"]["assignment_coverage"]
    assert assign["delta"] == pytest.approx(0.0)


def test_compare_baseline_only_and_current_only_rows(tmp_path: Path):
    only_b = tmp_path / "only_b.json"
    only_c = tmp_path / "only_c.json"
    w = [{"speaker_track_id": "G0", "word": "a", "start_time_ms": 0, "end_time_ms": 10, "speaker_tag": "t"}]
    only_b.write_text(json.dumps(_bundle(job_id="b_only", words=w)), encoding="utf-8")
    only_c.write_text(json.dumps(_bundle(job_id="c_only", words=w)), encoding="utf-8")
    baseline = build_benchmark_report([only_b, only_c])
    current = build_benchmark_report([only_c])
    diff = compare_benchmark_reports(current, baseline)
    bases = {row["match_basis"] for row in diff["per_clip"]}
    assert bases == {"job_id", "baseline_only"}
    baseline_rows = [r for r in diff["per_clip"] if r["match_basis"] == "baseline_only"]
    assert len(baseline_rows) == 1
    assert baseline_rows[0]["job_id"] == "b_only"
    assert baseline_rows[0]["current_source_path"] is None


def test_compare_match_by_job_id_when_paths_differ(tmp_path: Path):
    p_b = tmp_path / "baseline_name.json"
    p_c = tmp_path / "renamed.json"
    w = [{"speaker_track_id": "G0", "word": "a", "start_time_ms": 0, "end_time_ms": 10, "speaker_tag": "t"}]
    p_b.write_text(json.dumps(_bundle(job_id="shared", words=w, processing_ms=10)), encoding="utf-8")
    p_c.write_text(json.dumps(_bundle(job_id="shared", words=w, processing_ms=20)), encoding="utf-8")
    baseline = build_benchmark_report([p_b])
    current = build_benchmark_report([p_c])
    diff = compare_benchmark_reports(current, baseline)
    assert len(diff["per_clip"]) == 1
    row = diff["per_clip"][0]
    assert row["match_basis"] == "job_id"
    assert row["baseline_source_path"] != row["current_source_path"]


def test_compare_rejects_missing_clips():
    with pytest.raises(ValueError, match="clips"):
        compare_benchmark_reports({"summary": {}}, {"summary": {}, "clips": []})


def test_load_benchmark_report_roundtrip(tmp_path: Path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"version": 1, "clips": [], "summary": {}}), encoding="utf-8")
    assert load_benchmark_report(p)["version"] == 1


def test_load_benchmark_report_rejects_non_object(tmp_path: Path):
    p = tmp_path / "x.json"
    p.write_text("[1]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        load_benchmark_report(p)


def test_cli_baseline_writes_comparison_next_to_output(tmp_path: Path):
    m = tmp_path / "m.json"
    m.write_text(
        json.dumps(
            _bundle(
                job_id="cli",
                words=[
                    {
                        "speaker_track_id": "G0",
                        "word": "a",
                        "start_time_ms": 0,
                        "end_time_ms": 10,
                        "speaker_tag": "t",
                    }
                ],
                processing_ms=1,
            )
        ),
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            build_benchmark_report(
                [
                    m,
                ]
            )
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out" / "report.json"
    comp = tmp_path / "out" / "report.comparison.json"
    root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.pipeline.phase1.benchmark_corpus",
            str(m),
            "-o",
            str(out),
            "--baseline-report",
            str(baseline_path),
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert out.is_file()
    assert comp.is_file()
    loaded = json.loads(comp.read_text(encoding="utf-8"))
    assert loaded["version"] == BENCHMARK_COMPARISON_VERSION
    assert "aggregate" in loaded


def test_cli_baseline_stdout_without_comparison_output_exits_3(tmp_path: Path):
    m = tmp_path / "m.json"
    m.write_text(
        json.dumps(
            _bundle(
                job_id="x",
                words=[
                    {
                        "speaker_track_id": "G0",
                        "word": "a",
                        "start_time_ms": 0,
                        "end_time_ms": 10,
                        "speaker_tag": "t",
                    }
                ],
            )
        ),
        encoding="utf-8",
    )
    baseline_path = tmp_path / "b.json"
    baseline_path.write_text(json.dumps(build_benchmark_report([m])), encoding="utf-8")
    root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.pipeline.phase1.benchmark_corpus",
            str(m),
            "--baseline-report",
            str(baseline_path),
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 3
    assert "comparison-output" in proc.stderr


def test_cli_explicit_comparison_output(tmp_path: Path):
    m = tmp_path / "m.json"
    m.write_text(
        json.dumps(
            _bundle(
                job_id="y",
                words=[
                    {
                        "speaker_track_id": "G0",
                        "word": "a",
                        "start_time_ms": 0,
                        "end_time_ms": 10,
                        "speaker_tag": "t",
                    }
                ],
            )
        ),
        encoding="utf-8",
    )
    baseline_path = tmp_path / "b.json"
    baseline_path.write_text(json.dumps(build_benchmark_report([m])), encoding="utf-8")
    comp_path = tmp_path / "custom.comp.json"
    root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.pipeline.phase1.benchmark_corpus",
            str(m),
            "--baseline-report",
            str(baseline_path),
            "--comparison-output",
            str(comp_path),
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert comp_path.is_file()
