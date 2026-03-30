from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "do_phase1"
    / "run_benchmark_corpus.py"
)


def _load_subject():
    spec = importlib.util.spec_from_file_location("run_benchmark_corpus", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_manifest_paths_dedupes_and_sorts(tmp_path: Path):
    mod = _load_subject()
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "one.json").write_text("{}", encoding="utf-8")
    nested = b / "nested"
    nested.mkdir()
    (nested / "two.json").write_text("{}", encoding="utf-8")

    out = mod.collect_manifest_paths([a, b, a])
    names = [p.name for p in out]
    assert names == ["one.json", "two.json"]


def test_evaluate_wave4_exit_gates_passes_on_improvement():
    mod = _load_subject()
    comparison = {
        "aggregate": {
            "summary_ratio_means": {
                "assignment_coverage": {"delta": 0.03},
                "with_scored_candidate_ratio": {"delta": 0.01},
                "unknown_rate": {"delta": -0.02},
                "overlap_camera_consistency_ratio": {"delta": 0.04},
            },
            "wallclock_ms_means": {
                "total_ms": {"delta": -500.0},
            },
        }
    }
    gate = mod.evaluate_wave4_exit_gates(comparison)
    assert gate["passed"] is True
    assert all(c["passed"] for c in gate["checks"])


def test_evaluate_wave4_exit_gates_fails_when_missing_or_regressed():
    mod = _load_subject()
    comparison = {
        "aggregate": {
            "summary_ratio_means": {
                "assignment_coverage": {"delta": -0.01},
                "with_scored_candidate_ratio": {"delta": 0.0},
                "unknown_rate": {"delta": 0.03},
                "overlap_camera_consistency_ratio": {"delta": 0.0},
            },
            "wallclock_ms_means": {
                "total_ms": {"delta": 1200.0},
            },
        }
    }
    gate = mod.evaluate_wave4_exit_gates(comparison)
    assert gate["passed"] is False
    assert any(c["name"] == "assignment_coverage_delta" and c["passed"] is False for c in gate["checks"])
    assert any(c["name"] == "unknown_rate_delta" and c["passed"] is False for c in gate["checks"])
    assert any(c["name"] == "total_wallclock_ms_delta" and c["passed"] is False for c in gate["checks"])


def test_main_generates_current_baseline_comparison_and_gate(tmp_path: Path):
    mod = _load_subject()
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    out_dir = tmp_path / "out"
    current.mkdir()
    baseline.mkdir()

    current_payload = {
        "phase_1_audio": {
            "words": [
                    {
                        "text": "hi",
                        "speaker_local": "spk_1",
                        "speaker_track_id": "track_1",
                        "with_scored_candidate": True,
                    },
            ]
                ,
                "speaker_candidate_debug": [
                    {"candidates": [{"local_track_id": "track_1", "score": 0.9}]},
                ],
        },
        "phase_1_visual": {
            "speaker_follow_bindings_local": [
                {"start_ms": 0, "end_ms": 500, "speaker_local": "spk_1"},
            ]
        },
        "timings": {"ingest_ms": 20, "processing_ms": 80, "upload_ms": 10},
        "job_id": "clip_1",
    }
    baseline_payload = {
        "phase_1_audio": {
            "words": [
                    {
                        "text": "hi",
                        "speaker_local": "unknown",
                        "speaker_track_id": "",
                        "with_scored_candidate": False,
                    },
            ]
                ,
                "speaker_candidate_debug": [
                    {"candidates": []},
                ],
        },
        "phase_1_visual": {
            "speaker_follow_bindings_local": [
                {"start_ms": 0, "end_ms": 500, "speaker_local": "unknown"},
            ]
        },
        "timings": {"ingest_ms": 25, "processing_ms": 120, "upload_ms": 15},
        "job_id": "clip_1",
    }

    (current / "c1.json").write_text(json.dumps(current_payload), encoding="utf-8")
    (baseline / "b1.json").write_text(json.dumps(baseline_payload), encoding="utf-8")

    rc = mod.main(
        [
            "--current-corpus-dir",
            str(current),
            "--baseline-corpus-dir",
            str(baseline),
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["current_report_path"] is not None
    assert summary["comparison_path"] is not None
    assert summary["gate_path"] is not None
    assert summary["gate_passed"] is True
