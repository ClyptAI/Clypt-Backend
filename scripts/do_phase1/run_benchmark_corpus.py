#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from backend.pipeline.phase1.benchmark_corpus import (
    build_benchmark_report,
    compare_benchmark_reports,
    load_benchmark_report,
)


class Wave4GateThresholds:
    def __init__(
        self,
        *,
        min_assignment_coverage_delta: float = 0.0,
        min_with_scored_candidate_ratio_delta: float = 0.0,
        max_unknown_rate_delta: float = 0.0,
        min_overlap_camera_consistency_delta: float = 0.0,
        max_total_wallclock_ms_delta: float = 0.0,
        min_canonical_face_stream_coverage_delta: float = 0.0,
        min_identity_fragmentation_reduction_delta: float = 0.0,
        max_decode_overhead_ratio_delta: float = 0.0,
        max_decode_before_after_size_ratio_delta: float = 0.0,
    ) -> None:
        self.min_assignment_coverage_delta = min_assignment_coverage_delta
        self.min_with_scored_candidate_ratio_delta = min_with_scored_candidate_ratio_delta
        self.max_unknown_rate_delta = max_unknown_rate_delta
        self.min_overlap_camera_consistency_delta = min_overlap_camera_consistency_delta
        self.max_total_wallclock_ms_delta = max_total_wallclock_ms_delta
        self.min_canonical_face_stream_coverage_delta = min_canonical_face_stream_coverage_delta
        self.min_identity_fragmentation_reduction_delta = min_identity_fragmentation_reduction_delta
        self.max_decode_overhead_ratio_delta = max_decode_overhead_ratio_delta
        self.max_decode_before_after_size_ratio_delta = max_decode_before_after_size_ratio_delta


def _safe_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _comparison_delta(comparison: Mapping[str, Any], bucket: str, key: str) -> float | None:
    agg = comparison.get("aggregate")
    if not isinstance(agg, Mapping):
        return None
    group = agg.get(bucket)
    if not isinstance(group, Mapping):
        return None
    triplet = group.get(key)
    if not isinstance(triplet, Mapping):
        return None
    return _safe_float(triplet.get("delta"))


def evaluate_wave4_exit_gates(
    comparison: Mapping[str, Any],
    thresholds: Wave4GateThresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or Wave4GateThresholds()
    checks: list[dict[str, Any]] = []

    def _add_check(
        name: str,
        delta: float | None,
        op: str,
        target: float,
        guidance: str,
        *,
        required: bool = True,
    ) -> None:
        if delta is None:
            checks.append(
                {
                    "name": name,
                    "passed": not required,
                    "delta": None,
                    "operator": op,
                    "target": target,
                    "guidance": guidance,
                    "reason": "missing_metric_optional" if not required else "missing_metric",
                    "required": required,
                }
            )
            return
        passed = delta >= target if op == ">=" else delta <= target
        checks.append(
            {
                "name": name,
                "passed": passed,
                "delta": delta,
                "operator": op,
                "target": target,
                "guidance": guidance,
                "required": required,
            }
        )

    _add_check(
        "assignment_coverage_delta",
        _comparison_delta(comparison, "summary_ratio_means", "assignment_coverage"),
        ">=",
        t.min_assignment_coverage_delta,
        "Higher is better.",
    )
    _add_check(
        "with_scored_candidate_ratio_delta",
        _comparison_delta(comparison, "summary_ratio_means", "with_scored_candidate_ratio"),
        ">=",
        t.min_with_scored_candidate_ratio_delta,
        "Higher is better.",
    )
    _add_check(
        "unknown_rate_delta",
        _comparison_delta(comparison, "summary_ratio_means", "unknown_rate"),
        "<=",
        t.max_unknown_rate_delta,
        "Lower is better.",
    )
    _add_check(
        "overlap_camera_consistency_ratio_delta",
        _comparison_delta(comparison, "summary_ratio_means", "overlap_camera_consistency_ratio"),
        ">=",
        t.min_overlap_camera_consistency_delta,
        "Higher is better.",
        required=False,
    )
    _add_check(
        "total_wallclock_ms_delta",
        _comparison_delta(comparison, "wallclock_ms_means", "total_ms"),
        "<=",
        t.max_total_wallclock_ms_delta,
        "Lower is better.",
    )
    _add_check(
        "canonical_face_stream_coverage_delta",
        _comparison_delta(comparison, "summary_ratio_means", "canonical_face_stream_coverage"),
        ">=",
        t.min_canonical_face_stream_coverage_delta,
        "Higher is better.",
        required=False,
    )
    _add_check(
        "identity_fragmentation_reduction_ratio_delta",
        _comparison_delta(comparison, "summary_ratio_means", "identity_fragmentation_reduction_ratio"),
        ">=",
        t.min_identity_fragmentation_reduction_delta,
        "Higher is better.",
        required=False,
    )
    _add_check(
        "decode_overhead_ratio_delta",
        _comparison_delta(comparison, "summary_ratio_means", "decode_overhead_ratio"),
        "<=",
        t.max_decode_overhead_ratio_delta,
        "Lower is better.",
        required=False,
    )
    _add_check(
        "decode_before_after_size_ratio_delta",
        _comparison_delta(comparison, "summary_ratio_means", "decode_before_after_size_ratio"),
        "<=",
        t.max_decode_before_after_size_ratio_delta,
        "Lower is better.",
        required=False,
    )

    return {
        "passed": all(bool(c["passed"]) for c in checks),
        "checks": checks,
    }


def collect_manifest_paths(corpus_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for root in corpus_dirs:
        resolved = root.expanduser().resolve()
        if not resolved.exists():
            continue
        for candidate in sorted(resolved.rglob("*.json")):
            p = str(candidate.resolve())
            if p in seen:
                continue
            seen.add(p)
            paths.append(candidate.resolve())
    return paths


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_output_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(".tmp") / "phase1-benchmark" / ts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase1 benchmark report(s) from corpus directories and optional baseline gate checks."
    )
    parser.add_argument(
        "--current-corpus-dir",
        action="append",
        required=True,
        type=Path,
        help="Directory containing current-run Phase1 manifest/bundle JSON files. Can be provided multiple times.",
    )
    parser.add_argument(
        "--baseline-corpus-dir",
        action="append",
        default=[],
        type=Path,
        help="Optional baseline corpus directory; if provided a baseline report is generated and compared.",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=None,
        help="Optional prebuilt baseline benchmark report JSON. Overrides --baseline-corpus-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated report artifacts (default: timestamped .tmp path).",
    )
    parser.add_argument("--min-assignment-coverage-delta", type=float, default=0.0)
    parser.add_argument("--min-with-scored-candidate-ratio-delta", type=float, default=0.0)
    parser.add_argument("--max-unknown-rate-delta", type=float, default=0.0)
    parser.add_argument("--min-overlap-consistency-delta", type=float, default=0.0)
    parser.add_argument("--max-total-wallclock-ms-delta", type=float, default=0.0)
    parser.add_argument("--min-canonical-face-coverage-delta", type=float, default=0.0)
    parser.add_argument("--min-fragmentation-reduction-delta", type=float, default=0.0)
    parser.add_argument("--max-decode-overhead-ratio-delta", type=float, default=0.0)
    parser.add_argument("--max-decode-before-after-size-ratio-delta", type=float, default=0.0)
    args = parser.parse_args(argv)

    out_dir = (args.output_dir or _default_output_dir()).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    current_paths = collect_manifest_paths(list(args.current_corpus_dir))
    if not current_paths:
        print("No JSON manifests found in --current-corpus-dir", file=sys.stderr)
        return 2

    current_report = build_benchmark_report(current_paths)
    current_report_path = out_dir / "current_report.json"
    _write_json(current_report_path, current_report)

    summary: dict[str, Any] = {
        "output_dir": str(out_dir),
        "current_report_path": str(current_report_path),
        "baseline_report_path": None,
        "comparison_path": None,
        "gate_path": None,
        "gate_passed": None,
    }

    if args.baseline_report is not None or args.baseline_corpus_dir:
        if args.baseline_report is not None:
            baseline_report = load_benchmark_report(args.baseline_report.expanduser().resolve())
        else:
            baseline_paths = collect_manifest_paths(list(args.baseline_corpus_dir))
            if not baseline_paths:
                print("No JSON manifests found in --baseline-corpus-dir", file=sys.stderr)
                return 2
            baseline_report = build_benchmark_report(baseline_paths)
            baseline_report_path = out_dir / "baseline_report.json"
            _write_json(baseline_report_path, baseline_report)
            summary["baseline_report_path"] = str(baseline_report_path)

        if summary["baseline_report_path"] is None and args.baseline_report is not None:
            summary["baseline_report_path"] = str(args.baseline_report.expanduser().resolve())

        comparison = compare_benchmark_reports(current_report, baseline_report)
        comparison_path = out_dir / "comparison_report.json"
        _write_json(comparison_path, comparison)
        summary["comparison_path"] = str(comparison_path)

        gate = evaluate_wave4_exit_gates(
            comparison,
            thresholds=Wave4GateThresholds(
                min_assignment_coverage_delta=float(args.min_assignment_coverage_delta),
                min_with_scored_candidate_ratio_delta=float(args.min_with_scored_candidate_ratio_delta),
                max_unknown_rate_delta=float(args.max_unknown_rate_delta),
                min_overlap_camera_consistency_delta=float(args.min_overlap_consistency_delta),
                max_total_wallclock_ms_delta=float(args.max_total_wallclock_ms_delta),
                min_canonical_face_stream_coverage_delta=float(args.min_canonical_face_coverage_delta),
                min_identity_fragmentation_reduction_delta=float(args.min_fragmentation_reduction_delta),
                max_decode_overhead_ratio_delta=float(args.max_decode_overhead_ratio_delta),
                max_decode_before_after_size_ratio_delta=float(
                    args.max_decode_before_after_size_ratio_delta
                ),
            ),
        )
        gate_path = out_dir / "wave4_gate.json"
        _write_json(gate_path, gate)
        summary["gate_path"] = str(gate_path)
        summary["gate_passed"] = bool(gate["passed"])

    summary_path = out_dir / "summary.json"
    _write_json(summary_path, summary)
    print(str(summary_path))

    if current_report["summary"]["failed_clip_count"] > 0:
        return 2
    if summary["gate_passed"] is False:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
