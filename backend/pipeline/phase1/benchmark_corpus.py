"""Load Phase 1 manifest JSON files, compute scorecards, aggregate benchmark reports.

Standalone from workers: uses only ``metrics_scorecard.compute_phase1_scorecard``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from backend.pipeline.phase1.metrics_scorecard import compute_phase1_scorecard

BENCHMARK_REPORT_VERSION = 1
BENCHMARK_COMPARISON_VERSION = 1

_RATIO_SUMMARY_KEYS = (
    "assignment_coverage",
    "with_scored_candidate_ratio",
    "unknown_rate",
    "overlap_camera_consistency_ratio",
)
_WALLCLOCK_MS_KEYS = ("ingest_ms", "processing_ms", "upload_ms", "total_ms")


def _as_mapping(v: Any) -> dict[str, Any] | None:
    return v if isinstance(v, Mapping) else None


def extract_ledgers_from_payload(data: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str | None, dict[str, int]]:
    """Return ``(phase_1_audio, phase_1_visual, job_id, job_timings_ms)`` from a JSON root object.

    Supports:

    - Full Phase 1 manifest: ``artifacts.transcript``, ``artifacts.visual_tracking``,
      ``metadata.timings``, ``job_id``.
    - Loose bundle: ``phase_1_audio``, ``phase_1_visual``, optional ``job_id``,
      ``metadata.timings`` or top-level ``timings``.
    """
    job_id = data.get("job_id")
    job_id_str = str(job_id) if job_id is not None else None

    artifacts = _as_mapping(data.get("artifacts"))
    if artifacts is not None:
        tr = _as_mapping(artifacts.get("transcript"))
        vis = _as_mapping(artifacts.get("visual_tracking"))
        if tr is None or vis is None:
            raise ValueError("manifest artifacts must include transcript and visual_tracking objects")
        meta = _as_mapping(data.get("metadata")) or {}
        timings = _timings_to_job_ms(_as_mapping(meta.get("timings")))
        return dict(tr), dict(vis), job_id_str, timings

    audio = _as_mapping(data.get("phase_1_audio"))
    visual = _as_mapping(data.get("phase_1_visual"))
    if audio is not None and visual is not None:
        meta = _as_mapping(data.get("metadata")) or {}
        t1 = _timings_to_job_ms(_as_mapping(meta.get("timings")))
        t2 = _timings_to_job_ms(_as_mapping(data.get("timings")))
        timings = {**t2, **t1}
        return dict(audio), dict(visual), job_id_str, timings

    raise ValueError(
        "expected Phase 1 manifest (artifacts.transcript + artifacts.visual_tracking) "
        "or bundle (phase_1_audio + phase_1_visual)"
    )


def _timings_to_job_ms(timings: Mapping[str, Any] | None) -> dict[str, int]:
    if not timings:
        return {}
    out: dict[str, int] = {}
    for key in ("ingest_ms", "processing_ms", "upload_ms"):
        raw = timings.get(key)
        if raw is None:
            continue
        try:
            out[key] = int(max(0, int(raw)))
        except (TypeError, ValueError):
            continue
    return out


def _nullable_float_stats(values: list[float | None]) -> dict[str, Any]:
    present = [v for v in values if v is not None]
    if not present:
        return {"mean": None, "min": None, "max": None, "n": 0}
    return {
        "mean": sum(present) / len(present),
        "min": min(present),
        "max": max(present),
        "n": len(present),
    }


def _float_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "min": None, "max": None, "n": 0}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def load_json_manifest_path(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def scorecard_for_payload(data: Mapping[str, Any]) -> dict[str, Any]:
    audio, visual, _job_id, timings = extract_ledgers_from_payload(data)
    return compute_phase1_scorecard(audio, visual, job_timings_ms=timings or None)


def aggregate_scorecard_summary(scorecards: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary over successful per-clip scorecards (nullable ratios + timings)."""
    assign = [sc.get("assignment_coverage") for sc in scorecards]
    assign_typed: list[float | None] = [float(x) if isinstance(x, (int, float)) else None for x in assign]

    scored = [sc.get("with_scored_candidate_ratio") for sc in scorecards]
    scored_typed: list[float | None] = [float(x) if isinstance(x, (int, float)) else None for x in scored]

    unknown = [sc.get("unknown_rate") for sc in scorecards]
    unknown_typed: list[float | None] = [float(x) if isinstance(x, (int, float)) else None for x in unknown]

    overlap = [sc.get("overlap_camera_consistency_ratio") for sc in scorecards]
    overlap_typed: list[float | None] = [float(x) if isinstance(x, (int, float)) else None for x in overlap]

    ingest: list[float] = []
    proc: list[float] = []
    upload: list[float] = []
    total: list[float] = []
    for sc in scorecards:
        wc = sc.get("wallclock_ms")
        if not isinstance(wc, Mapping):
            continue
        for key, bucket in (
            ("ingest_ms", ingest),
            ("processing_ms", proc),
            ("upload_ms", upload),
            ("total_ms", total),
        ):
            v = wc.get(key)
            if isinstance(v, (int, float)):
                bucket.append(float(v))

    stage_keys: set[str] = set()
    for sc in scorecards:
        sw = sc.get("stage_wallclock_s")
        if isinstance(sw, Mapping):
            stage_keys.update(str(k) for k in sw.keys())

    stage_summary: dict[str, Any] = {}
    for sk in sorted(stage_keys):
        vals: list[float] = []
        for sc in scorecards:
            sw = sc.get("stage_wallclock_s")
            if not isinstance(sw, Mapping):
                continue
            v = sw.get(sk)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        stage_summary[sk] = _float_stats(vals)

    return {
        "assignment_coverage": _nullable_float_stats(assign_typed),
        "with_scored_candidate_ratio": _nullable_float_stats(scored_typed),
        "unknown_rate": _nullable_float_stats(unknown_typed),
        "overlap_camera_consistency_ratio": _nullable_float_stats(overlap_typed),
        "wallclock_ms": {
            "ingest_ms": _float_stats(ingest),
            "processing_ms": _float_stats(proc),
            "upload_ms": _float_stats(upload),
            "total_ms": _float_stats(total),
        },
        "stage_wallclock_s": stage_summary,
    }


def _safe_float_metric(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _delta_triplet(baseline_val: Any, current_val: Any) -> dict[str, Any]:
    b = _safe_float_metric(baseline_val)
    c = _safe_float_metric(current_val)
    delta: float | None = None
    if b is not None and c is not None:
        delta = c - b
    return {"baseline": b, "current": c, "delta": delta}


def _mean_from_summary_block(summary: Mapping[str, Any], key: str) -> float | None:
    block = summary.get(key)
    if not isinstance(block, Mapping):
        return None
    return _safe_float_metric(block.get("mean"))


def _wallclock_mean_from_summary(summary: Mapping[str, Any], subkey: str) -> float | None:
    wc = summary.get("wallclock_ms")
    if not isinstance(wc, Mapping):
        return None
    inner = wc.get(subkey)
    if not isinstance(inner, Mapping):
        return None
    return _safe_float_metric(inner.get("mean"))


def _stage_means_from_summary(summary: Mapping[str, Any]) -> dict[str, float | None]:
    sw = summary.get("stage_wallclock_s")
    if not isinstance(sw, Mapping):
        return {}
    out: dict[str, float | None] = {}
    for sk, block in sw.items():
        if isinstance(block, Mapping):
            out[str(sk)] = _safe_float_metric(block.get("mean"))
    return out


def load_benchmark_report(path: Path) -> dict[str, Any]:
    """Load a benchmark report JSON written by :func:`build_benchmark_report`."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("benchmark report root must be a JSON object")
    return data


def compare_benchmark_reports(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two benchmark reports (current minus baseline).

    Produces aggregate summary deltas (means) and a per-clip table matched by
    ``job_id`` when both sides set it, otherwise by ``source_path``.
    """
    b_clips = baseline.get("clips")
    c_clips = current.get("clips")
    if not isinstance(b_clips, list) or not isinstance(c_clips, list):
        raise ValueError("both reports must have a 'clips' array")

    b_summary = baseline.get("summary")
    c_summary = current.get("summary")
    if not isinstance(b_summary, Mapping) or not isinstance(c_summary, Mapping):
        raise ValueError("both reports must have a 'summary' object")

    aggregate_ratios: dict[str, Any] = {}
    for key in _RATIO_SUMMARY_KEYS:
        bm = _mean_from_summary_block(b_summary, key)
        cm = _mean_from_summary_block(c_summary, key)
        aggregate_ratios[key] = _delta_triplet(bm, cm)

    wallclock_agg: dict[str, Any] = {}
    for wk in _WALLCLOCK_MS_KEYS:
        bm = _wallclock_mean_from_summary(b_summary, wk)
        cm = _wallclock_mean_from_summary(c_summary, wk)
        wallclock_agg[wk] = _delta_triplet(bm, cm)

    b_stages = _stage_means_from_summary(b_summary)
    c_stages = _stage_means_from_summary(c_summary)
    stage_keys = sorted(set(b_stages) | set(c_stages))
    stage_agg: dict[str, Any] = {}
    for sk in stage_keys:
        stage_agg[sk] = _delta_triplet(b_stages.get(sk), c_stages.get(sk))

    # --- per-clip matching ---
    def _clip_row(raw: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "source_path": raw.get("source_path"),
            "job_id": raw.get("job_id"),
            "scorecard": raw.get("scorecard"),
        }

    baseline_rows = [_clip_row(x) for x in b_clips if isinstance(x, Mapping)]
    current_rows = [_clip_row(x) for x in c_clips if isinstance(x, Mapping)]

    unused_baseline: set[int] = set(range(len(baseline_rows)))

    def _by_job() -> dict[str, int]:
        m: dict[str, int] = {}
        for i, row in enumerate(baseline_rows):
            jid = row.get("job_id")
            if jid is not None and str(jid).strip() != "":
                sj = str(jid)
                if sj not in m:
                    m[sj] = i
        return m

    def _by_path() -> dict[str, int]:
        m: dict[str, int] = {}
        for i, row in enumerate(baseline_rows):
            p = row.get("source_path")
            if isinstance(p, str) and p:
                if p not in m:
                    m[p] = i
        return m

    job_map = _by_job()
    path_map = _by_path()

    def _scorecard_metric_keys(sc: Mapping[str, Any] | None) -> dict[str, Any]:
        if not isinstance(sc, Mapping):
            return {}
        out: dict[str, Any] = {}
        for k in _RATIO_SUMMARY_KEYS:
            out[k] = sc.get(k)
        wc = sc.get("wallclock_ms")
        if isinstance(wc, Mapping):
            for wk in _WALLCLOCK_MS_KEYS:
                out[f"wallclock_ms.{wk}"] = wc.get(wk)
        sw = sc.get("stage_wallclock_s")
        if isinstance(sw, Mapping):
            for sk, sv in sw.items():
                out[f"stage_wallclock_s.{str(sk)}"] = sv
        return out

    def _merge_metric_keys(bsc: Mapping[str, Any] | None, csc: Mapping[str, Any] | None) -> list[str]:
        kb = set(_scorecard_metric_keys(bsc).keys())
        kc = set(_scorecard_metric_keys(csc).keys())
        # Stable: ratios first, then wallclock, then stages sorted
        ordered: list[str] = []
        for k in _RATIO_SUMMARY_KEYS:
            if k in kb or k in kc:
                ordered.append(k)
        for wk in _WALLCLOCK_MS_KEYS:
            kk = f"wallclock_ms.{wk}"
            if kk in kb or kk in kc:
                ordered.append(kk)
        stage_suffixes = sorted(
            {k.replace("stage_wallclock_s.", "", 1) for k in (kb | kc) if k.startswith("stage_wallclock_s.")}
        )
        for suf in stage_suffixes:
            kk = f"stage_wallclock_s.{suf}"
            if kk in kb or kk in kc:
                ordered.append(kk)
        return ordered

    per_clip: list[dict[str, Any]] = []

    for cur in current_rows:
        c_path = cur["source_path"] if isinstance(cur["source_path"], str) else None
        c_jid = cur["job_id"]
        c_jid_s = str(c_jid) if c_jid is not None and str(c_jid).strip() != "" else None
        c_sc = cur["scorecard"] if isinstance(cur["scorecard"], Mapping) else None

        bi: int | None = None
        basis: str | None = None
        if c_jid_s is not None and c_jid_s in job_map:
            cand = job_map[c_jid_s]
            if cand in unused_baseline:
                bi = cand
                basis = "job_id"
        if bi is None and c_path and c_path in path_map:
            cand = path_map[c_path]
            if cand in unused_baseline:
                bi = cand
                basis = "source_path"

        row: dict[str, Any] = {
            "match_basis": basis or "unmatched_current",
            "job_id": c_jid_s,
            "current_source_path": c_path,
            "baseline_source_path": None,
            "metrics": {},
        }

        if bi is not None:
            unused_baseline.discard(bi)
            base = baseline_rows[bi]
            b_path = base["source_path"] if isinstance(base["source_path"], str) else None
            b_sc = base["scorecard"] if isinstance(base["scorecard"], Mapping) else None
            row["baseline_source_path"] = b_path
            if row["job_id"] is None and base.get("job_id") is not None:
                row["job_id"] = str(base["job_id"])
            for mk in _merge_metric_keys(b_sc, c_sc):
                bv = _scorecard_metric_keys(b_sc).get(mk)
                cv = _scorecard_metric_keys(c_sc).get(mk)
                row["metrics"][mk] = _delta_triplet(bv, cv)
        else:
            for mk in _merge_metric_keys(None, c_sc):
                cv = _scorecard_metric_keys(c_sc).get(mk)
                row["metrics"][mk] = _delta_triplet(None, cv)

        per_clip.append(row)

    for bi in sorted(unused_baseline):
        base = baseline_rows[bi]
        b_path = base["source_path"] if isinstance(base["source_path"], str) else None
        b_jid = base["job_id"]
        b_jid_s = str(b_jid) if b_jid is not None and str(b_jid).strip() != "" else None
        b_sc = base["scorecard"] if isinstance(base["scorecard"], Mapping) else None
        row: dict[str, Any] = {
            "match_basis": "baseline_only",
            "job_id": b_jid_s,
            "current_source_path": None,
            "baseline_source_path": b_path,
            "metrics": {},
        }
        for mk in _merge_metric_keys(b_sc, None):
            bv = _scorecard_metric_keys(b_sc).get(mk)
            row["metrics"][mk] = _delta_triplet(bv, None)
        per_clip.append(row)

    return {
        "version": BENCHMARK_COMPARISON_VERSION,
        "baseline_report_version": baseline.get("version"),
        "current_report_version": current.get("version"),
        "baseline_generated_at": baseline.get("generated_at"),
        "current_generated_at": current.get("generated_at"),
        "aggregate": {
            "summary_ratio_means": aggregate_ratios,
            "wallclock_ms_means": wallclock_agg,
            "stage_wallclock_s_means": stage_agg,
        },
        "per_clip": per_clip,
    }


def build_benchmark_report(
    manifest_paths: list[Path],
    *,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Load each JSON path, compute a scorecard, return a benchmark report dict."""
    ts = now_utc or datetime.now(timezone.utc)
    clips_out: list[dict[str, Any]] = []
    successful_cards: list[dict[str, Any]] = []

    for p in manifest_paths:
        resolved = p.resolve()
        entry: dict[str, Any] = {
            "source_path": str(resolved),
            "job_id": None,
            "scorecard": None,
            "error": None,
        }
        try:
            raw = load_json_manifest_path(resolved)
            audio, visual, job_id, timings = extract_ledgers_from_payload(raw)
            entry["job_id"] = job_id
            card = compute_phase1_scorecard(audio, visual, job_timings_ms=timings or None)
            entry["scorecard"] = card
            successful_cards.append(card)
        except Exception as exc:  # noqa: BLE001 — surface per-file errors in the report
            entry["error"] = f"{type(exc).__name__}: {exc}"

        clips_out.append(entry)

    summary = aggregate_scorecard_summary(successful_cards)
    summary["clip_count"] = len(clips_out)
    summary["successful_clip_count"] = len(successful_cards)
    summary["failed_clip_count"] = len(clips_out) - len(successful_cards)

    return {
        "version": BENCHMARK_REPORT_VERSION,
        "generated_at": ts.isoformat().replace("+00:00", "Z"),
        "clips": clips_out,
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1 benchmark corpus: scorecards from manifest JSON files.")
    parser.add_argument(
        "manifests",
        nargs="+",
        type=Path,
        help="One or more local JSON files (Phase 1 manifest or audio/visual bundle).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write benchmark report JSON to this path (default: stdout).",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=None,
        help="Optional frozen benchmark report JSON to diff against the report built from manifests.",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=None,
        help="Write benchmark-vs-baseline comparison JSON here. "
        "If omitted with --baseline-report, defaults to <output-stem>.comparison.json next to -o/--output.",
    )
    args = parser.parse_args(argv)

    report = build_benchmark_report(list(args.manifests))
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    if args.baseline_report is not None:
        baseline_path = args.baseline_report.expanduser().resolve()
        baseline_data = load_benchmark_report(baseline_path)
        comparison = compare_benchmark_reports(report, baseline_data)
        comp_text = json.dumps(comparison, indent=2, sort_keys=True) + "\n"
        comp_out = args.comparison_output
        if comp_out is None:
            if args.output is None:
                print(
                    "benchmark_corpus: --baseline-report requires --comparison-output when report goes to stdout",
                    file=sys.stderr,
                )
                return 3
            comp_out = args.output.with_name(args.output.stem + ".comparison.json")
        comp_out = comp_out.expanduser().resolve()
        comp_out.parent.mkdir(parents=True, exist_ok=True)
        comp_out.write_text(comp_text, encoding="utf-8")

    return 0 if report["summary"]["failed_clip_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
