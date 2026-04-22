#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.phase1_runtime.models import Phase1Workspace
from backend.phase1_runtime.visual import V31VisualExtractor
from backend.phase1_runtime.visual_config import VisualPipelineConfig


def _parse_batch_sizes(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one batch size is required.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase1 RF-DETR + ByteTrack visual extraction only."
    )
    parser.add_argument("--video-path", required=True)
    parser.add_argument(
        "--work-root",
        default="backend/outputs/tmp_visual_only",
        help="Workspace root for temporary visual-only runs.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run id. Defaults to a timestamp-based value.",
    )
    parser.add_argument(
        "--benchmark-batches",
        type=_parse_batch_sizes,
        default=None,
        help="Comma-separated static TensorRT batch sizes to benchmark, e.g. 16,24,32.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the JSON summary.",
    )
    return parser


def _run_visual_once(*, video_path: Path, work_root: Path, run_id: str, config: VisualPipelineConfig) -> dict:
    workspace = Phase1Workspace.create(root=work_root, run_id=run_id)
    workspace.video_path.parent.mkdir(parents=True, exist_ok=True)
    if workspace.video_path != video_path:
        workspace.video_path.unlink(missing_ok=True)
        workspace.video_path.symlink_to(video_path)

    extractor = V31VisualExtractor(visual_config=config)
    started = time.perf_counter()
    payload = extractor.extract(video_path=video_path, workspace=workspace)
    wall_ms = (time.perf_counter() - started) * 1000.0
    shot_changes = list(payload.get("shot_changes") or [])
    tracking_metrics = dict(payload.get("tracking_metrics") or {})
    return {
        "run_id": run_id,
        "batch_size": config.detector_batch_size,
        "wall_ms": round(wall_ms, 1),
        "tracking_metrics": tracking_metrics,
        "track_rows": len(payload["tracks"]),
        "person_segments": len(payload["person_detections"]),
        "shot_count": int(tracking_metrics.get("shot_count", len(shot_changes))),
    }


def main() -> int:
    args = build_parser().parse_args()
    video_path = Path(args.video_path).expanduser().resolve()
    work_root = Path(args.work_root).expanduser().resolve()
    if not video_path.exists():
        raise SystemExit(f"Video path does not exist: {video_path}")

    base_config = VisualPipelineConfig.from_env()
    batch_sizes = args.benchmark_batches or (base_config.detector_batch_size,)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    results = []
    for batch_size in batch_sizes:
        config = replace(base_config, detector_batch_size=batch_size)
        run_id = args.run_id or f"tmp_rfdetr_visual_{batch_size}_{timestamp}"
        if len(batch_sizes) > 1 and args.run_id is not None:
            run_id = f"{args.run_id}_b{batch_size}"
        results.append(
            _run_visual_once(
                video_path=video_path,
                work_root=work_root,
                run_id=run_id,
                config=config,
            )
        )

    payload = {
        "video_path": str(video_path),
        "benchmark_batch_sizes": list(batch_sizes),
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
