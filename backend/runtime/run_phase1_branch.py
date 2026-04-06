from __future__ import annotations

import argparse
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.phase1_runtime.branch_io import (
    build_branch_paths,
    read_branch_request,
    read_branch_status,
    write_branch_result,
    write_branch_status,
)
from backend.phase1_runtime.branch_models import BranchKind, BranchResultEnvelope, BranchStatus
from backend.phase1_runtime.branches.audio_branch import run_audio_branch
from backend.phase1_runtime.branches.visual_branch import run_visual_branch
from backend.phase1_runtime.branches.yamnet_branch import run_yamnet_branch
from backend.phase1_runtime.models import Phase1Workspace
from backend.phase1_runtime.visual import SimpleVisualExtractor
from backend.phase1_runtime.visual_config import VisualPipelineConfig
from backend.providers import ForcedAlignmentProvider, VibeVoiceASRProvider
from backend.providers.emotion2vec import Emotion2VecPlusProvider
from backend.providers.yamnet import YAMNetProvider

logger = logging.getLogger(__name__)
UTC = timezone.utc


def _now() -> str:
    return datetime.now(UTC).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single Phase 1 branch worker.")
    parser.add_argument("--request-path", required=True, help="Path to branch request.json")
    return parser


def build_branch_dependencies(*, request, workspace: Phase1Workspace) -> dict[str, Any]:
    dependencies: dict[str, Any] = {}
    if request.branch == BranchKind.VISUAL:
        dependencies["visual_extractor"] = SimpleVisualExtractor(
            visual_config=VisualPipelineConfig.from_env()
        )
    elif request.branch == BranchKind.AUDIO:
        dependencies["vibevoice_provider"] = VibeVoiceASRProvider()
        dependencies["forced_aligner"] = ForcedAlignmentProvider()
        dependencies["emotion_provider"] = Emotion2VecPlusProvider()
    elif request.branch == BranchKind.YAMNET:
        dependencies["yamnet_provider"] = YAMNetProvider(device="cpu")
    else:  # pragma: no cover - exhaustive guard
        raise ValueError(f"Unsupported branch: {request.branch}")
    return dependencies


def _write_status(
    *,
    paths,
    branch: BranchKind,
    state: str,
    message: str | None = None,
    pid: int | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> None:
    write_branch_status(
        paths.status_path,
        BranchStatus(
            branch=branch,
            state=state,  # type: ignore[arg-type]
            message=message,
            pid=pid,
            started_at=started_at if started_at is not None else (_now() if state == "running" else None),
            updated_at=_now(),
            completed_at=completed_at if completed_at is not None else (_now() if state in {"succeeded", "failed"} else None),
        ),
    )


def _run_requested_branch(*, request, workspace: Phase1Workspace, dependencies: dict[str, Any]) -> dict[str, Any]:
    if request.branch == BranchKind.VISUAL:
        return run_visual_branch(
            workspace=workspace,
            visual_extractor=dependencies["visual_extractor"],
        )
    if request.branch == BranchKind.AUDIO:
        return run_audio_branch(
            request=request,
            workspace=workspace,
            vibevoice_provider=dependencies["vibevoice_provider"],
            forced_aligner=dependencies["forced_aligner"],
            emotion_provider=dependencies["emotion_provider"],
        )
    if request.branch == BranchKind.YAMNET:
        return run_yamnet_branch(
            workspace=workspace,
            yamnet_provider=dependencies["yamnet_provider"],
        )
    raise ValueError(f"Unsupported branch: {request.branch}")


def run_branch_request(*, request_path: Path) -> int:
    request = read_branch_request(request_path)
    branch_root = request_path.parents[2]
    workspace = Phase1Workspace.create(root=branch_root.parent, run_id=branch_root.name)
    run_root = branch_root
    paths = build_branch_paths(run_root=run_root, branch=request.branch)

    _write_status(paths=paths, branch=request.branch, state="running", message="branch running")
    started_at = read_branch_status(paths.status_path).started_at

    try:
        dependencies = build_branch_dependencies(request=request, workspace=workspace)
        result = _run_requested_branch(request=request, workspace=workspace, dependencies=dependencies)
    except Exception as exc:
        _write_status(
            paths=paths,
            branch=request.branch,
            state="failed",
            message=str(exc) or type(exc).__name__,
            pid=os.getpid(),
            started_at=started_at,
        )
        envelope = BranchResultEnvelope(
            branch=request.branch,
            ok=False,
            error={
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "branch": request.branch.value,
                "branch_log_path": str(paths.log_path),
            },
        )
        write_branch_result(paths.result_path, envelope)
        logger.exception("Phase 1 branch %s failed", request.branch.value)
        return 1

    envelope = BranchResultEnvelope(branch=request.branch, ok=True, result=result)
    write_branch_result(paths.result_path, envelope)
    _write_status(
        paths=paths,
        branch=request.branch,
        state="succeeded",
        message="branch succeeded",
        started_at=started_at,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_branch_request(request_path=Path(args.request_path))


if __name__ == "__main__":
    raise SystemExit(main())
