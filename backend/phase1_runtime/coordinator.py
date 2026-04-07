from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
import json
from pathlib import Path

from .branch_io import (
    build_branch_paths,
    read_branch_result,
    read_branch_status,
    write_branch_request,
    write_branch_status,
)
from .branch_models import BranchKind, BranchRequest, BranchResultEnvelope, BranchStatus
from .models import Phase1SidecarOutputs, Phase1Workspace


@dataclass(slots=True)
class _RunningBranch:
    branch: BranchKind
    request: BranchRequest
    process: subprocess.Popen
    paths: object
    started_at: float
    log_handle: object


class Phase1BranchFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        failing_branch: str,
        branch_log_path: str,
        summary_entry: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.failing_branch = failing_branch
        self.branch_log_path = branch_log_path
        self.summary_entry = summary_entry


class Phase1BranchTimeout(TimeoutError):
    def __init__(
        self,
        message: str,
        *,
        failing_branch: str,
        branch_log_path: str,
    ) -> None:
        super().__init__(message)
        self.failing_branch = failing_branch
        self.branch_log_path = branch_log_path


_YAMNET_STRIPPED_ENV_KEYS = {
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_ROOT",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_REQUIRE_CUDA",
    "GPU_DEVICE_ORDINAL",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
}


def _build_branch_request(
    *,
    branch: BranchKind,
    source_url: str,
    workspace: Phase1Workspace,
) -> BranchRequest:
    return BranchRequest(
        job_id=workspace.run_id,
        run_id=workspace.run_id,
        branch=branch,
        source_url=source_url,
    )


def _build_branch_env(*, branch: BranchKind) -> dict[str, str]:
    env = os.environ.copy()
    if branch is BranchKind.YAMNET:
        env = {key: value for key, value in env.items() if key not in _YAMNET_STRIPPED_ENV_KEYS}
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["NVIDIA_VISIBLE_DEVICES"] = "void"
    return env


def _launch_branch(*, branch: BranchKind, source_url: str, workspace: Phase1Workspace) -> _RunningBranch:
    paths = build_branch_paths(run_root=workspace.root, branch=branch)
    request = _build_branch_request(branch=branch, source_url=source_url, workspace=workspace)
    write_branch_request(paths.request_path, request)
    write_branch_status(
        paths.status_path,
        BranchStatus(
            branch=branch,
            state="queued",
            message="branch queued",
        ),
    )
    log_handle = paths.log_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "backend.runtime.run_phase1_branch",
                "--request-path",
                str(paths.request_path),
            ],
            env=_build_branch_env(branch=branch),
            stdout=log_handle,
            stderr=log_handle,
            text=True,
        )
    except Exception as exc:
        write_branch_status(
            paths.status_path,
            BranchStatus(
                branch=branch,
                state="failed",
                message=str(exc) or type(exc).__name__,
            ),
        )
        log_handle.write(f"Failed to launch branch {branch.value}: {exc}\n")
        log_handle.flush()
        log_handle.close()
        raise Phase1BranchFailure(
            str(exc) or f"Phase 1 branch {branch.value} failed to launch",
            failing_branch=branch.value,
            branch_log_path=str(paths.log_path),
            summary_entry={
                "request_path": str(paths.request_path),
                "status_path": str(paths.status_path),
                "result_path": str(paths.result_path),
                "log_path": str(paths.log_path),
                "pid": None,
                "status": read_branch_status(paths.status_path).model_dump(mode="json"),
            },
        ) from exc
    return _RunningBranch(
        branch=branch,
        request=request,
        process=process,
        paths=paths,
        started_at=time.monotonic(),
        log_handle=log_handle,
    )


def _branch_summary_path(workspace: Phase1Workspace) -> Path:
    return workspace.metadata_dir / "branch_summary.json"


def _read_status_or_none(running: _RunningBranch) -> dict[str, object] | None:
    try:
        return read_branch_status(running.paths.status_path).model_dump(mode="json")
    except Exception:
        return None


def _append_branch_log_line(running: _RunningBranch, message: str) -> None:
    running.log_handle.write(f"{message}\n")
    running.log_handle.flush()


def _record_branch_termination(
    termination_details: dict[BranchKind, dict[str, object]],
    *,
    running: _RunningBranch,
    reason: str,
) -> None:
    termination_details[running.branch] = {
        "reason": reason,
        "pid": running.process.pid,
        "branch": running.branch.value,
    }


def _write_branch_summary(
    *,
    workspace: Phase1Workspace,
    running_branches: dict[BranchKind, _RunningBranch],
    failed_branch: BranchKind | None = None,
    branch_results: dict[BranchKind, BranchResultEnvelope] | None = None,
    termination_details: dict[BranchKind, dict[str, object]] | None = None,
    extra_branches: dict[str, dict[str, object]] | None = None,
) -> None:
    summary = {
        "run_id": workspace.run_id,
        "failed_branch": failed_branch.value if failed_branch is not None else None,
        "branches": {},
    }
    for branch, running in running_branches.items():
        branch_entry: dict[str, object] = {
            "request_path": str(running.paths.request_path),
            "status_path": str(running.paths.status_path),
            "result_path": str(running.paths.result_path),
            "log_path": str(running.paths.log_path),
            "pid": running.process.pid,
            "status": _read_status_or_none(running),
        }
        if branch_results and branch in branch_results:
            branch_entry["result"] = branch_results[branch].model_dump(mode="json")
        if termination_details and branch in termination_details:
            branch_entry["termination"] = termination_details[branch]
        summary["branches"][branch.value] = branch_entry
    if extra_branches:
        for branch_name, branch_entry in extra_branches.items():
            summary["branches"].setdefault(branch_name, branch_entry)
    _branch_summary_path(workspace).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _load_branch_result(running: _RunningBranch) -> BranchResultEnvelope:
    return read_branch_result(running.paths.result_path)


def _load_branch_result_or_raise(running: _RunningBranch) -> BranchResultEnvelope:
    try:
        return _load_branch_result(running)
    except Exception as exc:
        raise Phase1BranchFailure(
            f"Phase 1 branch {running.branch.value} exited successfully but result.json was missing or invalid",
            failing_branch=running.branch.value,
            branch_log_path=str(running.paths.log_path),
        ) from exc


def _terminate_branch(running: _RunningBranch) -> None:
    if running.process.poll() is not None:
        return
    running.process.terminate()
    try:
        running.process.wait(timeout=1.0)
    except Exception:
        running.process.kill()
        try:
            running.process.wait(timeout=1.0)
        except Exception:
            pass


def _try_terminate_failed_branch(running: _RunningBranch) -> None:
    try:
        running.process.terminate()
    except Exception:
        pass


def _close_branch_log_handle(running: _RunningBranch) -> None:
    if not running.log_handle.closed:
        running.log_handle.close()


def _cleanup_started_branches(running_branches: dict[BranchKind, _RunningBranch]) -> None:
    for running in running_branches.values():
        _terminate_branch(running)
        _close_branch_log_handle(running)


def _cleanup_started_branches_with_details(
    running_branches: dict[BranchKind, _RunningBranch],
    *,
    failed_branch: BranchKind,
    termination_details: dict[BranchKind, dict[str, object]],
    reason: str,
) -> None:
    for branch, running in running_branches.items():
        if branch is failed_branch:
            _close_branch_log_handle(running)
            continue
        _record_branch_termination(
            termination_details,
            running=running,
            reason=reason,
        )
        _terminate_branch(running)
        _close_branch_log_handle(running)


def _stop_siblings(
    running_branches: dict[BranchKind, _RunningBranch],
    *,
    failed_branch: BranchKind,
) -> dict[BranchKind, dict[str, object]]:
    terminated: dict[BranchKind, dict[str, object]] = {}
    for branch, running in running_branches.items():
        if branch is failed_branch:
            continue
        if running.process.poll() is not None:
            continue
        message = (
            f"Coordinator terminating sibling branch {branch.value} "
            f"(pid={running.process.pid}) because {failed_branch.value} failed"
        )
        _append_branch_log_line(running, message)
        terminated[branch] = {
            "reason": f"{failed_branch.value} failed",
            "pid": running.process.pid,
            "branch": branch.value,
        }
        _terminate_branch(running)
    return terminated


def _branch_failure_error(running: _RunningBranch) -> RuntimeError:
    try:
        envelope = _load_branch_result(running)
    except Exception:
        return Phase1BranchFailure(
            f"Phase 1 branch {running.branch.value} failed",
            failing_branch=running.branch.value,
            branch_log_path=str(running.paths.log_path),
        )

    if envelope.ok:
        return Phase1BranchFailure(
            f"Phase 1 branch {running.branch.value} failed without an error envelope",
            failing_branch=running.branch.value,
            branch_log_path=str(running.paths.log_path),
        )
    error_message = str((envelope.error or {}).get("error_message") or "").strip()
    if error_message:
        return Phase1BranchFailure(
            error_message,
            failing_branch=running.branch.value,
            branch_log_path=str(
                (envelope.error or {}).get("branch_log_path") or running.paths.log_path
            ),
        )
    return Phase1BranchFailure(
        f"Phase 1 branch {running.branch.value} failed",
        failing_branch=running.branch.value,
        branch_log_path=str(running.paths.log_path),
    )


def _join_outputs(
    *,
    source_url: str,
    video_gcs_uri: str,
    workspace: Phase1Workspace,
    branch_results: dict[BranchKind, BranchResultEnvelope],
) -> Phase1SidecarOutputs:
    visual_result = branch_results[BranchKind.VISUAL].result or {}
    audio_result = branch_results[BranchKind.AUDIO].result or {}
    yamnet_result = branch_results[BranchKind.YAMNET].result or {}
    return Phase1SidecarOutputs(
        phase1_audio={
            "source_audio": source_url,
            "video_gcs_uri": video_gcs_uri,
            "local_video_path": str(workspace.video_path),
            "local_audio_path": str(workspace.audio_path),
        },
        diarization_payload=dict(audio_result["diarization_payload"]),
        phase1_visual=dict(visual_result["phase1_visual"]),
        emotion2vec_payload=dict(audio_result["emotion2vec_payload"]),
        yamnet_payload=dict(yamnet_result["yamnet_payload"]),
    )


def run_phase1_sidecars_coordinator(
    *,
    source_url: str,
    video_gcs_uri: str,
    workspace: Phase1Workspace,
    branch_timeout_s: float = 1800.0,
    poll_interval_s: float = 0.1,
) -> Phase1SidecarOutputs:
    running_branches: dict[BranchKind, _RunningBranch] = {}
    branch_results: dict[BranchKind, BranchResultEnvelope] = {}
    termination_details: dict[BranchKind, dict[str, object]] = {}
    failed_branch: BranchKind | None = None
    try:
        for branch in (BranchKind.VISUAL, BranchKind.AUDIO, BranchKind.YAMNET):
            running_branches[branch] = _launch_branch(
                branch=branch,
                source_url=source_url,
                workspace=workspace,
            )
        _write_branch_summary(workspace=workspace, running_branches=running_branches)
    except Phase1BranchFailure as exc:
        failed_branch = BranchKind(exc.failing_branch)
        _cleanup_started_branches_with_details(
            running_branches,
            failed_branch=failed_branch,
            termination_details=termination_details,
            reason=f"{failed_branch.value} failed to launch",
        )
        _write_branch_summary(
            workspace=workspace,
            running_branches=running_branches,
            failed_branch=failed_branch,
            branch_results=branch_results,
            termination_details=termination_details,
            extra_branches={exc.failing_branch: dict(exc.summary_entry or {})},
        )
        raise
    except Exception:
        _write_branch_summary(
            workspace=workspace,
            running_branches=running_branches,
            failed_branch=failed_branch,
            branch_results=branch_results,
            termination_details=termination_details,
        )
        _cleanup_started_branches(running_branches)
        raise

    try:
        while len(branch_results) < len(running_branches):
            now = time.monotonic()
            made_progress = False
            for branch, running in running_branches.items():
                if branch in branch_results:
                    continue
                return_code = running.process.poll()
                if return_code is None:
                    if now - running.started_at > branch_timeout_s:
                        termination_details.update(
                            _stop_siblings(running_branches, failed_branch=branch)
                        )
                        _terminate_branch(running)
                        _record_branch_termination(
                            termination_details,
                            running=running,
                            reason=f"{branch.value} timed out",
                        )
                        failed_branch = branch
                        raise Phase1BranchTimeout(
                            f"Phase 1 branch {branch.value} exceeded timeout of {branch_timeout_s:.1f}s",
                            failing_branch=branch.value,
                            branch_log_path=str(running.paths.log_path),
                        )
                    continue
                if return_code != 0:
                    _try_terminate_failed_branch(running)
                    _record_branch_termination(
                        termination_details,
                        running=running,
                        reason=f"{branch.value} failed",
                    )
                    termination_details.update(
                        _stop_siblings(running_branches, failed_branch=branch)
                    )
                    failed_branch = branch
                    raise _branch_failure_error(running)
                try:
                    branch_results[branch] = _load_branch_result_or_raise(running)
                except Phase1BranchFailure:
                    _record_branch_termination(
                        termination_details,
                        running=running,
                        reason=f"{branch.value} result invalid",
                    )
                    termination_details.update(
                        _stop_siblings(running_branches, failed_branch=branch)
                    )
                    failed_branch = branch
                    raise
                _write_branch_summary(
                    workspace=workspace,
                    running_branches=running_branches,
                    failed_branch=failed_branch,
                    branch_results=branch_results,
                    termination_details=termination_details,
                )
                made_progress = True
            if len(branch_results) == len(running_branches):
                break
            if not made_progress and poll_interval_s > 0:
                time.sleep(poll_interval_s)
        return _join_outputs(
            source_url=source_url,
            video_gcs_uri=video_gcs_uri,
            workspace=workspace,
            branch_results=branch_results,
        )
    except (Phase1BranchFailure, Phase1BranchTimeout):
        if failed_branch is None:
            raise
        _write_branch_summary(
            workspace=workspace,
            running_branches=running_branches,
            failed_branch=failed_branch,
            branch_results=branch_results,
            termination_details=termination_details,
        )
        raise
    finally:
        _write_branch_summary(
            workspace=workspace,
            running_branches=running_branches,
            failed_branch=failed_branch,
            branch_results=branch_results,
            termination_details=termination_details,
        )
        for running in running_branches.values():
            _close_branch_log_handle(running)


__all__ = ["Phase1BranchFailure", "Phase1BranchTimeout", "run_phase1_sidecars_coordinator"]
