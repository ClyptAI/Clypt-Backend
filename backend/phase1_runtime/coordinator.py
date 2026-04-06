from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass

from .branch_io import build_branch_paths, read_branch_result, write_branch_request
from .branch_models import BranchKind, BranchRequest, BranchResultEnvelope
from .models import Phase1SidecarOutputs, Phase1Workspace


@dataclass(slots=True)
class _RunningBranch:
    branch: BranchKind
    request: BranchRequest
    process: subprocess.Popen
    paths: object
    started_at: float
    log_handle: object


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
    return env


def _launch_branch(*, branch: BranchKind, source_url: str, workspace: Phase1Workspace) -> _RunningBranch:
    paths = build_branch_paths(run_root=workspace.root, branch=branch)
    request = _build_branch_request(branch=branch, source_url=source_url, workspace=workspace)
    write_branch_request(paths.request_path, request)
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
    except Exception:
        log_handle.close()
        raise
    return _RunningBranch(
        branch=branch,
        request=request,
        process=process,
        paths=paths,
        started_at=time.monotonic(),
        log_handle=log_handle,
    )


def _load_branch_result(running: _RunningBranch) -> BranchResultEnvelope:
    return read_branch_result(running.paths.result_path)


def _load_branch_result_or_raise(running: _RunningBranch) -> BranchResultEnvelope:
    try:
        return _load_branch_result(running)
    except Exception as exc:
        raise RuntimeError(
            f"Phase 1 branch {running.branch.value} exited successfully but result.json was missing or invalid"
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


def _close_branch_log_handle(running: _RunningBranch) -> None:
    if not running.log_handle.closed:
        running.log_handle.close()


def _cleanup_started_branches(running_branches: dict[BranchKind, _RunningBranch]) -> None:
    for running in running_branches.values():
        _terminate_branch(running)
        _close_branch_log_handle(running)


def _stop_siblings(running_branches: dict[BranchKind, _RunningBranch], *, failed_branch: BranchKind) -> None:
    for branch, running in running_branches.items():
        if branch is failed_branch:
            continue
        _terminate_branch(running)


def _branch_failure_error(running: _RunningBranch) -> RuntimeError:
    try:
        envelope = _load_branch_result(running)
    except Exception:
        return RuntimeError(f"Phase 1 branch {running.branch.value} failed")

    if envelope.ok:
        return RuntimeError(f"Phase 1 branch {running.branch.value} failed without an error envelope")
    error_message = str((envelope.error or {}).get("error_message") or "").strip()
    if error_message:
        return RuntimeError(error_message)
    return RuntimeError(f"Phase 1 branch {running.branch.value} failed")


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
    try:
        for branch in (BranchKind.VISUAL, BranchKind.AUDIO, BranchKind.YAMNET):
            running_branches[branch] = _launch_branch(
                branch=branch,
                source_url=source_url,
                workspace=workspace,
            )
    except Exception:
        _cleanup_started_branches(running_branches)
        raise

    branch_results: dict[BranchKind, BranchResultEnvelope] = {}
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
                        _stop_siblings(running_branches, failed_branch=branch)
                        _terminate_branch(running)
                        raise TimeoutError(
                            f"Phase 1 branch {branch.value} exceeded timeout of {branch_timeout_s:.1f}s"
                        )
                    continue
                if return_code != 0:
                    _stop_siblings(running_branches, failed_branch=branch)
                    raise _branch_failure_error(running)
                try:
                    branch_results[branch] = _load_branch_result_or_raise(running)
                except RuntimeError:
                    _stop_siblings(running_branches, failed_branch=branch)
                    raise
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
    finally:
        for running in running_branches.values():
            _close_branch_log_handle(running)


__all__ = ["run_phase1_sidecars_coordinator"]
