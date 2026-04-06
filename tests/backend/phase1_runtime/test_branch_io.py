from __future__ import annotations

from backend.phase1_runtime.branch_io import (
    BranchPaths,
    build_branch_paths,
    read_branch_request,
    read_branch_result,
    read_branch_status,
    write_branch_request,
    write_branch_result,
    write_branch_status,
)
from backend.phase1_runtime.branch_models import BranchKind, BranchRequest, BranchResultEnvelope, BranchStatus
import pytest


def test_build_branch_paths_creates_stable_layout(tmp_path):
    paths = build_branch_paths(run_root=tmp_path, branch=BranchKind.VISUAL)

    assert isinstance(paths, BranchPaths)
    assert paths.branch_root.exists()
    assert paths.request_path.name == "request.json"
    assert paths.result_path.name == "result.json"
    assert paths.status_path.name == "status.json"
    assert paths.log_path.name == "branch.log"


def test_branch_request_round_trip(tmp_path):
    paths = build_branch_paths(run_root=tmp_path, branch=BranchKind.AUDIO)
    request = BranchRequest(
        job_id="job_123",
        run_id="run_123",
        branch=BranchKind.AUDIO,
        source_url="https://youtube.com/watch?v=test",
        runtime_controls={"run_phase14": True},
    )

    write_branch_request(paths.request_path, request)

    loaded = read_branch_request(paths.request_path)
    assert loaded == request


def test_branch_request_rejects_missing_source():
    with pytest.raises(ValueError, match="exactly one of source_url or source_path"):
        BranchRequest(
            job_id="job_123",
            run_id="run_123",
            branch=BranchKind.AUDIO,
        )


def test_branch_request_rejects_both_sources():
    with pytest.raises(ValueError, match="exactly one of source_url or source_path"):
        BranchRequest(
            job_id="job_123",
            run_id="run_123",
            branch=BranchKind.AUDIO,
            source_url="https://youtube.com/watch?v=test",
            source_path="/tmp/source.mp4",
        )


def test_branch_result_envelope_round_trip(tmp_path):
    paths = build_branch_paths(run_root=tmp_path, branch=BranchKind.YAMNET)
    result = BranchResultEnvelope(
        branch=BranchKind.YAMNET,
        result={"events": [{"label": "music", "score": 0.9}]},
    )

    write_branch_result(paths.result_path, result)

    loaded = read_branch_result(paths.result_path)
    assert loaded == result


def test_branch_result_envelope_rejects_error_when_ok():
    with pytest.raises(ValueError, match="ok=True requires result and forbids error"):
        BranchResultEnvelope(
            branch=BranchKind.YAMNET,
            ok=True,
            result={"events": []},
            error={"message": "boom"},
        )


def test_branch_result_envelope_rejects_missing_error_on_failure():
    with pytest.raises(ValueError, match="ok=False requires error and forbids result"):
        BranchResultEnvelope(
            branch=BranchKind.YAMNET,
            ok=False,
            result={"events": []},
        )


def test_branch_status_round_trip(tmp_path):
    paths = build_branch_paths(run_root=tmp_path, branch=BranchKind.VISUAL)
    status = BranchStatus(
        branch=BranchKind.VISUAL,
        state="running",
        message="branch is running",
        pid=4321,
    )

    write_branch_status(paths.status_path, status)

    loaded = read_branch_status(paths.status_path)
    assert loaded == status
