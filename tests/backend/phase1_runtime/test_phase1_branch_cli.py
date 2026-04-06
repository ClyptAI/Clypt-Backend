from __future__ import annotations

import json
from pathlib import Path
import pytest


def test_branch_cli_dispatches_visual_branch_and_writes_result_files(tmp_path: Path, monkeypatch):
    from backend.phase1_runtime.branch_io import build_branch_paths, write_branch_request
    from backend.phase1_runtime.branch_models import BranchKind, BranchRequest
    import backend.runtime.run_phase1_branch as cli

    run_root = tmp_path / "runs" / "run_001"
    request = BranchRequest(
        job_id="job_123",
        run_id="run_001",
        branch=BranchKind.VISUAL,
        source_path="/tmp/source.mp4",
    )
    paths = build_branch_paths(run_root=run_root, branch=BranchKind.VISUAL)
    write_branch_request(paths.request_path, request)

    seen: dict[str, object] = {}

    class _FakeVisualExtractor:
        def extract(self, *, video_path: Path, workspace):
            seen["video_path"] = video_path.name
            seen["workspace_root"] = str(workspace.root)
            seen["video_path_full"] = str(workspace.video_path)
            return {"video_metadata": {"fps": 30.0}, "shot_changes": [], "tracks": [{"track_id": "t1"}]}

    monkeypatch.setattr(
        cli,
        "build_branch_dependencies",
        lambda request, workspace: {
            "visual_extractor": _FakeVisualExtractor(),
        },
    )

    exit_code = cli.main(["--request-path", str(paths.request_path)])

    assert exit_code == 0
    assert seen == {
        "video_path": "source_video.mp4",
        "workspace_root": str(run_root),
        "video_path_full": str(run_root / "source_video.mp4"),
    }
    status = json.loads(paths.status_path.read_text(encoding="utf-8"))
    assert status["state"] == "succeeded"
    assert status["started_at"] is not None
    assert status["completed_at"] is not None
    result = json.loads(paths.result_path.read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert result["result"]["phase1_visual"]["tracks"] == [{"track_id": "t1"}]


def test_branch_cli_exits_nonzero_and_writes_failure_status(tmp_path: Path, monkeypatch):
    from backend.phase1_runtime.branch_io import build_branch_paths, write_branch_request
    from backend.phase1_runtime.branch_models import BranchKind, BranchRequest
    import backend.runtime.run_phase1_branch as cli

    run_root = tmp_path / "runs" / "run_999"
    request = BranchRequest(
        job_id="job_999",
        run_id="run_999",
        branch=BranchKind.YAMNET,
        source_path="/tmp/source.mp4",
    )
    paths = build_branch_paths(run_root=run_root, branch=BranchKind.YAMNET)
    write_branch_request(paths.request_path, request)

    monkeypatch.setattr(
        cli,
        "build_branch_dependencies",
        lambda request, workspace: {"yamnet_provider": object()},
    )

    def _fake_run_yamnet_branch(*, workspace, yamnet_provider):
        assert str(workspace.root) == str(run_root)
        raise RuntimeError("yamnet boom")

    monkeypatch.setattr(cli, "run_yamnet_branch", _fake_run_yamnet_branch)

    exit_code = cli.main(["--request-path", str(paths.request_path)])

    assert exit_code == 1
    status = json.loads(paths.status_path.read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["started_at"] is not None
    assert status["completed_at"] is not None
    result = json.loads(paths.result_path.read_text(encoding="utf-8"))
    assert result["ok"] is False
    assert result["error"]["error_message"] == "yamnet boom"


def test_branch_cli_writes_status_json_before_exiting_nonzero(tmp_path: Path, monkeypatch):
    from backend.phase1_runtime.branch_io import build_branch_paths, write_branch_request
    from backend.phase1_runtime.branch_models import BranchKind, BranchRequest
    import backend.runtime.run_phase1_branch as cli

    run_root = tmp_path / "runs" / "run_404"
    request = BranchRequest(
        job_id="job_404",
        run_id="run_404",
        branch=BranchKind.YAMNET,
        source_path="/tmp/source.mp4",
    )
    paths = build_branch_paths(run_root=run_root, branch=BranchKind.YAMNET)
    write_branch_request(paths.request_path, request)

    monkeypatch.setattr(
        cli,
        "build_branch_dependencies",
        lambda request, workspace: {"yamnet_provider": object()},
    )
    monkeypatch.setattr(
        cli,
        "run_yamnet_branch",
        lambda *, workspace, yamnet_provider: (_ for _ in ()).throw(RuntimeError("yamnet boom")),
    )

    original_write_branch_result = cli.write_branch_result
    seen: dict[str, object] = {}

    def _recording_write_branch_result(path, envelope):
        status = json.loads(paths.status_path.read_text(encoding="utf-8"))
        seen["status_before_result"] = status
        return original_write_branch_result(path, envelope)

    monkeypatch.setattr(cli, "write_branch_result", _recording_write_branch_result)

    exit_code = cli.main(["--request-path", str(paths.request_path)])

    assert exit_code == 1
    assert seen["status_before_result"]["state"] == "failed"
    assert seen["status_before_result"]["completed_at"] is not None


def test_branch_cli_does_not_clobber_success_result_when_success_status_write_fails(
    tmp_path: Path,
    monkeypatch,
):
    from backend.phase1_runtime.branch_io import build_branch_paths, write_branch_request, write_branch_status
    from backend.phase1_runtime.branch_models import BranchKind, BranchRequest
    import backend.runtime.run_phase1_branch as cli

    run_root = tmp_path / "runs" / "run_777"
    request = BranchRequest(
        job_id="job_777",
        run_id="run_777",
        branch=BranchKind.VISUAL,
        source_path="/tmp/source.mp4",
    )
    paths = build_branch_paths(run_root=run_root, branch=BranchKind.VISUAL)
    write_branch_request(paths.request_path, request)

    class _FakeVisualExtractor:
        def extract(self, *, video_path: Path, workspace):
            return {"video_metadata": {"fps": 24.0}, "shot_changes": [], "tracks": [{"track_id": "t777"}]}

    monkeypatch.setattr(
        cli,
        "build_branch_dependencies",
        lambda request, workspace: {"visual_extractor": _FakeVisualExtractor()},
    )

    original_write_branch_status = write_branch_status

    def _failing_write_branch_status(path, status):
        if status.state == "succeeded":
            raise OSError("status write failed")
        return original_write_branch_status(path, status)

    monkeypatch.setattr(cli, "write_branch_status", _failing_write_branch_status)

    with pytest.raises(OSError, match="status write failed"):
        cli.main(["--request-path", str(paths.request_path)])

    result = json.loads(paths.result_path.read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert result["result"]["phase1_visual"]["tracks"] == [{"track_id": "t777"}]
