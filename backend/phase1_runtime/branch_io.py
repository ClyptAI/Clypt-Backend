from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .branch_models import BranchKind, BranchRequest, BranchResultEnvelope, BranchStatus

TModel = TypeVar("TModel", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class BranchPaths:
    branch_root: Path
    request_path: Path
    result_path: Path
    status_path: Path
    log_path: Path


def build_branch_paths(*, run_root: Path, branch: BranchKind) -> BranchPaths:
    branch_root = run_root / "branches" / branch.value
    branch_root.mkdir(parents=True, exist_ok=True)
    return BranchPaths(
        branch_root=branch_root,
        request_path=branch_root / "request.json",
        result_path=branch_root / "result.json",
        status_path=branch_root / "status.json",
        log_path=branch_root / "branch.log",
    )


def _write_json_model(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as tmp_file:
            tmp_file.write(json.dumps(model.model_dump(mode="json"), indent=2, ensure_ascii=True))
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            tmp_path = Path(tmp_file.name)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _read_json_model(path: Path, model_type: type[TModel]) -> TModel:
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))


def write_branch_request(path: Path, request: BranchRequest) -> None:
    _write_json_model(path, request)


def read_branch_request(path: Path) -> BranchRequest:
    return _read_json_model(path, BranchRequest)


def write_branch_result(path: Path, result: BranchResultEnvelope) -> None:
    _write_json_model(path, result)


def read_branch_result(path: Path) -> BranchResultEnvelope:
    return _read_json_model(path, BranchResultEnvelope)


def write_branch_status(path: Path, status: BranchStatus) -> None:
    _write_json_model(path, status)


def read_branch_status(path: Path) -> BranchStatus:
    return _read_json_model(path, BranchStatus)


__all__ = [
    "BranchPaths",
    "build_branch_paths",
    "read_branch_request",
    "read_branch_result",
    "read_branch_status",
    "write_branch_request",
    "write_branch_result",
    "write_branch_status",
]
