"""FastAPI dependency injection for the v1 API routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Request

from backend.pipeline.artifacts import V31RunPaths
from backend.repository.phase14_repository import Phase14Repository


def get_repo(request: Request) -> Phase14Repository:
    return request.app.state.repo


def get_artifact_root(request: Request) -> Path:
    return request.app.state.artifact_root


def get_run_paths(request: Request, run_id: str) -> V31RunPaths:
    root: Path = request.app.state.artifact_root
    return V31RunPaths(run_id=run_id, root=root)
