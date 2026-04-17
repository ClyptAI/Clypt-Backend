"""Grounding state endpoints — per-clip bounding box editor state.

Grounding state is persisted as JSON files on disk under the artifact root
at {artifact_root}/{run_id}/grounding/{clip_id}.json until a proper database
table is added.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from .deps import get_artifact_root
from .schemas import GroundingClipState

router = APIRouter(prefix="/runs/{run_id}/clips/{clip_id}", tags=["grounding"])


def _grounding_path(artifact_root: Path, run_id: str, clip_id: str) -> Path:
    return artifact_root / run_id / "grounding" / f"{clip_id}.json"


@router.get("/grounding", response_model=GroundingClipState)
def get_grounding(
    run_id: str,
    clip_id: str,
    artifact_root: Path = Depends(get_artifact_root),
) -> GroundingClipState:
    path = _grounding_path(artifact_root, run_id, clip_id)
    if not path.exists():
        # Return empty default state
        return GroundingClipState(
            run_id=run_id,
            clip_id=clip_id,
            shots=[],
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return GroundingClipState(**data)


@router.put("/grounding", response_model=GroundingClipState)
def put_grounding(
    run_id: str,
    clip_id: str,
    body: GroundingClipState,
    artifact_root: Path = Depends(get_artifact_root),
) -> GroundingClipState:
    body.run_id = run_id
    body.clip_id = clip_id
    body.updated_at = datetime.now(timezone.utc).isoformat()

    path = _grounding_path(artifact_root, run_id, clip_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(body.model_dump(mode="json"), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return body
