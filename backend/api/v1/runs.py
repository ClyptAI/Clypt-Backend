"""Run management endpoints: list, get, create."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from backend.repository.phase14_repository import Phase14Repository
from backend.repository.models import RunRecord

from .deps import get_repo
from .schemas import (
    PhaseNumber,
    PhaseStatus,
    PhaseStatusEntry,
    RunCreateRequest,
    RunDetail,
    RunListItem,
    RunMeta,
)

router = APIRouter(prefix="/runs", tags=["runs"])

# Map internal RunStatus to frontend PhaseNumber + PhaseStatus
_STATUS_TO_PHASE: dict[str, tuple[PhaseNumber, PhaseStatus]] = {
    "PHASE1_DONE": (1, "completed"),
    "PHASE24_QUEUED": (2, "pending"),
    "PHASE24_RUNNING": (2, "running"),
    "PHASE24_DONE": (4, "completed"),
    "FAILED": (1, "failed"),
}

_PHASE_NAME_MAP: dict[str, tuple[PhaseNumber, str]] = {
    "phase1": (1, "Timeline Foundation"),
    "phase2": (2, "Node Construction"),
    "phase3": (3, "Graph Construction"),
    "phase4": (4, "Candidate Generation"),
    "phase5": (5, "Grounding"),
    "phase6": (6, "Render"),
}


def _run_to_list_item(run: RunRecord, repo: Phase14Repository) -> RunListItem:
    phase_num, phase_status = _STATUS_TO_PHASE.get(run.status, (1, "pending"))
    candidates = repo.list_candidates(run_id=run.run_id)
    return RunListItem(
        run_id=run.run_id,
        source_url=run.source_url or "",
        created_at=run.created_at.isoformat(),
        display_name=run.metadata.get("display_name"),
        latest_phase=phase_num,
        latest_status=phase_status,
        clip_count=len(candidates),
    )


def _run_to_detail(run: RunRecord, repo: Phase14Repository) -> RunDetail:
    metrics = repo.list_phase_metrics(run_id=run.run_id)
    nodes = repo.list_nodes(run_id=run.run_id)
    edges = repo.list_edges(run_id=run.run_id)
    candidates = repo.list_candidates(run_id=run.run_id)

    phases: list[PhaseStatusEntry] = []
    for metric in metrics:
        phase_info = _PHASE_NAME_MAP.get(metric.phase_name)
        if phase_info is None:
            continue
        phase_num, phase_label = phase_info
        status: PhaseStatus
        if metric.status == "succeeded":
            status = "completed"
        elif metric.status == "running":
            status = "running"
        elif metric.status == "failed":
            status = "failed"
        else:
            status = "pending"
        elapsed = metric.duration_ms / 1000.0 if metric.duration_ms else None
        phases.append(PhaseStatusEntry(
            phase=phase_num,
            name=phase_label,
            status=status,
            elapsed_s=elapsed,
        ))

    return RunDetail(
        run_id=run.run_id,
        source_url=run.source_url or "",
        created_at=run.created_at.isoformat(),
        display_name=run.metadata.get("display_name"),
        phases=phases,
        node_count=len(nodes),
        edge_count=len(edges),
        clip_count=len(candidates),
    )


@router.get("", response_model=list[RunListItem])
def list_runs(repo: Phase14Repository = Depends(get_repo)) -> list[RunListItem]:
    # The repository doesn't have a list_runs method yet.
    # For now we return runs stored in metadata. This will need a repo method added.
    # Temporary: return empty list if list_runs doesn't exist.
    if not hasattr(repo, "list_runs"):
        return []
    runs: list[RunRecord] = repo.list_runs()  # type: ignore[attr-defined]
    return [_run_to_list_item(r, repo) for r in runs]


@router.get("/{run_id}", response_model=RunDetail)
def get_run(run_id: str, repo: Phase14Repository = Depends(get_repo)) -> RunDetail:
    run = repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return _run_to_detail(run, repo)


@router.post("", response_model=RunMeta, status_code=201)
def create_run(
    body: RunCreateRequest,
    repo: Phase14Repository = Depends(get_repo),
) -> RunMeta:
    now = datetime.now(timezone.utc)
    run_id = f"run_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    record = RunRecord(
        run_id=run_id,
        source_url=body.source_url,
        status="PHASE1_DONE",  # initial status — will be updated by pipeline
        created_at=now,
        updated_at=now,
        metadata={"display_name": body.display_name} if body.display_name else {},
    )
    repo.upsert_run(record)
    return RunMeta(
        run_id=run_id,
        source_url=body.source_url,
        created_at=now.isoformat(),
        display_name=body.display_name,
    )
