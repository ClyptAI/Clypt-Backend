"""Clip candidate endpoints with approval/rejection."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.repository.phase14_repository import Phase14Repository

from .deps import get_repo
from .schemas import ClipCandidate

router = APIRouter(prefix="/runs/{run_id}/clips", tags=["clips"])


def _record_to_schema(rec) -> ClipCandidate:
    return ClipCandidate(
        clip_id=rec.clip_id,
        node_ids=rec.node_ids,
        start_ms=rec.start_ms,
        end_ms=rec.end_ms,
        score=rec.score,
        rationale=rec.rationale,
        source_prompt_ids=rec.source_prompt_ids,
        seed_node_id=rec.seed_node_id,
        subgraph_id=rec.subgraph_id,
        query_aligned=rec.query_aligned,
        pool_rank=rec.pool_rank,
        score_breakdown=rec.score_breakdown,
    )


def _find_candidate(repo: Phase14Repository, run_id: str, clip_id: str):
    records = repo.list_candidates(run_id=run_id)
    for r in records:
        if r.clip_id == clip_id:
            return r
    return None


@router.get("", response_model=list[ClipCandidate])
def list_clips(
    run_id: str,
    repo: Phase14Repository = Depends(get_repo),
) -> list[ClipCandidate]:
    records = repo.list_candidates(run_id=run_id)
    return [_record_to_schema(r) for r in records]


@router.get("/{clip_id}", response_model=ClipCandidate)
def get_clip(
    run_id: str,
    clip_id: str,
    repo: Phase14Repository = Depends(get_repo),
) -> ClipCandidate:
    rec = _find_candidate(repo, run_id, clip_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="clip not found")
    return _record_to_schema(rec)


@router.post("/{clip_id}/approve", response_model=ClipCandidate)
def approve_clip(
    run_id: str,
    clip_id: str,
    repo: Phase14Repository = Depends(get_repo),
) -> ClipCandidate:
    # TODO: persist approval state once repo method is added
    rec = _find_candidate(repo, run_id, clip_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="clip not found")
    return _record_to_schema(rec)


@router.post("/{clip_id}/reject", response_model=ClipCandidate)
def reject_clip(
    run_id: str,
    clip_id: str,
    repo: Phase14Repository = Depends(get_repo),
) -> ClipCandidate:
    # TODO: persist rejection state once repo method is added
    rec = _find_candidate(repo, run_id, clip_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="clip not found")
    return _record_to_schema(rec)
