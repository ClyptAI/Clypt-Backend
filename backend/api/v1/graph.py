"""Graph endpoints: nodes and edges."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.repository.phase14_repository import Phase14Repository

from .deps import get_repo
from .schemas import SemanticGraphEdge, SemanticGraphNode, SemanticNodeEvidence

router = APIRouter(prefix="/runs/{run_id}", tags=["graph"])


def _node_record_to_schema(rec) -> SemanticGraphNode:
    evidence = rec.evidence if isinstance(rec.evidence, dict) else {}
    return SemanticGraphNode(
        node_id=rec.node_id,
        node_type=rec.node_type,
        start_ms=rec.start_ms,
        end_ms=rec.end_ms,
        source_turn_ids=rec.source_turn_ids,
        word_ids=rec.word_ids,
        transcript_text=rec.transcript_text,
        node_flags=rec.node_flags,
        summary=rec.summary,
        evidence=SemanticNodeEvidence(
            emotion_labels=evidence.get("emotion_labels", []),
            audio_events=evidence.get("audio_events", []),
        ),
        semantic_embedding=rec.semantic_embedding,
        multimodal_embedding=rec.multimodal_embedding,
    )


def _edge_record_to_schema(rec) -> SemanticGraphEdge:
    return SemanticGraphEdge(
        source_node_id=rec.source_node_id,
        target_node_id=rec.target_node_id,
        edge_type=rec.edge_type,
        rationale=rec.rationale,
        confidence=rec.confidence,
        support_count=rec.support_count,
        batch_ids=rec.batch_ids,
    )


@router.get("/nodes", response_model=list[SemanticGraphNode])
def list_nodes(
    run_id: str,
    repo: Phase14Repository = Depends(get_repo),
) -> list[SemanticGraphNode]:
    records = repo.list_nodes(run_id=run_id)
    return [_node_record_to_schema(r) for r in records]


@router.get("/nodes/{node_id}", response_model=SemanticGraphNode)
def get_node(
    run_id: str,
    node_id: str,
    repo: Phase14Repository = Depends(get_repo),
) -> SemanticGraphNode:
    records = repo.list_nodes(run_id=run_id)
    for r in records:
        if r.node_id == node_id:
            return _node_record_to_schema(r)
    raise HTTPException(status_code=404, detail="node not found")


@router.get("/edges", response_model=list[SemanticGraphEdge])
def list_edges(
    run_id: str,
    repo: Phase14Repository = Depends(get_repo),
) -> list[SemanticGraphEdge]:
    records = repo.list_edges(run_id=run_id)
    return [_edge_record_to_schema(r) for r in records]
