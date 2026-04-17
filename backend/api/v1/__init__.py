from __future__ import annotations

from fastapi import APIRouter

from .runs import router as runs_router
from .graph import router as graph_router
from .clips import router as clips_router
from .timeline import router as timeline_router
from .grounding import router as grounding_router
from .render import router as render_router

router = APIRouter(prefix="/v1")
router.include_router(runs_router)
router.include_router(graph_router)
router.include_router(clips_router)
router.include_router(timeline_router)
router.include_router(grounding_router)
router.include_router(render_router)

__all__ = ["router"]
