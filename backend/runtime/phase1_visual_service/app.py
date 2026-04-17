from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .deps import AppDeps, get_app_deps

logger = logging.getLogger(__name__)


class VisualExtractRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str | None = None
    video_path: str


def _require_bearer(request: Request, expected_auth_token: str) -> None:
    header = request.headers.get("authorization") or ""
    if not header.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")
    token = header.split(" ", 1)[1].strip()
    if token != expected_auth_token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid bearer token")


def create_app(
    *,
    visual_extractor: Any | None = None,
    expected_auth_token: str | None = None,
) -> FastAPI:
    app = FastAPI(title="Clypt Phase 1 Visual Service", version="1.0.0")

    def _deps() -> AppDeps:
        if visual_extractor is not None and expected_auth_token is not None:
            return AppDeps(
                visual_extractor=visual_extractor,
                expected_auth_token=expected_auth_token,
            )
        return get_app_deps()

    @app.get("/health")
    async def health(deps: AppDeps = Depends(_deps)) -> dict[str, Any]:
        active_extractor = deps.visual_extractor
        return {
            "status": "ok",
            "visual_ready": active_extractor is not None,
        }

    @app.post("/tasks/visual-extract")
    async def visual_extract(
        body: VisualExtractRequestBody,
        request: Request,
        deps: AppDeps = Depends(_deps),
    ) -> JSONResponse:
        _require_bearer(request, deps.expected_auth_token)
        extractor = deps.visual_extractor
        try:
            payload = extractor.extract(video_path=Path(body.video_path), workspace=None)
        except Exception as exc:
            logger.exception("[phase1_visual_service] visual extract failed run_id=%s", body.run_id or "-")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"visual extraction failed: {exc}",
            ) from exc
        return JSONResponse(payload)

    return app


app = create_app()


__all__ = ["app", "create_app"]
