from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict

from backend.runtime.phase24_local_queue import Phase24LocalQueue


class Phase26EnqueueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    payload: dict[str, Any]


def _require_bearer(request: Request, expected_auth_token: str) -> None:
    header = request.headers.get("authorization") or ""
    if not header.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")
    token = header.split(" ", 1)[1].strip()
    if token != expected_auth_token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid bearer token")


def create_app(*, queue: Phase24LocalQueue, expected_auth_token: str) -> FastAPI:
    app = FastAPI(title="Clypt Phase26 Dispatch Service", version="1.0.0")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "queue_backend": "local_sqlite"}

    @app.post("/tasks/phase26-enqueue")
    async def phase26_enqueue(body: Phase26EnqueueRequest, request: Request) -> dict[str, Any]:
        _require_bearer(request, expected_auth_token)
        task_id = queue.enqueue(body.run_id, body.payload)
        return {
            "run_id": body.run_id,
            "status": "queued",
            "task_name": f"local-sqlite:{task_id}",
        }

    return app


__all__ = ["create_app"]
