"""FastAPI application factory for the Clypt V1 API server.

Usage:
    # With Spanner (production):
    python -m backend.api.v1.app

    # With Spanner + custom artifact root:
    CLYPT_ARTIFACT_ROOT=/path/to/outputs python -m backend.api.v1.app

Environment variables:
    GOOGLE_CLOUD_PROJECT  — GCP project (required for Spanner)
    CLYPT_SPANNER_INSTANCE — Spanner instance (default: clypt-phase14)
    CLYPT_SPANNER_DATABASE — Spanner database (default: clypt_phase14)
    CLYPT_ARTIFACT_ROOT    — Root directory for pipeline output artifacts
                             (default: backend/outputs/v3_1_phase1_work)
    CLYPT_API_PORT         — Server port (default: 8000)
    CLYPT_CORS_ORIGINS     — Comma-separated allowed CORS origins
                             (default: http://localhost:8080,http://localhost:5173)
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.v1 import router as v1_router


def create_app(
    *,
    repo=None,
    artifact_root: Path | str | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    app = FastAPI(title="Clypt V3.1 API", version="0.1.0")

    # CORS
    origins = cors_origins or _default_cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store dependencies on app.state for injection via deps.py
    app.state.repo = repo
    app.state.artifact_root = Path(
        artifact_root
        or os.getenv("CLYPT_ARTIFACT_ROOT", "backend/outputs/v3_1_phase1_work")
    )

    # Mount v1 routes
    app.include_router(v1_router)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    return app


def _default_cors_origins() -> list[str]:
    env = os.getenv("CLYPT_CORS_ORIGINS", "")
    if env:
        return [o.strip() for o in env.split(",") if o.strip()]
    return [
        "http://localhost:8080",
        "http://localhost:5173",
        "http://localhost:3000",
    ]


def _build_repo():
    """Build the Spanner repository from environment variables."""
    from dataclasses import dataclass

    @dataclass
    class _SpannerSettings:
        project: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        instance: str = os.getenv("CLYPT_SPANNER_INSTANCE", "clypt-phase14")
        database: str = os.getenv("CLYPT_SPANNER_DATABASE", "clypt_phase14")
        ddl_operation_timeout_s: float = 600.0

    from backend.repository.spanner_phase14_repository import SpannerPhase14Repository
    return SpannerPhase14Repository.from_settings(settings=_SpannerSettings())


def main():
    import uvicorn

    port = int(os.getenv("CLYPT_API_PORT", "8000"))
    repo = _build_repo()
    app = create_app(repo=repo)

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
