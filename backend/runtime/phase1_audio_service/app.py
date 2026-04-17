"""FastAPI application for the RTX 6000 Ada Phase 1 audio host.

Exposes three routes consumed by the H200 orchestrator:

* ``POST /tasks/phase1-audio`` — run VibeVoice → NFA → emotion2vec+ → YAMNet
  and return the merged payload in a single round trip. Audio goes over the
  wire once (via GCS URI), and the chain stays hot on one GPU.
* ``POST /tasks/node-media-prep`` — extract Phase 2 node clips with ffmpeg
  NVENC and upload them to GCS.
* ``GET /health`` — readiness probe for the systemd unit and the H200-side
  ``RemoteAudioChainClient.healthcheck()``.

Authentication is a shared bearer token passed in the ``Authorization`` header
(``Bearer <token>``). The expected token is read from
``CLYPT_PHASE1_AUDIO_HOST_AUTH_TOKEN`` (or ``CLYPT_PHASE1_AUDIO_HOST_TOKEN``).

Concurrency: audio runs serialize on a per-process asyncio semaphore
(``audio_chain`` stays hot on one GPU). Node-media prep can accept multiple
concurrent requests — each request owns its own bounded ffmpeg thread pool
inside :func:`backend.runtime.phase1_audio_service.node_media_prep.run_node_media_prep`.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .audio_chain import run_audio_chain
from .deps import AppDeps, get_app_deps
from .node_media_prep import NodeMediaPrepRequest, run_node_media_prep

logger = logging.getLogger(__name__)


class Phase1AudioRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio_gcs_uri: str
    source_url: str | None = None
    video_gcs_uri: str | None = None
    run_id: str | None = None


class StageEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    stage_name: str
    status: str
    duration_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_payload: dict[str, Any] | None = None


def _require_bearer(request: Request, deps: AppDeps) -> None:
    header = request.headers.get("authorization") or ""
    if not header.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
        )
    token = header.split(" ", 1)[1].strip()
    if token != deps.expected_auth_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="invalid bearer token",
        )


def create_app() -> FastAPI:
    app = FastAPI(title="Clypt Phase 1 Audio Host", version="1.0.0")
    audio_lock = asyncio.Lock()

    @app.get("/health")
    async def health(deps: AppDeps = Depends(get_app_deps)) -> dict[str, Any]:
        return {
            "status": "ok",
            "scratch_root": str(deps.scratch_root),
            "audio_chain_ready": True,
            "node_media_prep_ready": True,
        }

    @app.post("/tasks/phase1-audio")
    async def phase1_audio(
        body: Phase1AudioRequestBody,
        request: Request,
        deps: AppDeps = Depends(get_app_deps),
    ) -> JSONResponse:
        _require_bearer(request, deps)

        async with audio_lock:
            stage_events: list[dict[str, Any]] = []

            def _record(
                *,
                stage_name: str,
                status: str,
                duration_ms: float | None = None,
                metadata: dict[str, Any] | None = None,
                error_payload: dict[str, Any] | None = None,
            ) -> None:
                stage_events.append(
                    {
                        "stage_name": stage_name,
                        "status": status,
                        "duration_ms": duration_ms,
                        "metadata": metadata or {},
                        "error_payload": error_payload,
                    }
                )

            t_start = time.perf_counter()
            with tempfile.TemporaryDirectory(
                prefix=f"phase1-audio-{body.run_id or 'anon'}-",
                dir=str(deps.scratch_root),
            ) as tmp_root_str:
                tmp_root = Path(tmp_root_str)
                audio_path = tmp_root / "source_audio.wav"
                logger.info(
                    "[phase1_audio_service] downloading audio run_id=%s uri=%s",
                    body.run_id or "-",
                    body.audio_gcs_uri,
                )
                try:
                    deps.storage_client.download_file(
                        gcs_uri=body.audio_gcs_uri,
                        local_path=audio_path,
                    )
                except Exception as exc:
                    logger.exception("[phase1_audio_service] audio download failed")
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"failed to download {body.audio_gcs_uri}: {exc}",
                    ) from exc

                try:
                    chain_output = await asyncio.to_thread(
                        run_audio_chain,
                        audio_path=audio_path,
                        vibevoice_provider=deps.vibevoice_provider,
                        forced_aligner=deps.forced_aligner,
                        emotion_provider=deps.emotion_provider,
                        yamnet_provider=deps.yamnet_provider,
                        audio_gcs_uri=body.audio_gcs_uri,
                        stage_event_recorder=_record,
                    )
                except Exception as exc:
                    logger.exception(
                        "[phase1_audio_service] audio chain failed run_id=%s",
                        body.run_id or "-",
                    )
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"audio chain failed: {exc}",
                    ) from exc

            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            logger.info(
                "[phase1_audio_service] phase1-audio done run_id=%s in %.1f ms",
                body.run_id or "-",
                elapsed_ms,
            )
            response_payload: dict[str, Any] = {
                "run_id": body.run_id,
                "turns": chain_output["turns"],
                "diarization_payload": chain_output["diarization_payload"],
                "emotion2vec_payload": chain_output["emotion2vec_payload"],
                "yamnet_payload": chain_output["yamnet_payload"],
                "stage_events": stage_events,
                "elapsed_ms": elapsed_ms,
            }
            return JSONResponse(response_payload)

    @app.post("/tasks/node-media-prep")
    async def node_media_prep(
        request: Request,
        deps: AppDeps = Depends(get_app_deps),
    ) -> JSONResponse:
        _require_bearer(request, deps)

        try:
            raw_body = await request.json()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"request body must be JSON: {exc}",
            ) from exc
        if not isinstance(raw_body, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="request body must be a JSON object",
            )

        try:
            parsed = NodeMediaPrepRequest.from_payload(raw_body)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        t_start = time.perf_counter()
        try:
            result = await asyncio.to_thread(
                run_node_media_prep,
                request=parsed,
                storage_client=deps.storage_client,
                scratch_root=deps.scratch_root,
            )
        except Exception as exc:
            logger.exception(
                "[phase1_audio_service] node-media-prep failed run_id=%s",
                parsed.run_id,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"node-media-prep failed: {exc}",
            ) from exc

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        result["elapsed_ms"] = elapsed_ms
        logger.info(
            "[phase1_audio_service] node-media-prep done run_id=%s nodes=%d in %.1f ms",
            parsed.run_id,
            len(parsed.nodes),
            elapsed_ms,
        )
        return JSONResponse(result)

    return app


# Uvicorn entrypoint: `uvicorn backend.runtime.phase1_audio_service.app:app`
app = create_app()


__all__ = ["app", "create_app", "Phase1AudioRequestBody", "StageEvent"]
