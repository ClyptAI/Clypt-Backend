"""FastAPI application for the RTX 6000 Ada VibeVoice ASR + node-media-prep host.

Exposes three routes consumed by the H200 orchestrator:

* ``POST /tasks/vibevoice-asr`` — run VibeVoice ASR over a GCS-hosted audio
  file and return the raw VibeVoice turns. Audio goes over the wire once
  (via GCS URI) and vLLM stays hot on the local GPU.
* ``POST /tasks/node-media-prep`` — extract Phase 2 node clips with ffmpeg
  NVENC and upload them to GCS.
* ``GET /health`` — readiness probe for the systemd unit and the H200-side
  ``RemoteVibeVoiceAsrClient.healthcheck()``.

Authentication is a shared bearer token passed in the ``Authorization`` header
(``Bearer <token>``). The expected token is read from
``CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN`` (or, for one release, the
legacy ``CLYPT_PHASE1_AUDIO_HOST_AUTH_TOKEN`` /
``CLYPT_PHASE1_AUDIO_HOST_TOKEN`` aliases).

NFA, emotion2vec+ and YAMNet used to run on this host too; they were moved
back to the H200 because NFA's global-alignment pass needs ~17 GiB of VRAM
that it could not get while co-located with vLLM on the 48 GiB RTX card.
See docs/ERROR_LOG.md 2026-04-17.

Concurrency: ASR requests serialize on a per-process asyncio lock (vLLM
runs max-num-seqs 1). Node-media prep can accept multiple concurrent
requests — each request owns its own bounded ffmpeg thread pool inside
:func:`backend.runtime.phase1_audio_service.node_media_prep.run_node_media_prep`.
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

from .deps import AppDeps, get_app_deps
from .node_media_prep import NodeMediaPrepRequest, run_node_media_prep

logger = logging.getLogger(__name__)


class VibeVoiceAsrRequestBody(BaseModel):
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


def _run_vibevoice_asr(
    *,
    vibevoice_provider: Any,
    audio_path: Path,
    audio_gcs_uri: str | None,
) -> list[dict[str, Any]]:
    """Invoke the in-process VibeVoice vLLM provider and return raw turns.

    Tries the GCS-aware signature first so vLLM can stream audio directly
    from the URL it already has (avoiding a second upload); falls back to
    the local-file signature if the provider does not accept ``audio_gcs_uri``.
    """
    audio_path_str = str(audio_path)
    if audio_gcs_uri:
        try:
            return vibevoice_provider.run(
                audio_path=audio_path_str,
                audio_gcs_uri=audio_gcs_uri,
            )
        except TypeError as exc:
            if "audio_gcs_uri" not in str(exc):
                raise
    return vibevoice_provider.run(audio_path=audio_path_str)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Clypt VibeVoice ASR + node-media-prep host",
        version="2.0.0",
    )
    asr_lock = asyncio.Lock()

    @app.get("/health")
    async def health(deps: AppDeps = Depends(get_app_deps)) -> dict[str, Any]:
        return {
            "status": "ok",
            "scratch_root": str(deps.scratch_root),
            "vibevoice_asr_ready": True,
            "node_media_prep_ready": True,
        }

    @app.post("/tasks/vibevoice-asr")
    async def vibevoice_asr(
        body: VibeVoiceAsrRequestBody,
        request: Request,
        deps: AppDeps = Depends(get_app_deps),
    ) -> JSONResponse:
        _require_bearer(request, deps)

        async with asr_lock:
            stage_events: list[dict[str, Any]] = []

            t_start = time.perf_counter()
            with tempfile.TemporaryDirectory(
                prefix=f"vibevoice-asr-{body.run_id or 'anon'}-",
                dir=str(deps.scratch_root),
            ) as tmp_root_str:
                tmp_root = Path(tmp_root_str)
                audio_path = tmp_root / "source_audio.wav"
                logger.info(
                    "[vibevoice_asr_service] downloading audio run_id=%s uri=%s",
                    body.run_id or "-",
                    body.audio_gcs_uri,
                )
                try:
                    deps.storage_client.download_file(
                        gcs_uri=body.audio_gcs_uri,
                        local_path=audio_path,
                    )
                except Exception as exc:
                    logger.exception("[vibevoice_asr_service] audio download failed")
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"failed to download {body.audio_gcs_uri}: {exc}",
                    ) from exc

                t_asr = time.perf_counter()
                try:
                    turns = await asyncio.to_thread(
                        _run_vibevoice_asr,
                        vibevoice_provider=deps.vibevoice_provider,
                        audio_path=audio_path,
                        audio_gcs_uri=body.audio_gcs_uri,
                    )
                except Exception as exc:
                    duration_ms = (time.perf_counter() - t_asr) * 1000.0
                    stage_events.append(
                        {
                            "stage_name": "vibevoice_asr",
                            "status": "failed",
                            "duration_ms": duration_ms,
                            "metadata": {},
                            "error_payload": {
                                "code": exc.__class__.__name__,
                                "message": str(exc)[:2048],
                            },
                        }
                    )
                    logger.exception(
                        "[vibevoice_asr_service] VibeVoice ASR failed run_id=%s",
                        body.run_id or "-",
                    )
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"VibeVoice ASR failed: {exc}",
                    ) from exc
                stage_events.append(
                    {
                        "stage_name": "vibevoice_asr",
                        "status": "succeeded",
                        "duration_ms": (time.perf_counter() - t_asr) * 1000.0,
                        "metadata": {"turn_count": len(turns)},
                        "error_payload": None,
                    }
                )

            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            logger.info(
                "[vibevoice_asr_service] vibevoice-asr done run_id=%s turns=%d in %.1f ms",
                body.run_id or "-",
                len(turns),
                elapsed_ms,
            )
            response_payload: dict[str, Any] = {
                "run_id": body.run_id,
                "turns": list(turns),
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
                "[vibevoice_asr_service] node-media-prep failed run_id=%s",
                parsed.run_id,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"node-media-prep failed: {exc}",
            ) from exc

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        result["elapsed_ms"] = elapsed_ms
        logger.info(
            "[vibevoice_asr_service] node-media-prep done run_id=%s nodes=%d in %.1f ms",
            parsed.run_id,
            len(parsed.nodes),
            elapsed_ms,
        )
        return JSONResponse(result)

    return app


# Uvicorn entrypoint: `uvicorn backend.runtime.phase1_audio_service.app:app`
app = create_app()


__all__ = ["app", "create_app", "VibeVoiceAsrRequestBody", "StageEvent"]
