from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import tempfile
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException

from backend.pipeline.semantics.media_embeddings import prepare_node_media_embeddings
from backend.pipeline.semantics.media_prep_contracts import (
    NodeMediaPrepRequest,
    NodeMediaPrepResponse,
)
from backend.providers import (
    VibeVoiceVLLMProvider,
    build_gcs_uri_url_resolver,
    load_provider_settings,
)
from backend.providers.phase1_asr_contracts import Phase1ASRRequest, Phase1ASRResponse
from backend.providers.storage import GCSStorageClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _NodeWindow:
    node_id: str
    start_ms: int
    end_ms: int


@dataclass(slots=True)
class Phase24MediaPrepService:
    storage_client: Any

    def handle_request(self, payload: NodeMediaPrepRequest) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix=f"media-prep-{payload.run_id}-") as temp_dir:
            source_video_path = Path(temp_dir) / "source_video.mp4"
            self.storage_client.download_file(
                gcs_uri=payload.source_video_gcs_uri,
                local_path=source_video_path,
            )
            descriptors = prepare_node_media_embeddings(
                nodes=[
                    _NodeWindow(
                        node_id=item.node_id,
                        start_ms=int(item.start_ms),
                        end_ms=int(item.end_ms),
                    )
                    for item in payload.items
                ],
                source_video_path=source_video_path,
                clips_dir=Path(temp_dir) / "node_media_clips",
                storage_client=self.storage_client,
                object_prefix=payload.object_prefix,
            )
            return NodeMediaPrepResponse(
                run_id=payload.run_id,
                items=[
                    {
                        "node_id": descriptor["node_id"],
                        "file_uri": descriptor["file_uri"],
                        "mime_type": descriptor.get("mime_type") or "video/mp4",
                    }
                    for descriptor in descriptors
                ],
            ).model_dump(mode="json", exclude_none=True)


@dataclass(slots=True)
class Phase1ASRService:
    storage_client: Any
    provider_kwargs: dict[str, Any]

    def handle_request(self, payload: Phase1ASRRequest) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="phase1-asr-") as temp_dir:
            suffix = Path(urlparse(payload.audio_gcs_uri).path).suffix or ".wav"
            audio_path = Path(temp_dir) / f"source_audio{suffix}"
            self.storage_client.download_file(
                gcs_uri=payload.audio_gcs_uri,
                local_path=audio_path,
            )
            provider_kwargs = dict(self.provider_kwargs)
            generation_config = (
                payload.generation_config.model_dump(mode="python", exclude_none=True)
                if payload.generation_config is not None
                else {}
            )
            provider_kwargs.update(generation_config)
            provider = VibeVoiceVLLMProvider(**provider_kwargs)
            turns = provider.run(
                audio_path=audio_path,
                context_info=payload.context_info,
                audio_gcs_uri=payload.audio_gcs_uri,
            )
            return Phase1ASRResponse(turns=turns).model_dump(mode="json", exclude_none=True)


def build_default_phase24_media_prep_service() -> Phase24MediaPrepService:
    settings = load_provider_settings()
    return Phase24MediaPrepService(
        storage_client=GCSStorageClient(settings=settings.storage),
    )


def build_default_phase1_asr_service() -> Phase1ASRService:
    settings = load_provider_settings()
    storage_client = GCSStorageClient(settings=settings.storage)
    vibevoice_kwargs = {
        "base_url": settings.vllm_vibevoice.base_url,
        "model": settings.vllm_vibevoice.model,
        "timeout_s": settings.vllm_vibevoice.timeout_s,
        "healthcheck_path": settings.vllm_vibevoice.healthcheck_path,
        "max_retries": settings.vllm_vibevoice.max_retries,
        "audio_mode": settings.vllm_vibevoice.audio_mode,
        "audio_gcs_url_resolver": build_gcs_uri_url_resolver(storage_client=storage_client),
        "hotwords_context": settings.vibevoice.hotwords_context,
        "max_new_tokens": settings.vibevoice.max_new_tokens,
        "do_sample": settings.vibevoice.do_sample,
        "temperature": settings.vibevoice.temperature,
        "top_p": settings.vibevoice.top_p,
        "repetition_penalty": settings.vibevoice.repetition_penalty,
        "num_beams": settings.vibevoice.num_beams,
    }
    VibeVoiceVLLMProvider(**vibevoice_kwargs).load()
    return Phase1ASRService(
        storage_client=storage_client,
        provider_kwargs=vibevoice_kwargs,
    )


def create_app(
    *,
    service: Phase24MediaPrepService | None = None,
    asr_service: Phase1ASRService | None = None,
) -> FastAPI:
    app = FastAPI(title="Clypt L4 Combined Service")
    app.state.service = service
    app.state.asr_service = asr_service

    def _service() -> Phase24MediaPrepService:
        if app.state.service is None:
            app.state.service = build_default_phase24_media_prep_service()
        return app.state.service

    def _asr_service() -> Phase1ASRService:
        if app.state.asr_service is None:
            app.state.asr_service = build_default_phase1_asr_service()
        return app.state.asr_service

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    def _handle(payload: NodeMediaPrepRequest) -> dict[str, Any]:
        try:
            return _service().handle_request(payload)
        except Exception as exc:
            logger.exception("phase24 media prep failed run_id=%s", payload.run_id)
            raise HTTPException(
                status_code=500,
                detail="phase24 media prep request failed; see service logs",
            ) from exc

    def _handle_asr(payload: Phase1ASRRequest) -> dict[str, Any]:
        try:
            return _asr_service().handle_request(payload)
        except Exception as exc:
            logger.exception("phase1 asr failed audio_gcs_uri=%s", payload.audio_gcs_uri)
            raise HTTPException(
                status_code=500,
                detail="phase1 asr request failed; see service logs",
            ) from exc

    @app.post("/")
    def root_task(payload: NodeMediaPrepRequest) -> dict[str, Any]:
        return _handle(payload)

    @app.post("/tasks/node-media-prep")
    def node_media_prep_task(payload: NodeMediaPrepRequest) -> dict[str, Any]:
        return _handle(payload)

    @app.post("/tasks/asr")
    def asr_task(payload: Phase1ASRRequest) -> dict[str, Any]:
        return _handle_asr(payload)

    return app


__all__ = [
    "Phase24MediaPrepService",
    "Phase1ASRService",
    "build_default_phase1_asr_service",
    "build_default_phase24_media_prep_service",
    "create_app",
]
