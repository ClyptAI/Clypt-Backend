from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .deps import AppDeps, get_app_deps
from .longform import run_longform_vibevoice_asr

logger = logging.getLogger(__name__)


class VibeVoiceAsrRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio_gcs_uri: str
    source_url: str | None = None
    video_gcs_uri: str | None = None
    run_id: str | None = None


def _require_bearer(request: Request, deps: AppDeps) -> None:
    header = request.headers.get("authorization") or ""
    if not header.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")
    token = header.split(" ", 1)[1].strip()
    if token != deps.expected_auth_token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid bearer token")


def _run_vibevoice_asr(*, vibevoice_provider: Any, audio_path: Path, audio_gcs_uri: str | None) -> list[dict[str, Any]]:
    return vibevoice_provider.run(audio_path=str(audio_path), audio_gcs_uri=audio_gcs_uri)


def _probe_audio_duration_s(audio_path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        stderr=subprocess.STDOUT,
    ).decode("utf-8").strip()
    return float(out)


def _extract_shard_audio(
    *,
    source_audio_path: Path,
    output_audio_path: Path,
    start_s: float,
    end_s: float,
) -> None:
    duration_s = end_s - start_s
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_s),
            "-t",
            str(duration_s),
            "-i",
            str(source_audio_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_audio_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Clypt Phase 1 VibeVoice Service", version="1.0.0")
    asr_lock = asyncio.Lock()

    @app.get("/health")
    async def health(deps: AppDeps = Depends(get_app_deps)) -> dict[str, Any]:
        return {
            "status": "ok",
            "scratch_root": str(deps.scratch_root),
            "vibevoice_asr_ready": True,
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
                try:
                    deps.storage_client.download_file(gcs_uri=body.audio_gcs_uri, local_path=audio_path)
                except Exception as exc:
                    logger.exception("[phase1_vibevoice_service] audio download failed")
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"failed to download {body.audio_gcs_uri}: {exc}",
                    ) from exc

                t_asr = time.perf_counter()
                try:
                    audio_duration_s = _probe_audio_duration_s(audio_path)
                    longform_settings = deps.longform_settings
                    single_pass_threshold_s = longform_settings.single_pass_max_minutes * 60
                    if longform_settings.enabled and audio_duration_s > single_pass_threshold_s:
                        outputs = await asyncio.to_thread(
                            run_longform_vibevoice_asr,
                            audio_path=audio_path,
                            canonical_audio_gcs_uri=body.audio_gcs_uri,
                            run_id=body.run_id or "anon",
                            vibevoice_provider=deps.vibevoice_provider,
                            storage_client=deps.storage_client,
                            speaker_verifier=deps.speaker_verifier,
                            duration_s=audio_duration_s,
                            single_pass_max_minutes=longform_settings.single_pass_max_minutes,
                            two_shard_max_minutes=longform_settings.two_shard_max_minutes,
                            four_shard_max_minutes=longform_settings.four_shard_max_minutes,
                            max_shards=longform_settings.max_shards,
                            threshold=longform_settings.speaker_match_threshold,
                            representative_clip_min_s=longform_settings.representative_clip_min_seconds,
                            representative_clip_max_s=longform_settings.representative_clip_max_seconds,
                            extract_shard_audio=_extract_shard_audio,
                        )
                        turns = outputs.turns
                        stage_events.extend(outputs.stage_events)
                    else:
                        turns = await asyncio.to_thread(
                            _run_vibevoice_asr,
                            vibevoice_provider=deps.vibevoice_provider,
                            audio_path=audio_path,
                            audio_gcs_uri=body.audio_gcs_uri,
                        )
                except Exception as exc:
                    stage_events.append(
                        {
                            "stage_name": "vibevoice_asr",
                            "status": "failed",
                            "duration_ms": (time.perf_counter() - t_asr) * 1000.0,
                            "metadata": {},
                            "error_payload": {
                                "code": exc.__class__.__name__,
                                "message": str(exc)[:2048],
                            },
                        }
                    )
                    logger.exception("[phase1_vibevoice_service] VibeVoice ASR failed")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"VibeVoice ASR failed: {exc}",
                    ) from exc
                stage_events.append(
                    {
                        "stage_name": "vibevoice_asr",
                        "status": "succeeded",
                        "duration_ms": (time.perf_counter() - t_asr) * 1000.0,
                        "metadata": {
                            "turn_count": len(turns),
                            "audio_duration_s": audio_duration_s,
                        },
                        "error_payload": None,
                    }
                )

            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return JSONResponse(
                {
                    "run_id": body.run_id,
                    "turns": list(turns),
                    "stage_events": stage_events,
                    "elapsed_ms": elapsed_ms,
                }
            )

    return app


app = create_app()


__all__ = ["app", "create_app"]
