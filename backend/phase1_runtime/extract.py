"""Phase 1 Scribe + Modal visual orchestration.

Phase 1 now submits the Modal RF-DETR-Seg visual job, runs ElevenLabs Scribe v2
synchronously from a signed GCS audio URL, adapts the Scribe response into the
canonical audio payloads, then hands off to Phase26 while RF-DETR-Seg is still
pending. There is no local VibeVoice/NFA/emotion2vec+/YAMNet fallback on this
branch.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from .models import Phase1Workspace
from .payloads import (
    Phase1AudioAssets,
    Phase1SidecarOutputs,
)

logger = logging.getLogger(__name__)


def _emit_stage_event(
    stage_event_logger: Callable[..., None] | None,
    *,
    stage_name: str,
    status: str,
    duration_ms: float | None = None,
    metadata: dict[str, Any] | None = None,
    error_payload: dict[str, Any] | None = None,
) -> None:
    if stage_event_logger is None:
        return
    try:
        stage_event_logger(
            stage_name=stage_name,
            status=status,
            duration_ms=duration_ms,
            metadata=metadata or {},
            error_payload=error_payload,
        )
    except Exception:  # pragma: no cover - defensive logging path
        logger.exception(
            "[extract] stage_event_logger failed for stage=%s status=%s",
            stage_name,
            status,
        )


def _call_scribe_provider(
    *,
    scribe_provider: Any,
    signed_audio_url: str,
    audio_gcs_uri: str,
    source_url: str,
    video_gcs_uri: str,
    run_id: str,
) -> dict[str, Any]:
    if hasattr(scribe_provider, "transcribe"):
        return scribe_provider.transcribe(
            source_url=signed_audio_url,
            audio_gcs_uri=audio_gcs_uri,
            original_source_url=source_url,
            video_gcs_uri=video_gcs_uri,
            run_id=run_id,
        )
    if hasattr(scribe_provider, "run"):
        return scribe_provider.run(
            source_url=signed_audio_url,
            audio_gcs_uri=audio_gcs_uri,
            original_source_url=source_url,
            video_gcs_uri=video_gcs_uri,
            run_id=run_id,
        )
    raise TypeError("scribe_provider must expose transcribe(...) or run(...)")


def _scribe_raw_dict(raw_scribe: Any) -> dict[str, Any]:
    if hasattr(raw_scribe, "raw"):
        raw_scribe = raw_scribe.raw
    if not isinstance(raw_scribe, dict):
        raise TypeError(
            f"Scribe provider returned {type(raw_scribe).__name__}; expected dict-like transcript."
        )
    return raw_scribe


def _run_phase1_sidecars_scribe(
    *,
    source_url: str,
    video_gcs_uri: str,
    audio_gcs_uri: str,
    signed_audio_url: str,
    source_context: dict[str, Any] | None,
    workspace: Phase1Workspace,
    scribe_provider: Any,
    visual_extractor: Any,
    scribe_turn_gap_ms: int,
    on_audio_chain_complete: Callable[[Phase1SidecarOutputs], None] | None,
    stage_event_logger: Callable[..., None] | None,
) -> Phase1SidecarOutputs:
    from .scribe_adapter import adapt_scribe_response

    t_total = time.perf_counter()
    logger.info("[extract] submitting Modal RF-DETR-Seg visual future ...")
    t_visual_submit = time.perf_counter()
    try:
        visual_future = visual_extractor.submit(
            run_id=workspace.run_id,
            video_gcs_uri=video_gcs_uri,
            video_path=workspace.video_path,
        )
    except Exception as exc:
        _emit_stage_event(
            stage_event_logger,
            stage_name="visual_extraction_submit",
            status="failed",
            duration_ms=(time.perf_counter() - t_visual_submit) * 1000.0,
            error_payload={"code": exc.__class__.__name__, "message": str(exc)[:2048]},
        )
        raise
    _emit_stage_event(
        stage_event_logger,
        stage_name="visual_extraction_submit",
        status="succeeded",
        duration_ms=(time.perf_counter() - t_visual_submit) * 1000.0,
        metadata={
            "backend": visual_future.backend,
            "call_id": visual_future.call_id,
            "source_video_gcs_uri": visual_future.source_video_gcs_uri,
        },
    )

    logger.info("[extract] starting ElevenLabs Scribe v2 transcription ...")
    t_scribe = time.perf_counter()
    try:
        raw_scribe = _call_scribe_provider(
            scribe_provider=scribe_provider,
            signed_audio_url=signed_audio_url,
            audio_gcs_uri=audio_gcs_uri,
            source_url=source_url,
            video_gcs_uri=video_gcs_uri,
            run_id=workspace.run_id,
        )
    except Exception as exc:
        _emit_stage_event(
            stage_event_logger,
            stage_name="scribe_transcription",
            status="failed",
            duration_ms=(time.perf_counter() - t_scribe) * 1000.0,
            metadata={"audio_gcs_uri": audio_gcs_uri},
            error_payload={"code": exc.__class__.__name__, "message": str(exc)[:2048]},
        )
        raise
    scribe_ms = (time.perf_counter() - t_scribe) * 1000.0
    raw_scribe_dict = _scribe_raw_dict(raw_scribe)
    _emit_stage_event(
        stage_event_logger,
        stage_name="scribe_transcription",
        status="succeeded",
        duration_ms=scribe_ms,
        metadata={
            "audio_gcs_uri": audio_gcs_uri,
            "word_count": len(raw_scribe_dict.get("words") or []),
            "language_code": raw_scribe_dict.get("language_code"),
        },
    )

    t_adapter = time.perf_counter()
    adapted_payloads = adapt_scribe_response(
        raw_scribe,
        turn_gap_ms=scribe_turn_gap_ms,
    )
    diarization_payload = adapted_payloads.diarization_payload
    emotion2vec_payload = adapted_payloads.emotion2vec_payload
    yamnet_payload = adapted_payloads.yamnet_payload
    _emit_stage_event(
        stage_event_logger,
        stage_name="scribe_adapter",
        status="succeeded",
        duration_ms=(time.perf_counter() - t_adapter) * 1000.0,
        metadata={
            "turn_count": len(diarization_payload.turns),
            "word_count": len(diarization_payload.words),
            "audio_event_count": len(yamnet_payload.events),
        },
    )

    outputs = Phase1SidecarOutputs(
        phase1_audio=Phase1AudioAssets(
            source_audio=source_url,
            video_gcs_uri=video_gcs_uri,
            audio_gcs_uri=audio_gcs_uri,
            local_video_path=str(workspace.video_path),
            local_audio_path=str(workspace.audio_path),
        ),
        diarization_payload=diarization_payload,
        phase1_visual_status="pending",
        phase1_visual=None,
        visual_future=visual_future,
        emotion2vec_payload=emotion2vec_payload,
        yamnet_payload=yamnet_payload,
        source_context=source_context,
    )
    if on_audio_chain_complete is not None:
        on_audio_chain_complete(outputs)
        logger.info("[extract] Scribe callback fired — Phase26 can start while RF-DETR-Seg runs")

    logger.info("[extract] Scribe audio path done in %.1f s", time.perf_counter() - t_total)
    return outputs


def run_phase1_sidecars(
    *,
    source_url: str,
    video_gcs_uri: str,
    audio_gcs_uri: str | None = None,
    signed_audio_url: str | None = None,
    source_context: dict[str, Any] | None = None,
    workspace: Phase1Workspace,
    scribe_provider: Any | None = None,
    scribe_turn_gap_ms: int = 1200,
    visual_extractor: Any,
    on_audio_chain_complete: Callable[[Phase1SidecarOutputs], None] | None = None,
    stage_event_logger: Callable[..., None] | None = None,
) -> Phase1SidecarOutputs:
    """Run Phase 1 sidecars through Scribe and the remote Modal visual worker."""
    if scribe_provider is None:
        raise ValueError("run_phase1_sidecars requires an ElevenLabs Scribe provider.")
    if audio_gcs_uri is None or not str(audio_gcs_uri).strip():
        raise ValueError("audio_gcs_uri is required for Scribe transcription.")
    if signed_audio_url is None or not str(signed_audio_url).strip():
        raise ValueError("signed_audio_url is required for Scribe transcription.")
    return _run_phase1_sidecars_scribe(
        source_url=source_url,
        video_gcs_uri=video_gcs_uri,
        audio_gcs_uri=str(audio_gcs_uri),
        signed_audio_url=str(signed_audio_url),
        source_context=source_context,
        workspace=workspace,
        scribe_provider=scribe_provider,
        visual_extractor=visual_extractor,
        scribe_turn_gap_ms=scribe_turn_gap_ms,
        on_audio_chain_complete=on_audio_chain_complete,
        stage_event_logger=stage_event_logger,
    )


__all__ = ["run_phase1_sidecars"]
