"""H200-side Phase 1 sidecar orchestration.

On the H200 the audio chain (VibeVoice + NFA + emotion2vec+ + YAMNet) runs
exclusively on the RTX 6000 Ada box via
:class:`backend.providers.audio_host_client.RemoteAudioChainClient`. The H200
itself runs only the visual extraction (RF-DETR) and coordinates the two
chains.

This module therefore keeps only:

* the concurrent overlap of visual extraction + a single remote audio call,
* the optional early-handoff callback so Phase 2-4 can start while RF-DETR is
  still finishing, and
* stage-event forwarding so Spanner telemetry is identical to the previous
  in-process execution.

All former in-process audio providers (``vibevoice_provider``, ``forced_aligner``,
``emotion_provider``, ``yamnet_provider``) have been removed from this module
and moved to :mod:`backend.runtime.phase1_audio_service.audio_chain` (RTX only).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from backend.providers.audio_host_client import (
    PhaseOneAudioResponse,
    RemoteAudioChainClient,
)

from .models import Phase1SidecarOutputs, Phase1Workspace

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


def run_phase1_sidecars(
    *,
    source_url: str,
    video_gcs_uri: str,
    audio_gcs_uri: str | None = None,
    workspace: Phase1Workspace,
    audio_host_client: RemoteAudioChainClient,
    visual_extractor: Any,
    on_audio_chain_complete: Any | None = None,
    stage_event_logger: Callable[..., None] | None = None,
) -> Phase1SidecarOutputs:
    """Run Phase 1 sidecar tasks against the remote audio host.

    Execution order (H200 perspective)::

        1a. Visual extraction (RF-DETR + ByteTrack) on H200  ─┐ concurrent
        1b. Remote audio chain call (VibeVoice vLLM → NFA →   │  via a
            emotion2vec+ → YAMNet) on RTX 6000 Ada            ─┘  pool.

        ↓ remote chain returns — Phase 2-4 may start immediately while
          RF-DETR continues on the H200.

    Phases 2-4 only depend on the audio-chain outputs (canonical_timeline,
    speech_emotion_timeline, audio_event_timeline). The early-handoff
    callback fires as soon as the remote audio call returns so Phase 2-4
    work can be queued before the visual future finishes, preserving the
    ~230 s savings we had on the ~13-min clip when the chain was in-process.
    """
    if audio_host_client is None:
        raise ValueError(
            "run_phase1_sidecars requires an audio_host_client; the Phase 1 audio "
            "chain has no in-process fallback on the H200."
        )
    if audio_gcs_uri is None or not str(audio_gcs_uri).strip():
        raise ValueError(
            "audio_gcs_uri is required: the RTX audio host fetches audio from GCS."
        )

    t_total = time.perf_counter()

    logger.info(
        "[extract] starting visual extraction (H200) + remote audio chain (RTX) in parallel ..."
    )
    t_overlap = time.perf_counter()

    with ThreadPoolExecutor(max_workers=2) as pool:
        t_visual = time.perf_counter()
        visual_future = pool.submit(
            visual_extractor.extract,
            video_path=workspace.video_path,
            workspace=workspace,
        )
        t_audio = time.perf_counter()
        audio_future = pool.submit(
            audio_host_client.run,
            audio_gcs_uri=audio_gcs_uri,
            source_url=source_url,
            video_gcs_uri=video_gcs_uri,
            run_id=workspace.run_id,
            stage_event_logger=stage_event_logger,
        )

        try:
            audio_response: PhaseOneAudioResponse = audio_future.result()
        except Exception as exc:
            # The client already re-emits a failure stage event, but we also
            # emit a summary audio_host_call failure if none was emitted yet.
            _emit_stage_event(
                stage_event_logger,
                stage_name="audio_host_call",
                status="failed",
                duration_ms=(time.perf_counter() - t_audio) * 1000.0,
                metadata={"audio_gcs_uri": audio_gcs_uri},
                error_payload={
                    "code": exc.__class__.__name__,
                    "message": str(exc)[:2048],
                },
            )
            raise

        diarization_payload = audio_response.diarization_payload
        emotion2vec_payload = audio_response.emotion2vec_payload
        yamnet_payload = audio_response.yamnet_payload

        logger.info(
            "[extract] remote audio chain done — starting Phase 2-4 handoff "
            "while RF-DETR continues on H200 ..."
        )

        if on_audio_chain_complete is not None:
            _partial = Phase1SidecarOutputs(
                phase1_audio={
                    "source_audio": source_url,
                    "video_gcs_uri": video_gcs_uri,
                    "audio_gcs_uri": audio_gcs_uri,
                    "local_video_path": str(workspace.video_path),
                    "local_audio_path": str(workspace.audio_path),
                },
                diarization_payload=diarization_payload,
                phase1_visual={},
                emotion2vec_payload=emotion2vec_payload,
                yamnet_payload=yamnet_payload,
            )
            on_audio_chain_complete(_partial)
            logger.info(
                "[extract] audio-chain callback fired — Phases 2-4 starting while RF-DETR finishes"
            )

        try:
            phase1_visual = visual_future.result()
        except Exception as exc:
            _emit_stage_event(
                stage_event_logger,
                stage_name="visual_extraction",
                status="failed",
                error_payload={"code": exc.__class__.__name__, "message": str(exc)[:2048]},
            )
            raise
        _emit_stage_event(
            stage_event_logger,
            stage_name="visual_extraction",
            status="succeeded",
            duration_ms=(time.perf_counter() - t_visual) * 1000.0,
            metadata={
                "shot_change_count": len((phase1_visual or {}).get("shot_changes") or []),
                "track_count": len((phase1_visual or {}).get("tracks") or []),
            },
        )

    logger.info(
        "[extract] visual + remote audio chain both done in %.1f s",
        time.perf_counter() - t_overlap,
    )

    logger.info("[extract] all sidecars done in %.1f s", time.perf_counter() - t_total)

    return Phase1SidecarOutputs(
        phase1_audio={
            "source_audio": source_url,
            "video_gcs_uri": video_gcs_uri,
            "audio_gcs_uri": audio_gcs_uri,
            "local_video_path": str(workspace.video_path),
            "local_audio_path": str(workspace.audio_path),
        },
        diarization_payload=diarization_payload,
        phase1_visual=phase1_visual,
        emotion2vec_payload=emotion2vec_payload,
        yamnet_payload=yamnet_payload,
    )


__all__ = ["run_phase1_sidecars"]
