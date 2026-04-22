"""Phase 1 sidecar orchestration.

Phase 1 runs visual extraction (RF-DETR + ByteTrack) and the post-VibeVoice
audio chain (NFA -> emotion2vec+ -> YAMNet) on the same host. VibeVoice ASR
is reached through :class:`backend.providers.audio_host_client.RemoteVibeVoiceAsrClient`,
which now normally targets the local Phase 1 service boundary.

Execution order (H200 perspective)::

    1a. Visual extraction (RF-DETR + ByteTrack)          ─┐ concurrent
    1b. VibeVoice ASR service call                       ─┘  via a pool.

        ↓ ASR returns — local NFA → emotion2vec+ → YAMNet starts
          concurrently with RF-DETR.

        ↓ local audio chain done — optional callback fires so Phase 2-4
          can start while RF-DETR continues.

Phase 2-4 only depend on the audio-chain outputs (canonical_timeline,
speech_emotion_timeline, audio_event_timeline). The early-handoff callback
fires as soon as the local audio chain is done so Phase 2-4 work can be
queued before the visual future finishes.

NFA/emotion/YAMNet remain local to the Phase 1 host and intentionally do not
get split into separate services.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from backend.pipeline.timeline.vibevoice_merge import merge_vibevoice_outputs
from backend.providers.audio_host_client import (
    RemoteVibeVoiceAsrClient,
    VibeVoiceAsrResponse,
)

from .models import Phase1Workspace
from .payloads import (
    DiarizationPayload,
    EmotionSegmentsPayload,
    Phase1AudioAssets,
    Phase1SidecarOutputs,
    VisualPayload,
    YamnetPayload,
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


def _is_forced_alignment_required() -> bool:
    return (os.getenv("CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT") or "1") == "1"


def _run_audio_chain(
    *,
    workspace: Phase1Workspace,
    vibevoice_turns: list[dict],
    forced_aligner: Any,
    emotion_provider: Any,
    yamnet_provider: Any,
    stage_event_logger: Callable[..., None] | None = None,
) -> tuple[DiarizationPayload, EmotionSegmentsPayload, YamnetPayload]:
    """NFA → emotion2vec+ → YAMNet, serial with each other to avoid CUDA graph
    conflicts between them. Returns ``(diarization_payload, emotion2vec_payload,
    yamnet_payload)``.

    Runs on the H200 in a ThreadPoolExecutor worker thread while RF-DETR
        visual extraction is still on the main GPU stream. The default target
        is an H200 with enough headroom to co-locate RF-DETR + NFA global
        alignment without starvation.
    """
    prelim_turns = []
    for idx, t in enumerate(vibevoice_turns, start=1):
        prelim_turns.append(
            {
                "turn_id": f"t_{idx:06d}",
                "speaker_id": f"SPEAKER_{int(t.get('Speaker') or 0)}",
                "start_ms": int(round(float(t.get("Start") or 0) * 1000)),
                "end_ms": int(round(float(t.get("End") or 0) * 1000)),
                "transcript_text": str(t.get("Content") or "").strip(),
            }
        )

    logger.info("[extract] starting forced-alignment on %d turns ...", len(prelim_turns))
    t_fa = time.perf_counter()
    try:
        word_alignments = forced_aligner.run(
            audio_path=workspace.audio_path,
            turns=prelim_turns,
        )
    except Exception as exc:
        _emit_stage_event(
            stage_event_logger,
            stage_name="forced_alignment",
            status="failed",
            error_payload={"code": exc.__class__.__name__, "message": str(exc)[:2048]},
        )
        raise
    forced_alignment_ms = (time.perf_counter() - t_fa) * 1000.0
    forced_alignment_metrics = (
        getattr(forced_aligner, "last_run_metrics", {})
        if getattr(forced_aligner, "last_run_metrics", None) is not None
        else {}
    )
    logger.info(
        "[extract] forced-alignment done in %.1f s — %d words",
        forced_alignment_ms / 1000.0,
        len(word_alignments),
    )
    alignable_turns = [
        t
        for t in prelim_turns
        if int(t.get("end_ms") or 0) > int(t.get("start_ms") or 0)
        and str(t.get("transcript_text") or "").strip()
    ]
    if alignable_turns and not word_alignments:
        msg = (
            "[extract] forced-alignment produced 0 words for non-empty ASR turns; "
            "this indicates NFA is unhealthy."
        )
        if _is_forced_alignment_required():
            _emit_stage_event(
                stage_event_logger,
                stage_name="forced_alignment",
                status="failed",
                duration_ms=forced_alignment_ms,
                metadata={
                    "turn_count": len(prelim_turns),
                    "word_count": len(word_alignments),
                    "forced_alignment_ms": forced_alignment_ms,
                    "forced_alignment_chunk_count": int(
                        forced_alignment_metrics.get("chunk_count") or 0
                    ),
                },
                error_payload={"code": "RuntimeError", "message": msg[:2048]},
            )
            raise RuntimeError(
                f"{msg} Set CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT=0 only for temporary debugging bypass."
            )
        logger.warning("%s Continuing because CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT=0.", msg)

    _emit_stage_event(
        stage_event_logger,
        stage_name="forced_alignment",
        status="succeeded",
        duration_ms=forced_alignment_ms,
        metadata={
            "turn_count": len(prelim_turns),
            "word_count": len(word_alignments),
            "forced_alignment_ms": forced_alignment_ms,
            "forced_alignment_chunk_count": int(forced_alignment_metrics.get("chunk_count") or 0),
        },
    )

    merged = merge_vibevoice_outputs(
        vibevoice_turns=vibevoice_turns,
        word_alignments=word_alignments,
    )
    turns = list(merged.get("turns") or [])

    diarization_payload = DiarizationPayload.model_validate(
        {
            "turns": [
                {
                    "turn_id": t["turn_id"],
                    "speaker_id": t["speaker_id"],
                    "start_ms": t["start_ms"],
                    "end_ms": t["end_ms"],
                    "transcript_text": t["transcript_text"],
                    "word_ids": t.get("word_ids", []),
                    "identification_match": None,
                }
                for t in turns
            ],
            "words": merged.get("words", []),
        }
    )

    logger.info("[extract] starting emotion2vec+ on %d turns ...", len(turns))
    t_emotion = time.perf_counter()
    try:
        emotion2vec_payload = EmotionSegmentsPayload.model_validate(
            emotion_provider.run(
                audio_path=workspace.audio_path,
                turns=turns,
            )
        )
    except Exception as exc:
        _emit_stage_event(
            stage_event_logger,
            stage_name="emotion2vec",
            status="failed",
            error_payload={"code": exc.__class__.__name__, "message": str(exc)[:2048]},
        )
        raise
    emotion_ms = (time.perf_counter() - t_emotion) * 1000.0
    emotion_metrics = (
        getattr(emotion_provider, "last_run_metrics", {})
        if getattr(emotion_provider, "last_run_metrics", None) is not None
        else {}
    )
    logger.info("[extract] emotion2vec+ done in %.1f s", emotion_ms / 1000.0)
    emotion_segments = len(emotion2vec_payload.segments)
    _emit_stage_event(
        stage_event_logger,
        stage_name="emotion2vec",
        status="succeeded",
        duration_ms=emotion_ms,
        metadata={
            "turn_count": len(turns),
            "segment_count": emotion_segments,
            "emotion_clip_extract_ms": float(emotion_metrics.get("clip_extract_ms") or 0.0),
            "emotion_infer_ms": float(emotion_metrics.get("infer_ms") or 0.0),
        },
    )

    logger.info("[extract] starting YAMNet ...")
    t_yamnet = time.perf_counter()
    try:
        yamnet_payload = YamnetPayload.model_validate(
            yamnet_provider.run(audio_path=workspace.audio_path)
        )
    except Exception as exc:
        _emit_stage_event(
            stage_event_logger,
            stage_name="yamnet",
            status="failed",
            error_payload={"code": exc.__class__.__name__, "message": str(exc)[:2048]},
        )
        raise
    yamnet_event_count = len(yamnet_payload.events)
    yamnet_ms = (time.perf_counter() - t_yamnet) * 1000.0
    logger.info(
        "[extract] YAMNet done in %.1f s — %d events",
        yamnet_ms / 1000.0,
        yamnet_event_count,
    )
    _emit_stage_event(
        stage_event_logger,
        stage_name="yamnet",
        status="succeeded",
        duration_ms=yamnet_ms,
        metadata={"event_count": yamnet_event_count, "yamnet_ms": yamnet_ms},
    )

    return diarization_payload, emotion2vec_payload, yamnet_payload


def run_phase1_sidecars(
    *,
    source_url: str,
    video_gcs_uri: str,
    audio_gcs_uri: str | None = None,
    source_context: dict[str, Any] | None = None,
    workspace: Phase1Workspace,
    vibevoice_asr_client: RemoteVibeVoiceAsrClient,
    forced_aligner: Any,
    visual_extractor: Any,
    emotion_provider: Any,
    yamnet_provider: Any,
    on_audio_chain_complete: Callable[[Phase1SidecarOutputs], None] | None = None,
    stage_event_logger: Callable[..., None] | None = None,
) -> Phase1SidecarOutputs:
    """Run Phase 1 sidecar tasks with service-routed VibeVoice ASR + local audio chain.

    Execution order::

        1a. Visual extraction (RF-DETR + ByteTrack)           ─┐ concurrent
        1b. VibeVoice ASR service call                        ─┘

            ↓ ASR done — NFA → emotion2vec+ → YAMNet starts locally,
              concurrent with RF-DETR, serial with each other.

    Phase 2-4 only depend on the audio-chain outputs. The early-handoff
    callback fires as soon as the local audio chain returns so Phase 2-4
    work can be queued before the visual future finishes.
    """
    if vibevoice_asr_client is None:
        raise ValueError(
            "run_phase1_sidecars requires a vibevoice_asr_client; VibeVoice ASR "
            "must be routed through the Phase 1 service boundary."
        )
    if audio_gcs_uri is None or not str(audio_gcs_uri).strip():
        raise ValueError(
            "audio_gcs_uri is required: the VibeVoice ASR service fetches audio "
            "from GCS."
        )
    if not getattr(vibevoice_asr_client, "supports_concurrent_visual", False):
        raise RuntimeError(
            "Phase 1 requires a concurrent-capable VibeVoice ASR client."
        )

    t_total = time.perf_counter()

    logger.info(
        "[extract] starting visual extraction (H200) + remote VibeVoice ASR (RTX) "
        "in parallel ..."
    )
    t_overlap = time.perf_counter()

    with ThreadPoolExecutor(max_workers=3) as pool:
        t_visual = time.perf_counter()
        visual_future = pool.submit(
            visual_extractor.extract,
            video_path=workspace.video_path,
            workspace=workspace,
        )
        t_asr = time.perf_counter()
        asr_future = pool.submit(
            vibevoice_asr_client.run,
            audio_gcs_uri=audio_gcs_uri,
            source_url=source_url,
            video_gcs_uri=video_gcs_uri,
            run_id=workspace.run_id,
            stage_event_logger=stage_event_logger,
        )

        try:
            asr_response: VibeVoiceAsrResponse = asr_future.result()
        except Exception as exc:
            # The client already re-emits a failure stage event, but we also
            # emit a summary one in case none reached the logger.
            _emit_stage_event(
                stage_event_logger,
                stage_name="vibevoice_asr",
                status="failed",
                duration_ms=(time.perf_counter() - t_asr) * 1000.0,
                metadata={"audio_gcs_uri": audio_gcs_uri},
                error_payload={
                    "code": exc.__class__.__name__,
                    "message": str(exc)[:2048],
                },
            )
            raise

        vibevoice_turns = asr_response.turns
        logger.info(
            "[extract] remote VibeVoice ASR done (%d turns) — starting local audio "
            "chain while RF-DETR continues ...",
            len(vibevoice_turns),
        )

        audio_chain_future = pool.submit(
            _run_audio_chain,
            workspace=workspace,
            vibevoice_turns=vibevoice_turns,
            forced_aligner=forced_aligner,
            emotion_provider=emotion_provider,
            yamnet_provider=yamnet_provider,
            stage_event_logger=stage_event_logger,
        )

        diarization_payload, emotion2vec_payload, yamnet_payload = (
            audio_chain_future.result()
        )

        if on_audio_chain_complete is not None:
            _partial = Phase1SidecarOutputs(
                phase1_audio=Phase1AudioAssets(
                    source_audio=source_url,
                    video_gcs_uri=video_gcs_uri,
                    audio_gcs_uri=str(audio_gcs_uri),
                    local_video_path=str(workspace.video_path),
                    local_audio_path=str(workspace.audio_path),
                ),
                diarization_payload=diarization_payload,
                phase1_visual=VisualPayload(),
                emotion2vec_payload=emotion2vec_payload,
                yamnet_payload=yamnet_payload,
                source_context=source_context,
            )
            on_audio_chain_complete(_partial)
            logger.info(
                "[extract] audio-chain callback fired — Phases 2-4 starting while "
                "RF-DETR finishes"
            )

        try:
            phase1_visual = VisualPayload.model_validate(visual_future.result())
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
                "shot_change_count": len(phase1_visual.shot_changes),
                "track_count": len(phase1_visual.tracks),
            },
        )

    logger.info(
        "[extract] visual + audio chain both done in %.1f s",
        time.perf_counter() - t_overlap,
    )

    logger.info("[extract] all sidecars done in %.1f s", time.perf_counter() - t_total)

    return Phase1SidecarOutputs(
        phase1_audio=Phase1AudioAssets(
            source_audio=source_url,
            video_gcs_uri=video_gcs_uri,
            audio_gcs_uri=str(audio_gcs_uri),
            local_video_path=str(workspace.video_path),
            local_audio_path=str(workspace.audio_path),
        ),
        diarization_payload=diarization_payload,
        phase1_visual=phase1_visual,
        emotion2vec_payload=emotion2vec_payload,
        yamnet_payload=yamnet_payload,
        source_context=source_context,
    )


__all__ = ["run_phase1_sidecars"]
