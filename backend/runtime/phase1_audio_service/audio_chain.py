"""Pure-Python Phase 1 audio chain run on the RTX 6000 Ada audio host.

This module is imported **only** on the RTX box. It runs the full audio
pipeline — VibeVoice ASR (via the local vLLM service on 127.0.0.1:8000) → NeMo
Forced Aligner → emotion2vec+ → YAMNet — serially on the hot GPU and returns
the merged payload expected by downstream Phase 1 code.

The H200 orchestrator invokes this chain via
``backend.providers.audio_host_client.RemoteAudioChainClient``; the FastAPI
route in :mod:`backend.runtime.phase1_audio_service.app` wraps this function.

The function is intentionally pure and provider-agnostic so it can be unit
tested with stub providers without spinning up any real ML services.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

from backend.pipeline.timeline.vibevoice_merge import merge_vibevoice_outputs

logger = logging.getLogger(__name__)


StageEventRecorder = Callable[..., None]


def _record_stage_event(
    recorder: StageEventRecorder | None,
    *,
    stage_name: str,
    status: str,
    duration_ms: float | None = None,
    metadata: dict[str, Any] | None = None,
    error_payload: dict[str, Any] | None = None,
) -> None:
    if recorder is None:
        return
    try:
        recorder(
            stage_name=stage_name,
            status=status,
            duration_ms=duration_ms,
            metadata=metadata or {},
            error_payload=error_payload,
        )
    except Exception:  # pragma: no cover - defensive
        logger.exception(
            "[audio_chain] stage recorder failed stage=%s status=%s",
            stage_name,
            status,
        )


def _is_forced_alignment_required() -> bool:
    return (os.getenv("CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT") or "1") == "1"


def run_audio_chain(
    *,
    audio_path: Path,
    vibevoice_provider: Any,
    forced_aligner: Any,
    emotion_provider: Any,
    yamnet_provider: Any,
    audio_gcs_uri: str | None = None,
    stage_event_recorder: StageEventRecorder | None = None,
) -> dict[str, Any]:
    """Run the full Phase 1 audio chain on the hot GPU.

    Returns a dict in the exact shape emitted by ``POST /tasks/phase1-audio``:

        {
            "turns": [...],            # raw VibeVoice turns
            "diarization_payload": {...},
            "emotion2vec_payload": {...},
            "yamnet_payload": {...},
        }

    Stage timing/metadata is emitted through ``stage_event_recorder`` (if
    provided). The FastAPI route collects these events and forwards them in
    the HTTP response so the H200 can re-emit them to its Spanner metrics.

    Parameters
    ----------
    audio_path:
        Local filesystem path to the 16 kHz mono WAV the chain should ingest.
    vibevoice_provider / forced_aligner / emotion_provider / yamnet_provider:
        Concrete providers constructed in the FastAPI app's ``deps`` module.
    audio_gcs_uri:
        Optional GCS pointer. VibeVoice's ``url`` audio mode uses this when
        available; for local-file mode it is ignored.
    stage_event_recorder:
        Optional callable invoked for each stage with the same keyword args as
        the H200-side ``stage_event_logger`` so the service can mirror them.
    """
    audio_path_str = str(audio_path)

    # ------------------------------------------------------------------
    # VibeVoice ASR
    # ------------------------------------------------------------------
    logger.info("[audio_chain] starting VibeVoice ASR ...")
    t_asr = time.perf_counter()
    try:
        if audio_gcs_uri:
            try:
                vibevoice_turns = vibevoice_provider.run(
                    audio_path=audio_path_str,
                    audio_gcs_uri=audio_gcs_uri,
                )
            except TypeError as exc:
                if "audio_gcs_uri" not in str(exc):
                    raise
                vibevoice_turns = vibevoice_provider.run(audio_path=audio_path_str)
        else:
            vibevoice_turns = vibevoice_provider.run(audio_path=audio_path_str)
    except Exception as exc:
        _record_stage_event(
            stage_event_recorder,
            stage_name="vibevoice_asr",
            status="failed",
            duration_ms=(time.perf_counter() - t_asr) * 1000.0,
            error_payload={"code": exc.__class__.__name__, "message": str(exc)[:2048]},
        )
        raise
    _record_stage_event(
        stage_event_recorder,
        stage_name="vibevoice_asr",
        status="succeeded",
        duration_ms=(time.perf_counter() - t_asr) * 1000.0,
        metadata={"turn_count": len(vibevoice_turns)},
    )

    # ------------------------------------------------------------------
    # Forced alignment
    # ------------------------------------------------------------------
    prelim_turns = [
        {
            "turn_id": f"t_{idx:06d}",
            "speaker_id": f"SPEAKER_{int(t.get('Speaker') or 0)}",
            "start_ms": int(round(float(t.get("Start") or 0) * 1000)),
            "end_ms": int(round(float(t.get("End") or 0) * 1000)),
            "transcript_text": str(t.get("Content") or "").strip(),
        }
        for idx, t in enumerate(vibevoice_turns, start=1)
    ]

    logger.info(
        "[audio_chain] starting forced-alignment on %d turns ...",
        len(prelim_turns),
    )
    t_fa = time.perf_counter()
    try:
        word_alignments = forced_aligner.run(
            audio_path=audio_path_str,
            turns=prelim_turns,
        )
    except Exception as exc:
        _record_stage_event(
            stage_event_recorder,
            stage_name="forced_alignment",
            status="failed",
            error_payload={"code": exc.__class__.__name__, "message": str(exc)[:2048]},
        )
        raise

    alignable_turns = [
        t
        for t in prelim_turns
        if int(t.get("end_ms") or 0) > int(t.get("start_ms") or 0)
        and str(t.get("transcript_text") or "").strip()
    ]
    if alignable_turns and not word_alignments:
        msg = (
            "[audio_chain] forced-alignment produced 0 words for non-empty ASR turns; "
            "this indicates NFA is unhealthy."
        )
        if _is_forced_alignment_required():
            _record_stage_event(
                stage_event_recorder,
                stage_name="forced_alignment",
                status="failed",
                duration_ms=(time.perf_counter() - t_fa) * 1000.0,
                metadata={
                    "turn_count": len(prelim_turns),
                    "word_count": len(word_alignments),
                },
                error_payload={"code": "RuntimeError", "message": msg[:2048]},
            )
            raise RuntimeError(
                f"{msg} Set CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT=0 only for temporary debugging bypass."
            )
        logger.warning(
            "%s Continuing because CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT=0.",
            msg,
        )

    _record_stage_event(
        stage_event_recorder,
        stage_name="forced_alignment",
        status="succeeded",
        duration_ms=(time.perf_counter() - t_fa) * 1000.0,
        metadata={"turn_count": len(prelim_turns), "word_count": len(word_alignments)},
    )

    merged = merge_vibevoice_outputs(
        vibevoice_turns=vibevoice_turns,
        word_alignments=word_alignments,
    )
    turns = list(merged.get("turns") or [])
    diarization_payload: dict[str, Any] = {
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

    # ------------------------------------------------------------------
    # emotion2vec+
    # ------------------------------------------------------------------
    logger.info("[audio_chain] starting emotion2vec+ on %d turns ...", len(turns))
    t_emotion = time.perf_counter()
    try:
        emotion2vec_payload = emotion_provider.run(
            audio_path=audio_path_str,
            turns=turns,
        )
    except Exception as exc:
        _record_stage_event(
            stage_event_recorder,
            stage_name="emotion2vec",
            status="failed",
            error_payload={"code": exc.__class__.__name__, "message": str(exc)[:2048]},
        )
        raise
    emotion_segments = len((emotion2vec_payload or {}).get("segments") or [])
    _record_stage_event(
        stage_event_recorder,
        stage_name="emotion2vec",
        status="succeeded",
        duration_ms=(time.perf_counter() - t_emotion) * 1000.0,
        metadata={"turn_count": len(turns), "segment_count": emotion_segments},
    )

    # ------------------------------------------------------------------
    # YAMNet
    # ------------------------------------------------------------------
    logger.info("[audio_chain] starting YAMNet ...")
    t_yamnet = time.perf_counter()
    try:
        yamnet_payload = yamnet_provider.run(audio_path=audio_path_str)
    except Exception as exc:
        _record_stage_event(
            stage_event_recorder,
            stage_name="yamnet",
            status="failed",
            error_payload={"code": exc.__class__.__name__, "message": str(exc)[:2048]},
        )
        raise
    yamnet_event_count = len((yamnet_payload or {}).get("events") or [])
    _record_stage_event(
        stage_event_recorder,
        stage_name="yamnet",
        status="succeeded",
        duration_ms=(time.perf_counter() - t_yamnet) * 1000.0,
        metadata={"event_count": yamnet_event_count},
    )

    return {
        "turns": list(vibevoice_turns),
        "diarization_payload": diarization_payload,
        "emotion2vec_payload": dict(emotion2vec_payload or {}),
        "yamnet_payload": dict(yamnet_payload or {}),
    }


__all__ = ["run_audio_chain"]
