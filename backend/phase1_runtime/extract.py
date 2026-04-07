from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from backend.pipeline.timeline.vibevoice_merge import merge_vibevoice_outputs

from .models import Phase1SidecarOutputs, Phase1Workspace

logger = logging.getLogger(__name__)


def _run_audio_chain(
    *,
    workspace: Phase1Workspace,
    vibevoice_turns: list[dict],
    forced_aligner: Any,
    emotion_provider: Any,
    yamnet_provider: Any,
) -> tuple[dict, Any, Any]:
    """
    NFA → emotion2vec+ → YAMNet, always serial with each other to avoid
    CUDA graph conflicts between them.  Returns (diarization_payload,
    emotion2vec_payload, yamnet_payload).

    On the vLLM path this runs in a ThreadPoolExecutor worker thread while
    RF-DETR visual extraction is still running on the main GPU stream.
    """
    # Build prelim turns for the aligner
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
    word_alignments = forced_aligner.run(
        audio_path=workspace.audio_path,
        turns=prelim_turns,
    )
    logger.info(
        "[extract] forced-alignment done in %.1f s — %d words",
        time.perf_counter() - t_fa,
        len(word_alignments),
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

    logger.info("[extract] starting emotion2vec+ on %d turns ...", len(turns))
    t_emotion = time.perf_counter()
    emotion2vec_payload = emotion_provider.run(
        audio_path=workspace.audio_path,
        turns=turns,
    )
    logger.info("[extract] emotion2vec+ done in %.1f s", time.perf_counter() - t_emotion)

    logger.info("[extract] starting YAMNet ...")
    t_yamnet = time.perf_counter()
    yamnet_payload = yamnet_provider.run(audio_path=workspace.audio_path)
    yamnet_event_count = len((yamnet_payload or {}).get("events") or [])
    logger.info(
        "[extract] YAMNet done in %.1f s — %d events",
        time.perf_counter() - t_yamnet,
        yamnet_event_count,
    )

    return diarization_payload, emotion2vec_payload, yamnet_payload


def run_phase1_sidecars(
    *,
    source_url: str,
    video_gcs_uri: str,
    workspace: Phase1Workspace,
    vibevoice_provider: Any,
    forced_aligner: Any,
    visual_extractor: Any,
    emotion_provider: Any,
    yamnet_provider: Any,
    on_audio_chain_complete: Any | None = None,
) -> Phase1SidecarOutputs:
    """
    Run all Phase 1 sidecar tasks on the GPU.

    Serial order (native/hf backends — all GPU, no thread contention):
      1. Visual extraction (RF-DETR + ByteTrack)
      2. VibeVoice ASR (diarization + transcription)
      3. NeMo Forced Aligner (word-level timestamps)
      4. emotion2vec+
      5. YAMNet

    Overlapped order (vLLM backend — ASR is an HTTP call, not a GPU op):
      1a. Visual extraction  ─┐ concurrent via ThreadPoolExecutor(max_workers=3)
      1b. vLLM ASR           ─┘
          ↓ ASR done — audio chain starts immediately, visual still running
      2.  NeMo Forced Aligner  ─┐
      3.  emotion2vec+          ├─ concurrent with RF-DETR, serial with each other
      4.  YAMNet               ─┘

    Phases 2–4 only need the audio chain outputs (canonical_timeline,
    speech_emotion_timeline, audio_event_timeline).  RF-DETR output is only
    needed by Phase 5.  Starting the audio chain as soon as ASR finishes
    means Phases 2–4 artifacts are ready ~230 s earlier on a ~13-min clip.
    """
    t_total = time.perf_counter()

    # VibeVoiceVLLMProvider sets supports_concurrent_visual=True because it
    # calls a persistent HTTP service rather than the local GPU, so there is
    # no CUDA memory contention with visual extraction.
    _overlap = getattr(vibevoice_provider, "supports_concurrent_visual", False)

    if _overlap:
        # --- Overlapped: visual ∥ ASR, then audio chain immediately after ASR -
        logger.info("[extract] starting visual extraction + vLLM ASR in parallel ...")
        t_overlap = time.perf_counter()

        with ThreadPoolExecutor(max_workers=3) as pool:
            visual_future = pool.submit(
                visual_extractor.extract,
                video_path=workspace.video_path,
                workspace=workspace,
            )
            asr_future = pool.submit(
                vibevoice_provider.run,
                audio_path=workspace.audio_path,
            )

            # Block until ASR completes, then immediately start the audio chain
            # as a third worker — RF-DETR is still running concurrently.
            vibevoice_turns = asr_future.result()
            logger.info(
                "[extract] ASR done (%d turns) — starting audio chain "
                "while RF-DETR still running ...",
                len(vibevoice_turns),
            )
            audio_chain_future = pool.submit(
                _run_audio_chain,
                workspace=workspace,
                vibevoice_turns=vibevoice_turns,
                forced_aligner=forced_aligner,
                emotion_provider=emotion_provider,
                yamnet_provider=yamnet_provider,
            )

            # Block until audio chain completes — fire callback immediately,
            # before waiting for RF-DETR.
            diarization_payload, emotion2vec_payload, yamnet_payload = (
                audio_chain_future.result()
            )

            if on_audio_chain_complete is not None:
                _partial = Phase1SidecarOutputs(
                    phase1_audio={
                        "source_audio": source_url,
                        "video_gcs_uri": video_gcs_uri,
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

            # Now wait for RF-DETR (may already be done if it finished first).
            phase1_visual = visual_future.result()

        logger.info(
            "[extract] visual + audio chain both done in %.1f s",
            time.perf_counter() - t_overlap,
        )

    else:
        # --- Serial: visual → ASR → audio chain ------------------------------
        logger.info("[extract] starting visual extraction ...")
        t_visual = time.perf_counter()
        phase1_visual = visual_extractor.extract(
            video_path=workspace.video_path,
            workspace=workspace,
        )
        logger.info(
            "[extract] visual extraction done in %.1f s", time.perf_counter() - t_visual
        )

        logger.info("[extract] starting VibeVoice ASR ...")
        t_vv = time.perf_counter()
        vibevoice_turns = vibevoice_provider.run(audio_path=workspace.audio_path)
        logger.info(
            "[extract] VibeVoice done in %.1f s — %d turns",
            time.perf_counter() - t_vv,
            len(vibevoice_turns),
        )

        diarization_payload, emotion2vec_payload, yamnet_payload = _run_audio_chain(
            workspace=workspace,
            vibevoice_turns=vibevoice_turns,
            forced_aligner=forced_aligner,
            emotion_provider=emotion_provider,
            yamnet_provider=yamnet_provider,
        )

    logger.info("[extract] all sidecars done in %.1f s", time.perf_counter() - t_total)

    return Phase1SidecarOutputs(
        phase1_audio={
            "source_audio": source_url,
            "video_gcs_uri": video_gcs_uri,
            "local_video_path": str(workspace.video_path),
            "local_audio_path": str(workspace.audio_path),
        },
        diarization_payload=diarization_payload,
        phase1_visual=phase1_visual,
        emotion2vec_payload=emotion2vec_payload,
        yamnet_payload=yamnet_payload,
    )


__all__ = ["run_phase1_sidecars"]
