from __future__ import annotations

import logging
import time
from typing import Any

from backend.pipeline.timeline.vibevoice_merge import merge_vibevoice_outputs

from .models import Phase1SidecarOutputs, Phase1Workspace

logger = logging.getLogger(__name__)


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
) -> Phase1SidecarOutputs:
    """
    Run all Phase 1 sidecar tasks serially on the GPU.

    Serial order (all on the GPU, no thread contention):
      1. Visual extraction (RF-DETR + ByteTrack)
      2. VibeVoice ASR (diarization + transcription)
      3. NeMo Forced Aligner (word-level timestamps)
      4. emotion2vec+
      5. YAMNet
    """
    t_total = time.perf_counter()

    # --- 1. Visual extraction --------------------------------------------
    logger.info("[extract] starting visual extraction ...")
    t_visual = time.perf_counter()
    phase1_visual = visual_extractor.extract(
        video_path=workspace.video_path,
        workspace=workspace,
    )
    logger.info("[extract] visual extraction done in %.1f s", time.perf_counter() - t_visual)

    # --- 2. VibeVoice ASR ------------------------------------------------
    logger.info("[extract] starting VibeVoice ASR ...")
    t_vv = time.perf_counter()
    vibevoice_turns = vibevoice_provider.run(audio_path=workspace.audio_path)
    logger.info(
        "[extract] VibeVoice done in %.1f s — %d turns",
        time.perf_counter() - t_vv,
        len(vibevoice_turns),
    )

    # --- 3. ctc-forced-aligner -------------------------------------------
    # Build preliminary turn dicts for the aligner so it gets start_ms/end_ms/transcript_text
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

    # --- Merge VibeVoice + alignments into canonical {words, turns} ------
    merged = merge_vibevoice_outputs(
        vibevoice_turns=vibevoice_turns,
        word_alignments=word_alignments,
    )
    turns = list(merged.get("turns") or [])

    # Build diarization_payload in a shape downstream can use
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

    # --- 4. emotion2vec+ -------------------------------------------------
    logger.info("[extract] starting emotion2vec+ on %d turns ...", len(turns))
    t_emotion = time.perf_counter()
    emotion2vec_payload = emotion_provider.run(
        audio_path=workspace.audio_path,
        turns=turns,
    )
    logger.info("[extract] emotion2vec+ done in %.1f s", time.perf_counter() - t_emotion)

    # --- 5. YAMNet -------------------------------------------------------
    logger.info("[extract] starting YAMNet ...")
    t_yamnet = time.perf_counter()
    yamnet_payload = yamnet_provider.run(audio_path=workspace.audio_path)
    yamnet_event_count = len((yamnet_payload or {}).get("events") or [])
    logger.info(
        "[extract] YAMNet done in %.1f s — %d events",
        time.perf_counter() - t_yamnet,
        yamnet_event_count,
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
