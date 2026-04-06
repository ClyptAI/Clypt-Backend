from __future__ import annotations

import logging
import time
from typing import Any

from backend.pipeline.timeline.vibevoice_merge import merge_vibevoice_outputs

from .coordinator import run_phase1_sidecars_coordinator
from .models import Phase1SidecarOutputs, Phase1Workspace

logger = logging.getLogger(__name__)


def _build_phase1_audio_payload(
    *,
    source_url: str,
    video_gcs_uri: str,
    workspace: Phase1Workspace,
) -> dict[str, str]:
    return {
        "source_audio": source_url,
        "video_gcs_uri": video_gcs_uri,
        "local_video_path": str(workspace.video_path),
        "local_audio_path": str(workspace.audio_path),
    }


def run_parallel_phase1_sidecars(
    *,
    source_url: str,
    video_gcs_uri: str,
    workspace: Phase1Workspace,
    branch_timeout_s: float = 1800.0,
    poll_interval_s: float = 0.1,
) -> Phase1SidecarOutputs:
    return run_phase1_sidecars_coordinator(
        source_url=source_url,
        video_gcs_uri=video_gcs_uri,
        workspace=workspace,
        branch_timeout_s=branch_timeout_s,
        poll_interval_s=poll_interval_s,
    )


def run_phase1_sidecars(
    *,
    source_url: str,
    video_gcs_uri: str,
    workspace: Phase1Workspace,
    vibevoice_provider: Any | None = None,
    forced_aligner: Any | None = None,
    visual_extractor: Any | None = None,
    emotion_provider: Any | None = None,
    yamnet_provider: Any | None = None,
) -> Phase1SidecarOutputs:
    if all(
        provider is not None
        for provider in (
            vibevoice_provider,
            forced_aligner,
            visual_extractor,
            emotion_provider,
            yamnet_provider,
        )
    ):
        t_total = time.perf_counter()

        logger.info("[extract] starting visual extraction ...")
        t_visual = time.perf_counter()
        phase1_visual = visual_extractor.extract(
            video_path=workspace.video_path,
            workspace=workspace,
        )
        logger.info("[extract] visual extraction done in %.1f s", time.perf_counter() - t_visual)

        logger.info("[extract] starting VibeVoice ASR ...")
        t_vv = time.perf_counter()
        vibevoice_turns = vibevoice_provider.run(audio_path=workspace.audio_path)
        logger.info(
            "[extract] VibeVoice done in %.1f s — %d turns",
            time.perf_counter() - t_vv,
            len(vibevoice_turns),
        )

        prelim_turns = []
        for idx, turn in enumerate(vibevoice_turns, start=1):
            prelim_turns.append(
                {
                    "turn_id": f"t_{idx:06d}",
                    "speaker_id": f"SPEAKER_{int(turn.get('Speaker') or 0)}",
                    "start_ms": int(round(float(turn.get("Start") or 0) * 1000)),
                    "end_ms": int(round(float(turn.get("End") or 0) * 1000)),
                    "transcript_text": str(turn.get("Content") or "").strip(),
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
                    "turn_id": turn["turn_id"],
                    "speaker_id": turn["speaker_id"],
                    "start_ms": turn["start_ms"],
                    "end_ms": turn["end_ms"],
                    "transcript_text": turn["transcript_text"],
                    "word_ids": turn.get("word_ids", []),
                    "identification_match": None,
                }
                for turn in turns
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

        logger.info("[extract] all sidecars done in %.1f s", time.perf_counter() - t_total)

        return Phase1SidecarOutputs(
            phase1_audio=_build_phase1_audio_payload(
                source_url=source_url,
                video_gcs_uri=video_gcs_uri,
                workspace=workspace,
            ),
            diarization_payload=diarization_payload,
            phase1_visual=phase1_visual,
            emotion2vec_payload=emotion2vec_payload,
            yamnet_payload=yamnet_payload,
        )

    return run_parallel_phase1_sidecars(
        source_url=source_url,
        video_gcs_uri=video_gcs_uri,
        workspace=workspace,
    )


__all__ = ["run_parallel_phase1_sidecars", "run_phase1_sidecars"]
