from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from backend.pipeline.timeline.pyannote_merge import merge_pyannote_outputs

from .models import Phase1SidecarOutputs, Phase1Workspace

logger = logging.getLogger(__name__)


def run_phase1_sidecars(
    *,
    source_url: str,
    video_gcs_uri: str,
    workspace: Phase1Workspace,
    pyannote_client: Any,
    visual_extractor: Any,
    emotion_provider: Any,
    yamnet_provider: Any,
    identify_voiceprints: list[str] | None = None,
) -> Phase1SidecarOutputs:
    logger.info("[extract] submitting pyannote diarization (parallel) ...")
    logger.info("[extract] starting visual extraction (main thread) ...")
    t_sidecars = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as executor:
        diarize_future = executor.submit(
            pyannote_client.run_diarize,
            media_url=video_gcs_uri,
        )
        t_visual = time.perf_counter()
        phase1_visual = visual_extractor.extract(
            video_path=workspace.video_path,
            workspace=workspace,
        )
        logger.info(
            "[extract] visual extraction done in %.1f s — waiting for pyannote ...",
            time.perf_counter() - t_visual,
        )
        pyannote_payload = diarize_future.result()

    turns_count = len(list((pyannote_payload or {}).get("turns") or []))
    logger.info(
        "[extract] pyannote done — %d turns — total parallel phase: %.1f s",
        turns_count,
        time.perf_counter() - t_sidecars,
    )

    identify_payload = None
    if identify_voiceprints:
        logger.info("[extract] running pyannote identify (%d voiceprints) ...", len(identify_voiceprints))
        identify_payload = pyannote_client.run_identify(
            media_url=video_gcs_uri,
            voiceprint_ids=identify_voiceprints,
        )

    merged = merge_pyannote_outputs(
        diarize_payload=pyannote_payload,
        identify_payload=identify_payload,
    )
    turns = list(merged.get("turns") or [])
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

    return Phase1SidecarOutputs(
        phase1_audio={
            "source_audio": source_url,
            "video_gcs_uri": video_gcs_uri,
            "local_video_path": str(workspace.video_path),
            "local_audio_path": str(workspace.audio_path),
        },
        pyannote_payload=pyannote_payload,
        identify_payload=identify_payload,
        phase1_visual=phase1_visual,
        emotion2vec_payload=emotion2vec_payload,
        yamnet_payload=yamnet_payload,
    )


__all__ = ["run_phase1_sidecars"]
