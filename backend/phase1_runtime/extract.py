from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from backend.pipeline.timeline.pyannote_merge import merge_pyannote_outputs

from .models import Phase1SidecarOutputs, Phase1Workspace


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
    with ThreadPoolExecutor(max_workers=1) as executor:
        diarize_future = executor.submit(
            pyannote_client.run_diarize,
            media_url=video_gcs_uri,
        )
        phase1_visual = visual_extractor.extract(
            video_path=workspace.video_path,
            workspace=workspace,
        )
        pyannote_payload = diarize_future.result()

    identify_payload = None
    if identify_voiceprints:
        identify_payload = pyannote_client.run_identify(
            media_url=video_gcs_uri,
            voiceprint_ids=identify_voiceprints,
        )

    merged = merge_pyannote_outputs(
        diarize_payload=pyannote_payload,
        identify_payload=identify_payload,
    )
    turns = list(merged.get("turns") or [])
    emotion2vec_payload = emotion_provider.run(
        audio_path=workspace.audio_path,
        turns=turns,
    )
    yamnet_payload = yamnet_provider.run(audio_path=workspace.audio_path)

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
