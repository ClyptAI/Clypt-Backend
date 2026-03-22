from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

from backend.do_phase1_service.models import PersistedPhase1Manifest
from backend.do_phase1_service.storage import GCSStorage, StorageBackend, persist_phase1_outputs
from backend.pipeline.phase_1_modal_pipeline import (
    call_modal_worker,
    download_media,
    enrich_visual_ledger_for_downstream,
    upload_video_to_gcs,
    validate_phase_handoff,
)


def run_extraction_job(
    *,
    source_url: str,
    job_id: str,
    output_dir: str | Path,
    storage: StorageBackend | None = None,
) -> PersistedPhase1Manifest:
    storage = storage or GCSStorage()
    output_dir = Path(output_dir)
    job_output_dir = output_dir / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)

    video_path, audio_path = download_media(source_url)
    modal_result = _resolve_modal_result(video_path=video_path, audio_path=audio_path, youtube_url=source_url)
    if modal_result.get("status") != "success":
        raise RuntimeError(modal_result.get("message", "phase 1 extraction failed"))

    phase_1_visual = modal_result.get("phase_1_visual") or modal_result.get("phase_1a_visual")
    phase_1_audio = modal_result.get("phase_1_audio") or modal_result.get("phase_1a_audio")
    if phase_1_visual is None or phase_1_audio is None:
        raise RuntimeError("modal worker did not return phase_1_visual and phase_1_audio")

    canonical_video_uri = upload_video_to_gcs(video_path)
    phase_1_visual = enrich_visual_ledger_for_downstream(phase_1_visual, phase_1_audio, video_path)
    phase_1_visual["video_gcs_uri"] = canonical_video_uri
    phase_1_audio["video_gcs_uri"] = canonical_video_uri
    validate_phase_handoff(phase_1_visual, phase_1_audio)

    return persist_phase1_outputs(
        storage=storage,
        output_dir=job_output_dir,
        job_id=job_id,
        source_url=source_url,
        canonical_video_uri=canonical_video_uri,
        phase_1_audio=phase_1_audio,
        phase_1_visual=phase_1_visual,
    )


def _resolve_modal_result(*, video_path: str, audio_path: str, youtube_url: str) -> dict:
    result = call_modal_worker(video_path, audio_path, youtube_url)
    if inspect.isawaitable(result):
        return asyncio.run(result)
    return result
