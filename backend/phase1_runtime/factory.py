from __future__ import annotations

import logging
from pathlib import Path

from backend.providers import (
    Emotion2VecPlusProvider,
    ForcedAlignmentProvider,
    RemotePhase1VisualClient,
    RemotePhase26DispatchClient,
    RemoteVibeVoiceAsrClient,
    YAMNetProvider,
    load_phase1_host_settings,
)
from backend.providers.storage import GCSStorageClient
from backend.repository import SpannerPhase14Repository

from .input_resolver import Phase1InputResolver
from .runner import Phase1JobRunner

logger = logging.getLogger(__name__)

def _build_phase14_repository(*, settings) -> SpannerPhase14Repository | None:
    # Opt-out path: if Spanner is not fully configured, Phase 1 runs without
    # durable persistence (in-memory only). Once configured, we deliberately
    # do NOT catch init exceptions: rotated credentials, DDL drift, or missing
    # IAM must fail fast instead of silently disabling persistence while GPUs
    # keep burning. See docs/ERROR_LOG.md 2026-04-16 (F0.3).
    if not settings.spanner.is_configured:
        logger.info(
            "Spanner Phase14 repository disabled (spanner settings not fully "
            "configured); Phase 1 will run without durable persistence."
        )
        return None
    repository = SpannerPhase14Repository.from_settings(settings=settings.spanner)
    repository.bootstrap_schema()
    return repository


def build_default_phase1_job_runner(*, working_root: str | Path | None = None) -> Phase1JobRunner:
    settings = load_phase1_host_settings()
    storage_client = GCSStorageClient(settings=settings.storage)

    if settings.phase1_visual_service is None:
        raise ValueError(
            "Phase 1 host requires CLYPT_PHASE1_VISUAL_SERVICE_URL and "
            "CLYPT_PHASE1_VISUAL_SERVICE_AUTH_TOKEN."
        )
    if settings.phase26_dispatch_service is None:
        raise ValueError(
            "Phase 1 host requires CLYPT_PHASE24_DISPATCH_URL and "
            "CLYPT_PHASE24_DISPATCH_AUTH_TOKEN."
        )

    vibevoice_asr_client = RemoteVibeVoiceAsrClient(
        settings=settings.vibevoice_asr_service,
    )
    visual_extractor = RemotePhase1VisualClient(settings=settings.phase1_visual_service)
    forced_aligner = ForcedAlignmentProvider()
    emotion_provider = Emotion2VecPlusProvider()
    yamnet_provider = YAMNetProvider(
        device="gpu" if settings.phase1_runtime.run_yamnet_on_gpu else "cpu",
    )

    phase14_repository = _build_phase14_repository(settings=settings)
    phase24_task_queue_client = RemotePhase26DispatchClient(
        settings=settings.phase26_dispatch_service,
    )
    input_resolver = None
    input_mode = (settings.phase1_runtime.input_mode or "test_bank").strip().lower()
    if input_mode != "test_bank":
        raise ValueError(
            f"Unsupported CLYPT_PHASE1_INPUT_MODE={settings.phase1_runtime.input_mode!r}; expected 'test_bank'."
        )
    if settings.phase1_runtime.test_bank_path:
        input_resolver = Phase1InputResolver.from_mapping_file(settings.phase1_runtime.test_bank_path)
    elif settings.phase1_runtime.test_bank_strict:
        raise ValueError(
            "CLYPT_PHASE1_INPUT_MODE=test_bank requires CLYPT_PHASE1_TEST_BANK_PATH when strict mode is enabled."
        )
    else:
        logger.warning(
            "CLYPT_PHASE1_INPUT_MODE=test_bank with strict=0 and no mapping path; "
            "source_url jobs will fail unless source_path is provided."
        )
    return Phase1JobRunner(
        working_root=Path(working_root or settings.phase1_runtime.working_root),
        storage_client=storage_client,
        vibevoice_asr_client=vibevoice_asr_client,
        forced_aligner=forced_aligner,
        visual_extractor=visual_extractor,
        emotion_provider=emotion_provider,
        yamnet_provider=yamnet_provider,
        phase24_task_queue_client=phase24_task_queue_client,
        phase14_repository=phase14_repository,
        phase24_query_version=settings.phase24_worker.query_version,
        input_resolver=input_resolver,
        input_resolver_strict=settings.phase1_runtime.test_bank_strict,
    )


__all__ = ["build_default_phase1_job_runner"]
