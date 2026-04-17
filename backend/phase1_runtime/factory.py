from __future__ import annotations

import logging
from pathlib import Path

from backend.providers import (
    Emotion2VecPlusProvider,
    ForcedAlignmentProvider,
    RemoteVibeVoiceAsrClient,
    YAMNetProvider,
    load_provider_settings,
)
from backend.providers.storage import GCSStorageClient
from backend.repository import SpannerPhase14Repository
from backend.runtime.phase24_local_dispatcher import Phase24LocalDispatcherClient
from backend.runtime.phase24_local_queue import Phase24LocalQueue

from .input_resolver import Phase1InputResolver
from .runner import Phase1JobRunner
from .visual import SimpleVisualExtractor
from .visual_config import VisualPipelineConfig

logger = logging.getLogger(__name__)


def _build_phase24_local_dispatcher(*, settings) -> Phase24LocalDispatcherClient:
    queue = Phase24LocalQueue(path=settings.phase24_local_queue.path)
    return Phase24LocalDispatcherClient(queue=queue)


def _build_phase14_repository(*, settings) -> SpannerPhase14Repository | None:
    try:
        repository = SpannerPhase14Repository.from_settings(settings=settings.spanner)
        repository.bootstrap_schema()
        return repository
    except Exception as exc:  # pragma: no cover - depends on optional credentials/deps
        logger.warning("failed to initialize Spanner Phase14 repository: %s", exc)
        return None


def build_default_phase1_job_runner(*, working_root: str | Path | None = None) -> Phase1JobRunner:
    settings = load_provider_settings()
    if settings.phase24_local_queue.queue_backend != "local_sqlite":
        raise ValueError(
            "Phase1 local runtime requires CLYPT_PHASE24_QUEUE_BACKEND=local_sqlite for Phase 2–4 "
            f"(got {settings.phase24_local_queue.queue_backend!r})."
        )
    storage_client = GCSStorageClient(settings=settings.storage)

    # VibeVoice ASR is the only audio stage that is remote: it runs on the
    # RTX 6000 Ada host. NFA + emotion2vec+ + YAMNet live here on the H200
    # in-process because the RTX's 48 GiB cannot support them alongside
    # VibeVoice vLLM without NFA global-alignment OOMing. See
    # docs/ERROR_LOG.md 2026-04-17.
    vibevoice_asr_client = RemoteVibeVoiceAsrClient(
        settings=settings.vibevoice_asr_service,
    )
    forced_aligner = ForcedAlignmentProvider()
    emotion_provider = Emotion2VecPlusProvider()
    yamnet_provider = YAMNetProvider(
        device="gpu" if settings.phase1_runtime.run_yamnet_on_gpu else "cpu",
    )

    phase14_repository = _build_phase14_repository(settings=settings)
    phase24_task_queue_client = _build_phase24_local_dispatcher(settings=settings)
    visual_config = VisualPipelineConfig.from_env()
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
        visual_extractor=SimpleVisualExtractor(visual_config=visual_config),
        emotion_provider=emotion_provider,
        yamnet_provider=yamnet_provider,
        phase24_task_queue_client=phase24_task_queue_client,
        phase14_repository=phase14_repository,
        phase24_query_version=settings.phase24_worker.query_version,
        input_resolver=input_resolver,
        input_resolver_strict=settings.phase1_runtime.test_bank_strict,
    )


__all__ = ["build_default_phase1_job_runner"]
