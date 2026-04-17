from __future__ import annotations

import logging
from pathlib import Path

from backend.providers import (
    RemoteAudioChainClient,
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

    # The Phase 1 audio chain runs exclusively on the RTX 6000 Ada audio host.
    # There is no in-process VibeVoice/NFA/emotion/YAMNet provider on the H200.
    audio_host_client = RemoteAudioChainClient(settings=settings.audio_host)

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
        audio_host_client=audio_host_client,
        visual_extractor=SimpleVisualExtractor(visual_config=visual_config),
        phase24_task_queue_client=phase24_task_queue_client,
        phase14_repository=phase14_repository,
        phase24_query_version=settings.phase24_worker.query_version,
        input_resolver=input_resolver,
        input_resolver_strict=settings.phase1_runtime.test_bank_strict,
    )


__all__ = ["build_default_phase1_job_runner"]
