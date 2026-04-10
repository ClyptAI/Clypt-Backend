from __future__ import annotations

import logging
from pathlib import Path

from backend.providers import (
    ForcedAlignmentProvider,
    Phase24TaskQueueClient,
    VibeVoiceVLLMProvider,
    VertexEmbeddingClient,
    VertexGeminiClient,
    load_provider_settings,
)
from backend.providers.emotion2vec import Emotion2VecPlusProvider
from backend.providers.storage import GCSStorageClient
from backend.providers.yamnet import YAMNetProvider
from backend.repository import SpannerPhase14Repository
from backend.runtime.phase14_live import V31LivePhase14Runner

from .input_resolver import Phase1InputResolver
from .runner import Phase1JobRunner
from .visual import SimpleVisualExtractor
from .visual_config import VisualPipelineConfig

logger = logging.getLogger(__name__)


def _build_phase24_task_queue_client(*, settings) -> Phase24TaskQueueClient | None:
    if not settings.cloud_tasks.worker_url:
        logger.warning(
            "CLYPT_PHASE24_WORKER_URL is not set; run_phase14 queue mode will fail fast."
        )
        return None
    try:
        from google.cloud import tasks_v2
    except ImportError:  # pragma: no cover - optional dependency in minimal environments
        logger.warning(
            "google-cloud-tasks is unavailable; run_phase14 queue mode will fail fast."
        )
        return None
    return Phase24TaskQueueClient(
        settings=settings.cloud_tasks,
        tasks_client=tasks_v2.CloudTasksClient(),
    )


def _build_phase14_repository(*, settings) -> SpannerPhase14Repository | None:
    try:
        return SpannerPhase14Repository.from_settings(settings=settings.spanner)
    except Exception as exc:  # pragma: no cover - depends on optional credentials/deps
        logger.warning("failed to initialize Spanner Phase14 repository: %s", exc)
        return None


def build_default_phase1_job_runner(*, working_root: str | Path | None = None) -> Phase1JobRunner:
    settings = load_provider_settings()
    vv = settings.vllm_vibevoice
    vibevoice_provider = VibeVoiceVLLMProvider(
        base_url=vv.base_url,
        model=vv.model,
        timeout_s=vv.timeout_s,
        healthcheck_path=vv.healthcheck_path,
        max_retries=vv.max_retries,
        audio_mode=vv.audio_mode,
        hotwords_context=settings.vibevoice.hotwords_context,
        max_new_tokens=settings.vibevoice.max_new_tokens,
        do_sample=settings.vibevoice.do_sample,
        temperature=settings.vibevoice.temperature,
        top_p=settings.vibevoice.top_p,
        repetition_penalty=settings.vibevoice.repetition_penalty,
        num_beams=settings.vibevoice.num_beams,
    )

    forced_aligner = ForcedAlignmentProvider()
    embedding_client = VertexEmbeddingClient(settings=settings.vertex)
    llm_client = VertexGeminiClient(settings=settings.vertex)
    storage_client = GCSStorageClient(settings=settings.storage)
    phase14_repository = _build_phase14_repository(settings=settings)
    phase14_runner = V31LivePhase14Runner.from_env(
        llm_client=llm_client,
        embedding_client=embedding_client,
        flash_model=settings.vertex.flash_model,
        storage_client=storage_client,
        repository=phase14_repository,
        query_version=settings.phase24_worker.query_version,
        debug_snapshots=settings.phase24_worker.debug_snapshots,
    )
    phase24_task_queue_client = _build_phase24_task_queue_client(settings=settings)
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
        vibevoice_provider=vibevoice_provider,
        forced_aligner=forced_aligner,
        visual_extractor=SimpleVisualExtractor(visual_config=visual_config),
        emotion_provider=Emotion2VecPlusProvider(),
        yamnet_provider=YAMNetProvider(
            device="gpu" if settings.phase1_runtime.run_yamnet_on_gpu else "cpu"
        ),
        phase14_runner=phase14_runner,
        phase24_task_queue_client=phase24_task_queue_client,
        phase14_repository=phase14_repository,
        phase24_worker_url=settings.cloud_tasks.worker_url,
        phase24_query_version=settings.phase24_worker.query_version,
        input_resolver=input_resolver,
        input_resolver_strict=settings.phase1_runtime.test_bank_strict,
    )


__all__ = ["build_default_phase1_job_runner"]
