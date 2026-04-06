from __future__ import annotations

from pathlib import Path

from backend.providers import (
    ForcedAlignmentProvider,
    VibeVoiceASRProvider,
    VertexEmbeddingClient,
    VertexGeminiClient,
    load_provider_settings,
)
from backend.providers.emotion2vec import Emotion2VecPlusProvider
from backend.providers.storage import GCSStorageClient
from backend.providers.yamnet import YAMNetProvider
from backend.runtime.phase14_live import V31LivePhase14Runner

from .extract import run_parallel_phase1_sidecars
from .runner import Phase1JobRunner
from .visual import SimpleVisualExtractor
from .visual_config import VisualPipelineConfig


def _validate_parallel_runtime_contract(settings) -> None:
    runtime = settings.phase1_runtime
    if not runtime.phase1_parallel_enabled:
        return
    if runtime.phase1_parallel_gpu_branch_limit != 2:
        raise ValueError(
            "Phase 1 parallel runtime requires CLYPT_PHASE1_PARALLEL_GPU_BRANCH_LIMIT=2 for V1."
        )
    if runtime.yamnet_branch_device != "cpu":
        raise ValueError(
            "Phase 1 parallel runtime requires CLYPT_PHASE1_YAMNET_BRANCH_DEVICE=cpu for V1."
        )


def build_default_phase1_job_runner(*, working_root: str | Path | None = None) -> Phase1JobRunner:
    settings = load_provider_settings()
    _validate_parallel_runtime_contract(settings)
    vibevoice_provider = VibeVoiceASRProvider(
        backend=settings.vibevoice.backend,
        native_venv_python=settings.vibevoice.native_venv_python or None,
        model_id=settings.vibevoice.model_id,
        flash_attention=settings.vibevoice.flash_attention,
        liger_kernel=settings.vibevoice.liger_kernel,
        hotwords_context=settings.vibevoice.hotwords_context,
        system_prompt=settings.vibevoice.system_prompt or None,
        max_new_tokens=settings.vibevoice.max_new_tokens,
        do_sample=settings.vibevoice.do_sample,
        temperature=settings.vibevoice.temperature,
        top_p=settings.vibevoice.top_p,
        repetition_penalty=settings.vibevoice.repetition_penalty,
        num_beams=settings.vibevoice.num_beams,
        attn_implementation=settings.vibevoice.attn_implementation,
        subprocess_timeout_s=settings.vibevoice.subprocess_timeout_s,
    )
    forced_aligner = ForcedAlignmentProvider()
    embedding_client = VertexEmbeddingClient(settings=settings.vertex)
    llm_client = VertexGeminiClient(settings=settings.vertex)
    storage_client = GCSStorageClient(settings=settings.storage)
    phase14_runner = V31LivePhase14Runner.from_env(
        llm_client=llm_client,
        embedding_client=embedding_client,
        storage_client=storage_client,
    )
    visual_config = VisualPipelineConfig.from_env()

    run_phase1_sidecars = None
    if settings.phase1_runtime.phase1_parallel_enabled:
        def run_phase1_sidecars(*, source_url: str, video_gcs_uri: str, workspace):
            return run_parallel_phase1_sidecars(
                source_url=source_url,
                video_gcs_uri=video_gcs_uri,
                workspace=workspace,
                branch_timeout_s=settings.phase1_runtime.branch_timeout_s,
                poll_interval_s=settings.phase1_runtime.branch_poll_interval_s,
            )

        run_phase1_sidecars.__name__ = "run_parallel_phase1_sidecars"

    return Phase1JobRunner(
        working_root=Path(working_root or settings.phase1_runtime.working_root),
        storage_client=storage_client,
        vibevoice_provider=vibevoice_provider,
        forced_aligner=forced_aligner,
        visual_extractor=SimpleVisualExtractor(visual_config=visual_config),
        emotion_provider=Emotion2VecPlusProvider(),
        yamnet_provider=YAMNetProvider(device=settings.phase1_runtime.yamnet_branch_device),
        run_phase1_sidecars=run_phase1_sidecars,
        phase14_runner=phase14_runner,
    )


__all__ = ["build_default_phase1_job_runner"]
