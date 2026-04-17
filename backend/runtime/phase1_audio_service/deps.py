"""Per-process dependencies for the RTX 6000 Ada audio host FastAPI service.

This module constructs the heavy ML providers (VibeVoice vLLM client, NeMo
Forced Aligner, emotion2vec+, YAMNet) and the GCS storage client once per
worker process and caches them. The FastAPI routes import ``get_app_deps()``
to reuse the singletons so each HTTP request stays hot on the GPU.

The audio chain itself is serial on the GPU; we accept a single in-flight
request at a time (enforced in ``app.py`` with an asyncio semaphore) so the
providers can be plain synchronous callables without thread-safety concerns.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.providers import (
    ForcedAlignmentProvider,
    VibeVoiceVLLMProvider,
    build_gcs_uri_url_resolver,
    load_audio_host_settings,
)
from backend.providers.emotion2vec import Emotion2VecPlusProvider
from backend.providers.storage import GCSStorageClient
from backend.providers.yamnet import YAMNetProvider

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AppDeps:
    """Singleton bundle of RTX-side providers and config."""

    vibevoice_provider: Any
    forced_aligner: Any
    emotion_provider: Any
    yamnet_provider: Any
    storage_client: Any
    scratch_root: Path
    expected_auth_token: str


def _resolve_scratch_root() -> Path:
    root = os.getenv("CLYPT_PHASE1_AUDIO_SCRATCH_ROOT") or "/opt/clypt-phase1/scratch"
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_expected_auth_token() -> str:
    token = os.getenv("CLYPT_PHASE1_AUDIO_HOST_AUTH_TOKEN") or os.getenv(
        "CLYPT_PHASE1_AUDIO_HOST_TOKEN"
    )
    if not token:
        raise RuntimeError(
            "CLYPT_PHASE1_AUDIO_HOST_AUTH_TOKEN (preferred) or CLYPT_PHASE1_AUDIO_HOST_TOKEN "
            "must be set on the audio host so incoming requests can be authenticated."
        )
    return token.strip()


@lru_cache(maxsize=1)
def get_app_deps() -> AppDeps:
    """Build and cache the RTX-side audio providers."""
    settings = load_audio_host_settings()
    storage_client = GCSStorageClient(settings=settings.storage)

    vv = settings.vllm_vibevoice
    vibevoice_provider = VibeVoiceVLLMProvider(
        base_url=vv.base_url,
        model=vv.model,
        timeout_s=vv.timeout_s,
        healthcheck_path=vv.healthcheck_path,
        max_retries=vv.max_retries,
        audio_mode=vv.audio_mode,
        audio_gcs_url_resolver=build_gcs_uri_url_resolver(storage_client=storage_client),
        hotwords_context=settings.vibevoice.hotwords_context,
        max_new_tokens=settings.vibevoice.max_new_tokens,
        do_sample=settings.vibevoice.do_sample,
        temperature=settings.vibevoice.temperature,
        top_p=settings.vibevoice.top_p,
        repetition_penalty=settings.vibevoice.repetition_penalty,
        num_beams=settings.vibevoice.num_beams,
    )

    forced_aligner = ForcedAlignmentProvider()
    emotion_provider = Emotion2VecPlusProvider()
    yamnet_provider = YAMNetProvider(
        device="gpu" if settings.phase1_runtime.run_yamnet_on_gpu else "cpu"
    )

    deps = AppDeps(
        vibevoice_provider=vibevoice_provider,
        forced_aligner=forced_aligner,
        emotion_provider=emotion_provider,
        yamnet_provider=yamnet_provider,
        storage_client=storage_client,
        scratch_root=_resolve_scratch_root(),
        expected_auth_token=_resolve_expected_auth_token(),
    )
    logger.info(
        "[phase1_audio_service.deps] initialized providers scratch_root=%s vibevoice_base=%s",
        deps.scratch_root,
        vv.base_url,
    )
    return deps


__all__ = ["AppDeps", "get_app_deps"]
