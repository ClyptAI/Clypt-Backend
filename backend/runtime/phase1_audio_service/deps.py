"""Per-process dependencies for the RTX 6000 Ada VibeVoice ASR host.

This module constructs the VibeVoice vLLM client and the GCS storage client
once per worker process and caches them. The FastAPI routes import
``get_app_deps()`` to reuse the singletons so each HTTP request stays hot on
the GPU.

NFA/emotion2vec+/YAMNet no longer live here — they are back on the H200.
See docs/ERROR_LOG.md 2026-04-17.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.providers import (
    VibeVoiceVLLMProvider,
    build_gcs_uri_url_resolver,
    load_audio_host_settings,
)
from backend.providers.config import _VIBEVOICE_ASR_SERVICE_ENV_ALIASES, _getenv_with_aliases
from backend.providers.storage import GCSStorageClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AppDeps:
    """Singleton bundle of RTX-side providers and config."""

    vibevoice_provider: Any
    storage_client: Any
    scratch_root: Path
    expected_auth_token: str


def _resolve_scratch_root() -> Path:
    root = _getenv_with_aliases(
        "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_SCRATCH_ROOT",
        _VIBEVOICE_ASR_SERVICE_ENV_ALIASES[
            "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_SCRATCH_ROOT"
        ],
    ) or "/opt/clypt-phase1/scratch"
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_expected_auth_token() -> str:
    # Legacy CLYPT_PHASE1_AUDIO_HOST_* auth-token aliases accepted through
    # 2026-05-17 (commit 393abaee). Drop together with the AudioHostSettings
    # alias.
    token = _getenv_with_aliases(
        "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN",
        _VIBEVOICE_ASR_SERVICE_ENV_ALIASES[
            "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN"
        ],
    )
    if not token:
        raise RuntimeError(
            "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN must be set on the RTX "
            "VibeVoice ASR host so incoming requests can be authenticated. The legacy "
            "CLYPT_PHASE1_AUDIO_HOST_AUTH_TOKEN / CLYPT_PHASE1_AUDIO_HOST_TOKEN aliases "
            "are accepted through 2026-05-17 (commit 393abaee)."
        )
    return token.strip()


@lru_cache(maxsize=1)
def get_app_deps() -> AppDeps:
    """Build and cache the RTX-side VibeVoice provider + GCS client."""
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

    deps = AppDeps(
        vibevoice_provider=vibevoice_provider,
        storage_client=storage_client,
        scratch_root=_resolve_scratch_root(),
        expected_auth_token=_resolve_expected_auth_token(),
    )
    logger.info(
        "[phase1_audio_service.deps] initialized VibeVoice ASR provider "
        "scratch_root=%s vibevoice_base=%s",
        deps.scratch_root,
        vv.base_url,
    )
    return deps


__all__ = ["AppDeps", "get_app_deps"]
