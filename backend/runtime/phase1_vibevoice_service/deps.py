from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.providers import (
    EcapaTdnnSpeakerVerifier,
    VibeVoiceVLLMProvider,
    build_gcs_uri_url_resolver,
    load_audio_host_settings,
)
from backend.providers.config import _VIBEVOICE_ASR_SERVICE_ENV_ALIASES, _getenv_with_aliases
from backend.providers.storage import GCSStorageClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AppDeps:
    vibevoice_provider: Any
    speaker_verifier: Any
    storage_client: Any
    longform_settings: Any
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
    token = _getenv_with_aliases(
        "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN",
        _VIBEVOICE_ASR_SERVICE_ENV_ALIASES[
            "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN"
        ],
    )
    if not token:
        raise RuntimeError(
            "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN must be set on the Phase 1 host."
        )
    return token.strip()


@lru_cache(maxsize=1)
def get_app_deps() -> AppDeps:
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
    longform_settings = settings.vibevoice_longform
    if longform_settings.verifier_backend != "ecapa_tdnn":
        raise RuntimeError(
            f"Unsupported VIBEVOICE_LONGFORM_VERIFIER_BACKEND={longform_settings.verifier_backend!r}; "
            "expected 'ecapa_tdnn'."
        )
    verifier_savedir = None
    if longform_settings.verifier_cache_dir:
        verifier_savedir = str(Path(longform_settings.verifier_cache_dir).expanduser())
    speaker_verifier = EcapaTdnnSpeakerVerifier(
        model_id=longform_settings.verifier_model_id,
        device=longform_settings.verifier_device,
        savedir=verifier_savedir,
    )
    deps = AppDeps(
        vibevoice_provider=vibevoice_provider,
        speaker_verifier=speaker_verifier,
        storage_client=storage_client,
        longform_settings=longform_settings,
        scratch_root=_resolve_scratch_root(),
        expected_auth_token=_resolve_expected_auth_token(),
    )
    logger.info(
        "[phase1_vibevoice_service.deps] initialized VibeVoice ASR provider scratch_root=%s vibevoice_base=%s",
        deps.scratch_root,
        vv.base_url,
    )
    return deps


__all__ = ["AppDeps", "get_app_deps"]
