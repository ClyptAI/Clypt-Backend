from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.phase1_runtime.visual import V31VisualExtractor
from backend.phase1_runtime.visual_config import VisualPipelineConfig
from backend.providers.config import load_phase1_host_settings


@dataclass(slots=True)
class AppDeps:
    visual_extractor: Any
    expected_auth_token: str


@lru_cache(maxsize=1)
def get_app_deps() -> AppDeps:
    settings = load_phase1_host_settings()
    visual_settings = settings.phase1_visual_service
    if visual_settings is None:
        raise RuntimeError("CLYPT_PHASE1_VISUAL_SERVICE_* envs must be set on the Phase 1 host.")
    return AppDeps(
        visual_extractor=V31VisualExtractor(visual_config=VisualPipelineConfig.from_env()),
        expected_auth_token=visual_settings.auth_token,
    )


__all__ = ["AppDeps", "get_app_deps"]
