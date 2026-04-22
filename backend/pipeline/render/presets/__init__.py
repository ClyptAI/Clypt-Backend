from __future__ import annotations

from .registry import CAPTION_PRESET_REGISTRY
from ..contracts import CaptionPreset


def load_caption_presets() -> dict[str, CaptionPreset]:
    return {
        preset_id: CaptionPreset.model_validate(payload)
        for preset_id, payload in CAPTION_PRESET_REGISTRY.items()
    }


__all__ = [
    "CAPTION_PRESET_REGISTRY",
    "load_caption_presets",
]
