from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def payload_to_dict(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json")
    if isinstance(payload, Mapping):
        return dict(payload)
    raise TypeError(f"Unsupported payload type: {type(payload).__name__}")
