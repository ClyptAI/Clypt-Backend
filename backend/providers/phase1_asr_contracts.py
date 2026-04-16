from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class Phase1ASRGenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int | None = None
    do_sample: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    num_beams: int | None = None


class Phase1ASRRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio_gcs_uri: str
    context_info: str | None = None
    generation_config: Phase1ASRGenerationConfig | None = None


class Phase1ASRResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turns: list[dict[str, Any]]


__all__ = [
    "Phase1ASRGenerationConfig",
    "Phase1ASRRequest",
    "Phase1ASRResponse",
]
