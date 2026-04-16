from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator


class NodeMediaPrepItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    start_ms: NonNegativeInt
    end_ms: NonNegativeInt

    @model_validator(mode="after")
    def _validate_time_order(self) -> "NodeMediaPrepItem":
        if self.start_ms > self.end_ms:
            raise ValueError("start_ms must be <= end_ms")
        return self


class NodeMediaPrepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    source_video_gcs_uri: str
    object_prefix: str
    items: list[NodeMediaPrepItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_items(self) -> "NodeMediaPrepRequest":
        if not self.items:
            raise ValueError("items must contain at least one node media request")
        return self


class NodeMediaDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    file_uri: str
    mime_type: str = "video/mp4"
    local_path: str | None = None


class NodeMediaPrepResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    items: list[NodeMediaDescriptor] = Field(default_factory=list)


__all__ = [
    "NodeMediaDescriptor",
    "NodeMediaPrepItem",
    "NodeMediaPrepRequest",
    "NodeMediaPrepResponse",
]
