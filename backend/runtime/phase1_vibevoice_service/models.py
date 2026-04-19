from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ShardPlan:
    index: int
    shard_count: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(slots=True)
class ShardAsrResult:
    plan: ShardPlan
    turns: list[dict[str, Any]]
    audio_path: Path | None = None
    audio_gcs_uri: str | None = None
    representative_clips: dict[int, Path] = field(default_factory=dict)


@dataclass(slots=True)
class LongFormAsrOutputs:
    turns: list[dict[str, Any]]
    stage_events: list[dict[str, Any]]
    shard_results: list[ShardAsrResult] = field(default_factory=list)


__all__ = [
    "LongFormAsrOutputs",
    "ShardAsrResult",
    "ShardPlan",
]
