"""Shared analysis video decode / prep metadata for Phase 1 stages.

Tracking, face extraction, and speaker binding should reuse one prepared
analysis context instead of calling prep helpers independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass
class Phase1AnalysisContext:
    """Caches the result of a single `_prepare_*` call for a source video."""

    source_video_path: str
    _payload: dict[str, Any] | None = None
    _prepare_invocations: int = 0
    _cache_hits: int = field(default=0, repr=False)

    def ensure_prepared(
        self,
        worker: Any,
        *,
        tracking_mode: str,
        prepare_direct: Callable[[str], dict[str, Any]] | None = None,
        prepare_proxy: Callable[[str], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run at most one prepare for this source path and return the shared dict.

        When ``prepare_*`` callables are omitted, ``worker`` is called using the
        same rules as :meth:`ClyptWorker._run_tracking` (direct vs analysis/proxy prep).
        """
        if self._payload is not None:
            self._cache_hits += 1
            return self._payload

        self._prepare_invocations += 1
        if prepare_direct is not None and prepare_proxy is not None:
            if tracking_mode == "direct":
                self._payload = dict(prepare_direct(self.source_video_path))
            else:
                self._payload = dict(prepare_proxy(self.source_video_path))
        else:
            if tracking_mode == "direct":
                self._payload = dict(worker._prepare_direct_analysis_context(self.source_video_path))
            else:
                self._payload = dict(worker._prepare_analysis_video(self.source_video_path))
        return self._payload

    def as_dict(self) -> dict[str, Any]:
        """Return the prepared analysis metadata, or {} if not prepared yet."""
        return dict(self._payload) if self._payload is not None else {}

    def bind_payload(self, payload: Mapping[str, Any]) -> None:
        """Attach an externally prepared dict (tests or advanced callers)."""
        self._payload = dict(payload)

    @property
    def analysis_video_path(self) -> str:
        d = self.as_dict()
        return str(d.get("analysis_video_path", self.source_video_path))

    @property
    def prepare_invocations(self) -> int:
        return int(self._prepare_invocations)

    @property
    def cache_hits(self) -> int:
        return int(self._cache_hits)


__all__ = ["Phase1AnalysisContext"]
