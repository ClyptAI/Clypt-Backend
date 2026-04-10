from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import ExternalSignal


@dataclass(slots=True)
class TrendSpygClient:
    max_items: int = 40

    def fetch_related(self, *, query: str) -> list[dict[str, Any]]:
        try:
            from trendspyg import Trends  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency in runtime env
            raise RuntimeError(
                "trendspyg is required when CLYPT_ENABLE_TREND_SIGNALS=1; install trendspyg."
            ) from exc

        client = Trends()
        last_error: Exception | None = None
        attempted_supported_method = False

        for method_name in ("related_queries", "related_topics", "query", "search"):
            method = getattr(client, method_name, None)
            if not callable(method):
                continue
            attempted_supported_method = True
            try:
                payload = method(query)
            except TypeError:
                try:
                    payload = method(keyword=query)
                except Exception as exc:
                    last_error = exc
                    continue
            except Exception as exc:
                raise RuntimeError(f"trendspyg query failed for {query!r} using {method_name}") from exc

            flattened = _flatten_items(payload)
            if flattened:
                return flattened[: max(1, self.max_items)]
            if payload is not None:
                return []
            last_error = RuntimeError(f"trendspyg method {method_name} returned no results for query {query!r}")

        if not attempted_supported_method:
            raise RuntimeError("trendspyg does not expose a supported related trend method")
        if last_error is not None:
            raise RuntimeError(f"trendspyg could not resolve related trends for {query!r}") from last_error
        return []


def _flatten_items(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        out: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict):
                out.append(item)
            else:
                out.append({"title": str(item)})
        return out
    if isinstance(payload, dict):
        out: list[dict[str, Any]] = []
        for value in payload.values():
            if isinstance(value, list):
                out.extend(_flatten_items(value))
            elif isinstance(value, dict):
                out.append(value)
            elif value is not None:
                out.append({"title": str(value)})
        return out
    return [{"title": str(payload)}]


def to_external_signals_from_trends(*, query: str, items: list[dict[str, Any]]) -> list[ExternalSignal]:
    signals: list[ExternalSignal] = []
    for idx, item in enumerate(items, start=1):
        title = str(item.get("title") or item.get("query") or item.get("topic") or "").strip()
        if not title:
            continue
        value = item.get("value")
        try:
            score = float(value)
        except Exception:
            score = 0.0
        signal_id = f"trend:{query}:{idx:03d}"
        signals.append(
            ExternalSignal(
                signal_id=signal_id,
                signal_type="trend_query",
                source_platform="google_trends",
                source_id=signal_id,
                text=title,
                engagement_score=score,
                metadata={
                    "seed_query": query,
                    "raw": item,
                },
            )
        )
    return signals


__all__ = [
    "TrendSpygClient",
    "to_external_signals_from_trends",
]
