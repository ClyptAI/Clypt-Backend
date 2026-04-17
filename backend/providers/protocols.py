from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMGenerateJsonClient(Protocol):
    def generate_json(
        self,
        *,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        response_schema: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]: ...


@runtime_checkable
class EmbeddingClient(Protocol):
    def embed_texts(
        self,
        texts: Iterable[str],
        *,
        task_type: str | None = None,
        model: str | None = None,
    ) -> list[list[float]]: ...

    def embed_media_uris(
        self,
        media_items: Iterable[dict[str, str]],
        *,
        model: str | None = None,
    ) -> list[list[float]]: ...
