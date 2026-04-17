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


from pathlib import Path

from backend.phase1_runtime.payloads import Phase1SidecarOutputs


@runtime_checkable
class StorageClient(Protocol):
    def upload_file(self, *, local_path: Path, object_name: str) -> str: ...

    def download_file(self, *, gcs_uri: str, local_path: Path) -> Path: ...

    def get_https_url(self, gcs_uri: str, expiry_hours: int = 24) -> str: ...

    def exists(self, gcs_uri: str) -> bool: ...


class NodeMediaPreparerCallable(Protocol):
    def __call__(
        self,
        *,
        nodes: list[Any],
        paths: Any,
        phase1_outputs: Phase1SidecarOutputs,
    ) -> list[dict[str, str]]: ...
