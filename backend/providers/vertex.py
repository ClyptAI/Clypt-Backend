from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

from .config import VertexSettings

try:
    from google.genai import types
except ImportError:
    types = None

_PRIORITY_PAYGO_HEADERS = {
    "X-Vertex-AI-LLM-Shared-Request-Type": "priority",
    "X-Vertex-AI-LLM-Request-Type": "shared",
}


def _extract_embedding_values(item: Any) -> list[float]:
    values = getattr(item, "values", None)
    if values is None and isinstance(item, dict):
        values = item.get("values")
    if values is None:
        raise ValueError("embedding response item is missing values")
    return [float(value) for value in values]


def _build_default_sdk_client(*, settings: VertexSettings, location: str, headers: dict | None = None):
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is required for live Vertex AI execution."
        ) from exc
    try:
        from google.genai.types import HttpOptions
        http_options = HttpOptions(headers=headers or {})
    except Exception:
        http_options = None
    kwargs = dict(vertexai=True, project=settings.project, location=location)
    if http_options is not None:
        kwargs["http_options"] = http_options
    return genai.Client(**kwargs)


class VertexGeminiClient:
    def __init__(self, *, settings: VertexSettings, sdk_client: Any | None = None) -> None:
        self.settings = settings
        self._sdk = sdk_client or _build_default_sdk_client(
            settings=settings,
            location=settings.generation_location,
            headers=_PRIORITY_PAYGO_HEADERS,
        )

    def generate_json(
        self,
        *,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.0,
        response_schema: dict | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        if types is not None:
            config_kwargs: dict[str, Any] = dict(
                temperature=temperature,
                response_mime_type="application/json",
            )
            # Explicitly disable thinking for Flash and Pro — use model defaults
            try:
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass
            if response_schema is not None:
                config_kwargs["response_schema"] = response_schema
            if max_output_tokens is not None:
                config_kwargs["max_output_tokens"] = max_output_tokens
            _config = types.GenerateContentConfig(**config_kwargs)
        else:
            _config = {"temperature": temperature, "response_mime_type": "application/json"}
        response = self._sdk.models.generate_content(
            model=model or self.settings.generation_model,
            contents=prompt,
            config=_config,
        )
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise ValueError("Vertex Gemini returned no text payload.")
        return json.loads(text)


class VertexEmbeddingClient:
    def __init__(self, *, settings: VertexSettings, sdk_client: Any | None = None) -> None:
        self.settings = settings
        self._sdk = sdk_client or _build_default_sdk_client(
            settings=settings,
            location=settings.embedding_location,
        )

    def embed_texts(
        self,
        texts: Iterable[str],
        *,
        task_type: str | None = None,
        model: str | None = None,
    ) -> list[list[float]]:
        text_list = [str(text) for text in texts]
        if not text_list:
            return []
        # embed_content treats a list of strings as a single multimodal document,
        # returning 1 embedding for the whole batch. Call once per text instead.
        _model = model or self.settings.embedding_model
        config = {"task_type": task_type} if task_type else None

        def _embed_one(text):
            response = self._sdk.models.embed_content(
                model=_model,
                contents=text,
                config=config,
            )
            raw = getattr(response, "embeddings", None)
            if raw is None:
                raise ValueError("Vertex embeddings response is missing embeddings")
            return _extract_embedding_values(raw[0])

        with ThreadPoolExecutor(max_workers=min(len(text_list), 10)) as pool:
            futures = [pool.submit(_embed_one, t) for t in text_list]
            return [f.result() for f in futures]

    def embed_media_uris(
        self,
        media_items: Iterable[dict[str, str]],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        items = [dict(item) for item in media_items]
        if not items:
            return []
        try:
            from google.genai import types as _types
        except ImportError:
            _types = None

        # The embedding API only accepts 1 video/media part per call.
        # Call once per item and collect results.
        _model = model or self.settings.embedding_model

        def _embed_one(item):
            if item.get("file_uri"):
                if _types is None:
                    content: Any = {
                        "file_uri": item["file_uri"],
                        "mime_type": item.get("mime_type") or "application/octet-stream",
                    }
                else:
                    content = _types.Part.from_uri(
                        file_uri=item["file_uri"],
                        mime_type=item.get("mime_type") or "application/octet-stream",
                    )
            elif item.get("descriptor"):
                content = str(item["descriptor"])
            else:
                raise ValueError("each media embedding item must include file_uri or descriptor")
            response = self._sdk.models.embed_content(
                model=_model,
                contents=content,
                config=None,
            )
            raw = getattr(response, "embeddings", None)
            if raw is None:
                raise ValueError("Vertex embeddings response is missing embeddings")
            return _extract_embedding_values(raw[0])

        with ThreadPoolExecutor(max_workers=min(len(items), 10)) as pool:
            futures = [pool.submit(_embed_one, item) for item in items]
            return [f.result() for f in futures]


__all__ = [
    "VertexEmbeddingClient",
    "VertexGeminiClient",
]
