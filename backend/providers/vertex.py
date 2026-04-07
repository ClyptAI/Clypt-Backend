from __future__ import annotations

import json
from typing import Any, Iterable

from .config import VertexSettings

try:
    from google.genai import types
except ImportError:
    types = None


def _extract_embedding_values(item: Any) -> list[float]:
    values = getattr(item, "values", None)
    if values is None and isinstance(item, dict):
        values = item.get("values")
    if values is None:
        raise ValueError("embedding response item is missing values")
    return [float(value) for value in values]


def _build_default_sdk_client(*, settings: VertexSettings, location: str):
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is required for live Vertex AI execution."
        ) from exc
    return genai.Client(
        vertexai=True,
        project=settings.project,
        location=location,
    )


class VertexGeminiClient:
    def __init__(self, *, settings: VertexSettings, sdk_client: Any | None = None) -> None:
        self.settings = settings
        self._sdk = sdk_client or _build_default_sdk_client(
            settings=settings,
            location=settings.generation_location,
        )

    def generate_json(
        self,
        *,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        if types is not None:
            _config = types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.HIGH,
                ),
            )
        else:
            _config = {
                "temperature": temperature,
                "response_mime_type": "application/json",
            }
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
        config = {"task_type": task_type} if task_type else None
        response = self._sdk.models.embed_content(
            model=model or self.settings.embedding_model,
            contents=text_list,
            config=config,
        )
        raw_embeddings = getattr(response, "embeddings", None)
        if raw_embeddings is None:
            raise ValueError("Vertex embeddings response is missing embeddings")
        return [_extract_embedding_values(item) for item in raw_embeddings]

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
            from google.genai import types
        except ImportError:
            types = None

        contents: list[Any] = []
        for item in items:
            if item.get("file_uri"):
                if types is None:
                    contents.append(
                        {
                            "file_uri": item["file_uri"],
                            "mime_type": item.get("mime_type") or "application/octet-stream",
                        }
                    )
                else:
                    contents.append(
                        types.Part.from_uri(
                            file_uri=item["file_uri"],
                            mime_type=item.get("mime_type") or "application/octet-stream",
                        )
                    )
            elif item.get("descriptor"):
                contents.append(str(item["descriptor"]))
            else:
                raise ValueError("each media embedding item must include file_uri or descriptor")

        response = self._sdk.models.embed_content(
            model=model or self.settings.embedding_model,
            contents=contents,
            config=None,
        )
        raw_embeddings = getattr(response, "embeddings", None)
        if raw_embeddings is None:
            raise ValueError("Vertex embeddings response is missing embeddings")
        return [_extract_embedding_values(item) for item in raw_embeddings]


__all__ = [
    "VertexEmbeddingClient",
    "VertexGeminiClient",
]
