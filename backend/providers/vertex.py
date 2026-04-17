from __future__ import annotations

import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

from .config import VertexSettings

try:
    from google.genai import types
except ImportError:
    types = None

_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
_TRANSIENT_ERROR_TOKENS = (
    "resource_exhausted",
    "too many requests",
    "rate limit",
    "temporarily unavailable",
    "service unavailable",
    "deadline exceeded",
    "connection reset",
)

logger = logging.getLogger(__name__)
_THINKING_DISABLED_LEVELS = {"", "0", "false", "none", "off", "disabled"}


def _coerce_parsed_json(response: Any) -> dict[str, Any] | None:
    parsed = getattr(response, "parsed", None)
    if parsed is None:
        return None
    if isinstance(parsed, dict):
        return parsed
    if hasattr(parsed, "model_dump"):
        dumped = parsed.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    raise ValueError(f"Vertex generation parsed payload must be an object, got {type(parsed).__name__}")


def _extract_finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return None
    first = candidates[0]
    finish = getattr(first, "finish_reason", None)
    if finish is None and isinstance(first, dict):
        finish = first.get("finish_reason")
    if finish is None:
        return None
    if hasattr(finish, "name"):
        return str(finish.name)
    return str(finish)


def _extract_usage_summary(response: Any) -> str:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return "unknown"
    try:
        prompt_tokens = getattr(usage, "prompt_token_count", None)
        candidates_tokens = getattr(usage, "candidates_token_count", None)
        total_tokens = getattr(usage, "total_token_count", None)
        return (
            f"prompt_token_count={prompt_tokens}, "
            f"candidates_token_count={candidates_tokens}, "
            f"total_token_count={total_tokens}"
        )
    except Exception:
        return "unavailable"


def _extract_embedding_values(item: Any) -> list[float]:
    values = getattr(item, "values", None)
    if values is None and isinstance(item, dict):
        values = item.get("values")
    if values is None:
        raise ValueError("embedding response item is missing values")
    return [float(value) for value in values]


def _status_code_from_exception(exc: Exception) -> int | None:
    for attr in ("status_code", "code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("status_code", "code"):
            value = getattr(response, attr, None)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)

    match = re.search(r"\b([1-5][0-9]{2})\b", str(exc))
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(exc, "headers", None)
    if headers is None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) if response is not None else None
    if not headers:
        return None
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _is_transient_vertex_exception(exc: Exception) -> bool:
    status = _status_code_from_exception(exc)
    if status in _TRANSIENT_STATUS_CODES:
        return True
    message = str(exc).lower()
    return any(token in message for token in _TRANSIENT_ERROR_TOKENS)


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
    kwargs: dict[str, Any]
    if location == "__developer__":
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is required when using the Developer API backend."
            )
        kwargs = dict(api_key=settings.gemini_api_key)
    else:
        kwargs = dict(vertexai=True, project=settings.project, location=location)
    if http_options is not None:
        kwargs["http_options"] = http_options
    return genai.Client(**kwargs)


class VertexGenerationClient:
    def __init__(self, *, settings: VertexSettings, sdk_client: Any | None = None) -> None:
        self.settings = settings
        self._api_max_retries = max(0, int(settings.generation_api_max_retries))
        self._api_initial_backoff_s = max(0.0, float(settings.generation_api_initial_backoff_s))
        self._api_max_backoff_s = max(0.0, float(settings.generation_api_max_backoff_s))
        self._api_backoff_multiplier = max(1.0, float(settings.generation_api_backoff_multiplier))
        self._api_jitter_ratio = max(0.0, float(settings.generation_api_jitter_ratio))
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is required for Developer API generation."
            )
        self._backend = "developer"
        self._sdk = sdk_client or _build_default_sdk_client(
            settings=settings,
            location="__developer__",
            headers=None,
        )

    def _call_with_retry(self, *, operation: str, model: str, fn):
        for retry_index in range(self._api_max_retries + 1):
            attempt = retry_index + 1
            try:
                return fn()
            except Exception as exc:
                if retry_index >= self._api_max_retries or not _is_transient_vertex_exception(exc):
                    raise
                base_delay = min(
                    self._api_max_backoff_s,
                    self._api_initial_backoff_s
                    * (self._api_backoff_multiplier ** retry_index),
                )
                retry_after_s = _retry_after_seconds(exc)
                if retry_after_s is not None:
                    base_delay = max(base_delay, retry_after_s)
                jitter = (
                    base_delay * self._api_jitter_ratio * random.random()
                    if base_delay > 0.0 and self._api_jitter_ratio > 0.0
                    else 0.0
                )
                sleep_s = base_delay + jitter
                logger.warning(
                    "[vertex] transient %s error for model=%s attempt=%d/%d status=%s; retrying in %.2fs: %s",
                    operation,
                    model,
                    attempt,
                    self._api_max_retries + 1,
                    _status_code_from_exception(exc),
                    sleep_s,
                    exc,
                )
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

    def generate_json(
        self,
        *,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.0,
        response_schema: dict | None = None,
        max_output_tokens: int | None = None,
        thinking_level: str | None = None,
    ) -> dict[str, Any]:
        resolved_model = model or self.settings.generation_model
        if types is not None:
            config_kwargs: dict[str, Any] = dict(
                temperature=temperature,
                response_mime_type="application/json",
            )
            normalized_thinking_level = str(thinking_level or "").strip().lower()
            if normalized_thinking_level not in _THINKING_DISABLED_LEVELS:
                requested_level = str(thinking_level).strip().upper()
                try:
                    resolved_level = types.ThinkingLevel(requested_level)
                except Exception as exc:
                    raise ValueError(
                        "Unsupported thinking_level="
                        f"{thinking_level!r}; expected one of "
                        "MINIMAL|LOW|MEDIUM|HIGH."
                    ) from exc
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=resolved_level
                )
            if response_schema is not None:
                config_kwargs["response_schema"] = response_schema
            if max_output_tokens is not None:
                config_kwargs["max_output_tokens"] = max_output_tokens
            _config = types.GenerateContentConfig(**config_kwargs)
        else:
            _config = {"temperature": temperature, "response_mime_type": "application/json"}
            normalized_thinking_level = str(thinking_level or "").strip().lower()
            if normalized_thinking_level not in _THINKING_DISABLED_LEVELS:
                _config["thinking_config"] = {"thinking_level": str(thinking_level).strip().upper()}
        response = self._call_with_retry(
            operation="generate_content",
            model=resolved_model,
            fn=lambda: self._sdk.models.generate_content(
                model=resolved_model,
                contents=prompt,
                config=_config,
            ),
        )
        parsed_payload = _coerce_parsed_json(response)
        if parsed_payload is None:
            text = (getattr(response, "text", None) or "").strip()
            snippet = text[:300].replace("\n", "\\n") if text else ""
            finish_reason = _extract_finish_reason(response)
            usage_summary = _extract_usage_summary(response)
            raise ValueError(
                "Vertex generation did not return SDK-parsed JSON object; failing fast "
                f"(model={resolved_model}, finish_reason={finish_reason}, usage={usage_summary}, "
                f"has_text={bool(text)}, text_snippet={snippet!r})."
            )
        if not isinstance(parsed_payload, dict):
            raise ValueError(
                f"Vertex generation parsed payload must be an object, got {type(parsed_payload).__name__} "
                f"(model={resolved_model})."
            )
        return parsed_payload


class VertexEmbeddingClient:
    def __init__(self, *, settings: VertexSettings, sdk_client: Any | None = None) -> None:
        self.settings = settings
        self._api_max_retries = max(0, int(settings.embedding_api_max_retries))
        self._api_initial_backoff_s = max(0.0, float(settings.embedding_api_initial_backoff_s))
        self._api_max_backoff_s = max(0.0, float(settings.embedding_api_max_backoff_s))
        self._api_backoff_multiplier = max(1.0, float(settings.embedding_api_backoff_multiplier))
        self._api_jitter_ratio = max(0.0, float(settings.embedding_api_jitter_ratio))
        backend = (settings.embedding_backend or "vertex").strip().lower()
        if backend == "developer" and not settings.gemini_api_key:
            logger.warning(
                "[vertex] VERTEX_EMBEDDING_BACKEND=developer but GEMINI_API_KEY is missing; falling back to Vertex backend."
            )
            backend = "vertex"
        self._backend = backend
        self._sdk = sdk_client or _build_default_sdk_client(
            settings=settings,
            location="__developer__" if self._backend == "developer" else settings.embedding_location,
        )

    def _call_with_retry(self, *, operation: str, model: str, fn):
        for retry_index in range(self._api_max_retries + 1):
            attempt = retry_index + 1
            try:
                return fn()
            except Exception as exc:
                if retry_index >= self._api_max_retries or not _is_transient_vertex_exception(exc):
                    raise
                base_delay = min(
                    self._api_max_backoff_s,
                    self._api_initial_backoff_s
                    * (self._api_backoff_multiplier ** retry_index),
                )
                retry_after_s = _retry_after_seconds(exc)
                if retry_after_s is not None:
                    base_delay = max(base_delay, retry_after_s)
                jitter = (
                    base_delay * self._api_jitter_ratio * random.random()
                    if base_delay > 0.0 and self._api_jitter_ratio > 0.0
                    else 0.0
                )
                sleep_s = base_delay + jitter
                logger.warning(
                    "[vertex] transient %s error for model=%s attempt=%d/%d status=%s; retrying in %.2fs: %s",
                    operation,
                    model,
                    attempt,
                    self._api_max_retries + 1,
                    _status_code_from_exception(exc),
                    sleep_s,
                    exc,
                )
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

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
            response = self._call_with_retry(
                operation="embed_content_text",
                model=_model,
                fn=lambda: self._sdk.models.embed_content(
                    model=_model,
                    contents=text,
                    config=config,
                ),
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
        if self._backend != "vertex":
            raise RuntimeError(
                "Multimodal URI embeddings require Vertex backend. Set VERTEX_EMBEDDING_BACKEND=vertex."
            )
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
            response = self._call_with_retry(
                operation="embed_content_media",
                model=_model,
                fn=lambda: self._sdk.models.embed_content(
                    model=_model,
                    contents=content,
                    config=None,
                ),
            )
            raw = getattr(response, "embeddings", None)
            if raw is None:
                raise ValueError("Vertex embeddings response is missing embeddings")
            return _extract_embedding_values(raw[0])

        # Up to 32 concurrent Gemini embedding calls — each call is I/O bound
        # and the API handles the concurrency. For 17 nodes at 10 workers we'd
        # need 2 rounds; at 32 all fit in one round, halving the latency.
        with ThreadPoolExecutor(max_workers=min(len(items), 32)) as pool:
            futures = [pool.submit(_embed_one, item) for item in items]
            return [f.result() for f in futures]


__all__ = [
    "VertexEmbeddingClient",
    "VertexGenerationClient",
]
