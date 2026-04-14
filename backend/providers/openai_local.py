from __future__ import annotations

import json
import logging
import random
import re
import time
import urllib.error
import urllib.request
from typing import Any

from .config import LocalGenerationSettings

logger = logging.getLogger(__name__)

_TRANSIENT_HTTP_STATUS = {429, 500, 502, 503, 504}


def _parse_json_content(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("LLM response content is empty")
    fence = re.match(r"^```(?:json)?\s*\n?", text)
    if fence:
        end = text.rfind("```")
        if end > fence.end():
            text = text[fence.end() : end].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response content is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            f"LLM JSON response must be an object, got {type(parsed).__name__}"
        )
    return parsed


def _validate_schema_subset(value: Any, schema: dict[str, Any], *, path: str = "$") -> None:
    """Lightweight JSON-schema subset: object required keys, nested objects/arrays, enums."""
    stype = schema.get("type")
    if stype == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected object, got {type(value).__name__}")
        required = schema.get("required") or []
        for key in required:
            if key not in value:
                raise ValueError(f"{path}: missing required key {key!r}")
        props = schema.get("properties") or {}
        for key, subschema in props.items():
            if key not in value:
                continue
            if not isinstance(subschema, dict):
                continue
            _validate_schema_subset(value[key], subschema, path=f"{path}.{key}")
        return
    if stype == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected array, got {type(value).__name__}")
        items = schema.get("items")
        if isinstance(items, dict):
            for i, item in enumerate(value):
                _validate_schema_subset(item, items, path=f"{path}[{i}]")
        return
    if stype == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path}: expected string, got {type(value).__name__}")
        enum = schema.get("enum")
        if enum is not None and value not in enum:
            raise ValueError(f"{path}: expected one of {enum!r}, got {value!r}")
        return
    if stype == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}: expected number, got {type(value).__name__}")
        return
    if stype == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path}: expected integer, got {type(value).__name__}")
        return
    if stype == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{path}: expected boolean, got {type(value).__name__}")
        return


def _resolve_enable_thinking(
    *,
    default: bool,
    thinking_level: str | None,
) -> bool:
    if thinking_level is None:
        return default
    normalized = str(thinking_level).strip().lower()
    if normalized in {"", "minimal", "low"}:
        return False
    if normalized in {"medium", "high"}:
        return True
    return default


def _is_transient_error(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _TRANSIENT_HTTP_STATUS
    if isinstance(exc, urllib.error.URLError):
        return True
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, OSError):
        # e.g. connection reset
        return True
    return False


class LocalOpenAIQwenClient:
    """OpenAI-compatible chat/completions client for self-hosted Qwen (stdlib HTTP)."""

    def __init__(self, *, settings: LocalGenerationSettings) -> None:
        self.settings = settings
        self._api_max_retries = max(0, int(settings.max_retries))
        self._api_initial_backoff_s = max(0.0, float(settings.initial_backoff_s))
        self._api_max_backoff_s = max(0.0, float(settings.max_backoff_s))
        self._api_backoff_multiplier = max(1.0, float(settings.backoff_multiplier))
        self._api_jitter_ratio = max(0.0, float(settings.jitter_ratio))

    def _chat_completions_url(self) -> str:
        return self.settings.base_url.rstrip("/") + "/chat/completions"

    def _call_with_retry(self, *, operation: str, model: str, fn):
        for retry_index in range(self._api_max_retries + 1):
            attempt = retry_index + 1
            try:
                return fn()
            except Exception as exc:
                if retry_index >= self._api_max_retries or not _is_transient_error(exc):
                    raise
                base_delay = min(
                    self._api_max_backoff_s,
                    self._api_initial_backoff_s
                    * (self._api_backoff_multiplier**retry_index),
                )
                jitter = (
                    base_delay * self._api_jitter_ratio * random.random()
                    if base_delay > 0.0 and self._api_jitter_ratio > 0.0
                    else 0.0
                )
                sleep_s = base_delay + jitter
                code = getattr(exc, "code", None)
                logger.warning(
                    "[local_openai] transient %s error for model=%s attempt=%d/%d code=%s; "
                    "retrying in %.2fs: %s",
                    operation,
                    model,
                    attempt,
                    self._api_max_retries + 1,
                    code,
                    sleep_s,
                    exc,
                )
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

    def _post_chat_completion(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        url = self._chat_completions_url()
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        timeout = float(self.settings.timeout_s)

        def _do_request() -> dict[str, Any]:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw)

        model = str(payload.get("model") or "")
        return self._call_with_retry(operation="chat_completions", model=model, fn=_do_request)

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
        resolved_model = model or self.settings.model
        if not (resolved_model or "").strip():
            raise ValueError(
                "Local LLM model is not configured; set CLYPT_LOCAL_LLM_MODEL or pass model=."
            )
        enable_thinking = _resolve_enable_thinking(
            default=bool(self.settings.enable_thinking),
            thinking_level=thinking_level,
        )
        payload: dict[str, Any] = {
            "model": resolved_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if max_output_tokens is not None:
            payload["max_tokens"] = int(max_output_tokens)

        data = self._post_chat_completion(payload=payload)
        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            raise ValueError("OpenAI-compatible response missing choices[]")
        first = choices[0]
        if not isinstance(first, dict):
            raise ValueError("OpenAI-compatible response choices[0] must be an object")
        message = first.get("message")
        if not isinstance(message, dict):
            raise ValueError("OpenAI-compatible response missing message object")
        content = message.get("content")
        if content is None:
            raise ValueError("OpenAI-compatible response missing message.content")
        if not isinstance(content, str):
            content = str(content)

        parsed = _parse_json_content(content)
        if response_schema is not None:
            _validate_schema_subset(parsed, response_schema)
        return parsed


__all__ = ["LocalOpenAIQwenClient"]
