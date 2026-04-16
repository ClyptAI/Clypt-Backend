from __future__ import annotations

import json
import logging
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import LocalGenerationSettings

logger = logging.getLogger(__name__)

_TRANSIENT_HTTP_STATUS = {429, 500, 502, 503, 504}
_FAILURE_DIR_ENV = "CLYPT_LOCAL_OPENAI_FAILURE_DIR"
_DEFAULT_FAILURE_DIR = Path("backend/outputs/local_openai_failures")


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


def _schema_name_for_request(schema: dict[str, Any]) -> str:
    """Derive a stable, OpenAI-compatible schema name for response_format."""
    explicit_name = str(schema.get("title") or "").strip()
    if explicit_name:
        candidate = re.sub(r"[^a-zA-Z0-9_-]+", "_", explicit_name)
    else:
        serialized = json.dumps(schema, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        candidate = f"clypt_schema_{abs(hash(serialized)) % 10_000_000:07d}"
    candidate = candidate.strip("_") or "clypt_schema"
    # Keep name compact and deterministic.
    return candidate[:64]


def _enforce_strict_object_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a schema copy with additionalProperties=false on every object node."""

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            walked: dict[str, Any] = {k: _walk(v) for k, v in node.items()}
            if walked.get("type") == "object":
                walked["additionalProperties"] = False
            return walked
        if isinstance(node, list):
            return [_walk(item) for item in node]
        return node

    return _walk(schema)


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


def _failure_dir() -> Path:
    configured = str(os.getenv(_FAILURE_DIR_ENV) or "").strip()
    return Path(configured) if configured else _DEFAULT_FAILURE_DIR


def _persist_failed_chat_completion(
    *,
    raw_response: str,
    request_payload: dict[str, Any],
    reason: str,
    response_payload: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    artifact_dir = _failure_dir()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = f"local_openai_{int(time.time() * 1000)}_{os.getpid()}"
    response_path = artifact_dir / f"{stem}.response.json"
    meta_path = artifact_dir / f"{stem}.meta.json"
    response_path.write_text(raw_response or "", encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "reason": reason,
                "request_payload": request_payload,
                "response_payload": response_payload,
                "metadata": metadata or {},
            },
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    logger.error(
        "[local_openai] persisted failed chat completion to %s (metadata=%s)",
        response_path,
        meta_path,
    )
    return str(response_path)


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

    def _post_chat_completion(self, *, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
        url = self._chat_completions_url()
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        timeout = float(self.settings.timeout_s)

        def _do_request() -> tuple[dict[str, Any], str]:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                artifact = _persist_failed_chat_completion(
                    raw_response=raw,
                    request_payload=payload,
                    reason="invalid_json_response",
                    metadata={"json_error": str(exc)},
                )
                raise ValueError(
                    f"OpenAI-compatible response is not valid JSON: {exc} (artifact={artifact})"
                ) from exc
            return parsed, raw

        model = str(payload.get("model") or "")
        return self._call_with_retry(operation="chat_completions", model=model, fn=_do_request)

    def generate_json(
        self,
        *,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        response_schema: dict | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        resolved_model = model or self.settings.model
        if not (resolved_model or "").strip():
            raise ValueError(
                "Local LLM model is not configured; set CLYPT_LOCAL_LLM_MODEL or pass model=."
            )
        if response_schema is None:
            raise ValueError(
                "response_schema is required for strict server-side schema guarantees."
            )
        strict_response_schema = _enforce_strict_object_schema(response_schema)
        prompt_with_json = f"Return JSON only.\n{prompt}"
        resolved_temperature = (
            float(self.settings.temperature)
            if temperature is None
            else float(temperature)
        )
        payload: dict[str, Any] = {
            "model": resolved_model,
            "messages": [{"role": "user", "content": prompt_with_json}],
            "temperature": resolved_temperature,
            "top_p": float(self.settings.top_p),
            "presence_penalty": float(self.settings.presence_penalty),
            "repetition_penalty": float(self.settings.repetition_penalty),
            "extra_body": {
                "top_k": int(self.settings.top_k),
                "min_p": float(self.settings.min_p),
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": _schema_name_for_request(strict_response_schema),
                "strict": True,
                "schema": strict_response_schema,
            },
        }
        if max_output_tokens is not None:
            payload["max_tokens"] = int(max_output_tokens)

        data, raw_response = self._post_chat_completion(payload=payload)
        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            artifact = _persist_failed_chat_completion(
                raw_response=raw_response,
                request_payload=payload,
                reason="missing_choices",
                response_payload=data,
            )
            raise ValueError(f"OpenAI-compatible response missing choices[] (artifact={artifact})")
        first = choices[0]
        if not isinstance(first, dict):
            artifact = _persist_failed_chat_completion(
                raw_response=raw_response,
                request_payload=payload,
                reason="invalid_choice_object",
                response_payload=data,
                metadata={"choice_type": type(first).__name__},
            )
            raise ValueError(
                "OpenAI-compatible response choices[0] must be an object "
                f"(choice_type={type(first).__name__}, artifact={artifact})"
            )
        finish_reason = str(first.get("finish_reason") or "").strip()
        message = first.get("message")
        if not isinstance(message, dict):
            artifact = _persist_failed_chat_completion(
                raw_response=raw_response,
                request_payload=payload,
                reason="missing_message_object",
                response_payload=data,
                metadata={"message_type": type(message).__name__},
            )
            raise ValueError(
                "OpenAI-compatible response missing message object "
                f"(message_type={type(message).__name__}, artifact={artifact})"
            )
        content = message.get("content")
        if content is None:
            artifact = _persist_failed_chat_completion(
                raw_response=raw_response,
                request_payload=payload,
                reason="missing_message_content",
                response_payload=data,
                metadata={
                    "finish_reason": finish_reason,
                    "choice_keys": sorted(first.keys()),
                    "message_keys": sorted(message.keys()),
                    "has_reasoning_content": message.get("reasoning_content") is not None,
                    "reasoning_content_preview": (
                        None
                        if message.get("reasoning_content") is None
                        else str(message.get("reasoning_content"))[:500]
                    ),
                },
            )
            raise ValueError(
                "OpenAI-compatible response missing message.content "
                f"(finish_reason={finish_reason or 'unknown'}, "
                f"message_keys={sorted(message.keys())}, artifact={artifact})"
            )
        if not isinstance(content, str):
            content = str(content)

        response_chars = len(content)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        if finish_reason.lower() == "length":
            logger.warning(
                "[local_openai] structured output hit max_tokens for model=%s schema=%s max_output_tokens=%s "
                "prompt_tokens=%s completion_tokens=%s response_chars=%s",
                resolved_model,
                payload["response_format"]["json_schema"]["name"],
                max_output_tokens,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                response_chars,
            )
        try:
            parsed = _parse_json_content(content)
        except ValueError as exc:
            if finish_reason:
                raise ValueError(
                    f"{exc} (finish_reason={finish_reason}, max_output_tokens={max_output_tokens})"
                ) from exc
            raise
        _validate_schema_subset(parsed, strict_response_schema)
        return parsed


__all__ = ["LocalOpenAIQwenClient"]
