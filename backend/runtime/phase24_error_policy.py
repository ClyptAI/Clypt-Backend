from __future__ import annotations

import json
import urllib.error
from enum import Enum

try:
    from pydantic import ValidationError as PydanticValidationError
except Exception:  # pragma: no cover - optional import guard
    PydanticValidationError = tuple()  # type: ignore[assignment]


class Phase24FailureClass(str, Enum):
    TRANSIENT = "transient"
    NON_TRANSIENT = "non_transient"
    FAIL_FAST = "fail_fast"


class Phase24FailFastError(RuntimeError):
    """Raised when configured fail-fast thresholds are breached."""


_TRANSIENT_TOKENS = (
    "resource_exhausted",
    "too many requests",
    "rate limit",
    "temporarily unavailable",
    "service unavailable",
    "deadline exceeded",
    "connection reset",
    "timed out",
    "timeout",
)

_NON_TRANSIENT_HINTS = (
    "missing required",
    "must be an object",
    "valid json",
    "schema",
    "validation",
)

_FAIL_FAST_HINTS = (
    "connection refused",
    "enginecore",
    "xgrammar",
    "compile_json_schema",
    "structured output backend",
)


def classify_phase24_exception(exc: BaseException) -> Phase24FailureClass:
    if isinstance(exc, Phase24FailFastError):
        return Phase24FailureClass.FAIL_FAST
    if isinstance(exc, urllib.error.HTTPError):
        return (
            Phase24FailureClass.TRANSIENT
            if int(exc.code) in {429, 500, 502, 503, 504}
            else Phase24FailureClass.NON_TRANSIENT
        )
    if isinstance(exc, urllib.error.URLError):
        message = str(exc).lower()
        if any(token in message for token in _FAIL_FAST_HINTS):
            return Phase24FailureClass.FAIL_FAST
        return Phase24FailureClass.TRANSIENT
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        message = str(exc).lower()
        if any(token in message for token in _FAIL_FAST_HINTS):
            return Phase24FailureClass.FAIL_FAST
        return Phase24FailureClass.TRANSIENT
    if isinstance(exc, (json.JSONDecodeError, ValueError, TypeError)):
        return Phase24FailureClass.NON_TRANSIENT
    if PydanticValidationError and isinstance(exc, PydanticValidationError):
        return Phase24FailureClass.NON_TRANSIENT
    message = str(exc).lower()
    if any(token in message for token in _FAIL_FAST_HINTS):
        return Phase24FailureClass.FAIL_FAST
    if any(token in message for token in _NON_TRANSIENT_HINTS):
        return Phase24FailureClass.NON_TRANSIENT
    if any(token in message for token in _TRANSIENT_TOKENS):
        return Phase24FailureClass.TRANSIENT
    return Phase24FailureClass.NON_TRANSIENT


__all__ = [
    "Phase24FailFastError",
    "Phase24FailureClass",
    "classify_phase24_exception",
]
