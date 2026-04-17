from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_DEFAULT_INITIAL_BACKOFF_S = 1.0
_DEFAULT_MAX_BACKOFF_S = 8.0
_DEFAULT_MULTIPLIER = 2.0
_DEFAULT_JITTER_RATIO = 0.25

TRANSIENT_HTTP_STATUS: frozenset[int] = frozenset({408, 409, 429, 500, 502, 503, 504})


class RemoteServiceHTTPError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_retries: int,
    classify_transient: Callable[[BaseException], bool],
    log_prefix: str,
    operation: str,
    initial_backoff_s: float = _DEFAULT_INITIAL_BACKOFF_S,
    max_backoff_s: float = _DEFAULT_MAX_BACKOFF_S,
    multiplier: float = _DEFAULT_MULTIPLIER,
    jitter_ratio: float = _DEFAULT_JITTER_RATIO,
    retry_after_hook: Callable[[BaseException], float | None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
) -> T:
    """Retry ``fn`` with exponential backoff, jitter, and transient classification."""
    max_retries = max(0, int(max_retries))
    initial_backoff_s = max(0.0, float(initial_backoff_s))
    max_backoff_s = max(0.0, float(max_backoff_s))
    multiplier = max(1.0, float(multiplier))
    jitter_ratio = max(0.0, float(jitter_ratio))

    for retry_index in range(max_retries + 1):
        attempt = retry_index + 1
        try:
            return fn()
        except Exception as exc:
            if retry_index >= max_retries or not classify_transient(exc):
                raise

            backoff_s = min(max_backoff_s, initial_backoff_s * (multiplier**retry_index))
            if retry_after_hook is not None:
                retry_after_s = retry_after_hook(exc)
                if retry_after_s is not None:
                    backoff_s = max(backoff_s, max(0.0, float(retry_after_s)))
            jitter_s = (
                backoff_s * jitter_ratio * rng()
                if backoff_s > 0.0 and jitter_ratio > 0.0
                else 0.0
            )
            sleep_s = backoff_s + jitter_s
            logger.info(
                "%s transient %s attempt=%d/%d sleep=%.2fs: %s",
                log_prefix,
                operation,
                attempt,
                max_retries + 1,
                sleep_s,
                exc,
            )
            if sleep_s > 0.0:
                sleep(sleep_s)

