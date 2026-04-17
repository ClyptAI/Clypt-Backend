from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from typing import Any

from ._http_retry import RemoteServiceHTTPError, TRANSIENT_HTTP_STATUS, retry_with_backoff
from .config import Phase26DispatchServiceSettings


class RemotePhase26DispatchError(RemoteServiceHTTPError):
    """Raised when the Phase26 dispatch service rejects or fails a request."""


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, RemotePhase26DispatchError):
        return exc.status_code in TRANSIENT_HTTP_STATUS
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in TRANSIENT_HTTP_STATUS
    return isinstance(exc, (urllib.error.URLError, TimeoutError, OSError))


class RemotePhase26DispatchClient:
    _DEFAULT_MAX_RETRIES = 2

    def __init__(
        self,
        *,
        settings: Phase26DispatchServiceSettings,
        max_retries: int | None = None,
    ) -> None:
        self.settings = settings
        self._max_retries = (
            max(0, int(max_retries))
            if max_retries is not None
            else self._DEFAULT_MAX_RETRIES
        )
        self.backend = "remote_phase26"

    def _endpoint(self, path: str) -> str:
        return self.settings.service_url.rstrip("/") + path

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.auth_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _post_json(self, *, path: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        url = self._endpoint(path)
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers = self._auth_headers()

        def _do_request() -> dict[str, Any]:
            req = urllib.request.Request(url, data=body, method="POST", headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    raw = resp.read().decode("utf-8")
                    status = resp.status
            except urllib.error.HTTPError as exc:
                err_body = ""
                try:
                    err_body = exc.read().decode("utf-8", errors="replace")
                except Exception:  # pragma: no cover
                    err_body = ""
                raise RemotePhase26DispatchError(
                    f"phase26-dispatch service {path} returned HTTP {exc.code}: {err_body[:512] or exc.reason}",
                    status_code=exc.code,
                ) from exc
            if status >= 400:
                raise RemotePhase26DispatchError(
                    f"phase26-dispatch service {path} returned HTTP {status}",
                    status_code=status,
                )
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise RemotePhase26DispatchError(
                    f"phase26-dispatch service {path} response must be an object, got {type(parsed).__name__}"
                )
            return parsed

        return retry_with_backoff(
            _do_request,
            max_retries=self._max_retries,
            classify_transient=_is_transient,
            log_prefix="[phase26_dispatch_client]",
            operation="phase26_dispatch",
            initial_backoff_s=1.0,
            max_backoff_s=8.0,
            multiplier=2.0,
            jitter_ratio=0.25,
            sleep=time.sleep,
            rng=random.random,
        )

    def enqueue_phase24(
        self,
        *,
        run_id: str,
        payload: dict[str, Any],
        worker_url: str | None = None,
    ) -> str:
        _ = worker_url
        raw = self._post_json(
            path="/tasks/phase26-enqueue",
            payload={"run_id": run_id, "payload": payload},
            timeout_s=float(self.settings.timeout_s),
        )
        task_name = str(raw.get("task_name") or "").strip()
        if not task_name:
            raise RemotePhase26DispatchError("phase26-dispatch response missing task_name")
        return task_name


__all__ = ["RemotePhase26DispatchClient", "RemotePhase26DispatchError"]
