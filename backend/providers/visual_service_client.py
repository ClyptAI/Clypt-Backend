from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ._http_retry import RemoteServiceHTTPError, TRANSIENT_HTTP_STATUS, retry_with_backoff
from .config import Phase1VisualServiceSettings

logger = logging.getLogger(__name__)


class RemotePhase1VisualError(RemoteServiceHTTPError):
    """Raised when the Phase 1 visual service rejects or fails a request."""


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, RemotePhase1VisualError):
        return exc.status_code in TRANSIENT_HTTP_STATUS
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in TRANSIENT_HTTP_STATUS
    return isinstance(exc, (urllib.error.URLError, TimeoutError, OSError))


class RemotePhase1VisualClient:
    _DEFAULT_MAX_RETRIES = 2

    def __init__(
        self,
        *,
        settings: Phase1VisualServiceSettings,
        max_retries: int | None = None,
    ) -> None:
        self.settings = settings
        self._max_retries = (
            max(0, int(max_retries))
            if max_retries is not None
            else self._DEFAULT_MAX_RETRIES
        )

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
                raise RemotePhase1VisualError(
                    f"phase1-visual service {path} returned HTTP {exc.code}: {err_body[:512] or exc.reason}",
                    status_code=exc.code,
                ) from exc
            if status >= 400:
                raise RemotePhase1VisualError(
                    f"phase1-visual service {path} returned HTTP {status}",
                    status_code=status,
                )
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RemotePhase1VisualError(
                    f"phase1-visual service {path} returned non-JSON body: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise RemotePhase1VisualError(
                    f"phase1-visual service {path} response must be an object, got {type(parsed).__name__}"
                )
            return parsed

        return retry_with_backoff(
            _do_request,
            max_retries=self._max_retries,
            classify_transient=_is_transient,
            log_prefix="[phase1_visual_client]",
            operation="phase1_visual",
            initial_backoff_s=1.0,
            max_backoff_s=8.0,
            multiplier=2.0,
            jitter_ratio=0.25,
            sleep=time.sleep,
            rng=random.random,
        )

    def healthcheck(self) -> dict[str, Any]:
        url = self._endpoint(self.settings.healthcheck_path)
        req = urllib.request.Request(
            url,
            method="GET",
            headers={"Authorization": f"Bearer {self.settings.auth_token}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                raw = resp.read().decode("utf-8")
                status = resp.status
        except urllib.error.HTTPError as exc:
            raise RemotePhase1VisualError(
                f"phase1-visual service healthcheck returned HTTP {exc.code}",
                status_code=exc.code,
            ) from exc
        except Exception as exc:
            raise RemotePhase1VisualError(
                f"phase1-visual service healthcheck failed: {exc}"
            ) from exc
        if status >= 400:
            raise RemotePhase1VisualError(
                f"phase1-visual service healthcheck returned HTTP {status}",
                status_code=status,
            )
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {"raw": raw}

    def extract(self, *, video_path: Path, workspace: Any = None) -> dict[str, Any]:
        _ = workspace
        payload = {"video_path": str(Path(video_path))}
        logger.info("[phase1_visual_client] invoking visual-extract video=%s", video_path)
        return self._post_json(
            path="/tasks/visual-extract",
            payload=payload,
            timeout_s=float(self.settings.timeout_s),
        )


__all__ = ["RemotePhase1VisualClient", "RemotePhase1VisualError"]
