"""Remote client for the VibeVoice ASR service on the RTX 6000 Ada box.

The H200 orchestrator calls :meth:`RemoteVibeVoiceAsrClient.run` once per
Phase 1 job to invoke VibeVoice vLLM on the RTX. NFA/emotion2vec+/YAMNet
run **in-process** on the H200 afterwards — they are no longer the RTX's
responsibility (see docs/ERROR_LOG.md 2026-04-17).

Stage telemetry reported by the remote service is re-emitted through the
caller-supplied ``stage_event_logger`` so the H200 orchestrator preserves
the same ``vibevoice_asr`` stage event it produced when VibeVoice ran
locally.

HTTP is implemented with ``urllib.request`` to stay aligned with the existing
client patterns in ``backend/providers/openai_local.py`` — no extra runtime
dependency on the H200.
"""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from .config import VibeVoiceAsrServiceSettings

logger = logging.getLogger(__name__)

_TRANSIENT_HTTP_STATUS = {408, 409, 429, 500, 502, 503, 504}

StageEventLogger = Callable[..., None]


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, RemoteVibeVoiceAsrError):
        code = exc.status_code
        if code is None:
            return False
        return code in _TRANSIENT_HTTP_STATUS
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _TRANSIENT_HTTP_STATUS
    if isinstance(exc, urllib.error.URLError):
        return True
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, OSError):
        return True
    return False


def _emit(
    logger_fn: StageEventLogger | None,
    *,
    stage_name: str,
    status: str,
    duration_ms: float | None = None,
    metadata: dict[str, Any] | None = None,
    error_payload: dict[str, Any] | None = None,
) -> None:
    if logger_fn is None:
        return
    try:
        logger_fn(
            stage_name=stage_name,
            status=status,
            duration_ms=duration_ms,
            metadata=metadata or {},
            error_payload=error_payload,
        )
    except Exception:  # pragma: no cover - defensive
        logger.exception(
            "[vibevoice_asr_client] stage_event_logger re-emission failed "
            "stage=%s status=%s",
            stage_name,
            status,
        )


@dataclass(slots=True)
class VibeVoiceAsrResponse:
    """Typed view over the ``/tasks/vibevoice-asr`` response payload."""

    turns: list[dict[str, Any]]
    stage_events: list[dict[str, Any]]

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "VibeVoiceAsrResponse":
        def _as_list(key: str) -> list[dict[str, Any]]:
            value = payload.get(key) or []
            if not isinstance(value, list):
                raise ValueError(
                    f"vibevoice-asr response field {key!r} must be a list, "
                    f"got {type(value).__name__}"
                )
            return [dict(item) for item in value]

        return cls(
            turns=_as_list("turns"),
            stage_events=_as_list("stage_events"),
        )


# Deprecated alias retained for one release.
PhaseOneAudioResponse = VibeVoiceAsrResponse


class RemoteVibeVoiceAsrError(RuntimeError):
    """Raised when the VibeVoice ASR host rejects or fails a request."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# Deprecated alias retained for one release.
RemoteAudioChainError = RemoteVibeVoiceAsrError


class RemoteVibeVoiceAsrClient:
    """HTTP client for the RTX 6000 Ada ``POST /tasks/vibevoice-asr`` endpoint.

    The client issues a single request per Phase 1 job. The remote service
    is responsible for running VibeVoice vLLM on the hot GPU and returning
    the raw turns plus a stage-events trail. NFA/emotion2vec+/YAMNet are
    handled on the H200 side after this call returns.

    Parameters
    ----------
    settings:
        :class:`VibeVoiceAsrServiceSettings` loaded from env. ``service_url``
        is the private-VPC base URL of the RTX host
        (e.g. ``http://10.0.0.5:9100``).
    max_retries:
        Transient-error retries for connection-level failures. Kept small
        because the ASR call can be multi-minute for long content; retrying
        a long-running request repeatedly is worse than fail-fast.
    """

    _DEFAULT_MAX_RETRIES = 2
    _DEFAULT_INITIAL_BACKOFF_S = 1.0
    _DEFAULT_MAX_BACKOFF_S = 8.0
    _DEFAULT_BACKOFF_MULTIPLIER = 2.0
    _DEFAULT_JITTER_RATIO = 0.25

    def __init__(
        self,
        *,
        settings: VibeVoiceAsrServiceSettings,
        max_retries: int | None = None,
    ) -> None:
        self.settings = settings
        self._max_retries = (
            max(0, int(max_retries))
            if max_retries is not None
            else self._DEFAULT_MAX_RETRIES
        )

    @property
    def supports_concurrent_visual(self) -> bool:
        """Remote VibeVoice ASR runs concurrently with H200-local visual extraction."""
        return True

    def _endpoint(self, path: str) -> str:
        return self.settings.service_url.rstrip("/") + path

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.auth_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _post_json(
        self,
        *,
        path: str,
        payload: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        url = self._endpoint(path)
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers = self._auth_headers()

        def _do_request() -> dict[str, Any]:
            req = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers=headers,
            )
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
                message = (
                    f"vibevoice-asr host {path} returned HTTP {exc.code}: "
                    f"{err_body[:512] or exc.reason}"
                )
                raise RemoteVibeVoiceAsrError(message, status_code=exc.code) from exc

            if status >= 400:
                raise RemoteVibeVoiceAsrError(
                    f"vibevoice-asr host {path} returned HTTP {status}",
                    status_code=status,
                )
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RemoteVibeVoiceAsrError(
                    f"vibevoice-asr host {path} returned non-JSON body: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise RemoteVibeVoiceAsrError(
                    f"vibevoice-asr host {path} response must be an object, got "
                    f"{type(parsed).__name__}"
                )
            return parsed

        for retry_index in range(self._max_retries + 1):
            try:
                return _do_request()
            except Exception as exc:
                is_last = retry_index >= self._max_retries
                if is_last or not _is_transient(exc):
                    raise
                base_delay = min(
                    self._DEFAULT_MAX_BACKOFF_S,
                    self._DEFAULT_INITIAL_BACKOFF_S
                    * (self._DEFAULT_BACKOFF_MULTIPLIER**retry_index),
                )
                jitter = base_delay * self._DEFAULT_JITTER_RATIO * random.random()
                sleep_s = base_delay + jitter
                logger.warning(
                    "[vibevoice_asr_client] transient error on %s attempt=%d/%d; "
                    "retrying in %.2fs: %s",
                    path,
                    retry_index + 1,
                    self._max_retries + 1,
                    sleep_s,
                    exc,
                )
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
        raise RemoteVibeVoiceAsrError(
            f"vibevoice-asr host {path} retries exhausted"
        )

    def healthcheck(self) -> dict[str, Any]:
        """GET ``{healthcheck_path}`` — raises :class:`RemoteVibeVoiceAsrError` on failure."""
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
            raise RemoteVibeVoiceAsrError(
                f"vibevoice-asr host healthcheck returned HTTP {exc.code}",
                status_code=exc.code,
            ) from exc
        except Exception as exc:
            raise RemoteVibeVoiceAsrError(
                f"vibevoice-asr host healthcheck failed: {exc}"
            ) from exc
        if status >= 400:
            raise RemoteVibeVoiceAsrError(
                f"vibevoice-asr host healthcheck returned HTTP {status}",
                status_code=status,
            )
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {"raw": raw}

    def run(
        self,
        *,
        audio_gcs_uri: str,
        source_url: str | None = None,
        video_gcs_uri: str | None = None,
        run_id: str | None = None,
        stage_event_logger: StageEventLogger | None = None,
    ) -> VibeVoiceAsrResponse:
        """Invoke the remote VibeVoice ASR service for a single Phase 1 job.

        The server runs VibeVoice vLLM only and returns the raw turns plus a
        stage-events trail. Stage events are re-emitted through
        ``stage_event_logger`` so downstream telemetry matches the previous
        in-process execution.
        """
        if not audio_gcs_uri:
            raise ValueError(
                "RemoteVibeVoiceAsrClient.run requires a non-empty audio_gcs_uri"
            )

        payload: dict[str, Any] = {"audio_gcs_uri": audio_gcs_uri}
        if source_url:
            payload["source_url"] = source_url
        if video_gcs_uri:
            payload["video_gcs_uri"] = video_gcs_uri
        if run_id:
            payload["run_id"] = run_id

        t_start = time.perf_counter()
        logger.info(
            "[vibevoice_asr_client] invoking vibevoice-asr run_id=%s audio=%s",
            run_id or "-",
            audio_gcs_uri,
        )
        try:
            raw = self._post_json(
                path="/tasks/vibevoice-asr",
                payload=payload,
                timeout_s=float(self.settings.timeout_s),
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            _emit(
                stage_event_logger,
                stage_name="vibevoice_asr",
                status="failed",
                duration_ms=(time.perf_counter() - t_start) * 1000.0,
                metadata={"run_id": run_id or "", "status_code": status_code},
                error_payload={
                    "code": exc.__class__.__name__,
                    "message": str(exc)[:2048],
                },
            )
            raise

        try:
            response = VibeVoiceAsrResponse.from_json(raw)
        except Exception as exc:
            _emit(
                stage_event_logger,
                stage_name="vibevoice_asr",
                status="failed",
                duration_ms=(time.perf_counter() - t_start) * 1000.0,
                metadata={"run_id": run_id or ""},
                error_payload={
                    "code": "InvalidResponseShape",
                    "message": str(exc)[:2048],
                },
            )
            raise RemoteVibeVoiceAsrError(
                f"invalid vibevoice-asr response shape: {exc}"
            ) from exc

        for event in response.stage_events:
            if not isinstance(event, dict):
                continue
            stage_name = str(event.get("stage_name") or "").strip()
            status = str(event.get("status") or "").strip()
            if not stage_name or not status:
                continue
            duration_ms = event.get("duration_ms")
            if duration_ms is not None:
                try:
                    duration_ms = float(duration_ms)
                except (TypeError, ValueError):
                    duration_ms = None
            metadata = event.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            error_payload = event.get("error_payload")
            if error_payload is not None and not isinstance(error_payload, dict):
                error_payload = None
            _emit(
                stage_event_logger,
                stage_name=stage_name,
                status=status,
                duration_ms=duration_ms,
                metadata=metadata,
                error_payload=error_payload,
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "[vibevoice_asr_client] vibevoice-asr completed run_id=%s turns=%d in %.1f ms",
            run_id or "-",
            len(response.turns),
            elapsed_ms,
        )
        return response


# Deprecated alias retained for one release.
RemoteAudioChainClient = RemoteVibeVoiceAsrClient


__all__ = [
    "PhaseOneAudioResponse",  # deprecated alias of VibeVoiceAsrResponse
    "RemoteAudioChainClient",  # deprecated alias of RemoteVibeVoiceAsrClient
    "RemoteAudioChainError",  # deprecated alias of RemoteVibeVoiceAsrError
    "RemoteVibeVoiceAsrClient",
    "RemoteVibeVoiceAsrError",
    "VibeVoiceAsrResponse",
]
