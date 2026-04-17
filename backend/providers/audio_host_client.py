"""Remote client for the Phase 1 audio chain hosted on the RTX 6000 Ada box.

The H200 orchestrator calls ``RemoteAudioChainClient.run`` once per Phase 1 job.
The RTX host runs VibeVoice vLLM, NeMo Forced Aligner, emotion2vec+ and YAMNet
back-to-back on a single GPU, then returns the combined payload in one round
trip. This module is the **only** audio path on the H200 — there is no
in-process fallback.

Stage telemetry reported by the remote service is re-emitted through the
caller-supplied ``stage_event_logger`` so the H200 orchestrator preserves the
exact same stage-events stream it produced when the audio chain ran locally
(``vibevoice_asr`` → ``forced_alignment`` → ``emotion2vec`` → ``yamnet``).

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

from .config import AudioHostSettings

logger = logging.getLogger(__name__)

_TRANSIENT_HTTP_STATUS = {408, 409, 429, 500, 502, 503, 504}

StageEventLogger = Callable[..., None]


def _is_transient(exc: BaseException) -> bool:
    # Check the wrapped status first so retries work after ``_do_request``
    # translates the raw ``urllib.error.HTTPError`` into a
    # :class:`RemoteAudioChainError` with ``status_code``.
    if isinstance(exc, RemoteAudioChainError):
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
            "[audio_host_client] stage_event_logger re-emission failed stage=%s status=%s",
            stage_name,
            status,
        )


@dataclass(slots=True)
class PhaseOneAudioResponse:
    """Typed view over the ``/tasks/phase1-audio`` response payload."""

    turns: list[dict[str, Any]]
    diarization_payload: dict[str, Any]
    emotion2vec_payload: dict[str, Any]
    yamnet_payload: dict[str, Any]
    stage_events: list[dict[str, Any]]

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "PhaseOneAudioResponse":
        def _as_list(key: str) -> list[dict[str, Any]]:
            value = payload.get(key) or []
            if not isinstance(value, list):
                raise ValueError(
                    f"audio host response field {key!r} must be a list, "
                    f"got {type(value).__name__}"
                )
            return [dict(item) for item in value]

        def _as_dict(key: str) -> dict[str, Any]:
            value = payload.get(key) or {}
            if not isinstance(value, dict):
                raise ValueError(
                    f"audio host response field {key!r} must be an object, "
                    f"got {type(value).__name__}"
                )
            return dict(value)

        return cls(
            turns=_as_list("turns"),
            diarization_payload=_as_dict("diarization_payload"),
            emotion2vec_payload=_as_dict("emotion2vec_payload"),
            yamnet_payload=_as_dict("yamnet_payload"),
            stage_events=_as_list("stage_events"),
        )


class RemoteAudioChainError(RuntimeError):
    """Raised when the audio host rejects or fails a Phase 1 audio request."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RemoteAudioChainClient:
    """HTTP client for the RTX 6000 Ada ``POST /tasks/phase1-audio`` endpoint.

    The client issues a **single** request per Phase 1 job. The remote service
    is responsible for running VibeVoice vLLM → NFA → emotion2vec+ → YAMNet on
    the hot GPU and returning the combined payload plus a stage-events trail.

    Parameters
    ----------
    settings:
        :class:`AudioHostSettings` loaded from env. ``service_url`` is the
        private-VPC base URL of the audio host (e.g. ``http://10.0.0.5:9100``).
    max_retries:
        Transient-error retries for connection-level failures. Kept small
        because the full chain is long-running; retrying a 10-minute request
        repeatedly is worse than fail-fast.
    """

    _DEFAULT_MAX_RETRIES = 2
    _DEFAULT_INITIAL_BACKOFF_S = 1.0
    _DEFAULT_MAX_BACKOFF_S = 8.0
    _DEFAULT_BACKOFF_MULTIPLIER = 2.0
    _DEFAULT_JITTER_RATIO = 0.25

    def __init__(
        self,
        *,
        settings: AudioHostSettings,
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
        """Remote audio chain always runs concurrently with H200-local visual extraction."""
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
                    f"audio host {path} returned HTTP {exc.code}: "
                    f"{err_body[:512] or exc.reason}"
                )
                # Wrap with ``status_code`` so the outer retry loop can decide
                # whether the failure is transient (``_is_transient``).
                raise RemoteAudioChainError(message, status_code=exc.code) from exc

            if status >= 400:
                raise RemoteAudioChainError(
                    f"audio host {path} returned HTTP {status}",
                    status_code=status,
                )
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RemoteAudioChainError(
                    f"audio host {path} returned non-JSON body: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise RemoteAudioChainError(
                    f"audio host {path} response must be an object, got "
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
                    "[audio_host_client] transient error on %s attempt=%d/%d; "
                    "retrying in %.2fs: %s",
                    path,
                    retry_index + 1,
                    self._max_retries + 1,
                    sleep_s,
                    exc,
                )
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
        # Unreachable: the loop above either returns or raises.
        raise RemoteAudioChainError(f"audio host {path} retries exhausted")

    def healthcheck(self) -> dict[str, Any]:
        """GET ``{healthcheck_path}`` — raises :class:`RemoteAudioChainError` on failure."""
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
            raise RemoteAudioChainError(
                f"audio host healthcheck returned HTTP {exc.code}",
                status_code=exc.code,
            ) from exc
        except Exception as exc:
            raise RemoteAudioChainError(
                f"audio host healthcheck failed: {exc}"
            ) from exc
        if status >= 400:
            raise RemoteAudioChainError(
                f"audio host healthcheck returned HTTP {status}",
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
    ) -> PhaseOneAudioResponse:
        """Invoke the remote audio chain for a single Phase 1 job.

        The server runs VibeVoice ASR, NeMo Forced Aligner, emotion2vec+ and
        YAMNet sequentially on a hot GPU and returns the combined payload plus
        a stage-events trail. Stage events are re-emitted through
        ``stage_event_logger`` so downstream telemetry matches the previous
        in-process execution.
        """
        if not audio_gcs_uri:
            raise ValueError("RemoteAudioChainClient.run requires a non-empty audio_gcs_uri")

        payload: dict[str, Any] = {"audio_gcs_uri": audio_gcs_uri}
        if source_url:
            payload["source_url"] = source_url
        if video_gcs_uri:
            payload["video_gcs_uri"] = video_gcs_uri
        if run_id:
            payload["run_id"] = run_id

        t_start = time.perf_counter()
        logger.info(
            "[audio_host_client] invoking phase1-audio run_id=%s audio=%s",
            run_id or "-",
            audio_gcs_uri,
        )
        try:
            raw = self._post_json(
                path="/tasks/phase1-audio",
                payload=payload,
                timeout_s=float(self.settings.timeout_s),
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            _emit(
                stage_event_logger,
                stage_name="audio_host_call",
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
            response = PhaseOneAudioResponse.from_json(raw)
        except Exception as exc:
            _emit(
                stage_event_logger,
                stage_name="audio_host_call",
                status="failed",
                duration_ms=(time.perf_counter() - t_start) * 1000.0,
                metadata={"run_id": run_id or ""},
                error_payload={
                    "code": "InvalidResponseShape",
                    "message": str(exc)[:2048],
                },
            )
            raise RemoteAudioChainError(
                f"invalid audio host response shape: {exc}"
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
            "[audio_host_client] phase1-audio completed run_id=%s turns=%d words=%d "
            "emotion_segments=%d yamnet_events=%d in %.1f ms",
            run_id or "-",
            len(response.turns),
            len((response.diarization_payload or {}).get("words") or []),
            len((response.emotion2vec_payload or {}).get("segments") or []),
            len((response.yamnet_payload or {}).get("events") or []),
            elapsed_ms,
        )
        return response


__all__ = [
    "PhaseOneAudioResponse",
    "RemoteAudioChainClient",
    "RemoteAudioChainError",
]
