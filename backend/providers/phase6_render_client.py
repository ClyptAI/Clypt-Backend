"""Remote client for Phase 6 render/export."""

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
from .config import Phase6RenderSettings

logger = logging.getLogger(__name__)


class RemotePhase6RenderError(RemoteServiceHTTPError):
    """Raised when the Phase 6 render endpoint rejects or fails a request."""


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, RemotePhase6RenderError):
        return exc.status_code in TRANSIENT_HTTP_STATUS
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in TRANSIENT_HTTP_STATUS
    if isinstance(exc, urllib.error.URLError | TimeoutError | OSError):
        return True
    return False


def _extract_run_id(paths: Any) -> str:
    run_id = getattr(paths, "run_id", None)
    if not run_id:
        raise ValueError("paths.run_id is required for remote Phase 6 render")
    return str(run_id)


def _extract_video_gcs_uri(phase1_outputs: Any) -> str:
    phase1_audio = getattr(phase1_outputs, "phase1_audio", None)
    if phase1_audio is None and isinstance(phase1_outputs, dict):
        phase1_audio = phase1_outputs.get("phase1_audio")
    if hasattr(phase1_audio, "model_dump"):
        phase1_audio = phase1_audio.model_dump(mode="json")
    if isinstance(phase1_audio, dict):
        uri = (phase1_audio.get("video_gcs_uri") or "").strip()
    else:
        uri = str(getattr(phase1_audio, "video_gcs_uri", "") or "").strip()
    if not uri:
        raise ValueError(
            "RemotePhase6RenderClient requires phase1_outputs.phase1_audio.video_gcs_uri."
        )
    if not uri.startswith("gs://"):
        raise ValueError(f"phase1_outputs.phase1_audio.video_gcs_uri must be gs://, got {uri!r}")
    return uri


class RemotePhase6RenderClient:
    _DEFAULT_MAX_RETRIES = 2

    def __init__(
        self,
        *,
        settings: Phase6RenderSettings,
        storage_client: Any,
        max_retries: int | None = None,
    ) -> None:
        self.settings = settings
        self.storage_client = storage_client
        self._max_retries = (
            max(0, int(max_retries))
            if max_retries is not None
            else self._DEFAULT_MAX_RETRIES
        )

    def _endpoint(self, path: str) -> str:
        service_url = self.settings.service_url.rstrip("/")
        canonical_task_path = "/tasks/render-video"
        if service_url.endswith(path):
            return service_url
        if service_url.endswith(canonical_task_path):
            service_url = service_url[: -len(canonical_task_path)]
        return service_url + path

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.auth_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request_json(
        self,
        *,
        path: str,
        method: str,
        payload: dict[str, Any] | None,
        timeout_s: float,
    ) -> dict[str, Any]:
        url = self._endpoint(path)
        body = (
            json.dumps(payload, ensure_ascii=True).encode("utf-8")
            if payload is not None
            else None
        )
        headers = self._auth_headers()

        def _do_request() -> dict[str, Any]:
            req = urllib.request.Request(url, data=body, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    raw = resp.read().decode("utf-8")
                    status = resp.status
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    detail = ""
                raise RemotePhase6RenderError(
                    f"phase6-render {path} returned HTTP {exc.code}: {detail[:512] or exc.reason}",
                    status_code=exc.code,
                ) from exc

            if status >= 400:
                raise RemotePhase6RenderError(
                    f"phase6-render {path} returned HTTP {status}",
                    status_code=status,
                )
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RemotePhase6RenderError(
                    f"phase6-render {path} returned non-JSON body: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise RemotePhase6RenderError(
                    f"phase6-render {path} response must be an object, got {type(parsed).__name__}"
                )
            return parsed

        return retry_with_backoff(
            _do_request,
            max_retries=self._max_retries,
            classify_transient=_is_transient,
            log_prefix="[phase6_render_client]",
            operation="phase6_render",
            sleep=time.sleep,
            rng=random.random,
        )

    def _poll_result(self, *, call_id: str, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        while True:
            if time.monotonic() > deadline:
                raise RemotePhase6RenderError(
                    f"phase6-render timed out while polling call_id={call_id}"
                )
            raw = self._request_json(
                path=f"/tasks/render-video/result/{call_id}",
                method="GET",
                payload=None,
                timeout_s=min(30.0, max(5.0, timeout_s)),
            )
            if raw.get("status") == "pending":
                time.sleep(1.0)
                continue
            return raw

    def _upload_artifacts(self, *, run_id: str, artifact_paths: dict[str, str]) -> dict[str, str]:
        uploaded: dict[str, str] = {}
        for key, local in artifact_paths.items():
            local_path = Path(local)
            if not local_path.exists():
                continue
            object_name = f"phase14/{run_id}/render_inputs/{local_path.name}"
            uploaded[key] = self.storage_client.upload_file(
                local_path=local_path,
                object_name=object_name,
            )
        return uploaded

    def __call__(
        self,
        *,
        paths: Any,
        phase1_outputs: Any,
        artifact_paths: dict[str, str],
    ) -> dict[str, Any]:
        run_id = _extract_run_id(paths)
        render_plan_path = getattr(paths, "render_plan", None)
        if render_plan_path is None:
            raise ValueError("paths.render_plan is required for remote Phase 6 render")
        render_plan = json.loads(Path(render_plan_path).read_text(encoding="utf-8"))
        artifact_gcs_uris = self._upload_artifacts(run_id=run_id, artifact_paths=artifact_paths)
        submit = self._request_json(
            path="/tasks/render-video",
            method="POST",
            payload={
                "run_id": run_id,
                "source_video_gcs_uri": _extract_video_gcs_uri(phase1_outputs),
                "artifact_gcs_uris": artifact_gcs_uris,
                "clips": [
                    {
                        "clip_id": clip["clip_id"],
                        "clip_start_ms": int(clip["clip_start_ms"]),
                        "clip_end_ms": int(clip["clip_end_ms"]),
                    }
                    for clip in render_plan.get("clips", [])
                ],
            },
            timeout_s=min(60.0, self.settings.timeout_s),
        )
        call_id = str(submit.get("call_id") or "").strip()
        if not call_id:
            raise RemotePhase6RenderError("phase6-render submit response missing call_id")
        return self._poll_result(call_id=call_id, timeout_s=self.settings.timeout_s)


__all__ = ["RemotePhase6RenderClient", "RemotePhase6RenderError"]
