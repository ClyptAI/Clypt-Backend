from __future__ import annotations

import gzip
import json
import random
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.phase1_runtime.payloads import VisualFuturePayload
from backend.providers.storage import GCSStorageClient

from ._http_retry import RemoteServiceHTTPError, TRANSIENT_HTTP_STATUS, retry_with_backoff
from .config import Phase1VisualServiceSettings


class RemoteVisualExtractError(RemoteServiceHTTPError):
    """Raised when the Modal visual-extract endpoint rejects or fails a request."""


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, RemoteVisualExtractError):
        return exc.status_code in TRANSIENT_HTTP_STATUS
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in TRANSIENT_HTTP_STATUS
    return isinstance(exc, urllib.error.URLError | TimeoutError | OSError)


class RemoteVisualExtractClient:
    _DEFAULT_MAX_RETRIES = 2
    _DEFAULT_POLL_INTERVAL_S = 1.0

    def __init__(
        self,
        *,
        settings: Phase1VisualServiceSettings,
        storage_client: GCSStorageClient | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.settings = settings
        self._storage_client = storage_client
        self._max_retries = (
            max(0, int(max_retries))
            if max_retries is not None
            else self._DEFAULT_MAX_RETRIES
        )

    def _endpoint(self, path: str) -> str:
        service_url = self.settings.service_url.rstrip("/")
        canonical_task_path = "/tasks/visual-extract"
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
                except Exception:  # pragma: no cover
                    detail = ""
                raise RemoteVisualExtractError(
                    f"visual-extract {path} returned HTTP {exc.code}: {detail[:512] or exc.reason}",
                    status_code=exc.code,
                ) from exc
            if status >= 400:
                raise RemoteVisualExtractError(
                    f"visual-extract {path} returned HTTP {status}",
                    status_code=status,
                )
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RemoteVisualExtractError(
                    f"visual-extract {path} returned non-JSON body: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise RemoteVisualExtractError(
                    f"visual-extract {path} response must be an object, got {type(parsed).__name__}"
                )
            return parsed

        return retry_with_backoff(
            _do_request,
            max_retries=self._max_retries,
            classify_transient=_is_transient,
            log_prefix="[visual_extract_client]",
            operation="visual_extract",
            sleep=time.sleep,
            rng=random.random,
        )

    def _hydrate_phase1_visual_artifact(self, raw: dict[str, Any]) -> dict[str, Any]:
        if isinstance(raw.get("phase1_visual"), dict):
            return raw
        artifact_uri = str(raw.get("phase1_visual_gcs_uri") or "").strip()
        if not artifact_uri:
            return raw
        if self._storage_client is None:
            raise RemoteVisualExtractError(
                "visual-extract result returned phase1_visual_gcs_uri but no storage_client is configured"
            )
        encoding = str(raw.get("phase1_visual_encoding") or "json_gzip_v1").strip().lower()
        if encoding != "json_gzip_v1":
            raise RemoteVisualExtractError(
                f"unsupported phase1_visual_encoding={encoding!r} for visual-extract result"
            )
        with tempfile.TemporaryDirectory(prefix="clypt-visual-result-") as tmp_dir:
            local_path = Path(tmp_dir) / "phase1_visual.json.gz"
            self._storage_client.download_file(gcs_uri=artifact_uri, local_path=local_path)
            with gzip.open(local_path, "rt", encoding="utf-8") as fh:
                phase1_visual = json.load(fh)
        if not isinstance(phase1_visual, dict):
            raise RemoteVisualExtractError(
                f"phase1_visual artifact must decode to an object, got {type(phase1_visual).__name__}"
            )
        hydrated = dict(raw)
        hydrated["phase1_visual"] = phase1_visual
        return hydrated

    def submit(
        self,
        *,
        run_id: str,
        video_gcs_uri: str,
        video_path: Path | str | None = None,
        source_video_sha256: str | None = None,
    ) -> VisualFuturePayload:
        payload: dict[str, Any] = {
            "run_id": run_id,
            "video_gcs_uri": video_gcs_uri,
            "source_video_gcs_uri": video_gcs_uri,
            "submitted_at_ms": time.time() * 1000.0,
        }
        if video_path is not None:
            payload["video_path"] = str(video_path)
        if source_video_sha256:
            payload["source_video_sha256"] = source_video_sha256
        raw = self._request_json(
            path="/tasks/visual-extract",
            method="POST",
            payload=payload,
            timeout_s=min(60.0, float(self.settings.timeout_s)),
        )
        call_id = str(raw.get("call_id") or "").strip()
        if not call_id:
            raise RemoteVisualExtractError("visual-extract submit response missing call_id")
        result_path = str(raw.get("result_path") or f"/tasks/visual-extract/result/{call_id}")
        return VisualFuturePayload(
            backend="modal_rfdetr_l40s",
            call_id=call_id,
            service_url=self._endpoint("/tasks/visual-extract"),
            result_url=self._endpoint(result_path),
            source_video_gcs_uri=video_gcs_uri,
            source_video_sha256=source_video_sha256,
            submitted_at=datetime.now(timezone.utc).isoformat(),
        )

    def wait_for_result(self, *, visual_future: VisualFuturePayload | dict[str, Any]) -> dict[str, Any]:
        future = (
            visual_future
            if isinstance(visual_future, VisualFuturePayload)
            else VisualFuturePayload.model_validate(visual_future)
        )
        deadline = time.monotonic() + max(0.1, float(self.settings.timeout_s))
        while True:
            if time.monotonic() > deadline:
                raise RemoteVisualExtractError(
                    f"visual-extract timed out while polling call_id={future.call_id}"
                )
            raw = self._request_json(
                path=f"/tasks/visual-extract/result/{future.call_id}",
                method="GET",
                payload=None,
                timeout_s=min(30.0, max(5.0, float(self.settings.timeout_s))),
            )
            status = str(raw.get("status") or "").strip().lower()
            if status in {"pending", "running", "submitted"}:
                time.sleep(self._DEFAULT_POLL_INTERVAL_S)
                continue
            if status in {"", "succeeded", "success", "completed"}:
                return self._hydrate_phase1_visual_artifact(raw)
            raise RemoteVisualExtractError(
                f"visual-extract poll for call_id={future.call_id} returned terminal status {status!r}"
            )


__all__ = ["RemoteVisualExtractClient", "RemoteVisualExtractError"]
