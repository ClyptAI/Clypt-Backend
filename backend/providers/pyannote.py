from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from .config import PyannoteSettings

logger = logging.getLogger(__name__)

_TERMINAL_SUCCESS = {"succeeded", "completed", "done"}
_TERMINAL_FAILURE = {"failed", "error", "canceled", "cancelled"}
_ACTIVE_STATES = {"queued", "pending", "running", "processing", "created"}


class PyannoteCloudClient:
    def __init__(
        self,
        *,
        settings: PyannoteSettings,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(
            base_url=settings.base_url,
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
            timeout=settings.timeout_s,
        )

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> "PyannoteCloudClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def run_diarize(self, *, media_url: str) -> dict[str, Any]:
        job_id = self._submit_job(
            endpoint="/v1/diarize",
            payload={
                "url": media_url,
                "model": self.settings.diarize_model,
                "transcription": True,
                "transcriptionConfig": {
                    "model": self.settings.transcription_model,
                },
            },
        )
        logger.info("[pyannote] diarize job submitted — id=%s", job_id)
        return self._wait_for_output(job_id=job_id, label="diarize")

    def run_identify(self, *, media_url: str, voiceprint_ids: list[str]) -> dict[str, Any]:
        job_id = self._submit_job(
            endpoint="/v1/identify",
            payload={
                "url": media_url,
                "voiceprints": list(voiceprint_ids),
            },
        )
        logger.info("[pyannote] identify job submitted — id=%s", job_id)
        return self._wait_for_output(job_id=job_id, label="identify")

    def _submit_job(self, *, endpoint: str, payload: dict[str, Any]) -> str:
        response = self._http.post(endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
        job_id = (
            data.get("jobId")
            or data.get("job_id")
            or data.get("id")
        )
        if not job_id:
            raise RuntimeError(f"pyannote job submission returned no job id for {endpoint}")
        return str(job_id)

    def _wait_for_output(self, *, job_id: str, label: str = "job") -> dict[str, Any]:
        started_at = time.monotonic()
        last_status = ""
        poll_count = 0
        _log_every = max(1, int(10.0 / max(0.5, self.settings.poll_interval_s)))  # log ~every 10 s
        while True:
            response = self._http.get(f"/v1/jobs/{job_id}")
            response.raise_for_status()
            payload = response.json()
            status = str(payload.get("status") or "").strip().lower()
            elapsed = time.monotonic() - started_at
            poll_count += 1

            if status != last_status or poll_count % _log_every == 0:
                logger.info(
                    "[pyannote] %s  id=%s  status=%s  elapsed=%.0f s",
                    label, job_id, status or "unknown", elapsed,
                )
                last_status = status

            if status in _TERMINAL_SUCCESS:
                turns_out = len(list((payload.get("output") or payload).get("turns") or []))
                logger.info(
                    "[pyannote] %s complete in %.0f s — %d turns",
                    label, elapsed, turns_out,
                )
                return dict(payload.get("output") or payload)
            if status in _TERMINAL_FAILURE:
                raise RuntimeError(f"pyannote job {job_id} failed with status={status}")
            if status and status not in _ACTIVE_STATES:
                raise RuntimeError(f"pyannote job {job_id} returned unknown status={status}")
            if elapsed > self.settings.timeout_s:
                raise TimeoutError(f"timed out waiting for pyannote job {job_id}")
            time.sleep(max(0.0, self.settings.poll_interval_s))


__all__ = ["PyannoteCloudClient"]
