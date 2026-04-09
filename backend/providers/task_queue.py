from __future__ import annotations

import json
import re
from hashlib import sha1
from dataclasses import dataclass
from typing import Any

from .config import CloudTasksSettings

try:  # pragma: no cover - import guard for environments without google-api-core
    from google.api_core.exceptions import AlreadyExists
except ImportError:  # pragma: no cover
    class AlreadyExists(Exception):
        pass

_TASK_ID_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_TASK_ID_LENGTH = 500


def _task_id_for_run_id(run_id: str) -> str:
    run_slug = _TASK_ID_RE.sub("-", run_id).strip("-")
    if not run_slug:
        run_slug = "run"
    digest = sha1(run_id.encode("utf-8")).hexdigest()[:12]
    prefix = f"phase24-{run_slug}"
    max_prefix_len = _MAX_TASK_ID_LENGTH - len(digest) - 1
    if len(prefix) > max_prefix_len:
        prefix = prefix[:max_prefix_len].rstrip("-")
    if not prefix:
        prefix = "phase24"
    return f"{prefix}-{digest}"


@dataclass(slots=True)
class Phase24TaskQueueClient:
    settings: CloudTasksSettings
    tasks_client: Any

    @property
    def queue_path(self) -> str:
        return f"projects/{self.settings.project}/locations/{self.settings.location}/queues/{self.settings.queue}"

    def task_name_for_run_id(self, run_id: str) -> str:
        return f"{self.queue_path}/tasks/{_task_id_for_run_id(run_id)}"

    def enqueue_phase24(
        self,
        *,
        run_id: str,
        payload: dict[str, Any],
        worker_url: str | None = None,
    ) -> str:
        target_url = worker_url or self.settings.worker_url
        if not target_url:
            raise ValueError("worker_url is required to enqueue a phase24 task")

        http_request: dict[str, Any] = {
            "http_method": "POST",
            "url": target_url,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8"),
        }
        if self.settings.service_account_email:
            http_request["oidc_token"] = {
                "service_account_email": self.settings.service_account_email,
                "audience": target_url,
            }

        task = {
            "name": self.task_name_for_run_id(run_id),
            "http_request": http_request,
        }
        try:
            response = self.tasks_client.create_task(request={"parent": self.queue_path, "task": task})
        except AlreadyExists:
            return task["name"]
        return response.name


__all__ = [
    "AlreadyExists",
    "Phase24TaskQueueClient",
    "_task_id_for_run_id",
]
