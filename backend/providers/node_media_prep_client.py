"""Remote client for GPU node-media prep.

Phase26 needs one ~10-60 s ffmpeg clip (and a GCS upload) per semantic node
so downstream multimodal embeddings can be produced. This work runs through
the remote ``POST /tasks/node-media-prep`` endpoint and is the only
supported path for this pipeline (no local fallback).

The client exposes itself as a callable matching the ``node_media_preparer``
contract on :class:`backend.runtime.phase14_live.Phase14LiveRunner`:

    media = preparer(nodes=nodes, paths=paths, phase1_outputs=phase1_outputs)

Returned items match the ``prepare_node_media_embeddings`` shape so Phase 2's
downstream code (vertex multimodal embeddings + debug JSON) is untouched:

    {"node_id", "file_uri", "mime_type", "local_path"}

Here ``local_path`` is always an empty string — clips live only on the
remote media-prep worker and downstream code only consumes ``file_uri``.
"""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from ._http_retry import (
    RemoteServiceHTTPError,
    TRANSIENT_HTTP_STATUS,
    retry_with_backoff,
)
from .config import NodeMediaPrepSettings
from .storage import _normalize_node_media_object_prefix, parse_gcs_uri

logger = logging.getLogger(__name__)


class RemoteNodeMediaPrepError(RemoteServiceHTTPError):
    """Raised when the node-media-prep endpoint rejects or fails a request."""


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, RemoteNodeMediaPrepError):
        code = exc.status_code
        if code is None:
            return False
        return code in TRANSIENT_HTTP_STATUS
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in TRANSIENT_HTTP_STATUS
    if isinstance(exc, urllib.error.URLError):
        return True
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, OSError):
        return True
    return False


@dataclass(slots=True)
class _NodeSpec:
    node_id: str
    start_ms: int
    end_ms: int

    def to_json(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "start_ms": int(self.start_ms),
            "end_ms": int(self.end_ms),
        }


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
            "RemoteNodeMediaPrepClient requires phase1_outputs.phase1_audio to expose "
            "a non-empty 'video_gcs_uri'."
        )
    try:
        _bucket, _object_key = parse_gcs_uri(uri)
    except ValueError as exc:
        raise ValueError(
            f"phase1_outputs.phase1_audio['video_gcs_uri'] must be a gs:// URI, got {uri!r}"
        ) from exc
    return uri


def _extract_run_id(paths: Any) -> str:
    run_id = getattr(paths, "run_id", None)
    if not run_id:
        raise ValueError("paths.run_id is required for remote node-media prep")
    return str(run_id)


def _node_specs(nodes: list[Any]) -> list[_NodeSpec]:
    specs: list[_NodeSpec] = []
    for node in nodes:
        node_id = getattr(node, "node_id", None)
        start_ms = getattr(node, "start_ms", None)
        end_ms = getattr(node, "end_ms", None)
        if isinstance(node, dict):
            node_id = node_id or node.get("node_id")
            start_ms = start_ms if start_ms is not None else node.get("start_ms")
            end_ms = end_ms if end_ms is not None else node.get("end_ms")
        if not node_id:
            raise ValueError("every node passed to RemoteNodeMediaPrepClient must have a node_id")
        try:
            start_int = int(start_ms) if start_ms is not None else 0
            end_int = int(end_ms) if end_ms is not None else 0
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"node {node_id!r} has non-numeric start_ms/end_ms"
            ) from exc
        if start_int > end_int:
            raise ValueError(
                f"node {node_id!r} has start_ms > end_ms ({start_int} > {end_int})"
            )
        specs.append(_NodeSpec(node_id=str(node_id), start_ms=start_int, end_ms=end_int))
    return specs


class RemoteNodeMediaPrepClient:
    """Callable that proxies node-clip extraction to the remote media-prep service.

    The Phase26 worker invokes this once per Phase 2 job with the full list
    of semantic nodes. The remote service extracts each clip with ffmpeg,
    uploads to GCS, and returns the media descriptors in the order requested.
    """

    _DEFAULT_MAX_RETRIES = 2
    _DEFAULT_INITIAL_BACKOFF_S = 1.0
    _DEFAULT_MAX_BACKOFF_S = 8.0
    _DEFAULT_BACKOFF_MULTIPLIER = 2.0
    _DEFAULT_JITTER_RATIO = 0.25

    def __init__(
        self,
        *,
        settings: NodeMediaPrepSettings,
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
                raise RemoteNodeMediaPrepError(
                    f"node-media-prep {path} returned HTTP {exc.code}: "
                    f"{err_body[:512] or exc.reason}",
                    status_code=exc.code,
                ) from exc

            if status >= 400:
                raise RemoteNodeMediaPrepError(
                    f"node-media-prep {path} returned HTTP {status}",
                    status_code=status,
                )
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RemoteNodeMediaPrepError(
                    f"node-media-prep {path} returned non-JSON body: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise RemoteNodeMediaPrepError(
                    f"node-media-prep {path} response must be an object, got "
                    f"{type(parsed).__name__}"
                )
            return parsed

        return retry_with_backoff(
            _do_request,
            max_retries=self._max_retries,
            classify_transient=_is_transient,
            log_prefix="[node_media_prep_client]",
            operation="node_media_prep",
            initial_backoff_s=self._DEFAULT_INITIAL_BACKOFF_S,
            max_backoff_s=self._DEFAULT_MAX_BACKOFF_S,
            multiplier=self._DEFAULT_BACKOFF_MULTIPLIER,
            jitter_ratio=self._DEFAULT_JITTER_RATIO,
            sleep=time.sleep,
            rng=random.random,
        )

    def __call__(
        self,
        *,
        nodes: list[Any],
        paths: Any,
        phase1_outputs: Any,
    ) -> list[dict[str, str]]:
        if not nodes:
            return []

        run_id = _extract_run_id(paths)
        video_gcs_uri = _extract_video_gcs_uri(phase1_outputs)
        specs = _node_specs(nodes)

        payload: dict[str, Any] = {
            "run_id": run_id,
            "video_gcs_uri": video_gcs_uri,
            "object_prefix": _normalize_node_media_object_prefix(
                "",
                f"phase14/{run_id}/node_media",
                job_id=run_id,
            ),
            "max_concurrency": int(self.settings.max_concurrency),
            "nodes": [spec.to_json() for spec in specs],
        }

        t_start = time.perf_counter()
        logger.info(
            "[node_media_prep_client] invoking node-media-prep run_id=%s nodes=%d video=%s",
            run_id,
            len(specs),
            video_gcs_uri,
        )
        raw = self._post_json(
            path="/tasks/node-media-prep",
            payload=payload,
            timeout_s=float(self.settings.timeout_s),
        )

        media = raw.get("media")
        if not isinstance(media, list):
            raise RemoteNodeMediaPrepError(
                "node-media-prep response must contain a 'media' list, got "
                f"{type(media).__name__}"
            )

        by_node: dict[str, dict[str, str]] = {}
        for item in media:
            if not isinstance(item, dict):
                raise RemoteNodeMediaPrepError(
                    "node-media-prep 'media' entries must be objects"
                )
            node_id = str(item.get("node_id") or "").strip()
            file_uri = str(item.get("file_uri") or "").strip()
            if not node_id or not file_uri:
                raise RemoteNodeMediaPrepError(
                    "node-media-prep media entry missing required node_id/file_uri"
                )
            try:
                _bucket, _object_key = parse_gcs_uri(file_uri)
            except ValueError as exc:
                raise RemoteNodeMediaPrepError(
                    f"node-media-prep file_uri must be a gs:// URI, got {file_uri!r}"
                ) from exc
            mime_type = str(item.get("mime_type") or "video/mp4")
            by_node[node_id] = {
                "node_id": node_id,
                "file_uri": file_uri,
                "mime_type": mime_type,
                # Clip bytes live only on the RTX box. Downstream consumers
                # (vertex embeddings, debug JSON dump) only read file_uri, so
                # an empty local_path is safe and keeps the shape stable.
                "local_path": "",
            }

        missing = [spec.node_id for spec in specs if spec.node_id not in by_node]
        if missing:
            raise RemoteNodeMediaPrepError(
                "node-media-prep response missing media for nodes: "
                f"{missing[:20]}{'…' if len(missing) > 20 else ''}"
            )

        ordered = [by_node[spec.node_id] for spec in specs]
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "[node_media_prep_client] node-media-prep completed run_id=%s nodes=%d in %.1f ms",
            run_id,
            len(ordered),
            elapsed_ms,
        )
        return ordered


__all__ = [
    "RemoteNodeMediaPrepClient",
    "RemoteNodeMediaPrepError",
]
