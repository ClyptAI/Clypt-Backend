"""Remote node-media prep runner (ffmpeg + GCS upload).

The Phase26 worker invokes this via the
``POST /tasks/node-media-prep`` endpoint. The media-prep worker:

1. Downloads the source video from GCS into a per-run local scratch dir
   (once, even if hundreds of nodes share it).
2. Runs ffmpeg NVENC (via
   :func:`backend.pipeline.semantics.media_embeddings.extract_node_clip`) for
   each node in a bounded thread pool.
3. Uploads each clip to ``gs://{bucket}/{object_prefix}/{node_id}.mp4``.
4. Returns the ordered list of media descriptors (``node_id``, ``file_uri``,
   ``mime_type``).

The caller never sees raw clip bytes; it only receives the GCS URIs.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.pipeline.semantics.media_embeddings import prepare_node_media_embeddings
from backend.providers.storage import _normalize_node_media_object_prefix, parse_gcs_uri

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NodeMediaPrepRequest:
    """Parsed ``POST /tasks/node-media-prep`` request body."""

    run_id: str
    video_gcs_uri: str
    object_prefix: str
    max_concurrency: int
    nodes: list[dict[str, Any]]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "NodeMediaPrepRequest":
        run_id = str(payload.get("run_id") or "").strip()
        video_gcs_uri = str(payload.get("video_gcs_uri") or "").strip()
        object_prefix = str(
            payload.get("object_prefix") or f"phase14/{run_id}/node_media"
        ).strip()
        try:
            max_concurrency = int(payload.get("max_concurrency") or 8)
        except (TypeError, ValueError):
            raise ValueError("max_concurrency must be an integer >= 1") from None
        nodes = payload.get("nodes") or []
        if not run_id:
            raise ValueError("run_id is required")
        try:
            _bucket, _object_key = parse_gcs_uri(video_gcs_uri)
        except ValueError as exc:
            raise ValueError("video_gcs_uri must be a gs:// URI") from exc
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if not isinstance(nodes, list):
            raise ValueError("nodes must be a list")
        cleaned: list[dict[str, Any]] = []
        for node in nodes:
            if not isinstance(node, dict):
                raise ValueError("each node must be an object")
            node_id = str(node.get("node_id") or "").strip()
            if not node_id:
                raise ValueError("node.node_id is required")
            try:
                start_ms = int(node.get("start_ms") or 0)
                end_ms = int(node.get("end_ms") or 0)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"node {node_id!r} has non-numeric start_ms/end_ms"
                ) from exc
            if start_ms > end_ms:
                raise ValueError(
                    f"node {node_id!r} start_ms > end_ms ({start_ms} > {end_ms})"
                )
            cleaned.append(
                {"node_id": node_id, "start_ms": start_ms, "end_ms": end_ms}
            )
        return cls(
            run_id=run_id,
            video_gcs_uri=video_gcs_uri,
            object_prefix=_normalize_node_media_object_prefix(
                "",
                object_prefix,
                job_id=run_id,
            ),
            max_concurrency=max_concurrency,
            nodes=cleaned,
        )


@dataclass(slots=True)
class _MinimalNode:
    """Duck-typed node used by :func:`prepare_node_media_embeddings`.

    The extractor only reads ``node_id``, ``start_ms``, ``end_ms``; we avoid
    instantiating the full pydantic ``SemanticGraphNode`` contract on the
    media-prep worker to keep request parsing cheap and decoupled from
    Phase 2-4's schema.
    """

    node_id: str
    start_ms: int
    end_ms: int


def _as_semantic_nodes(nodes: list[dict[str, Any]]) -> list[_MinimalNode]:
    return [
        _MinimalNode(
            node_id=str(n["node_id"]),
            start_ms=int(n["start_ms"]),
            end_ms=int(n["end_ms"]),
        )
        for n in nodes
    ]


def run_node_media_prep(
    *,
    request: NodeMediaPrepRequest,
    storage_client: Any,
    scratch_root: Path,
) -> dict[str, Any]:
    """Execute one ``POST /tasks/node-media-prep`` request on the media-prep worker.

    Parameters
    ----------
    request:
        Validated request payload (see :meth:`NodeMediaPrepRequest.from_payload`).
    storage_client:
        An object exposing ``download_file(gcs_uri, local_path)`` and
        ``upload_file(local_path, object_name) -> str`` — typically a
        :class:`backend.providers.storage.GCSStorageClient`.
    scratch_root:
        Per-worker scratch directory.

    Returns the JSON body served by the FastAPI route.
    """
    if not request.nodes:
        return {"run_id": request.run_id, "media": []}

    semantic_nodes = _as_semantic_nodes(request.nodes)
    storage_bucket = getattr(getattr(storage_client, "settings", None), "gcs_bucket", "") or ""
    object_prefix = _normalize_node_media_object_prefix(
        storage_bucket,
        request.object_prefix,
        job_id=request.run_id,
    )

    with tempfile.TemporaryDirectory(
        prefix=f"node-media-prep-{request.run_id}-", dir=str(scratch_root)
    ) as tmp_root_str:
        tmp_root = Path(tmp_root_str)
        source_video_path = tmp_root / "source_video.mp4"
        logger.info(
            "[node_media_prep] downloading source video run_id=%s uri=%s",
            request.run_id,
            request.video_gcs_uri,
        )
        storage_client.download_file(
            gcs_uri=request.video_gcs_uri,
            local_path=source_video_path,
        )

        clips_dir = tmp_root / "clips"
        descriptors = prepare_node_media_embeddings(
            nodes=semantic_nodes,
            source_video_path=source_video_path,
            clips_dir=clips_dir,
            storage_client=storage_client,
            object_prefix=object_prefix,
            max_concurrent=request.max_concurrency,
        )

    # The caller only needs the GCS URIs; strip local_path because the
    # TemporaryDirectory is already cleaned up by serialization time.
    media = [
        {
            "node_id": d["node_id"],
            "file_uri": d["file_uri"],
            "mime_type": d.get("mime_type", "video/mp4"),
        }
        for d in descriptors
    ]
    return {"run_id": request.run_id, "media": media}


__all__ = ["NodeMediaPrepRequest", "run_node_media_prep"]
