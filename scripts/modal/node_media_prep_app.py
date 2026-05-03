"""Compatibility shim for node-media-prep.

Active deployment is `scripts.modal.media_worker_app`, which shares one L40S
between node-media-prep and render/export. This module preserves the historical
import path without defining a separate warm GPU function.
"""

from __future__ import annotations

from .media_worker_app import (
    _build_storage_client,
    _require_ffmpeg,
    _require_node_media_runtime as _require_worker_runtime,
    app,
    health,
    image,
    media_gpu_job,
    media_worker as node_media_prep,
    modal,
    node_media_prep_result_route,
    node_media_prep_route,
    web_app,
)


class node_media_prep_job:
    @staticmethod
    def spawn(payload):
        return media_gpu_job.spawn("node_media_prep", payload)
