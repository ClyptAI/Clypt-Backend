"""Compatibility shim for Phase 6 render/export.

Active deployment is `scripts.modal.media_worker_app`, which shares one L40S
between node-media-prep and render/export. This module preserves the historical
import path without defining a separate warm GPU function.
"""

from __future__ import annotations

from .media_worker_app import (
    _build_storage_client,
    _require_ffmpeg,
    _require_render_runtime as _require_worker_runtime,
    app,
    health,
    image,
    media_gpu_job,
    media_worker as render_video,
    modal,
    render_video_result_route,
    render_video_route,
    web_app,
)


class render_video_job:
    @staticmethod
    def spawn(payload):
        return media_gpu_job.spawn("render_video", payload)
