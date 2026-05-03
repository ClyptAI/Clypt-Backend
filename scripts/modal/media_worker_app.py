"""Modal app for the shared media L40S worker.

The public ASGI surface stays CPU-only. A single warm L40S function dispatches
node-media-prep and Phase 6 render/export work so the deployment uses one media
GPU pool instead of one warm GPU per route.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import fastapi
import modal

from backend.providers.config import StorageSettings
from backend.providers.storage import GCSStorageClient
from backend.runtime.node_media_prep import NodeMediaPrepRequest, run_node_media_prep
from backend.runtime.phase6_render import Phase6RenderRequest, run_phase6_render

app = modal.App("clypt-media-l40s")
web_app = fastapi.FastAPI()
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("google-cloud-storage>=2.19.0")
    .pip_install("google-auth>=2.38.0")
    .pip_install_from_requirements("requirements-modal-media-l40s.txt")
    .add_local_python_source("backend")
    .add_local_dir("backend/assets", remote_path="/root/backend/assets")
)


def _require_codec(codec_cmd: list[str], expected: str) -> None:
    output = subprocess.check_output(codec_cmd, text=True)
    if expected not in output:
        raise RuntimeError(f"missing required ffmpeg codec support: {expected}")


def _require_ffmpeg() -> None:
    subprocess.check_output(["ffmpeg", "-version"], text=True)


def _require_node_media_runtime() -> None:
    _require_codec(["ffmpeg", "-encoders"], "h264_nvenc")
    _require_codec(["ffmpeg", "-decoders"], "h264_cuvid")


def _require_render_runtime() -> None:
    _require_codec(["ffmpeg", "-encoders"], "h264_nvenc")


def _expected_token(kind: str) -> str:
    if kind == "node_media_prep":
        return (
            os.environ.get("NODE_MEDIA_PREP_AUTH_TOKEN")
            or os.environ.get("CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN")
            or ""
        ).strip()
    if kind == "render_video":
        return (
            os.environ.get("PHASE6_RENDER_AUTH_TOKEN")
            or os.environ.get("CLYPT_PHASE24_PHASE6_RENDER_TOKEN")
            or ""
        ).strip()
    raise ValueError(f"unsupported media auth kind: {kind}")


def _require_auth_header(kind: str, authorization: str | None) -> None:
    expected_token = _expected_token(kind)
    if not expected_token:
        raise fastapi.HTTPException(
            status_code=500,
            detail=f"{kind} auth token is not configured",
        )
    if authorization != f"Bearer {expected_token}":
        raise fastapi.HTTPException(status_code=401, detail="unauthorized")


@lru_cache(maxsize=1)
def _build_storage_client() -> GCSStorageClient:
    settings = StorageSettings(gcs_bucket=os.environ["GCS_BUCKET"])

    credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON", "").strip()
    if not credentials_json:
        return GCSStorageClient(settings=settings)

    from google.auth import load_credentials_from_dict
    from google.cloud import storage

    info = json.loads(credentials_json)
    credentials, project_id = load_credentials_from_dict(
        info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    storage_client = storage.Client(
        project=project_id or os.environ.get("GOOGLE_CLOUD_PROJECT"),
        credentials=credentials,
    )
    return GCSStorageClient(settings=settings, storage_client=storage_client)


def _submit_media_job(kind: str, payload: dict[str, Any], result_path: str) -> fastapi.Response:
    call = media_gpu_job.spawn(kind, payload)
    return fastapi.responses.JSONResponse(
        status_code=202,
        content={
            "call_id": call.object_id,
            "status": "submitted",
            "result_path": f"{result_path}/{call.object_id}",
        },
    )


def _poll_media_job(call_id: str, *, result_name: str) -> dict[str, Any] | fastapi.Response:
    function_call = modal.FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=0)
    except TimeoutError:
        return fastapi.responses.JSONResponse(
            status_code=202,
            content={"call_id": call_id, "status": "pending"},
        )
    except (modal.exception.OutputExpiredError, modal.exception.NotFoundError):
        raise fastapi.HTTPException(status_code=404, detail="unknown or expired call_id")

    if not isinstance(result, dict):
        raise fastapi.HTTPException(
            status_code=500,
            detail=f"{result_name} returned non-object result: {type(result).__name__}",
        )
    response = dict(result)
    response.setdefault("call_id", call_id)
    response.setdefault("status", "succeeded")
    return response


@web_app.on_event("startup")
def _startup_checks() -> None:
    _require_ffmpeg()


@web_app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@web_app.post("/tasks/node-media-prep")
def node_media_prep_route(
    payload: dict,
    authorization: str | None = fastapi.Header(default=None),
):
    _require_auth_header("node_media_prep", authorization)
    request = NodeMediaPrepRequest.from_payload(
        {**payload, "submitted_at_ms": time.time() * 1000.0}
    )
    return _submit_media_job(
        "node_media_prep",
        request.to_payload(),
        "/tasks/node-media-prep/result",
    )


@web_app.get("/tasks/node-media-prep/result/{call_id}")
def node_media_prep_result_route(
    call_id: str,
    authorization: str | None = fastapi.Header(default=None),
):
    _require_auth_header("node_media_prep", authorization)
    return _poll_media_job(call_id, result_name="node_media_prep")


@web_app.post("/tasks/render-video")
def render_video_route(
    payload: dict,
    authorization: str | None = fastapi.Header(default=None),
):
    _require_auth_header("render_video", authorization)
    request = Phase6RenderRequest.from_payload(payload)
    return _submit_media_job(
        "render_video",
        request.to_payload(),
        "/tasks/render-video/result",
    )


@web_app.get("/tasks/render-video/result/{call_id}")
def render_video_result_route(
    call_id: str,
    authorization: str | None = fastapi.Header(default=None),
):
    _require_auth_header("render_video", authorization)
    return _poll_media_job(call_id, result_name="render_video")


@app.function(
    image=image,
    gpu="L40S",
    min_containers=1,
    max_containers=1,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("clypt-media-l40s")],
)
def media_gpu_job(kind: str, payload: dict) -> dict:
    scratch_root = Path(tempfile.mkdtemp(prefix=f"clypt-modal-{kind}-"))
    storage_client = _build_storage_client()
    if kind == "node_media_prep":
        _require_node_media_runtime()
        return run_node_media_prep(
            request=NodeMediaPrepRequest.from_payload(payload),
            storage_client=storage_client,
            scratch_root=scratch_root,
        )
    if kind == "render_video":
        _require_render_runtime()
        return run_phase6_render(
            request=Phase6RenderRequest.from_payload(payload),
            storage_client=storage_client,
            scratch_root=scratch_root,
        )
    raise ValueError(f"unsupported media worker kind: {kind}")


@app.function(
    image=image,
    min_containers=1,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("clypt-media-l40s")],
)
@modal.asgi_app()
def media_worker():
    return web_app
