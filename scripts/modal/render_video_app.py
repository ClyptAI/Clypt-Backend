"""Modal app for Phase 6 render/export."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import fastapi
import modal

from backend.providers.config import StorageSettings
from backend.providers.storage import GCSStorageClient
from backend.runtime.phase6_render import Phase6RenderRequest, run_phase6_render

app = modal.App("clypt-render-video")
web_app = fastapi.FastAPI()
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install("google-cloud-storage>=2.19.0")
    .pip_install("google-auth>=2.38.0")
    .pip_install_from_requirements("requirements-do-phase26-h200.txt")
    .add_local_python_source("backend")
)


def _require_ffmpeg() -> None:
    subprocess.check_output(["ffmpeg", "-version"], text=True)


def _require_codec(codec_cmd: list[str], expected: str) -> None:
    output = subprocess.check_output(codec_cmd, text=True)
    if expected not in output:
        raise RuntimeError(f"missing required ffmpeg codec support: {expected}")


def _require_worker_runtime() -> None:
    _require_codec(["ffmpeg", "-encoders"], "h264_nvenc")


def _require_auth_header(authorization: str | None) -> None:
    expected_token = (
        os.environ.get("PHASE6_RENDER_AUTH_TOKEN")
        or os.environ.get("CLYPT_PHASE24_PHASE6_RENDER_TOKEN")
        or ""
    ).strip()
    if not expected_token:
        raise fastapi.HTTPException(
            status_code=500,
            detail="PHASE6_RENDER_AUTH_TOKEN is not configured",
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


@web_app.on_event("startup")
def _startup_checks() -> None:
    _require_ffmpeg()


@web_app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@web_app.post("/tasks/render-video")
def render_video_route(
    payload: dict,
    authorization: str | None = fastapi.Header(default=None),
):
    _require_auth_header(authorization)
    request = Phase6RenderRequest.from_payload(payload)
    call = render_video_job.spawn(request.to_payload())
    return fastapi.responses.JSONResponse(
        status_code=202,
        content={
            "call_id": call.object_id,
            "status": "submitted",
            "result_path": f"/tasks/render-video/result/{call.object_id}",
        },
    )


@web_app.get("/tasks/render-video/result/{call_id}")
def render_video_result_route(
    call_id: str,
    authorization: str | None = fastapi.Header(default=None),
):
    _require_auth_header(authorization)
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
            detail=f"render_video_job returned non-object result: {type(result).__name__}",
        )
    response = dict(result)
    response.setdefault("call_id", call_id)
    response.setdefault("status", "succeeded")
    return response


@app.function(
    image=image,
    gpu="L40S",
    min_containers=1,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("clypt-node-media-prep")],
)
def render_video_job(payload: dict) -> dict:
    _require_worker_runtime()
    request = Phase6RenderRequest.from_payload(payload)
    scratch_root = Path(tempfile.mkdtemp(prefix="clypt-modal-render-video-"))
    return run_phase6_render(
        request=request,
        storage_client=_build_storage_client(),
        scratch_root=scratch_root,
    )


@app.function(
    image=image,
    min_containers=1,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("clypt-node-media-prep")],
)
@modal.asgi_app()
def render_video():
    return web_app
