"""Modal L4 app for Clypt node-media-prep.

Uses min_containers=1 to keep one warm GPU worker, preserves the existing
RemoteNodeMediaPrepClient JSON contract, and validates ffmpeg NVENC/NVDEC
availability before serving work.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import fastapi
import modal

from backend.runtime.node_media_prep import NodeMediaPrepRequest, run_node_media_prep
from backend.providers.storage import GCSStorageClient
from backend.providers.config import StorageSettings

app = modal.App("clypt-node-media-prep")
web_app = fastapi.FastAPI()
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install("google-cloud-storage>=2.19.0")
    .pip_install("google-auth>=2.38.0")
    .pip_install_from_requirements("requirements-do-phase26-h200.txt")
    .add_local_python_source("backend")
)


def _require_codec(codec_cmd: list[str], expected: str) -> None:
    output = subprocess.check_output(codec_cmd, text=True)
    if expected not in output:
        raise RuntimeError(f"missing required ffmpeg codec support: {expected}")


def _require_auth_header(authorization: str | None) -> None:
    expected_token = (
        os.environ.get("NODE_MEDIA_PREP_AUTH_TOKEN")
        or os.environ.get("CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN")
        or ""
    ).strip()
    if not expected_token:
        raise fastapi.HTTPException(
            status_code=500,
            detail="NODE_MEDIA_PREP_AUTH_TOKEN is not configured",
        )
    expected_header = f"Bearer {expected_token}"
    if authorization != expected_header:
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
    _require_codec(["ffmpeg", "-encoders"], "h264_nvenc")
    _require_codec(["ffmpeg", "-decoders"], "h264_cuvid")


@web_app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@web_app.post("/tasks/node-media-prep")
def node_media_prep_route(
    payload: dict,
    authorization: str | None = fastapi.Header(default=None),
):
    _require_auth_header(authorization)
    request = NodeMediaPrepRequest.from_payload(payload)
    scratch_root = Path(tempfile.mkdtemp(prefix="clypt-modal-node-media-"))
    return run_node_media_prep(
        request=request,
        storage_client=_build_storage_client(),
        scratch_root=scratch_root,
    )


@app.function(
    image=image,
    gpu="L4",
    min_containers=1,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("clypt-node-media-prep")],
)
@modal.asgi_app()
def node_media_prep():
    return web_app
