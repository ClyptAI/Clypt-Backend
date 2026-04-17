"""Modal L4 app for Clypt node-media-prep.

Uses min_containers=1 to keep one warm GPU worker, preserves the existing
RemoteNodeMediaPrepClient JSON contract, and validates ffmpeg NVENC/NVDEC
availability before serving work.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import modal

from backend.runtime.node_media_prep import NodeMediaPrepRequest, run_node_media_prep
from backend.providers.storage import GCSStorageClient
from backend.providers.config import StorageSettings

app = modal.App("clypt-node-media-prep")
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install("google-cloud-storage>=2.19.0")
    .pip_install_from_requirements("requirements-do-phase26-h200.txt")
)


def _require_codec(codec_cmd: list[str], expected: str) -> None:
    output = subprocess.check_output(codec_cmd, text=True)
    if expected not in output:
        raise RuntimeError(f"missing required ffmpeg codec support: {expected}")


@app.function(
    image=image,
    gpu="L4",
    min_containers=1,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("clypt-node-media-prep")],
)
@modal.fastapi_endpoint(method="POST")
def node_media_prep(payload: dict):
    _require_codec(["ffmpeg", "-encoders"], "h264_nvenc")
    _require_codec(["ffmpeg", "-decoders"], "h264_cuvid")
    request = NodeMediaPrepRequest.from_payload(payload)
    scratch_root = Path(tempfile.mkdtemp(prefix="clypt-modal-node-media-"))
    storage_client = GCSStorageClient(
        settings=StorageSettings(gcs_bucket=os.environ["GCS_BUCKET"])
    )
    return run_node_media_prep(
        request=request,
        storage_client=storage_client,
        scratch_root=scratch_root,
    )
