"""Modal app for dedicated RF-DETR visual extraction on L40S."""

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

from backend.phase1_runtime.payloads import VisualPayload
from backend.phase1_runtime.visual import V31VisualExtractor
from backend.phase1_runtime.visual_config import VisualPipelineConfig
from backend.providers.config import StorageSettings
from backend.providers.storage import GCSStorageClient, parse_gcs_uri

app = modal.App("clypt-visual-l40s")
web_app = fastapi.FastAPI()
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("google-cloud-storage>=2.19.0")
    .pip_install("google-auth>=2.38.0")
    .pip_install_from_requirements("requirements-modal-visual-l40s.txt")
    .add_local_python_source("backend")
)


def _require_codec(codec_cmd: list[str], expected: str) -> None:
    output = subprocess.check_output(codec_cmd, text=True)
    if expected not in output:
        raise RuntimeError(f"missing required ffmpeg codec support: {expected}")


def _require_ffmpeg() -> None:
    subprocess.check_output(["ffmpeg", "-version"], text=True)


def _require_visual_runtime() -> None:
    _require_codec(["ffmpeg", "-hwaccels"], "cuda")
    _require_codec(["ffmpeg", "-filters"], "scale_cuda")
    subprocess.check_output(["trtexec", "--version"], text=True)
    import tensorrt  # noqa: F401
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Modal L40S RF-DETR visual extraction.")


def _require_auth_header(authorization: str | None) -> None:
    expected_token = (
        os.environ.get("VISUAL_EXTRACT_AUTH_TOKEN")
        or os.environ.get("CLYPT_PHASE1_VISUAL_SERVICE_AUTH_TOKEN")
        or ""
    ).strip()
    if not expected_token:
        raise fastapi.HTTPException(
            status_code=500,
            detail="VISUAL_EXTRACT_AUTH_TOKEN is not configured",
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


def _set_visual_defaults() -> None:
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_MODEL", "nano")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_BACKEND", "tensorrt_fp16")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_BATCH_SIZE", "16")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_THRESHOLD", "0.35")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_SHAPE", "640")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_TRACKER", "bytetrack")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_TRACKER_BUFFER", "30")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_TRACKER_MATCH_THRESH", "0.7")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_DECODE", "gpu")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_GPU_DECODE_BACKEND", "nvdec")
    os.environ.setdefault(
        "CLYPT_PHASE1_VISUAL_ARTIFACT_DIR",
        "/tmp/clypt-visual-artifacts",
    )


def _parse_payload(payload: dict[str, Any]) -> dict[str, str]:
    run_id = str(payload.get("run_id") or "").strip()
    video_gcs_uri = str(
        payload.get("video_gcs_uri") or payload.get("source_video_gcs_uri") or ""
    ).strip()
    if not run_id:
        raise ValueError("run_id is required")
    parse_gcs_uri(video_gcs_uri)
    return {"run_id": run_id, "video_gcs_uri": video_gcs_uri}


@web_app.on_event("startup")
def _startup_checks() -> None:
    _require_ffmpeg()


@web_app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@web_app.post("/tasks/visual-extract")
def visual_extract_route(
    payload: dict,
    authorization: str | None = fastapi.Header(default=None),
):
    _require_auth_header(authorization)
    parsed = _parse_payload(payload)
    call = visual_extract_job.spawn(
        {
            **parsed,
            "source_video_sha256": payload.get("source_video_sha256"),
            "submitted_at_ms": time.time() * 1000.0,
        }
    )
    return fastapi.responses.JSONResponse(
        status_code=202,
        content={
            "call_id": call.object_id,
            "status": "submitted",
            "result_path": f"/tasks/visual-extract/result/{call.object_id}",
        },
    )


@web_app.get("/tasks/visual-extract/result/{call_id}")
def visual_extract_result_route(
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
            detail=f"visual_extract_job returned non-object result: {type(result).__name__}",
        )
    response = dict(result)
    response.setdefault("call_id", call_id)
    response.setdefault("status", "succeeded")
    return response


@app.function(
    image=image,
    gpu="L40S",
    min_containers=1,
    max_containers=1,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("clypt-visual-l40s")],
)
def visual_extract_job(payload: dict) -> dict:
    _require_visual_runtime()
    parsed = _parse_payload(payload)
    storage_client = _build_storage_client()
    _set_visual_defaults()
    with tempfile.TemporaryDirectory(prefix=f"clypt-visual-{parsed['run_id']}-") as tmp:
        video_path = Path(tmp) / "source_video.mp4"
        storage_client.download_file(
            gcs_uri=parsed["video_gcs_uri"],
            local_path=video_path,
        )
        extractor = V31VisualExtractor(visual_config=VisualPipelineConfig.from_env())
        visual_payload = VisualPayload.model_validate(
            extractor.extract(video_path=video_path, workspace=None)
        )
    return {
        "run_id": parsed["run_id"],
        "source_video_gcs_uri": parsed["video_gcs_uri"],
        "source_video_sha256": payload.get("source_video_sha256"),
        "status": "succeeded",
        "visual_backend": "modal_l40s_tensorrt_nvdec",
        "queue_wait_ms": (
            max(0.0, time.time() * 1000.0 - float(payload["submitted_at_ms"]))
            if payload.get("submitted_at_ms") is not None
            else None
        ),
        "phase1_visual": visual_payload.model_dump(mode="json"),
    }


@app.function(
    image=image,
    min_containers=1,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("clypt-visual-l40s")],
)
@modal.asgi_app()
def visual_extract():
    return web_app
