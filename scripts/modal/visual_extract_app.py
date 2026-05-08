"""Modal app for dedicated RF-DETR-Seg visual extraction on L40S."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import fastapi
import modal

from backend.phase1_runtime.payloads import VisualPayload
from backend.phase1_runtime.visual import V31VisualExtractor
from backend.phase1_runtime.visual_config import VisualPipelineConfig
from backend.phase1_runtime.visual_warmup import (
    VisualWarmupSpec,
    load_visual_warmup_spec_from_env,
)
from backend.providers.config import StorageSettings
from backend.providers.storage import GCSStorageClient, parse_gcs_uri

app = modal.App("clypt-visual-l40s")
web_app = fastapi.FastAPI()
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "ca-certificates", "gnupg", "wget")
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb"
        " && dpkg -i /tmp/cuda-keyring.deb"
        " && rm /tmp/cuda-keyring.deb",
        "apt-get update"
        " && apt-get install -y --no-install-recommends libnvinfer-bin=10.16.1.11-1+cuda13.2"
        " && rm -rf /var/lib/apt/lists/*",
        "if ! command -v trtexec >/dev/null 2>&1; then"
        " found=$(find /usr -type f -name trtexec -print -quit);"
        ' test -n "$found";'
        ' ln -sf "$found" /usr/local/bin/trtexec;'
        " fi",
    )
    .pip_install("google-cloud-storage>=2.19.0")
    .pip_install("google-auth>=2.38.0")
    .pip_install_from_requirements("requirements-modal-visual-l40s.txt")
    .add_local_python_source("backend")
)

_VISUAL_READY_OBJECT_NAME = "service-state/modal-visual-l40s/visual_ready_state_v1.json"
_RUNTIME_FINGERPRINT_PATHS = (
    "scripts/modal/visual_extract_app.py",
    "backend/phase1_runtime/visual.py",
    "backend/phase1_runtime/visual_config.py",
    "backend/phase1_runtime/tensorrt_detector.py",
    "backend/phase1_runtime/pose_subject_validator.py",
    "backend/phase1_runtime/visual_warmup.py",
)
_GPU_CONTAINER_BOOT_ID = uuid.uuid4().hex
_GPU_READY_STATE: dict[str, Any] | None = None


def _require_codec(codec_cmd: list[str], expected: str) -> None:
    output = subprocess.check_output(codec_cmd, text=True)
    if expected not in output:
        raise RuntimeError(f"missing required ffmpeg codec support: {expected}")


def _require_ffmpeg() -> None:
    subprocess.check_output(["ffmpeg", "-version"], text=True)


def _require_visual_runtime() -> None:
    _require_codec(["ffmpeg", "-hwaccels"], "cuda")
    _require_codec(["ffmpeg", "-filters"], "scale_cuda")
    trtexec_check = subprocess.run(
        ["trtexec", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    if trtexec_check.returncode != 0:
        raise RuntimeError(
            "trtexec is required for Modal L40S TensorRT visual extraction, "
            f"but its health check failed: {trtexec_check.stderr[-1000:]}"
        )
    import tensorrt  # noqa: F401
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Modal L40S RF-DETR-Seg visual extraction.")


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
    os.environ.setdefault(
        "CLYPT_PHASE1_VISUAL_MODEL",
        os.environ.get("CLYPT_MODAL_VISUAL_MODEL", "seg_nano"),
    )
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_BACKEND", "tensorrt_fp16")
    os.environ.setdefault(
        "CLYPT_PHASE1_VISUAL_BATCH_SIZE",
        os.environ.get("CLYPT_MODAL_VISUAL_BATCH_SIZE", "16"),
    )
    os.environ.setdefault(
        "CLYPT_PHASE1_VISUAL_THRESHOLD",
        os.environ.get("CLYPT_MODAL_VISUAL_THRESHOLD", "0.85"),
    )
    os.environ.setdefault(
        "CLYPT_PHASE1_VISUAL_SHAPE",
        os.environ.get("CLYPT_MODAL_VISUAL_SHAPE", "648"),
    )
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_TRACKER", "bytetrack")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_TRACKER_BUFFER", "30")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_TRACKER_MATCH_THRESH", "0.7")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_DECODE", "gpu")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_GPU_DECODE_BACKEND", "nvdec")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_POSE_VALIDATION", "1")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_POSE_MODEL_PATH", "yolo11s-pose.pt")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_POSE_IMGSZ", "640")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_POSE_BATCH_SIZE", "16")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_POSE_MAX_SAMPLES_PER_TRACKLET", "24")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_POSE_MIN_RFDETR_CONFIDENCE", "0.85")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_POSE_MIN_HEAD_EVIDENCE_RATIO", "0.40")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_POSE_MIN_UPPER_BODY_ANCHOR_RATIO", "0.25")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_POSE_KEYPOINT_CONFIDENCE", "0.40")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_POSE_CONFIDENCE", "0.30")
    os.environ.setdefault("CLYPT_PHASE1_VISUAL_POSE_CROP_PADDING_RATIO", "0.08")
    os.environ.setdefault(
        "CLYPT_PHASE1_VISUAL_ARTIFACT_DIR",
        "/tmp/clypt-visual-artifacts",
    )


def _visual_ready_gcs_uri() -> str:
    object_name = os.environ.get(
        "CLYPT_PHASE1_VISUAL_READY_OBJECT_NAME",
        _VISUAL_READY_OBJECT_NAME,
    ).strip().strip("/")
    if not object_name:
        raise RuntimeError("CLYPT_PHASE1_VISUAL_READY_OBJECT_NAME resolved to an empty value")
    bucket = os.environ["GCS_BUCKET"].strip()
    return f"gs://{bucket}/{object_name}"


def _compute_visual_runtime_fingerprint() -> str:
    _set_visual_defaults()
    config = VisualPipelineConfig.from_env()
    repo_root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "detector_model": config.detector_model,
                "detector_backend": config.detector_backend,
                "detector_batch_size": config.detector_batch_size,
                "detector_resolution": config.detector_resolution,
                "detector_threshold": config.detector_threshold,
                "pose_validation_enabled": config.pose_validation_enabled,
                "pose_model_path": config.pose_model_path,
                "pose_imgsz": config.pose_imgsz,
                "pose_batch_size": config.pose_batch_size,
                "pose_min_head_evidence_ratio": config.pose_min_head_evidence_ratio,
                "pose_min_upper_body_anchor_ratio": config.pose_min_upper_body_anchor_ratio,
                "artifact_dir": config.detector_artifact_dir,
            },
            ensure_ascii=True,
            sort_keys=True,
        ).encode("utf-8")
    )
    for rel_path in _RUNTIME_FINGERPRINT_PATHS:
        file_path = repo_root / rel_path
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _parse_payload(payload: dict[str, Any]) -> dict[str, str]:
    run_id = str(payload.get("run_id") or "").strip()
    video_gcs_uri = str(
        payload.get("video_gcs_uri") or payload.get("source_video_gcs_uri") or ""
    ).strip()
    if not run_id:
        raise ValueError("run_id is required")
    parse_gcs_uri(video_gcs_uri)
    return {"run_id": run_id, "video_gcs_uri": video_gcs_uri}


def _read_json_artifact(
    *,
    storage_client: GCSStorageClient,
    gcs_uri: str,
) -> dict[str, Any] | None:
    if not storage_client.exists(gcs_uri):
        return None
    with tempfile.NamedTemporaryFile(prefix="clypt-visual-state-", suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        storage_client.download_file(gcs_uri=gcs_uri, local_path=tmp_path)
        return json.loads(tmp_path.read_text(encoding="utf-8"))
    finally:
        tmp_path.unlink(missing_ok=True)


def _write_json_artifact(
    *,
    storage_client: GCSStorageClient,
    gcs_uri: str,
    payload: dict[str, Any],
) -> str:
    _, object_name = parse_gcs_uri(gcs_uri)
    with tempfile.NamedTemporaryFile(prefix="clypt-visual-state-", suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return storage_client.upload_file(local_path=tmp_path, object_name=object_name)
    finally:
        tmp_path.unlink(missing_ok=True)


def _read_visual_readiness_state(storage_client: GCSStorageClient) -> dict[str, Any] | None:
    return _read_json_artifact(storage_client=storage_client, gcs_uri=_visual_ready_gcs_uri())


def _write_visual_readiness_state(
    *,
    storage_client: GCSStorageClient,
    payload: dict[str, Any],
) -> str:
    return _write_json_artifact(
        storage_client=storage_client,
        gcs_uri=_visual_ready_gcs_uri(),
        payload=payload,
    )


def _warmup_summary_from_phase1_visual(
    *,
    phase1_visual: dict[str, Any],
    warmup_spec: VisualWarmupSpec,
) -> dict[str, Any]:
    metrics = dict(phase1_visual.get("tracking_metrics") or {})
    emitted_track_rows = int(metrics.get("emitted_track_rows") or 0)
    pose_validated_tracklets = int(metrics.get("pose_validated_tracklets") or 0)
    pose_auto_follow_eligible_tracklets = int(
        metrics.get("pose_auto_follow_eligible_tracklets") or 0
    )
    return {
        "asset_id": warmup_spec.asset_id,
        "warmup_video_gcs_uri": warmup_spec.warmup_video_gcs_uri,
        "source_video_gcs_uri": warmup_spec.source_video_gcs_uri,
        "clip_start_ms": warmup_spec.clip_start_ms,
        "clip_end_ms": warmup_spec.clip_end_ms,
        "emitted_track_rows": emitted_track_rows,
        "pose_validated_tracklets": pose_validated_tracklets,
        "pose_auto_follow_eligible_tracklets": pose_auto_follow_eligible_tracklets,
        "tracking_metrics": metrics,
    }


def _validate_visual_warmup_result(
    *,
    phase1_visual: dict[str, Any],
    warmup_spec: VisualWarmupSpec,
) -> dict[str, Any]:
    summary = _warmup_summary_from_phase1_visual(
        phase1_visual=phase1_visual,
        warmup_spec=warmup_spec,
    )
    if summary["emitted_track_rows"] < int(warmup_spec.min_emitted_track_rows):
        raise RuntimeError(
            "visual warmup did not emit enough track rows: "
            f"{summary['emitted_track_rows']} < {warmup_spec.min_emitted_track_rows}"
        )
    if summary["pose_validated_tracklets"] < int(warmup_spec.min_pose_validated_tracklets):
        raise RuntimeError(
            "visual warmup did not exercise pose validation strongly enough: "
            f"{summary['pose_validated_tracklets']} < {warmup_spec.min_pose_validated_tracklets}"
        )
    return summary


def _build_visual_ready_state(
    *,
    source: str,
    run_id: str,
    video_gcs_uri: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    now_ms = time.time() * 1000.0
    return {
        "status": "ready",
        "source": source,
        "run_id": run_id,
        "video_gcs_uri": video_gcs_uri,
        "updated_at_ms": now_ms,
        "updated_at_unix_s": now_ms / 1000.0,
        "runtime_fingerprint": _compute_visual_runtime_fingerprint(),
        "gpu_container_boot_id": _GPU_CONTAINER_BOOT_ID,
        "warmup": summary,
    }


def _set_gpu_ready_state(
    *,
    source: str,
    run_id: str,
    video_gcs_uri: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    global _GPU_READY_STATE
    state = _build_visual_ready_state(
        source=source,
        run_id=run_id,
        video_gcs_uri=video_gcs_uri,
        summary=summary,
    )
    _GPU_READY_STATE = dict(state)
    return state


def _probe_gpu_ready_state() -> dict[str, Any]:
    expected_fingerprint = _compute_visual_runtime_fingerprint()
    state = dict(_GPU_READY_STATE or {})
    if not state:
        return {
            "status": "cold",
            "reason": "gpu_worker_not_warmed",
            "runtime_fingerprint": expected_fingerprint,
            "gpu_container_boot_id": _GPU_CONTAINER_BOOT_ID,
        }
    state_fingerprint = str(state.get("runtime_fingerprint") or "").strip()
    if state_fingerprint != expected_fingerprint:
        return {
            "status": "stale",
            "reason": "gpu_worker_fingerprint_mismatch",
            "runtime_fingerprint": expected_fingerprint,
            "gpu_container_boot_id": _GPU_CONTAINER_BOOT_ID,
            "state": state,
        }
    state["status"] = "ready"
    state["reason"] = "gpu_worker_warm"
    return state


def _mark_visual_ready_if_qualified(
    *,
    phase1_visual: dict[str, Any],
    storage_client: GCSStorageClient,
    source: str,
    run_id: str,
    video_gcs_uri: str,
) -> dict[str, Any] | None:
    warmup_spec = load_visual_warmup_spec_from_env()
    summary = _warmup_summary_from_phase1_visual(
        phase1_visual=phase1_visual,
        warmup_spec=warmup_spec,
    )
    if summary["emitted_track_rows"] <= 0 or summary["pose_validated_tracklets"] <= 0:
        return None
    state = _set_gpu_ready_state(
        source=source,
        run_id=run_id,
        video_gcs_uri=video_gcs_uri,
        summary=summary,
    )
    _write_visual_readiness_state(storage_client=storage_client, payload=state)
    return state


def _extract_phase1_visual(
    *,
    run_id: str,
    video_gcs_uri: str,
    storage_client: GCSStorageClient,
) -> dict[str, Any]:
    _set_visual_defaults()
    with tempfile.TemporaryDirectory(prefix=f"clypt-visual-{run_id}-") as tmp:
        video_path = Path(tmp) / "source_video.mp4"
        storage_client.download_file(
            gcs_uri=video_gcs_uri,
            local_path=video_path,
        )
        extractor = V31VisualExtractor(visual_config=VisualPipelineConfig.from_env())
        return extractor.extract(video_path=video_path, workspace=None)


def _poll_call_result(call_id: str, *, result_name: str) -> dict[str, Any] | fastapi.Response:
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


def _upload_mask_artifacts(
    *,
    phase1_visual: dict[str, Any],
    run_id: str,
    storage_client: GCSStorageClient,
) -> None:
    artifacts = list(phase1_visual.get("mask_artifacts") or [])
    uploaded: list[dict[str, Any]] = []
    for artifact in artifacts:
        local_path_raw = str(artifact.get("local_path") or "").strip()
        if not local_path_raw:
            raise RuntimeError("mask artifact is missing local_path before Modal upload")
        local_path = Path(local_path_raw)
        if not local_path.exists():
            raise RuntimeError(f"mask artifact local_path does not exist: {local_path}")
        object_name = f"phase14/{run_id}/visual/{local_path.name}"
        gcs_uri = storage_client.upload_file(local_path=local_path, object_name=object_name)
        clean = dict(artifact)
        clean.pop("local_path", None)
        clean["gcs_uri"] = gcs_uri
        uploaded.append(clean)
    phase1_visual["mask_artifacts"] = uploaded
    metrics = dict(phase1_visual.get("tracking_metrics") or {})
    metrics["mask_artifacts"] = uploaded
    phase1_visual["tracking_metrics"] = metrics


def _upload_phase1_visual_artifact(
    *,
    phase1_visual_payload: dict[str, Any],
    run_id: str,
    storage_client: GCSStorageClient,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(
        prefix=f"{run_id}-phase1-visual-",
        suffix=".json.gz",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with gzip.open(tmp_path, "wt", encoding="utf-8") as fh:
            json.dump(phase1_visual_payload, fh, ensure_ascii=True, separators=(",", ":"))
        object_name = f"phase14/{run_id}/visual/phase1_visual.json.gz"
        gcs_uri = storage_client.upload_file(local_path=tmp_path, object_name=object_name)
        return {
            "artifact_id": "phase1_visual_v1",
            "encoding": "json_gzip_v1",
            "bytes": tmp_path.stat().st_size,
            "gcs_uri": gcs_uri,
        }
    finally:
        tmp_path.unlink(missing_ok=True)


@web_app.on_event("startup")
def _startup_checks() -> None:
    _require_ffmpeg()


@web_app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "readiness_path": "/ready"}


@web_app.get("/ready")
def ready() -> fastapi.Response:
    _set_visual_defaults()
    storage_client = _build_storage_client()
    expected_fingerprint = _compute_visual_runtime_fingerprint()
    persisted_state = _read_visual_readiness_state(storage_client)
    gpu_state = visual_extract_job.remote({"job_kind": "ready_probe"})
    if not isinstance(gpu_state, dict):
        return fastapi.responses.JSONResponse(
            status_code=503,
            content={
                "status": "cold",
                "reason": "gpu_ready_probe_returned_non_object",
                "runtime_fingerprint": expected_fingerprint,
            },
        )
    gpu_status = str(gpu_state.get("status") or "").strip().lower()
    if gpu_status != "ready":
        return fastapi.responses.JSONResponse(status_code=503, content=gpu_state)
    if persisted_state is None:
        payload = dict(gpu_state)
        payload["persisted_state_status"] = "missing"
        return fastapi.responses.JSONResponse(status_code=200, content=payload)
    state_fingerprint = str(persisted_state.get("runtime_fingerprint") or "").strip()
    if state_fingerprint != expected_fingerprint:
        payload = dict(gpu_state)
        payload["persisted_state_status"] = "stale"
        payload["persisted_state"] = persisted_state
        return fastapi.responses.JSONResponse(status_code=200, content=payload)
    persisted_status = str(persisted_state.get("status") or "").strip().lower()
    if persisted_status != "ready":
        payload = dict(gpu_state)
        payload["persisted_state_status"] = "non_ready"
        payload["persisted_state"] = persisted_state
        return fastapi.responses.JSONResponse(status_code=200, content=payload)
    payload = dict(gpu_state)
    payload["persisted_state_status"] = "ready"
    payload["persisted_state"] = persisted_state
    return fastapi.responses.JSONResponse(status_code=200, content=payload)


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


@web_app.post("/tasks/visual-warmup")
def visual_warmup_route(
    payload: dict | None = None,
    authorization: str | None = fastapi.Header(default=None),
):
    _require_auth_header(authorization)
    _set_visual_defaults()
    warmup_spec = load_visual_warmup_spec_from_env()
    submitted_at_ms = time.time() * 1000.0
    storage_client = _build_storage_client()
    _write_visual_readiness_state(
        storage_client=storage_client,
        payload={
            "status": "warming",
            "source": "visual_warmup_route",
            "submitted_at_ms": submitted_at_ms,
            "runtime_fingerprint": _compute_visual_runtime_fingerprint(),
            "warmup": warmup_spec.to_payload(),
        },
    )
    call = visual_extract_job.spawn(
        {
            "job_kind": "warmup",
            "submitted_at_ms": submitted_at_ms,
            **dict(payload or {}),
        }
    )
    return fastapi.responses.JSONResponse(
        status_code=202,
        content={
            "call_id": call.object_id,
            "status": "submitted",
            "result_path": f"/tasks/visual-warmup/result/{call.object_id}",
        },
    )


@web_app.get("/tasks/visual-extract/result/{call_id}")
def visual_extract_result_route(
    call_id: str,
    authorization: str | None = fastapi.Header(default=None),
):
    _require_auth_header(authorization)
    return _poll_call_result(call_id, result_name="visual_extract_job")


@web_app.get("/tasks/visual-warmup/result/{call_id}")
def visual_warmup_result_route(
    call_id: str,
    authorization: str | None = fastapi.Header(default=None),
):
    _require_auth_header(authorization)
    return _poll_call_result(call_id, result_name="visual_warmup_job")


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
    storage_client = _build_storage_client()
    job_kind = str(payload.get("job_kind") or "extract").strip().lower()
    if job_kind == "ready_probe":
        return _probe_gpu_ready_state()
    if job_kind == "warmup":
        warmup_spec = load_visual_warmup_spec_from_env()
        run_id = f"visual_warmup_{warmup_spec.asset_id}"
        video_gcs_uri = str(
            payload.get("video_gcs_uri") or warmup_spec.warmup_video_gcs_uri
        ).strip()
        parse_gcs_uri(video_gcs_uri)
        phase1_visual = _extract_phase1_visual(
            run_id=run_id,
            video_gcs_uri=video_gcs_uri,
            storage_client=storage_client,
        )
        VisualPayload.model_validate(phase1_visual)
        summary = _validate_visual_warmup_result(
            phase1_visual=phase1_visual,
            warmup_spec=warmup_spec,
        )
        state = _set_gpu_ready_state(
            source="visual_warmup_job",
            run_id=run_id,
            video_gcs_uri=video_gcs_uri,
            summary=summary,
        )
        _write_visual_readiness_state(storage_client=storage_client, payload=state)
        return {
            "run_id": run_id,
            "job_kind": "warmup",
            "status": "succeeded",
            "visual_backend": "modal_l40s_tensorrt_nvdec",
            "queue_wait_ms": (
                max(0.0, time.time() * 1000.0 - float(payload["submitted_at_ms"]))
                if payload.get("submitted_at_ms") is not None
                else None
            ),
            "warmup": summary,
            "readiness": state,
        }

    parsed = _parse_payload(payload)
    phase1_visual = _extract_phase1_visual(
        run_id=parsed["run_id"],
        video_gcs_uri=parsed["video_gcs_uri"],
        storage_client=storage_client,
    )
    _upload_mask_artifacts(
        phase1_visual=phase1_visual,
        run_id=parsed["run_id"],
        storage_client=storage_client,
    )
    visual_payload = VisualPayload.model_validate(phase1_visual)
    visual_payload_json = visual_payload.model_dump(mode="json")
    visual_payload_artifact = _upload_phase1_visual_artifact(
        phase1_visual_payload=visual_payload_json,
        run_id=parsed["run_id"],
        storage_client=storage_client,
    )
    readiness_state = _mark_visual_ready_if_qualified(
        phase1_visual=phase1_visual,
        storage_client=storage_client,
        source="visual_extract_job",
        run_id=parsed["run_id"],
        video_gcs_uri=parsed["video_gcs_uri"],
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
        "phase1_visual_gcs_uri": visual_payload_artifact["gcs_uri"],
        "phase1_visual_encoding": visual_payload_artifact["encoding"],
        "phase1_visual_artifact": visual_payload_artifact,
        "readiness_updated": readiness_state is not None,
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
