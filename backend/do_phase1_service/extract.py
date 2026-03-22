from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import fcntl

from backend.do_phase1_service.models import PersistedPhase1Manifest
from backend.do_phase1_service.storage import GCSStorage, StorageBackend, persist_phase1_outputs
from backend.modal_worker import ClyptWorker, LRASD_MODEL_PATH, LRASD_REPO_ROOT
from backend.pipeline.phase_1_modal_pipeline import (
    download_media,
    enrich_visual_ledger_for_downstream,
    validate_phase_handoff,
)

DEFAULT_STATE_ROOT = Path(os.getenv("DO_PHASE1_STATE_ROOT", "/var/lib/clypt/do_phase1_service"))
DEFAULT_HOST_LOCK_PATH = Path(os.getenv("DO_PHASE1_HOST_LOCK_PATH", str(DEFAULT_STATE_ROOT / "extract.lock")))

REQUIRED_RUNTIME_MODULES = (
    "cv2",
    "numpy",
    "torch",
    "decord",
    "ultralytics",
    "omegaconf",
    "insightface",
    "nemo.collections.asr",
)


class LocalExtractionRuntimeError(RuntimeError):
    pass


def run_extraction_job(
    *,
    source_url: str,
    job_id: str,
    output_dir: str | Path,
    storage: StorageBackend | None = None,
    host_lock_path: str | Path | None = None,
) -> PersistedPhase1Manifest:
    storage = storage or GCSStorage()
    output_dir = Path(output_dir)
    job_output_dir = output_dir / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)

    with host_extraction_lock(host_lock_path or DEFAULT_HOST_LOCK_PATH):
        video_path, audio_path = download_media(source_url)
        modal_result = execute_local_extraction(
            video_path=video_path,
            audio_path=audio_path,
            youtube_url=source_url,
        )
        if modal_result.get("status") != "success":
            raise RuntimeError(modal_result.get("message", "phase 1 extraction failed"))

        phase_1_visual = modal_result.get("phase_1_visual") or modal_result.get("phase_1a_visual")
        phase_1_audio = modal_result.get("phase_1_audio") or modal_result.get("phase_1a_audio")
        if phase_1_visual is None or phase_1_audio is None:
            raise RuntimeError("modal worker did not return phase_1_visual and phase_1_audio")

        canonical_video_uri = storage.upload_file(video_path, f"phase_1/jobs/{job_id}/source_video.mp4")
        phase_1_visual = enrich_visual_ledger_for_downstream(phase_1_visual, phase_1_audio, video_path)
        phase_1_visual["video_gcs_uri"] = canonical_video_uri
        phase_1_audio["video_gcs_uri"] = canonical_video_uri
        validate_phase_handoff(phase_1_visual, phase_1_audio)

        manifest = persist_phase1_outputs(
            storage=storage,
            output_dir=job_output_dir,
            job_id=job_id,
            source_url=source_url,
            canonical_video_uri=canonical_video_uri,
            phase_1_audio=phase_1_audio,
            phase_1_visual=phase_1_visual,
        )

    return manifest


def execute_local_extraction(*, video_path: str, audio_path: str, youtube_url: str) -> dict:
    return get_local_extractor().extract(
        video_path=video_path,
        audio_path=audio_path,
        youtube_url=youtube_url,
    )


class LocalModalPhase1Extractor:
    """Run the existing Phase 1 extraction stack inside the DO worker process."""

    def __init__(self):
        ensure_local_runtime_prereqs()
        user_cls = ClyptWorker._get_user_cls()
        self._worker = user_cls()
        self._load_model = user_cls.load_model._get_raw_f()
        self._extract = user_cls.extract._get_raw_f()
        self._loaded = False

    def extract(self, *, video_path: str, audio_path: str, youtube_url: str) -> dict:
        self._ensure_loaded()
        return self._extract(
            self._worker,
            video_bytes=Path(video_path).read_bytes(),
            audio_wav_bytes=Path(audio_path).read_bytes(),
            youtube_url=youtube_url,
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            self._load_model(self._worker)
        except Exception as exc:
            raise LocalExtractionRuntimeError(
                "DigitalOcean Phase 1 local extraction runtime is not fully provisioned. "
                "The droplet must provide the Phase 1 ML dependency bundle used by "
                "backend/modal_worker.py, including GPU-enabled torch, NeMo ASR, "
                "Ultralytics, InsightFace/ONNX Runtime, Decord, OpenCV, OmegaConf, "
                f"and LR-ASD assets under {LRASD_REPO_ROOT} / {LRASD_MODEL_PATH}. "
                f"Underlying load error: {type(exc).__name__}: {exc}"
            ) from exc
        self._loaded = True


@lru_cache(maxsize=1)
def get_local_extractor() -> LocalModalPhase1Extractor:
    return LocalModalPhase1Extractor()


def ensure_local_runtime_prereqs() -> None:
    missing_modules = []
    for module_name in REQUIRED_RUNTIME_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing_modules.append(module_name)

    missing_paths = []
    for required_path in (Path(LRASD_REPO_ROOT), Path(LRASD_MODEL_PATH)):
        if not required_path.exists():
            missing_paths.append(str(required_path))

    if missing_modules or missing_paths:
        pieces = []
        if missing_modules:
            pieces.append(f"missing python modules: {', '.join(missing_modules)}")
        if missing_paths:
            pieces.append(f"missing runtime assets: {', '.join(missing_paths)}")
        raise LocalExtractionRuntimeError(
            "DigitalOcean Phase 1 local extraction runtime prerequisites are missing; "
            + "; ".join(pieces)
            + ". The droplet must provide the Phase 1 ML bundle expected by backend/modal_worker.py."
        )


@contextmanager
def host_extraction_lock(lock_path: str | Path) -> Iterator[Path]:
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield lock_path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
