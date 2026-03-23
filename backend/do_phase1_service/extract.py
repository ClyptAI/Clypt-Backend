from __future__ import annotations

import importlib
import os
import re
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterator, TextIO

import fcntl

from backend.do_phase1_service.models import PersistedPhase1Manifest
from backend.do_phase1_service.storage import GCSStorage, StorageBackend, persist_phase1_outputs
from backend.do_phase1_worker import ClyptWorker, LRASD_MODEL_PATH, LRASD_REPO_ROOT
from backend.pipeline import phase_1_do_pipeline as phase1_pipeline
from backend.pipeline.phase_1_do_pipeline import (
    download_media,
    enrich_visual_ledger_for_downstream,
    validate_phase_handoff,
)

DEFAULT_STATE_ROOT = Path(os.getenv("DO_PHASE1_STATE_ROOT", "/var/lib/clypt/do_phase1_service"))
DEFAULT_HOST_LOCK_PATH = Path(os.getenv("DO_PHASE1_HOST_LOCK_PATH", str(DEFAULT_STATE_ROOT / "extract.lock")))
DEFAULT_GPU_SLOT_POLL_INTERVAL_S = float(os.getenv("DO_PHASE1_GPU_SLOT_POLL_INTERVAL_S", "0.1"))

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


ProgressCallback = Callable[[str, str | None, float | None], None]


def run_extraction_job(
    *,
    source_url: str,
    job_id: str,
    runtime_controls: dict[str, object] | None = None,
    output_dir: str | Path,
    storage: StorageBackend | None = None,
    host_lock_path: str | Path | None = None,
    log_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PersistedPhase1Manifest:
    storage = storage or GCSStorage()
    output_dir = Path(output_dir)
    job_output_dir = output_dir / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_path) if log_path else output_dir / "logs" / f"{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ingest_started_at = time.perf_counter()

    progress = progress_callback or (lambda step, message=None, pct=None: None)
    progress("download_media", "Downloading source media", 0.02)

    with _job_pipeline_workspace(job_output_dir):
        with _capture_job_logs(log_path, on_line=lambda line: _forward_progress_line(line, progress)):
            print(f"[DO Phase 1] Starting job {job_id} for {source_url}")
            video_path, audio_path = download_media(source_url)
            processing_started_at = time.perf_counter()
            progress("awaiting_gpu_slot", "Waiting for a GPU extraction slot", 0.08)
            with host_extraction_slot(host_lock_path or DEFAULT_HOST_LOCK_PATH):
                progress("extracting", "Running ASR, tracking, clustering, and speaker binding", 0.1)
                with _runtime_controls_env(runtime_controls):
                    extraction_result = execute_local_extraction(
                        video_path=video_path,
                        audio_path=audio_path,
                        youtube_url=source_url,
                    )
                if extraction_result.get("status") != "success":
                    raise RuntimeError(extraction_result.get("message", "phase 1 extraction failed"))

                phase_1_visual = extraction_result.get("phase_1_visual") or extraction_result.get("phase_1a_visual")
                phase_1_audio = extraction_result.get("phase_1_audio") or extraction_result.get("phase_1a_audio")
                if phase_1_visual is None or phase_1_audio is None:
                    raise RuntimeError("phase 1 extraction did not return phase_1_visual and phase_1_audio")

            progress("uploading_source", "Uploading canonical source video to GCS", 0.92)
            canonical_video_uri = storage.upload_file(video_path, f"phase_1/jobs/{job_id}/source_video.mp4")
            progress("validating_outputs", "Validating Phase 1 handoff artifacts", 0.95)
            phase_1_visual = enrich_visual_ledger_for_downstream(phase_1_visual, phase_1_audio, video_path)
            phase_1_visual["video_gcs_uri"] = canonical_video_uri
            phase_1_audio["video_gcs_uri"] = canonical_video_uri
            validate_phase_handoff(phase_1_visual, phase_1_audio)
            upload_started_at = time.perf_counter()

            progress("persisting_manifest", "Persisting manifest and artifacts", 0.98)
            manifest = persist_phase1_outputs(
                storage=storage,
                output_dir=job_output_dir,
                job_id=job_id,
                source_url=source_url,
                canonical_video_uri=canonical_video_uri,
                phase_1_audio=phase_1_audio,
                phase_1_visual=phase_1_visual,
                timings={
                    "ingest_ms": int(round((processing_started_at - ingest_started_at) * 1000)),
                    "processing_ms": int(round((upload_started_at - processing_started_at) * 1000)),
                    "upload_ms": int(round((time.perf_counter() - upload_started_at) * 1000)),
                },
            )
            print(f"[DO Phase 1] Job {job_id} manifest persisted to {manifest.manifest_uri}")
    progress("complete", "Phase 1 job succeeded", 1.0)

    return manifest


def execute_local_extraction(*, video_path: str, audio_path: str, youtube_url: str) -> dict:
    return get_local_extractor().extract(
        video_path=video_path,
        audio_path=audio_path,
        youtube_url=youtube_url,
    )


class LocalDOPhase1Extractor:
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
                "backend/do_phase1_worker.py, including GPU-enabled torch, NeMo ASR, "
                "Ultralytics, InsightFace/ONNX Runtime, Decord, OpenCV, OmegaConf, "
                f"and LR-ASD assets under {LRASD_REPO_ROOT} / {LRASD_MODEL_PATH}. "
                f"Underlying load error: {type(exc).__name__}: {exc}"
            ) from exc
        self._loaded = True


@lru_cache(maxsize=1)
def get_local_extractor() -> LocalDOPhase1Extractor:
    return LocalDOPhase1Extractor()


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
            + ". The droplet must provide the Phase 1 ML bundle expected by backend/do_phase1_worker.py."
        )


@contextmanager
def host_extraction_lock(lock_path: str | Path) -> Iterator[Path]:
    with host_extraction_slot(lock_path, slot_count=1) as slot:
        yield slot[0]


@contextmanager
def host_extraction_slot(
    lock_path: str | Path,
    *,
    slot_count: int | None = None,
    poll_interval_s: float = DEFAULT_GPU_SLOT_POLL_INTERVAL_S,
) -> Iterator[tuple[Path, int]]:
    lock_path = Path(lock_path)
    slot_count = max(1, int(slot_count or os.getenv("DO_PHASE1_GPU_SLOTS", "1")))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = None
    acquired_path: Path | None = None
    acquired_slot = -1
    while lock_file is None:
        for slot_idx in range(slot_count):
            candidate = lock_path.parent / f"{lock_path.name}.slot{slot_idx}"
            handle = candidate.open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_file = handle
                acquired_path = candidate
                acquired_slot = slot_idx
                break
            except BlockingIOError:
                handle.close()
        if lock_file is None:
            time.sleep(max(0.01, float(poll_interval_s)))
    try:
        assert acquired_path is not None
        yield acquired_path, acquired_slot
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


@contextmanager
def _job_pipeline_workspace(job_output_dir: Path) -> Iterator[tuple[Path, Path]]:
    workspace_root = job_output_dir / "_workspace"
    download_dir = workspace_root / "downloads"
    output_dir = workspace_root / "outputs"
    download_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    previous = {
        "DOWNLOAD_DIR": phase1_pipeline.DOWNLOAD_DIR,
        "OUTPUT_DIR": phase1_pipeline.OUTPUT_DIR,
        "PHASE1_NDJSON_PATH": phase1_pipeline.PHASE1_NDJSON_PATH,
        "PHASE1_RUNTIME_CONTROLS_PATH": phase1_pipeline.PHASE1_RUNTIME_CONTROLS_PATH,
        "DETACHED_STATE_PATH": phase1_pipeline.DETACHED_STATE_PATH,
    }
    phase1_pipeline.DOWNLOAD_DIR = download_dir
    phase1_pipeline.OUTPUT_DIR = output_dir
    phase1_pipeline.PHASE1_NDJSON_PATH = output_dir / "phase_1_visual.ndjson"
    phase1_pipeline.PHASE1_RUNTIME_CONTROLS_PATH = output_dir / "phase_1_runtime_controls.json"
    phase1_pipeline.DETACHED_STATE_PATH = output_dir / "phase_1_detached_state.json"
    try:
        yield download_dir, output_dir
    finally:
        phase1_pipeline.DOWNLOAD_DIR = previous["DOWNLOAD_DIR"]
        phase1_pipeline.OUTPUT_DIR = previous["OUTPUT_DIR"]
        phase1_pipeline.PHASE1_NDJSON_PATH = previous["PHASE1_NDJSON_PATH"]
        phase1_pipeline.PHASE1_RUNTIME_CONTROLS_PATH = previous["PHASE1_RUNTIME_CONTROLS_PATH"]
        phase1_pipeline.DETACHED_STATE_PATH = previous["DETACHED_STATE_PATH"]


@contextmanager
def _capture_job_logs(log_path: Path, *, on_line: Callable[[str], None] | None = None) -> Iterator[Path]:
    with log_path.open("a", encoding="utf-8") as log_file:
        tee = _LineTee(sys.stdout, log_file, on_line=on_line)
        err_tee = _LineTee(sys.stderr, log_file, on_line=on_line)
        with redirect_stdout(tee), redirect_stderr(err_tee):
            yield log_path


def _forward_progress_line(line: str, callback: ProgressCallback) -> None:
    normalized = line.strip()
    if not normalized:
        return
    for pattern, step, message, pct in _PHASE1_PROGRESS_PATTERNS:
        if pattern.search(normalized):
            callback(step, message or normalized, pct)
            return


@contextmanager
def _runtime_controls_env(runtime_controls: dict[str, object] | None) -> Iterator[None]:
    runtime_controls = dict(runtime_controls or {})
    env_map = {
        "speaker_binding_mode": "CLYPT_SPEAKER_BINDING_MODE",
        "tracking_mode": "CLYPT_TRACKING_MODE",
        "shared_analysis_proxy_enabled": "CLYPT_SHARED_ANALYSIS_PROXY",
    }
    previous: dict[str, str | None] = {}
    try:
        for control_key, env_key in env_map.items():
            control_value = runtime_controls.get(control_key)
            if control_value in (None, ""):
                continue
            previous[env_key] = os.environ.get(env_key)
            os.environ[env_key] = str(control_value)

        profile_name = str(runtime_controls.get("profile_name", "") or "").strip().lower()
        evaluation_mode = bool(runtime_controls.get("evaluation_mode"))
        if evaluation_mode or profile_name in {"podcast_eval", "podcast", "eval", "test"}:
            previous["CLYPT_PHASE1_EVAL_PROFILE"] = os.environ.get("CLYPT_PHASE1_EVAL_PROFILE")
            os.environ["CLYPT_PHASE1_EVAL_PROFILE"] = profile_name or "eval"
        yield
    finally:
        for env_key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = old_value


class _LineTee:
    def __init__(
        self,
        original: TextIO,
        log_file: TextIO,
        *,
        on_line: Callable[[str], None] | None = None,
    ) -> None:
        self._original = original
        self._log_file = log_file
        self._on_line = on_line
        self._buffer = ""

    def write(self, data: str) -> int:
        self._original.write(data)
        self._original.flush()
        self._log_file.write(data)
        self._log_file.flush()
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if self._on_line is not None:
                self._on_line(line)
        return len(data)

    def flush(self) -> None:
        self._original.flush()
        self._log_file.flush()
        if self._buffer and self._on_line is not None:
            self._on_line(self._buffer)
            self._buffer = ""


_PHASE1_PROGRESS_PATTERNS: tuple[tuple[re.Pattern[str], str, str | None, float], ...] = (
    (
        re.compile(r"Downloading video\+audio stream"),
        "download_media",
        "Downloading source media",
        0.03,
    ),
    (
        re.compile(r"Re-encoding local video .* -> h264"),
        "transcoding",
        "Transcoding source video to H.264",
        0.08,
    ),
    (
        re.compile(r"Local H\.264 ready:"),
        "media_ready",
        "Source media prepared for extraction",
        0.1,
    ),
    (re.compile(r"^\[Phase 1\] Received video"), "received_media", "Media received by worker", 0.12),
    (
        re.compile(r"^\[Phase 1\] Step 1\+2/4"),
        "asr_tracking",
        "Running ASR and tracking",
        0.2,
    ),
    (
        re.compile(r"Running Parakeet ASR inference"),
        "asr",
        "Running ASR inference",
        0.25,
    ),
    (
        re.compile(r"Running YOLO26s .*tracking inference"),
        "tracking",
        "Running visual tracking",
        0.3,
    ),
    (
        re.compile(r"^\[Phase 1\] Step 3/4"),
        "clustering",
        "Clustering tracklets into global IDs",
        0.7,
    ),
    (
        re.compile(r"^\[Phase 1\] Step 4/4"),
        "speaker_binding",
        "Running speaker binding",
        0.82,
    ),
    (
        re.compile(r"^Tracking complete:"),
        "tracking_complete",
        "Tracking complete",
        0.65,
    ),
    (
        re.compile(r"^\[Phase 1\] Complete"),
        "phase1_complete",
        "Phase 1 extraction complete",
        0.9,
    ),
)
