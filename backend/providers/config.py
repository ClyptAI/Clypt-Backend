from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import subprocess


def _read_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _discover_gcloud_project() -> str | None:
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = (result.stdout or "").strip()
    if not value or value == "(unset)":
        return None
    return value


def _load_local_env_files() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    repo_root = Path.cwd()
    for filename in [".env", ".env.local"]:
        env_path = repo_root / filename
        if env_path.exists():
            load_dotenv(env_path, override=False)


@dataclass(slots=True)
class PyannoteSettings:
    api_key: str
    base_url: str = "https://api.pyannote.ai"
    diarize_model: str = "precision-2"
    transcription_model: str = "parakeet-tdt-0.6b-v3"
    poll_interval_s: float = 2.0
    timeout_s: float = 1800.0


@dataclass(slots=True)
class VertexSettings:
    project: str
    generation_location: str = "global"
    embedding_location: str = "us-central1"
    generation_model: str = "gemini-3.1-pro-preview"
    embedding_model: str = "gemini-embedding-2-preview"


@dataclass(slots=True)
class StorageSettings:
    gcs_bucket: str


@dataclass(slots=True)
class Phase1RuntimeSettings:
    working_root: Path = field(
        default_factory=lambda: Path(
            os.getenv("CLYPT_PHASE1_WORK_ROOT", "backend/outputs/v3_1_phase1_work")
        )
    )
    run_yamnet_on_gpu: bool = True
    keep_workdir: bool = False


@dataclass(slots=True)
class ProviderSettings:
    pyannote: PyannoteSettings
    vertex: VertexSettings
    storage: StorageSettings
    phase1_runtime: Phase1RuntimeSettings = field(default_factory=Phase1RuntimeSettings)


def load_provider_settings() -> ProviderSettings:
    _load_local_env_files()

    pyannote_api_key = _read_env("PYANNOTE_API_KEY")
    if not pyannote_api_key:
        raise ValueError("PYANNOTE_API_KEY is required for V3.1 cloud pyannote.")

    vertex_project = _read_env("GOOGLE_CLOUD_PROJECT") or _discover_gcloud_project()
    if not vertex_project:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT is required, or gcloud must have an active project configured."
        )

    gcs_bucket = _read_env("GCS_BUCKET", "CLYPT_GCS_BUCKET")
    if not gcs_bucket:
        raise ValueError("GCS_BUCKET or CLYPT_GCS_BUCKET is required for Phase 1 storage.")

    return ProviderSettings(
        pyannote=PyannoteSettings(
            api_key=pyannote_api_key,
            base_url=_read_env("PYANNOTE_BASE_URL") or "https://api.pyannote.ai",
            diarize_model=_read_env("PYANNOTE_DIARIZE_MODEL") or "precision-2",
            transcription_model=_read_env("PYANNOTE_TRANSCRIPTION_MODEL")
            or "parakeet-tdt-0.6b-v3",
            poll_interval_s=float(_read_env("PYANNOTE_POLL_INTERVAL_S") or "2.0"),
            timeout_s=float(_read_env("PYANNOTE_TIMEOUT_S") or "1800"),
        ),
        vertex=VertexSettings(
            project=vertex_project,
            generation_location=_read_env("VERTEX_GEMINI_LOCATION")
            or _read_env("GOOGLE_CLOUD_LOCATION", "VERTEX_LOCATION")
            or "global",
            embedding_location=_read_env("VERTEX_EMBEDDING_LOCATION") or "us-central1",
            generation_model=_read_env("VERTEX_GEMINI_MODEL") or "gemini-3.1-pro-preview",
            embedding_model=_read_env("VERTEX_EMBEDDING_MODEL") or "gemini-embedding-2-preview",
        ),
        storage=StorageSettings(gcs_bucket=gcs_bucket),
        phase1_runtime=Phase1RuntimeSettings(
            working_root=Path(
                _read_env("CLYPT_PHASE1_WORK_ROOT") or "backend/outputs/v3_1_phase1_work"
            ),
            run_yamnet_on_gpu=(_read_env("CLYPT_PHASE1_YAMNET_DEVICE") or "gpu").lower()
            == "gpu",
            keep_workdir=(_read_env("CLYPT_PHASE1_KEEP_WORKDIR") or "0") == "1",
        ),
    )


__all__ = [
    "Phase1RuntimeSettings",
    "ProviderSettings",
    "PyannoteSettings",
    "StorageSettings",
    "VertexSettings",
    "load_provider_settings",
]
