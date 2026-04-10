from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import subprocess

# Gradio-style list (comma-separated in .env); matches playground hotword usage.
_DEFAULT_HOTWORDS = (
    "I, you, he, she, it, we, they, me, him, her, us, them, "
    "my, your, his, hers, its, our, their, mine, yours, ours, theirs, "
    "this, that, these, those, who, whom, whose, which, what, "
    "and, but, or, nor, for, so, yet, after, although, as, because, before, if, since, "
    "that, though, unless, until, when, whenever, where, whereas, while, however, therefore, "
    "moreover, furthermore, also, additionally, meanwhile, consequently, otherwise, nevertheless, "
    "for example, in addition, on the other hand, similarly, likewise, in contrast, thus, hence, "
    "indeed, finally, first, second, third"
)


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
class VibeVoiceSettings:
    """Shared generation controls for the vLLM VibeVoice ASR path."""

    hotwords_context: str = _DEFAULT_HOTWORDS
    max_new_tokens: int = 32768
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    num_beams: int = 1


@dataclass(slots=True)
class VibeVoiceVLLMSettings:
    """Settings for the persistent vLLM VibeVoice ASR sidecar service."""

    base_url: str
    model: str = "vibevoice"
    timeout_s: float = 7200.0
    healthcheck_path: str = "/health"
    max_retries: int = 1
    audio_mode: str = "base64"


@dataclass(slots=True)
class VertexSettings:
    project: str
    generation_backend: str = "developer"
    embedding_backend: str = "vertex"
    gemini_api_key: str | None = None
    generation_location: str = "global"
    embedding_location: str = "us-central1"
    generation_model: str = "gemini-3.1-pro-preview"
    embedding_model: str = "gemini-embedding-2-preview"
    flash_model: str = "gemini-3-flash-preview"
    thinking_budget: int = 128
    api_max_retries: int = 6
    api_initial_backoff_s: float = 1.0
    api_max_backoff_s: float = 30.0
    api_backoff_multiplier: float = 2.0
    api_jitter_ratio: float = 0.2


@dataclass(slots=True)
class StorageSettings:
    gcs_bucket: str


@dataclass(slots=True)
class CloudTasksSettings:
    project: str = ""
    location: str = "us-central1"
    queue: str = "clypt-phase24"
    worker_url: str | None = None
    service_account_email: str | None = None


@dataclass(slots=True)
class SpannerSettings:
    project: str = ""
    instance: str = "clypt-phase14"
    database: str = "clypt_phase14"
    ddl_operation_timeout_s: float = 600.0


@dataclass(slots=True)
class Phase24WorkerSettings:
    service_name: str = "clypt-phase24-worker"
    environment: str = "dev"
    query_version: str = "v1"
    concurrency: int = 1
    debug_snapshots: bool = False
    max_attempts: int = 3


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
    vibevoice: VibeVoiceSettings
    vllm_vibevoice: VibeVoiceVLLMSettings
    vertex: VertexSettings
    storage: StorageSettings
    cloud_tasks: CloudTasksSettings = field(default_factory=CloudTasksSettings)
    spanner: SpannerSettings = field(default_factory=SpannerSettings)
    phase24_worker: Phase24WorkerSettings = field(default_factory=Phase24WorkerSettings)
    phase1_runtime: Phase1RuntimeSettings = field(default_factory=Phase1RuntimeSettings)


def load_provider_settings() -> ProviderSettings:
    _load_local_env_files()

    vertex_project = _read_env("GOOGLE_CLOUD_PROJECT") or _discover_gcloud_project()
    if not vertex_project:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT is required, or gcloud must have an active project configured."
        )

    gcs_bucket = _read_env("GCS_BUCKET", "CLYPT_GCS_BUCKET")
    if not gcs_bucket:
        raise ValueError("GCS_BUCKET or CLYPT_GCS_BUCKET is required for Phase 1 storage.")

    hotwords_context = _read_env("VIBEVOICE_HOTWORDS_CONTEXT") or _DEFAULT_HOTWORDS
    backend = (_read_env("VIBEVOICE_BACKEND") or "vllm").lower()
    if backend != "vllm":
        raise ValueError(
            f"Unsupported VIBEVOICE_BACKEND={backend!r}; only 'vllm' is supported on main."
        )

    vllm_base_url = _read_env("VIBEVOICE_VLLM_BASE_URL")
    if not vllm_base_url:
        raise ValueError("VIBEVOICE_VLLM_BASE_URL is required.")
    vllm_settings = VibeVoiceVLLMSettings(
        base_url=vllm_base_url,
        model=_read_env("VIBEVOICE_VLLM_MODEL") or "vibevoice",
        timeout_s=float(_read_env("VIBEVOICE_VLLM_TIMEOUT_S") or "7200"),
        healthcheck_path=_read_env("VIBEVOICE_VLLM_HEALTHCHECK_PATH") or "/health",
        max_retries=int(_read_env("VIBEVOICE_VLLM_MAX_RETRIES") or "1"),
        audio_mode=_read_env("VIBEVOICE_VLLM_AUDIO_MODE") or "base64",
    )

    generation_backend = (
        (_read_env("GENAI_GENERATION_BACKEND", "GENAI_BACKEND") or "developer")
        .strip()
        .lower()
    )
    if generation_backend not in {"developer", "vertex"}:
        raise ValueError(
            f"Unsupported GENAI_GENERATION_BACKEND={generation_backend!r}; expected 'developer' or 'vertex'."
        )
    embedding_backend = ((_read_env("GENAI_EMBEDDING_BACKEND") or "vertex").strip().lower())
    if embedding_backend not in {"developer", "vertex"}:
        raise ValueError(
            f"Unsupported GENAI_EMBEDDING_BACKEND={embedding_backend!r}; expected 'developer' or 'vertex'."
        )

    return ProviderSettings(
        vibevoice=VibeVoiceSettings(
            hotwords_context=hotwords_context,
            max_new_tokens=int(_read_env("VIBEVOICE_MAX_NEW_TOKENS") or "32768"),
            do_sample=(_read_env("VIBEVOICE_DO_SAMPLE") or "0") == "1",
            temperature=float(_read_env("VIBEVOICE_TEMPERATURE") or "0"),
            top_p=float(_read_env("VIBEVOICE_TOP_P") or "1"),
            repetition_penalty=float(_read_env("VIBEVOICE_REPETITION_PENALTY") or "1"),
            num_beams=int(_read_env("VIBEVOICE_NUM_BEAMS") or "1"),
        ),
        vllm_vibevoice=vllm_settings,
        vertex=VertexSettings(
            project=vertex_project,
            generation_backend=generation_backend,
            embedding_backend=embedding_backend,
            gemini_api_key=_read_env("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            generation_location=_read_env("VERTEX_GEMINI_LOCATION")
            or _read_env("GOOGLE_CLOUD_LOCATION", "VERTEX_LOCATION")
            or "global",
            embedding_location=_read_env("VERTEX_EMBEDDING_LOCATION") or "us-central1",
            generation_model=_read_env("VERTEX_GEMINI_MODEL") or "gemini-3.1-pro-preview",
            embedding_model=_read_env("VERTEX_EMBEDDING_MODEL") or "gemini-embedding-2-preview",
            flash_model=_read_env("VERTEX_FLASH_MODEL") or "gemini-3-flash-preview",
            thinking_budget=max(1, int(_read_env("VERTEX_THINKING_BUDGET") or "128")),
            api_max_retries=max(0, int(_read_env("VERTEX_API_MAX_RETRIES") or "6")),
            api_initial_backoff_s=max(
                0.0, float(_read_env("VERTEX_API_INITIAL_BACKOFF_S") or "1.0")
            ),
            api_max_backoff_s=max(
                0.0, float(_read_env("VERTEX_API_MAX_BACKOFF_S") or "30.0")
            ),
            api_backoff_multiplier=max(
                1.0, float(_read_env("VERTEX_API_BACKOFF_MULTIPLIER") or "2.0")
            ),
            api_jitter_ratio=max(
                0.0, float(_read_env("VERTEX_API_JITTER_RATIO") or "0.2")
            ),
        ),
        storage=StorageSettings(gcs_bucket=gcs_bucket),
        cloud_tasks=CloudTasksSettings(
            project=_read_env("CLYPT_PHASE24_PROJECT") or vertex_project,
            location=_read_env("CLYPT_PHASE24_TASKS_LOCATION") or "us-central1",
            queue=_read_env("CLYPT_PHASE24_TASKS_QUEUE") or "clypt-phase24",
            worker_url=_read_env("CLYPT_PHASE24_WORKER_URL"),
            service_account_email=_read_env("CLYPT_PHASE24_WORKER_SERVICE_ACCOUNT_EMAIL"),
        ),
        spanner=SpannerSettings(
            project=_read_env("CLYPT_SPANNER_PROJECT") or vertex_project,
            instance=_read_env("CLYPT_SPANNER_INSTANCE") or "clypt-phase14",
            database=_read_env("CLYPT_SPANNER_DATABASE") or "clypt_phase14",
            ddl_operation_timeout_s=float(
                _read_env("CLYPT_SPANNER_DDL_OPERATION_TIMEOUT_S") or "600"
            ),
        ),
        phase24_worker=Phase24WorkerSettings(
            service_name=_read_env("CLYPT_PHASE24_WORKER_SERVICE_NAME")
            or "clypt-phase24-worker",
            environment=_read_env("CLYPT_PHASE24_ENVIRONMENT") or "dev",
            query_version=_read_env("CLYPT_PHASE24_QUERY_VERSION") or "v1",
            concurrency=int(_read_env("CLYPT_PHASE24_CONCURRENCY") or "1"),
            debug_snapshots=(_read_env("CLYPT_DEBUG_SNAPSHOTS") or "0") == "1",
            max_attempts=int(_read_env("CLYPT_PHASE24_MAX_ATTEMPTS") or "3"),
        ),
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
    "StorageSettings",
    "CloudTasksSettings",
    "SpannerSettings",
    "Phase24WorkerSettings",
    "VertexSettings",
    "VibeVoiceSettings",
    "VibeVoiceVLLMSettings",
    "load_provider_settings",
]
