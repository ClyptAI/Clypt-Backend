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

_REMOVED_LOCAL_GENERATION_ENVS = (
    "CLYPT_LOCAL_LLM_ENABLE_THINKING",
)


def _read_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _read_bool_env(*names: str, default: bool = False) -> bool:
    raw = _read_env(*names)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "1.0", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "0.0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean value {raw!r} for {', '.join(names)}; "
        "expected one of 0/1, true/false, yes/no."
    )


def _read_int_env(*names: str, default: int) -> int:
    raw = _read_env(*names)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        parsed = float(raw)
        if not parsed.is_integer():
            raise ValueError(
                f"Invalid integer value {raw!r} for {', '.join(names)}; expected a whole number."
            )
        return int(parsed)


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


def _raise_if_removed_local_generation_env_present() -> None:
    for name in _REMOVED_LOCAL_GENERATION_ENVS:
        if os.getenv(name) is not None:
            raise ValueError(
                f"{name} has been removed because the local OpenAI-compatible Qwen path "
                "always runs with thinking disabled."
            )


@dataclass(slots=True)
class VibeVoiceSettings:
    """Shared generation controls for the vLLM VibeVoice ASR path."""

    hotwords_context: str = _DEFAULT_HOTWORDS
    max_new_tokens: int = 32768
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.03
    num_beams: int = 1


@dataclass(slots=True)
class VibeVoiceVLLMSettings:
    """Settings for the persistent vLLM VibeVoice ASR sidecar service."""

    base_url: str
    model: str = "vibevoice"
    timeout_s: float = 7200.0
    healthcheck_path: str = "/health"
    max_retries: int = 1
    audio_mode: str = "url"


@dataclass(slots=True)
class LocalGenerationSettings:
    """OpenAI-compatible local LLM (e.g. vLLM Qwen) for phase24 generation."""

    base_url: str = "http://127.0.0.1:8001/v1"
    model: str = ""
    timeout_s: float = 600.0
    max_retries: int = 6
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 30.0
    backoff_multiplier: float = 2.0
    jitter_ratio: float = 0.2
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0


@dataclass(slots=True)
class VertexSettings:
    project: str
    generation_backend: str = "local_openai"
    embedding_backend: str = "vertex"
    gemini_api_key: str | None = None
    generation_location: str = "global"
    embedding_location: str = "us-central1"
    generation_model: str = "Qwen/Qwen3.5-27B"
    embedding_model: str = "gemini-embedding-2-preview"
    flash_model: str = "Qwen/Qwen3.5-27B"
    generation_api_max_retries: int = 6
    generation_api_initial_backoff_s: float = 1.0
    generation_api_max_backoff_s: float = 30.0
    generation_api_backoff_multiplier: float = 2.0
    generation_api_jitter_ratio: float = 0.2
    embedding_api_max_retries: int = 6
    embedding_api_initial_backoff_s: float = 1.0
    embedding_api_max_backoff_s: float = 30.0
    embedding_api_backoff_multiplier: float = 2.0
    embedding_api_jitter_ratio: float = 0.2


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
    fail_fast_preemption_threshold: int = 0
    fail_fast_p95_latency_ms: float = 0.0
    admission_metrics_path: str | None = None
    block_on_phase1_active: bool = False
    max_vllm_queue_depth: int = 0
    max_vllm_decode_backlog: int = 0


@dataclass(slots=True)
class Phase24MediaPrepSettings:
    backend: str = "local"
    service_url: str | None = None
    auth_mode: str = "id_token"
    audience: str | None = None
    timeout_s: float = 600.0


@dataclass(slots=True)
class Phase24LocalQueueSettings:
    """SQLite-backed queue for Phase 2–4 when running Phase 1 locally (run_phase14)."""

    path: Path = field(
        default_factory=lambda: Path(
            os.getenv("CLYPT_PHASE24_LOCAL_QUEUE_PATH", "backend/outputs/phase24_local_queue.sqlite")
        )
    )
    poll_interval_ms: int = 500
    lease_timeout_s: int = 1800
    max_inflight: int = 1
    max_requests_per_worker: int = 0
    queue_backend: str = "local_sqlite"
    reclaim_expired_leases: bool = False
    fail_fast_on_stale_running: bool = True


@dataclass(slots=True)
class VLLMRuntimeSettings:
    """Runtime tuning surface for local vLLM-hosted generation."""

    profile: str = "conservative"
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    language_model_only: bool = False
    speculative_mode: str = "off"
    speculative_num_tokens: int | None = None


@dataclass(slots=True)
class Phase1RuntimeSettings:
    working_root: Path = field(
        default_factory=lambda: Path(
            os.getenv("CLYPT_PHASE1_WORK_ROOT", "backend/outputs/v3_1_phase1_work")
        )
    )
    run_yamnet_on_gpu: bool = False
    keep_workdir: bool = False
    input_mode: str = "test_bank"
    test_bank_path: str | None = None
    test_bank_strict: bool = True


@dataclass(slots=True)
class Phase1ASRSettings:
    backend: str = "vllm"
    service_url: str | None = None
    auth_mode: str = "id_token"
    audience: str | None = None
    timeout_s: float = 7200.0


@dataclass(slots=True)
class ProviderSettings:
    vibevoice: VibeVoiceSettings
    vllm_vibevoice: VibeVoiceVLLMSettings
    vertex: VertexSettings
    local_generation: LocalGenerationSettings
    storage: StorageSettings
    phase1_asr: Phase1ASRSettings = field(default_factory=Phase1ASRSettings)
    cloud_tasks: CloudTasksSettings = field(default_factory=CloudTasksSettings)
    spanner: SpannerSettings = field(default_factory=SpannerSettings)
    phase24_worker: Phase24WorkerSettings = field(default_factory=Phase24WorkerSettings)
    phase24_media_prep: Phase24MediaPrepSettings = field(default_factory=Phase24MediaPrepSettings)
    phase24_local_queue: Phase24LocalQueueSettings = field(default_factory=Phase24LocalQueueSettings)
    vllm_runtime: VLLMRuntimeSettings = field(default_factory=VLLMRuntimeSettings)
    phase1_runtime: Phase1RuntimeSettings = field(default_factory=Phase1RuntimeSettings)


def load_provider_settings() -> ProviderSettings:
    _load_local_env_files()
    _raise_if_removed_local_generation_env_present()

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
    phase1_asr = Phase1ASRSettings(
        backend=((_read_env("CLYPT_PHASE1_ASR_BACKEND") or "vllm").strip().lower()),
        service_url=_read_env("CLYPT_PHASE1_ASR_SERVICE_URL"),
        auth_mode=((_read_env("CLYPT_PHASE1_ASR_AUTH_MODE") or "id_token").strip().lower()),
        audience=_read_env("CLYPT_PHASE1_ASR_AUDIENCE"),
        timeout_s=float(_read_env("CLYPT_PHASE1_ASR_TIMEOUT_S") or "7200"),
    )
    if phase1_asr.backend not in {"vllm", "cloud_run_l4"}:
        raise ValueError(
            "Unsupported CLYPT_PHASE1_ASR_BACKEND="
            f"{phase1_asr.backend!r}; expected 'vllm' or 'cloud_run_l4'."
        )

    vllm_base_url = _read_env("VIBEVOICE_VLLM_BASE_URL")
    if phase1_asr.backend == "cloud_run_l4":
        if vllm_base_url:
            raise ValueError(
                "VIBEVOICE_VLLM_BASE_URL must be unset when CLYPT_PHASE1_ASR_BACKEND=cloud_run_l4."
            )
        if not phase1_asr.service_url:
            raise ValueError(
                "CLYPT_PHASE1_ASR_SERVICE_URL is required when CLYPT_PHASE1_ASR_BACKEND=cloud_run_l4."
            )
    elif not vllm_base_url:
        raise ValueError("VIBEVOICE_VLLM_BASE_URL is required.")
    vllm_settings = VibeVoiceVLLMSettings(
        base_url=vllm_base_url or "",
        model=_read_env("VIBEVOICE_VLLM_MODEL") or "vibevoice",
        timeout_s=float(_read_env("VIBEVOICE_VLLM_TIMEOUT_S") or "7200"),
        healthcheck_path=_read_env("VIBEVOICE_VLLM_HEALTHCHECK_PATH") or "/health",
        max_retries=int(_read_env("VIBEVOICE_VLLM_MAX_RETRIES") or "1"),
        audio_mode=_read_env("VIBEVOICE_VLLM_AUDIO_MODE") or "url",
    )

    generation_backend = ((_read_env("GENAI_GENERATION_BACKEND") or "local_openai").strip().lower())
    if generation_backend not in {"developer", "local_openai"}:
        raise ValueError(
            "Unsupported GENAI_GENERATION_BACKEND="
            f"{generation_backend!r}; expected 'developer' or 'local_openai'."
        )
    embedding_backend = ((_read_env("VERTEX_EMBEDDING_BACKEND") or "vertex").strip().lower())
    if embedding_backend not in {"developer", "vertex"}:
        raise ValueError(
            "Unsupported VERTEX_EMBEDDING_BACKEND="
            f"{embedding_backend!r}; expected 'developer' or 'vertex'."
        )

    local_generation = LocalGenerationSettings(
        base_url=_read_env("CLYPT_LOCAL_LLM_BASE_URL") or "http://127.0.0.1:8001/v1",
        model=_read_env("CLYPT_LOCAL_LLM_MODEL") or "",
        timeout_s=float(_read_env("CLYPT_LOCAL_LLM_TIMEOUT_S") or "600"),
        max_retries=max(0, int(_read_env("CLYPT_LOCAL_LLM_MAX_RETRIES") or "6")),
        initial_backoff_s=max(
            0.0,
            float(_read_env("CLYPT_LOCAL_LLM_INITIAL_BACKOFF_S") or "1.0"),
        ),
        max_backoff_s=max(
            0.0,
            float(_read_env("CLYPT_LOCAL_LLM_MAX_BACKOFF_S") or "30.0"),
        ),
        backoff_multiplier=max(
            1.0,
            float(_read_env("CLYPT_LOCAL_LLM_BACKOFF_MULTIPLIER") or "2.0"),
        ),
        jitter_ratio=max(
            0.0,
            float(_read_env("CLYPT_LOCAL_LLM_JITTER_RATIO") or "0.2"),
        ),
        temperature=float(_read_env("CLYPT_LOCAL_LLM_TEMPERATURE") or "0.7"),
        top_p=float(_read_env("CLYPT_LOCAL_LLM_TOP_P") or "0.8"),
        top_k=max(1, int(_read_env("CLYPT_LOCAL_LLM_TOP_K") or "20")),
        min_p=max(0.0, float(_read_env("CLYPT_LOCAL_LLM_MIN_P") or "0.0")),
        presence_penalty=float(_read_env("CLYPT_LOCAL_LLM_PRESENCE_PENALTY") or "1.5"),
        repetition_penalty=float(_read_env("CLYPT_LOCAL_LLM_REPETITION_PENALTY") or "1.0"),
    )

    return ProviderSettings(
        vibevoice=VibeVoiceSettings(
            hotwords_context=hotwords_context,
            max_new_tokens=int(_read_env("VIBEVOICE_MAX_NEW_TOKENS") or "32768"),
            do_sample=_read_bool_env("VIBEVOICE_DO_SAMPLE", default=False),
            temperature=float(_read_env("VIBEVOICE_TEMPERATURE") or "0"),
            top_p=float(_read_env("VIBEVOICE_TOP_P") or "1.0"),
            repetition_penalty=float(_read_env("VIBEVOICE_REPETITION_PENALTY") or "1.03"),
            num_beams=_read_int_env("VIBEVOICE_NUM_BEAMS", default=1),
        ),
        phase1_asr=phase1_asr,
        vllm_vibevoice=vllm_settings,
        local_generation=local_generation,
        vertex=VertexSettings(
            project=vertex_project,
            generation_backend=generation_backend,
            embedding_backend=embedding_backend,
            gemini_api_key=_read_env("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            generation_location=_read_env("GENAI_GENERATION_LOCATION")
            or _read_env("GOOGLE_CLOUD_LOCATION")
            or "global",
            embedding_location=_read_env("VERTEX_EMBEDDING_LOCATION") or "us-central1",
            generation_model=_read_env("GENAI_GENERATION_MODEL") or "Qwen/Qwen3.5-27B",
            embedding_model=_read_env("VERTEX_EMBEDDING_MODEL") or "gemini-embedding-2-preview",
            flash_model=_read_env("GENAI_FLASH_MODEL") or "Qwen/Qwen3.5-27B",
            generation_api_max_retries=max(
                0,
                int(
                    _read_env(
                        "GENAI_GENERATION_API_MAX_RETRIES",
                    )
                    or "6"
                ),
            ),
            generation_api_initial_backoff_s=max(
                0.0,
                float(
                    _read_env(
                        "GENAI_GENERATION_API_INITIAL_BACKOFF_S",
                    )
                    or "1.0"
                ),
            ),
            generation_api_max_backoff_s=max(
                0.0,
                float(
                    _read_env(
                        "GENAI_GENERATION_API_MAX_BACKOFF_S",
                    )
                    or "30.0"
                ),
            ),
            generation_api_backoff_multiplier=max(
                1.0,
                float(
                    _read_env(
                        "GENAI_GENERATION_API_BACKOFF_MULTIPLIER",
                    )
                    or "2.0"
                ),
            ),
            generation_api_jitter_ratio=max(
                0.0,
                float(
                    _read_env(
                        "GENAI_GENERATION_API_JITTER_RATIO",
                    )
                    or "0.2"
                ),
            ),
            embedding_api_max_retries=max(
                0,
                int(
                    _read_env(
                        "VERTEX_EMBEDDING_API_MAX_RETRIES",
                    )
                    or "6"
                ),
            ),
            embedding_api_initial_backoff_s=max(
                0.0,
                float(
                    _read_env(
                        "VERTEX_EMBEDDING_API_INITIAL_BACKOFF_S",
                    )
                    or "1.0"
                ),
            ),
            embedding_api_max_backoff_s=max(
                0.0,
                float(
                    _read_env(
                        "VERTEX_EMBEDDING_API_MAX_BACKOFF_S",
                    )
                    or "30.0"
                ),
            ),
            embedding_api_backoff_multiplier=max(
                1.0,
                float(
                    _read_env(
                        "VERTEX_EMBEDDING_API_BACKOFF_MULTIPLIER",
                    )
                    or "2.0"
                ),
            ),
            embedding_api_jitter_ratio=max(
                0.0,
                float(
                    _read_env(
                        "VERTEX_EMBEDDING_API_JITTER_RATIO",
                    )
                    or "0.2"
                ),
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
            fail_fast_preemption_threshold=int(
                _read_env("CLYPT_PHASE24_FAILFAST_PREEMPTION_THRESHOLD") or "0"
            ),
            fail_fast_p95_latency_ms=float(
                _read_env("CLYPT_PHASE24_FAILFAST_P95_LATENCY_MS") or "0"
            ),
            admission_metrics_path=_read_env("CLYPT_PHASE24_ADMISSION_METRICS_PATH"),
            block_on_phase1_active=_read_bool_env(
                "CLYPT_PHASE24_BLOCK_ON_PHASE1_ACTIVE", default=False
            ),
            max_vllm_queue_depth=int(
                _read_env("CLYPT_PHASE24_MAX_VLLM_QUEUE_DEPTH") or "0"
            ),
            max_vllm_decode_backlog=int(
                _read_env("CLYPT_PHASE24_MAX_VLLM_DECODE_BACKLOG") or "0"
            ),
        ),
        phase24_media_prep=Phase24MediaPrepSettings(
            backend=((_read_env("CLYPT_PHASE24_MEDIA_PREP_BACKEND") or "local").strip().lower()),
            service_url=_read_env("CLYPT_PHASE24_MEDIA_PREP_SERVICE_URL"),
            auth_mode=((_read_env("CLYPT_PHASE24_MEDIA_PREP_AUTH_MODE") or "id_token").strip().lower()),
            audience=_read_env("CLYPT_PHASE24_MEDIA_PREP_AUDIENCE"),
            timeout_s=float(_read_env("CLYPT_PHASE24_MEDIA_PREP_TIMEOUT_S") or "600"),
        ),
        phase24_local_queue=Phase24LocalQueueSettings(
            path=Path(
                _read_env("CLYPT_PHASE24_LOCAL_QUEUE_PATH")
                or "backend/outputs/phase24_local_queue.sqlite"
            ),
            poll_interval_ms=_read_int_env("CLYPT_PHASE24_LOCAL_POLL_INTERVAL_MS", default=500),
            lease_timeout_s=_read_int_env("CLYPT_PHASE24_LOCAL_LEASE_TIMEOUT_S", default=1800),
            max_inflight=_read_int_env("CLYPT_PHASE24_LOCAL_MAX_INFLIGHT", default=1),
            max_requests_per_worker=_read_int_env(
                "CLYPT_PHASE24_LOCAL_MAX_REQUESTS_PER_WORKER", default=0
            ),
            queue_backend=((_read_env("CLYPT_PHASE24_QUEUE_BACKEND") or "local_sqlite").strip().lower()),
            reclaim_expired_leases=_read_bool_env(
                "CLYPT_PHASE24_LOCAL_RECLAIM_EXPIRED_LEASES", default=False
            ),
            fail_fast_on_stale_running=_read_bool_env(
                "CLYPT_PHASE24_LOCAL_FAIL_FAST_ON_STALE_RUNNING", default=True
            ),
        ),
        vllm_runtime=VLLMRuntimeSettings(
            profile=((_read_env("CLYPT_VLLM_PROFILE") or "conservative").strip().lower()),
            max_num_seqs=(
                int(_read_env("CLYPT_VLLM_MAX_NUM_SEQS"))
                if _read_env("CLYPT_VLLM_MAX_NUM_SEQS") is not None
                else None
            ),
            max_num_batched_tokens=(
                int(_read_env("CLYPT_VLLM_MAX_NUM_BATCHED_TOKENS"))
                if _read_env("CLYPT_VLLM_MAX_NUM_BATCHED_TOKENS") is not None
                else None
            ),
            gpu_memory_utilization=(
                float(_read_env("CLYPT_VLLM_GPU_MEMORY_UTILIZATION"))
                if _read_env("CLYPT_VLLM_GPU_MEMORY_UTILIZATION") is not None
                else None
            ),
            max_model_len=(
                int(_read_env("CLYPT_VLLM_MAX_MODEL_LEN"))
                if _read_env("CLYPT_VLLM_MAX_MODEL_LEN") is not None
                else None
            ),
            language_model_only=_read_bool_env(
                "CLYPT_VLLM_LANGUAGE_MODEL_ONLY", default=False
            ),
            speculative_mode=((_read_env("CLYPT_VLLM_SPECULATIVE_MODE") or "off").strip().lower()),
            speculative_num_tokens=(
                int(_read_env("CLYPT_VLLM_SPECULATIVE_NUM_TOKENS"))
                if _read_env("CLYPT_VLLM_SPECULATIVE_NUM_TOKENS") is not None
                else None
            ),
        ),
        phase1_runtime=Phase1RuntimeSettings(
            working_root=Path(
                _read_env("CLYPT_PHASE1_WORK_ROOT") or "backend/outputs/v3_1_phase1_work"
            ),
            run_yamnet_on_gpu=(_read_env("CLYPT_PHASE1_YAMNET_DEVICE") or "cpu").lower()
            == "gpu",
            keep_workdir=(_read_env("CLYPT_PHASE1_KEEP_WORKDIR") or "0") == "1",
            input_mode=(_read_env("CLYPT_PHASE1_INPUT_MODE") or "test_bank").strip().lower(),
            test_bank_path=_read_env("CLYPT_PHASE1_TEST_BANK_PATH"),
            test_bank_strict=(_read_env("CLYPT_PHASE1_TEST_BANK_STRICT") or "1") == "1",
        ),
    )


__all__ = [
    "LocalGenerationSettings",
    "Phase24LocalQueueSettings",
    "Phase24MediaPrepSettings",
    "VLLMRuntimeSettings",
    "Phase1RuntimeSettings",
    "Phase1ASRSettings",
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
