from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import subprocess


_REMOVED_LOCAL_GENERATION_ENVS = (
    "CLYPT_LOCAL_LLM_ENABLE_THINKING",
)

_REMOVED_VLLM_RUNTIME_ENVS = (
    "CLYPT_VLLM_PROFILE",
    "CLYPT_VLLM_MAX_NUM_SEQS",
    "CLYPT_VLLM_MAX_NUM_BATCHED_TOKENS",
    "CLYPT_VLLM_GPU_MEMORY_UTILIZATION",
    "CLYPT_VLLM_MAX_MODEL_LEN",
    "CLYPT_VLLM_LANGUAGE_MODEL_ONLY",
    "CLYPT_VLLM_SPECULATIVE_MODE",
    "CLYPT_VLLM_SPECULATIVE_NUM_TOKENS",
    "CLYPT_PHASE24_MAX_VLLM_QUEUE_DEPTH",
    "CLYPT_PHASE24_MAX_VLLM_DECODE_BACKLOG",
)

_REMOVED_PHASE1_AUDIO_ENVS = (
    "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL",
    "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN",
    "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_TOKEN",
    "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_TIMEOUT_S",
    "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_HEALTHCHECK_PATH",
    "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_SCRATCH_ROOT",
    "CLYPT_PHASE1_AUDIO_HOST_URL",
    "CLYPT_PHASE1_AUDIO_HOST_AUTH_TOKEN",
    "CLYPT_PHASE1_AUDIO_HOST_TOKEN",
    "CLYPT_PHASE1_AUDIO_HOST_TIMEOUT_S",
    "CLYPT_PHASE1_AUDIO_HOST_HEALTHCHECK_PATH",
    "CLYPT_PHASE1_AUDIO_HOST_SCRATCH_ROOT",
    "CLYPT_PHASE1_AUDIO_SCRATCH_ROOT",
    "CLYPT_PHASE1_NFA_DEVICE",
    "CLYPT_PHASE1_EMOTION2VEC_DEVICE",
    "CLYPT_PHASE1_YAMNET_DEVICE",
    "CLYPT_PHASE1_ASR_FIRST_VISUAL_AFTER_ASR",
    "CLYPT_PHASE1_VLLM_SLEEP_AFTER_ASR",
    "CLYPT_PHASE1_VLLM_SLEEP_BASE_URL",
    "CLYPT_PHASE1_VLLM_SLEEP_LEVEL",
    "CLYPT_PHASE1_VLLM_SLEEP_TIMEOUT_S",
    "VIBEVOICE_BACKEND",
    "VIBEVOICE_VLLM_BASE_URL",
    "VIBEVOICE_VLLM_MODEL",
    "VIBEVOICE_VLLM_TIMEOUT_S",
    "VIBEVOICE_VLLM_HEALTHCHECK_PATH",
    "VIBEVOICE_VLLM_MAX_RETRIES",
    "VIBEVOICE_VLLM_AUDIO_MODE",
    "VIBEVOICE_HOTWORDS_CONTEXT",
    "VIBEVOICE_MAX_NEW_TOKENS",
    "VIBEVOICE_DO_SAMPLE",
    "VIBEVOICE_TEMPERATURE",
    "VIBEVOICE_TOP_P",
    "VIBEVOICE_REPETITION_PENALTY",
    "VIBEVOICE_NUM_BEAMS",
    "VIBEVOICE_LONGFORM_ENABLED",
    "VIBEVOICE_LONGFORM_SINGLE_PASS_MAX_MINUTES",
    "VIBEVOICE_LONGFORM_TWO_SHARD_MAX_MINUTES",
    "VIBEVOICE_LONGFORM_FOUR_SHARD_MAX_MINUTES",
    "VIBEVOICE_LONGFORM_MAX_SHARDS",
    "VIBEVOICE_LONGFORM_SPEAKER_MATCH_THRESHOLD",
    "VIBEVOICE_LONGFORM_REP_CLIP_MIN_SECONDS",
    "VIBEVOICE_LONGFORM_REP_CLIP_MAX_SECONDS",
    "VIBEVOICE_LONGFORM_VERIFIER_BACKEND",
    "VIBEVOICE_LONGFORM_VERIFIER_DEVICE",
    "VIBEVOICE_LONGFORM_VERIFIER_MODEL_ID",
    "VIBEVOICE_LONGFORM_VERIFIER_CACHE_DIR",
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
    normalized = raw.lower()
    if normalized in {"1", "1.0", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "0.0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean value {raw!r} for {', '.join(names)}; expected 0/1 or true/false."
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
    except (OSError, subprocess.SubprocessError):
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


def _raise_if_removed_env_present() -> None:
    for name in (
        *_REMOVED_LOCAL_GENERATION_ENVS,
        *_REMOVED_VLLM_RUNTIME_ENVS,
        *_REMOVED_PHASE1_AUDIO_ENVS,
    ):
        if os.getenv(name) is not None:
            raise ValueError(
                f"{name} has been removed on AMD-refactor. Phase1 now uses "
                "ElevenLabs Scribe v2 plus Modal RF-DETR; remove this env var."
            )


@dataclass(slots=True)
class LocalGenerationSettings:
    """OpenAI-compatible local LLM for Phase26 generation."""

    base_url: str = "http://127.0.0.1:8001/v1"
    model: str = ""
    timeout_s: float = 600.0
    max_retries: int = 6
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 30.0
    backoff_multiplier: float = 2.0
    jitter_ratio: float = 0.2
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 40
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0


@dataclass(slots=True)
class VertexSettings:
    project: str
    generation_backend: str = "local_openai"
    embedding_backend: str = "vertex"
    gemini_api_key: str | None = None
    generation_location: str = "global"
    embedding_location: str = "us-central1"
    generation_model: str = "Qwen/Qwen3.6-35B-A3B"
    embedding_model: str = "gemini-embedding-2-preview"
    flash_model: str = "Qwen/Qwen3.6-35B-A3B"
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
class SpannerSettings:
    project: str = ""
    instance: str = "clypt-spanner-v3"
    database: str = "clypt-graph-db-v3"
    ddl_operation_timeout_s: float = 600.0

    @property
    def is_configured(self) -> bool:
        return bool(self.project) and bool(self.instance) and bool(self.database)


@dataclass(slots=True)
class Phase24WorkerSettings:
    service_name: str = "clypt-phase26-worker"
    environment: str = "dev"
    query_version: str = "v1"
    concurrency: int = 1
    debug_snapshots: bool = False
    max_attempts: int = 3
    fail_fast_preemption_threshold: int = 0
    fail_fast_p95_latency_ms: float = 0.0
    admission_metrics_path: str | None = None
    block_on_phase1_active: bool = False


@dataclass(slots=True)
class Phase24LocalQueueSettings:
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
class Phase1RuntimeSettings:
    working_root: Path = field(
        default_factory=lambda: Path(
            os.getenv("CLYPT_PHASE1_WORK_ROOT", "backend/outputs/v3_1_phase1_work")
        )
    )
    input_mode: str = "test_bank"
    test_bank_path: str | None = None
    test_bank_strict: bool = True


@dataclass(slots=True)
class Phase1ASRSettings:
    backend: str = "elevenlabs_scribe_v2"
    timeout_s: float = 7200.0


@dataclass(slots=True)
class ElevenLabsScribeSettings:
    api_key: str
    model_id: str = "scribe_v2"
    endpoint_url: str = "https://api.elevenlabs.io/v1/speech-to-text"
    timeout_s: float = 7200.0
    max_retries: int = 2
    diarize: bool = True
    tag_audio_events: bool = True
    timestamps_granularity: str = "word"
    language_code: str = "en"
    num_speakers: int | None = None
    diarization_threshold: float | None = None
    temperature: float = 0.0
    seed: int | None = None
    keyterms: tuple[str, ...] = ()
    url_field: str = "source_url"
    signed_url_expiry_hours: int = 24
    turn_gap_ms: int = 1200


@dataclass(slots=True)
class Phase1VisualServiceSettings:
    """Settings for the dedicated Modal RF-DETR visual service."""

    service_url: str
    auth_token: str
    timeout_s: float = 7200.0
    healthcheck_path: str = "/health"


@dataclass(slots=True)
class Phase26DispatchServiceSettings:
    service_url: str
    auth_token: str
    timeout_s: float = 30.0
    healthcheck_path: str = "/health"


@dataclass(slots=True)
class NodeMediaPrepSettings:
    service_url: str
    auth_token: str
    timeout_s: float = 1800.0
    max_concurrency: int = 12


@dataclass(slots=True)
class Phase6RenderSettings:
    service_url: str
    auth_token: str
    timeout_s: float = 3600.0


@dataclass(slots=True)
class ProviderSettings:
    elevenlabs_scribe: ElevenLabsScribeSettings | None
    vertex: VertexSettings
    local_generation: LocalGenerationSettings
    storage: StorageSettings
    node_media_prep: NodeMediaPrepSettings
    phase6_render: Phase6RenderSettings | None = None
    phase1_visual_service: Phase1VisualServiceSettings | None = None
    phase26_dispatch_service: Phase26DispatchServiceSettings | None = None
    phase1_asr: Phase1ASRSettings = field(default_factory=Phase1ASRSettings)
    spanner: SpannerSettings = field(default_factory=SpannerSettings)
    phase24_worker: Phase24WorkerSettings = field(default_factory=Phase24WorkerSettings)
    phase24_local_queue: Phase24LocalQueueSettings = field(default_factory=Phase24LocalQueueSettings)
    phase1_runtime: Phase1RuntimeSettings = field(default_factory=Phase1RuntimeSettings)


def _load_scribe_settings(*, require: bool) -> tuple[Phase1ASRSettings, ElevenLabsScribeSettings | None]:
    phase1_audio_backend = (
        _read_env("CLYPT_PHASE1_AUDIO_BACKEND", "CLYPT_PHASE1_ASR_BACKEND")
        or "elevenlabs_scribe_v2"
    ).lower()
    if phase1_audio_backend != "elevenlabs_scribe_v2":
        raise ValueError(
            "Unsupported CLYPT_PHASE1_AUDIO_BACKEND/CLYPT_PHASE1_ASR_BACKEND="
            f"{phase1_audio_backend!r}; expected 'elevenlabs_scribe_v2'."
        )
    phase1_asr = Phase1ASRSettings(
        backend=phase1_audio_backend,
        timeout_s=float(_read_env("CLYPT_PHASE1_SCRIBE_TIMEOUT_S", "CLYPT_PHASE1_ASR_TIMEOUT_S") or "7200"),
    )

    api_key = _read_env("ELEVENLABS_API_KEY")
    if require and not api_key:
        raise ValueError("ELEVENLABS_API_KEY is required for CLYPT_PHASE1_AUDIO_BACKEND=elevenlabs_scribe_v2.")
    if not api_key:
        return phase1_asr, None

    num_speakers_raw = _read_env("CLYPT_PHASE1_SCRIBE_NUM_SPEAKERS")
    seed_raw = _read_env("CLYPT_PHASE1_SCRIBE_SEED")
    keyterms_raw = _read_env("CLYPT_PHASE1_SCRIBE_KEYTERMS")
    url_field = _read_env("CLYPT_PHASE1_SCRIBE_URL_FIELD") or "source_url"
    if url_field not in {"source_url", "cloud_storage_url"}:
        raise ValueError("CLYPT_PHASE1_SCRIBE_URL_FIELD must be 'source_url' or 'cloud_storage_url'.")
    return phase1_asr, ElevenLabsScribeSettings(
        api_key=api_key,
        model_id=_read_env("CLYPT_PHASE1_SCRIBE_MODEL_ID") or "scribe_v2",
        endpoint_url=_read_env("CLYPT_PHASE1_SCRIBE_ENDPOINT_URL")
        or "https://api.elevenlabs.io/v1/speech-to-text",
        timeout_s=float(_read_env("CLYPT_PHASE1_SCRIBE_TIMEOUT_S") or "7200"),
        max_retries=max(0, _read_int_env("CLYPT_PHASE1_SCRIBE_MAX_RETRIES", default=2)),
        diarize=_read_bool_env("CLYPT_PHASE1_SCRIBE_DIARIZE", default=True),
        tag_audio_events=_read_bool_env("CLYPT_PHASE1_SCRIBE_TAG_AUDIO_EVENTS", default=True),
        timestamps_granularity=_read_env("CLYPT_PHASE1_SCRIBE_TIMESTAMPS_GRANULARITY") or "word",
        language_code=_read_env("CLYPT_PHASE1_SCRIBE_LANGUAGE_CODE") or "en",
        num_speakers=int(num_speakers_raw) if num_speakers_raw else None,
        diarization_threshold=(
            float(_read_env("CLYPT_PHASE1_SCRIBE_DIARIZATION_THRESHOLD"))
            if _read_env("CLYPT_PHASE1_SCRIBE_DIARIZATION_THRESHOLD")
            else None
        ),
        temperature=float(_read_env("CLYPT_PHASE1_SCRIBE_TEMPERATURE") or "0"),
        seed=int(seed_raw) if seed_raw else None,
        keyterms=tuple(
            item.strip() for item in (keyterms_raw or "").split(",") if item.strip()
        ),
        url_field=url_field,
        signed_url_expiry_hours=_read_int_env("CLYPT_PHASE1_SCRIBE_SIGNED_URL_EXPIRY_HOURS", default=24),
        turn_gap_ms=_read_int_env("CLYPT_PHASE1_SCRIBE_TURN_GAP_MS", default=1200),
    )


def load_provider_settings(
    *,
    require_node_media_prep: bool = True,
    require_phase1_visual_service: bool = False,
    require_phase26_dispatch_service: bool = False,
) -> ProviderSettings:
    _load_local_env_files()
    _raise_if_removed_env_present()

    vertex_project = _read_env("GOOGLE_CLOUD_PROJECT") or _discover_gcloud_project()
    if not vertex_project:
        raise ValueError("GOOGLE_CLOUD_PROJECT is required, or gcloud must have an active project configured.")

    gcs_bucket = _read_env("GCS_BUCKET", "CLYPT_GCS_BUCKET")
    if not gcs_bucket:
        raise ValueError("GCS_BUCKET or CLYPT_GCS_BUCKET is required for storage.")

    require_scribe = require_phase1_visual_service or require_phase26_dispatch_service
    phase1_asr, elevenlabs_scribe = _load_scribe_settings(require=require_scribe)

    generation_backend = (_read_env("GENAI_GENERATION_BACKEND") or "local_openai").strip().lower()
    if generation_backend != "local_openai":
        raise ValueError("GENAI_GENERATION_BACKEND must be local_openai on AMD-refactor.")
    embedding_backend = (_read_env("GENAI_EMBEDDING_BACKEND") or "vertex").strip().lower()

    node_media_prep_url = _read_env("CLYPT_PHASE24_NODE_MEDIA_PREP_URL")
    node_media_prep_token = _read_env("CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN")
    if require_node_media_prep and not node_media_prep_url:
        raise ValueError("CLYPT_PHASE24_NODE_MEDIA_PREP_URL is required.")
    if require_node_media_prep and not node_media_prep_token:
        raise ValueError("CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN is required.")
    node_media_prep = NodeMediaPrepSettings(
        service_url=(node_media_prep_url or "").rstrip("/"),
        auth_token=node_media_prep_token or "",
        timeout_s=float(_read_env("CLYPT_PHASE24_NODE_MEDIA_PREP_TIMEOUT_S") or "1800"),
        max_concurrency=max(1, _read_int_env("CLYPT_PHASE24_NODE_MEDIA_PREP_MAX_CONCURRENCY", default=12)),
    )

    phase6_render = None
    phase6_render_url = _read_env("CLYPT_PHASE24_PHASE6_RENDER_URL")
    phase6_render_token = _read_env("CLYPT_PHASE24_PHASE6_RENDER_TOKEN")
    if phase6_render_url or phase6_render_token:
        phase6_render = Phase6RenderSettings(
            service_url=(phase6_render_url or "").rstrip("/"),
            auth_token=phase6_render_token or "",
            timeout_s=float(_read_env("CLYPT_PHASE24_PHASE6_RENDER_TIMEOUT_S") or "3600"),
        )

    visual_url = _read_env("CLYPT_PHASE1_VISUAL_SERVICE_URL")
    visual_token = _read_env("CLYPT_PHASE1_VISUAL_SERVICE_AUTH_TOKEN")
    if require_phase1_visual_service and not visual_url:
        raise ValueError("CLYPT_PHASE1_VISUAL_SERVICE_URL is required for the Modal RF-DETR visual service.")
    if require_phase1_visual_service and not visual_token:
        raise ValueError("CLYPT_PHASE1_VISUAL_SERVICE_AUTH_TOKEN is required for the Modal RF-DETR visual service.")
    phase1_visual_service = (
        Phase1VisualServiceSettings(
            service_url=(visual_url or "").rstrip("/"),
            auth_token=visual_token or "",
            timeout_s=float(_read_env("CLYPT_PHASE1_VISUAL_SERVICE_TIMEOUT_S") or "7200"),
            healthcheck_path=_read_env("CLYPT_PHASE1_VISUAL_SERVICE_HEALTHCHECK_PATH") or "/health",
        )
        if visual_url or visual_token
        else None
    )

    dispatch_url = _read_env("CLYPT_PHASE24_DISPATCH_URL")
    dispatch_token = _read_env("CLYPT_PHASE24_DISPATCH_AUTH_TOKEN")
    if require_phase26_dispatch_service and not dispatch_url:
        raise ValueError("CLYPT_PHASE24_DISPATCH_URL is required.")
    if require_phase26_dispatch_service and not dispatch_token:
        raise ValueError("CLYPT_PHASE24_DISPATCH_AUTH_TOKEN is required.")
    phase26_dispatch_service = (
        Phase26DispatchServiceSettings(
            service_url=(dispatch_url or "").rstrip("/"),
            auth_token=dispatch_token or "",
            timeout_s=float(_read_env("CLYPT_PHASE24_DISPATCH_TIMEOUT_S") or "30"),
            healthcheck_path=_read_env("CLYPT_PHASE24_DISPATCH_HEALTHCHECK_PATH") or "/health",
        )
        if dispatch_url or dispatch_token
        else None
    )

    local_generation = LocalGenerationSettings(
        base_url=_read_env("CLYPT_LOCAL_LLM_BASE_URL") or "http://127.0.0.1:8001/v1",
        model=_read_env("CLYPT_LOCAL_LLM_MODEL") or "",
        timeout_s=float(_read_env("CLYPT_LOCAL_LLM_TIMEOUT_S") or "600"),
        max_retries=max(0, int(_read_env("CLYPT_LOCAL_LLM_MAX_RETRIES") or "6")),
        initial_backoff_s=max(0.0, float(_read_env("CLYPT_LOCAL_LLM_INITIAL_BACKOFF_S") or "1.0")),
        max_backoff_s=max(0.0, float(_read_env("CLYPT_LOCAL_LLM_MAX_BACKOFF_S") or "30.0")),
        backoff_multiplier=max(1.0, float(_read_env("CLYPT_LOCAL_LLM_BACKOFF_MULTIPLIER") or "2.0")),
        jitter_ratio=max(0.0, float(_read_env("CLYPT_LOCAL_LLM_JITTER_RATIO") or "0.2")),
        temperature=float(_read_env("CLYPT_LOCAL_LLM_TEMPERATURE") or "0.0"),
        top_p=float(_read_env("CLYPT_LOCAL_LLM_TOP_P") or "1.0"),
        top_k=max(1, int(_read_env("CLYPT_LOCAL_LLM_TOP_K") or "40")),
        min_p=max(0.0, float(_read_env("CLYPT_LOCAL_LLM_MIN_P") or "0.0")),
        presence_penalty=float(_read_env("CLYPT_LOCAL_LLM_PRESENCE_PENALTY") or "0.0"),
        repetition_penalty=float(_read_env("CLYPT_LOCAL_LLM_REPETITION_PENALTY") or "1.0"),
    )

    return ProviderSettings(
        phase1_asr=phase1_asr,
        elevenlabs_scribe=elevenlabs_scribe,
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
            generation_model=_read_env("GENAI_GENERATION_MODEL") or "Qwen/Qwen3.6-35B-A3B",
            embedding_model=_read_env("VERTEX_EMBEDDING_MODEL") or "gemini-embedding-2-preview",
            flash_model=_read_env("GENAI_FLASH_MODEL") or "Qwen/Qwen3.6-35B-A3B",
            generation_api_max_retries=max(0, int(_read_env("GENAI_GENERATION_API_MAX_RETRIES") or "6")),
            generation_api_initial_backoff_s=max(0.0, float(_read_env("GENAI_GENERATION_API_INITIAL_BACKOFF_S") or "1.0")),
            generation_api_max_backoff_s=max(0.0, float(_read_env("GENAI_GENERATION_API_MAX_BACKOFF_S") or "30.0")),
            generation_api_backoff_multiplier=max(1.0, float(_read_env("GENAI_GENERATION_API_BACKOFF_MULTIPLIER") or "2.0")),
            generation_api_jitter_ratio=max(0.0, float(_read_env("GENAI_GENERATION_API_JITTER_RATIO") or "0.2")),
            embedding_api_max_retries=max(0, int(_read_env("VERTEX_EMBEDDING_API_MAX_RETRIES") or "6")),
            embedding_api_initial_backoff_s=max(0.0, float(_read_env("VERTEX_EMBEDDING_API_INITIAL_BACKOFF_S") or "1.0")),
            embedding_api_max_backoff_s=max(0.0, float(_read_env("VERTEX_EMBEDDING_API_MAX_BACKOFF_S") or "30.0")),
            embedding_api_backoff_multiplier=max(1.0, float(_read_env("VERTEX_EMBEDDING_API_BACKOFF_MULTIPLIER") or "2.0")),
            embedding_api_jitter_ratio=max(0.0, float(_read_env("VERTEX_EMBEDDING_API_JITTER_RATIO") or "0.2")),
        ),
        storage=StorageSettings(gcs_bucket=gcs_bucket),
        node_media_prep=node_media_prep,
        phase6_render=phase6_render,
        phase1_visual_service=phase1_visual_service,
        phase26_dispatch_service=phase26_dispatch_service,
        spanner=SpannerSettings(
            project=_read_env("CLYPT_SPANNER_PROJECT") or vertex_project,
            instance=_read_env("CLYPT_SPANNER_INSTANCE") or "clypt-spanner-v3",
            database=_read_env("CLYPT_SPANNER_DATABASE") or "clypt-graph-db-v3",
            ddl_operation_timeout_s=float(_read_env("CLYPT_SPANNER_DDL_OPERATION_TIMEOUT_S") or "600"),
        ),
        phase24_worker=Phase24WorkerSettings(
            service_name=_read_env("CLYPT_PHASE24_WORKER_SERVICE_NAME") or "clypt-phase26-worker",
            environment=_read_env("CLYPT_PHASE24_ENVIRONMENT") or "dev",
            query_version=_read_env("CLYPT_PHASE24_QUERY_VERSION") or "v1",
            concurrency=int(_read_env("CLYPT_PHASE24_CONCURRENCY") or "1"),
            debug_snapshots=_read_bool_env("CLYPT_DEBUG_SNAPSHOTS", default=False),
            max_attempts=int(_read_env("CLYPT_PHASE24_MAX_ATTEMPTS") or "3"),
            fail_fast_preemption_threshold=int(_read_env("CLYPT_PHASE24_FAILFAST_PREEMPTION_THRESHOLD") or "0"),
            fail_fast_p95_latency_ms=float(_read_env("CLYPT_PHASE24_FAILFAST_P95_LATENCY_MS") or "0"),
            admission_metrics_path=_read_env("CLYPT_PHASE24_ADMISSION_METRICS_PATH"),
            block_on_phase1_active=_read_bool_env("CLYPT_PHASE24_BLOCK_ON_PHASE1_ACTIVE", default=False),
        ),
        phase24_local_queue=Phase24LocalQueueSettings(
            path=Path(_read_env("CLYPT_PHASE24_LOCAL_QUEUE_PATH") or "backend/outputs/phase24_local_queue.sqlite"),
            poll_interval_ms=_read_int_env("CLYPT_PHASE24_LOCAL_POLL_INTERVAL_MS", default=500),
            lease_timeout_s=_read_int_env("CLYPT_PHASE24_LOCAL_LEASE_TIMEOUT_S", default=1800),
            max_inflight=_read_int_env("CLYPT_PHASE24_LOCAL_MAX_INFLIGHT", default=1),
            max_requests_per_worker=_read_int_env("CLYPT_PHASE24_LOCAL_MAX_REQUESTS_PER_WORKER", default=0),
            queue_backend=(_read_env("CLYPT_PHASE24_QUEUE_BACKEND") or "local_sqlite").lower(),
            reclaim_expired_leases=_read_bool_env("CLYPT_PHASE24_LOCAL_RECLAIM_EXPIRED_LEASES", default=False),
            fail_fast_on_stale_running=_read_bool_env("CLYPT_PHASE24_LOCAL_FAIL_FAST_ON_STALE_RUNNING", default=True),
        ),
        phase1_runtime=Phase1RuntimeSettings(
            working_root=Path(_read_env("CLYPT_PHASE1_WORK_ROOT") or "backend/outputs/v3_1_phase1_work"),
            input_mode=(_read_env("CLYPT_PHASE1_INPUT_MODE") or "test_bank").lower(),
            test_bank_path=_read_env("CLYPT_PHASE1_TEST_BANK_PATH"),
            test_bank_strict=_read_bool_env("CLYPT_PHASE1_TEST_BANK_STRICT", default=True),
        ),
    )


def load_phase1_host_settings() -> ProviderSettings:
    return load_provider_settings(
        require_node_media_prep=False,
        require_phase1_visual_service=True,
        require_phase26_dispatch_service=True,
    )


def load_phase26_host_settings() -> ProviderSettings:
    return load_provider_settings(
        require_node_media_prep=True,
        require_phase1_visual_service=False,
        require_phase26_dispatch_service=False,
    )


__all__ = [
    "ElevenLabsScribeSettings",
    "LocalGenerationSettings",
    "NodeMediaPrepSettings",
    "Phase1VisualServiceSettings",
    "Phase26DispatchServiceSettings",
    "Phase24LocalQueueSettings",
    "Phase1RuntimeSettings",
    "Phase1ASRSettings",
    "Phase6RenderSettings",
    "ProviderSettings",
    "StorageSettings",
    "SpannerSettings",
    "Phase24WorkerSettings",
    "VertexSettings",
    "load_phase1_host_settings",
    "load_phase26_host_settings",
    "load_provider_settings",
]
