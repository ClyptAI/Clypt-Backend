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

# Archaic env vars that were loaded but never consumed (pre-SGLang era).
# Hard-fail if any are set so stale deployments surface immediately.
_REMOVED_VLLM_RUNTIME_ENVS = (
    "CLYPT_VLLM_PROFILE",
    "CLYPT_VLLM_MAX_NUM_SEQS",
    "CLYPT_VLLM_MAX_NUM_BATCHED_TOKENS",
    "CLYPT_VLLM_GPU_MEMORY_UTILIZATION",
    "CLYPT_VLLM_MAX_MODEL_LEN",
    "CLYPT_VLLM_LANGUAGE_MODEL_ONLY",
    "CLYPT_VLLM_SPECULATIVE_MODE",
    "CLYPT_VLLM_SPECULATIVE_NUM_TOKENS",
    # Admission-gating keys whose metrics producers never shipped.
    "CLYPT_PHASE24_MAX_VLLM_QUEUE_DEPTH",
    "CLYPT_PHASE24_MAX_VLLM_DECODE_BACKLOG",
)


_VIBEVOICE_ASR_SERVICE_ENV_ALIASES: dict[str, tuple[str, ...]] = {
    "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL": ("CLYPT_PHASE1_AUDIO_HOST_URL",),
    "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN": (
        "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_TOKEN",
        "CLYPT_PHASE1_AUDIO_HOST_AUTH_TOKEN",
        "CLYPT_PHASE1_AUDIO_HOST_TOKEN",
    ),
    "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_TIMEOUT_S": ("CLYPT_PHASE1_AUDIO_HOST_TIMEOUT_S",),
    "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_HEALTHCHECK_PATH": (
        "CLYPT_PHASE1_AUDIO_HOST_HEALTHCHECK_PATH",
    ),
    "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_SCRATCH_ROOT": (
        "CLYPT_PHASE1_VIBEVOICE_ASR_SCRATCH_ROOT",
        "CLYPT_PHASE1_AUDIO_HOST_SCRATCH_ROOT",
        "CLYPT_PHASE1_AUDIO_SCRATCH_ROOT",
    ),
}


def _read_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _getenv_with_aliases(canonical: str, aliases: tuple[str, ...] = ()) -> str | None:
    value = os.getenv(canonical)
    if value is not None and value.strip():
        return value.strip()
    for legacy in aliases:
        value = os.getenv(legacy)
        if value is not None and value.strip():
            return value.strip()
    return None


def _normalize_hotwords_context(raw: str | None) -> str:
    return raw if raw is not None else _DEFAULT_HOTWORDS


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


def _raise_if_removed_local_generation_env_present() -> None:
    for name in _REMOVED_LOCAL_GENERATION_ENVS:
        if os.getenv(name) is not None:
            raise ValueError(
                f"{name} has been removed because the local OpenAI-compatible Qwen path "
                "always runs with thinking disabled."
            )
    for name in _REMOVED_VLLM_RUNTIME_ENVS:
        if os.getenv(name) is not None:
            raise ValueError(
                f"{name} has been removed; Phase 2-4 serves Qwen via SGLang and no "
                "code consumes this setting. Remove it from your env file."
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
    instance: str = "clypt-phase14"
    database: str = "clypt_phase14"
    ddl_operation_timeout_s: float = 600.0

    @property
    def is_configured(self) -> bool:
        """True iff Spanner is fully addressable (project/instance/database all set).

        Used as the durable-persistence opt-out gate: when False, Phase 1
        runs without Spanner (in-memory only). When True, any failure to
        construct the repository must propagate so GPU work is not
        silently decoupled from durable storage (rotated credentials,
        DDL drift, etc.).
        """
        return bool(self.project) and bool(self.instance) and bool(self.database)


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
class Phase1RuntimeSettings:
    working_root: Path = field(
        default_factory=lambda: Path(
            os.getenv("CLYPT_PHASE1_WORK_ROOT", "backend/outputs/v3_1_phase1_work")
        )
    )
    run_yamnet_on_gpu: bool = False
    input_mode: str = "test_bank"
    test_bank_path: str | None = None
    test_bank_strict: bool = True


@dataclass(slots=True)
class Phase1ASRSettings:
    backend: str = "vllm"
    timeout_s: float = 7200.0


@dataclass(slots=True)
class VibeVoiceAsrServiceSettings:
    """Required settings for the Phase 1 VibeVoice ASR service.

    Phase 1 still runs NFA + emotion2vec+ + YAMNet in-process; this setting
    covers only the HTTP boundary used for VibeVoice ASR. Fail-fast at load if
    URL or token are missing.
    """

    service_url: str
    auth_token: str
    timeout_s: float = 7200.0
    healthcheck_path: str = "/health"


# Legacy name kept for import compatibility with older call-sites during the
# one-release deprecation window. Prefer ``VibeVoiceAsrServiceSettings``.
AudioHostSettings = VibeVoiceAsrServiceSettings


@dataclass(slots=True)
class Phase1VisualServiceSettings:
    """Settings for the persistent local Phase 1 visual extraction service."""

    service_url: str
    auth_token: str
    timeout_s: float = 3600.0
    healthcheck_path: str = "/health"


@dataclass(slots=True)
class Phase26DispatchServiceSettings:
    """Settings for the remote Phase26 enqueue/dispatch service."""

    service_url: str
    auth_token: str
    timeout_s: float = 30.0
    healthcheck_path: str = "/health"


@dataclass(slots=True)
class NodeMediaPrepSettings:
    """Required settings for remote node-media-prep.

    The Phase 2-4 worker has no local fallback; the remote service is the only path.
    Fail-fast at load if URL or token are missing.
    """

    service_url: str
    auth_token: str
    timeout_s: float = 3600.0
    max_concurrency: int = 8


@dataclass(slots=True)
class ProviderSettings:
    vibevoice: VibeVoiceSettings
    vllm_vibevoice: VibeVoiceVLLMSettings
    vertex: VertexSettings
    local_generation: LocalGenerationSettings
    storage: StorageSettings
    vibevoice_asr_service: VibeVoiceAsrServiceSettings
    node_media_prep: NodeMediaPrepSettings
    phase1_visual_service: Phase1VisualServiceSettings | None = None
    phase26_dispatch_service: Phase26DispatchServiceSettings | None = None
    phase1_asr: Phase1ASRSettings = field(default_factory=Phase1ASRSettings)
    spanner: SpannerSettings = field(default_factory=SpannerSettings)
    phase24_worker: Phase24WorkerSettings = field(default_factory=Phase24WorkerSettings)
    phase24_local_queue: Phase24LocalQueueSettings = field(default_factory=Phase24LocalQueueSettings)
    phase1_runtime: Phase1RuntimeSettings = field(default_factory=Phase1RuntimeSettings)

    @property
    def audio_host(self) -> VibeVoiceAsrServiceSettings:
        """Legacy alias for ``vibevoice_asr_service`` accepted through 2026-05-17
        (commit 393abaee). Drop together with the ``AudioHostSettings`` alias.
        """
        return self.vibevoice_asr_service


@dataclass(slots=True)
class AudioHostProcessSettings:
    """Settings consumed by the Phase 1 VibeVoice service process itself.

    Distinct from ``VibeVoiceAsrServiceSettings``, which are the *caller-side*
    bearer token + URL the runner uses to reach this service. This dataclass holds
    only what the process needs to run: the co-located VibeVoice vLLM sidecar,
    VibeVoice generation controls, GCS bucket for URL resolution, and the
    Phase 1 working-root knob. NFA/emotion/YAMNet do not live in this process,
    so those settings are intentionally absent.
    """

    vibevoice: VibeVoiceSettings
    vllm_vibevoice: VibeVoiceVLLMSettings
    storage: StorageSettings
    phase1_runtime: Phase1RuntimeSettings = field(default_factory=Phase1RuntimeSettings)


def load_provider_settings(
    *,
    require_vibevoice_asr_service: bool = True,
    require_node_media_prep: bool = True,
    require_phase1_visual_service: bool = False,
    require_phase26_dispatch_service: bool = False,
) -> ProviderSettings:
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

    hotwords_context = _normalize_hotwords_context(_read_env("VIBEVOICE_HOTWORDS_CONTEXT"))
    backend = (_read_env("VIBEVOICE_BACKEND") or "vllm").lower()
    if backend != "vllm":
        raise ValueError(
            f"Unsupported VIBEVOICE_BACKEND={backend!r}; only 'vllm' is supported on main."
        )
    phase1_asr = Phase1ASRSettings(
        backend=((_read_env("CLYPT_PHASE1_ASR_BACKEND") or "vllm").strip().lower()),
        timeout_s=float(_read_env("CLYPT_PHASE1_ASR_TIMEOUT_S") or "7200"),
    )
    if phase1_asr.backend != "vllm":
        raise ValueError(
            "Unsupported CLYPT_PHASE1_ASR_BACKEND="
            f"{phase1_asr.backend!r}; only 'vllm' is supported."
        )

    # VIBEVOICE_VLLM_BASE_URL is intentionally optional here because the
    # general provider loader is used outside the dedicated Phase1 service
    # process. The active Phase1 host loader accepts VibeVoice settings, while
    # the service process loader (``load_audio_host_settings``) requires them.
    vllm_base_url = _read_env("VIBEVOICE_VLLM_BASE_URL")
    vllm_settings = VibeVoiceVLLMSettings(
        base_url=vllm_base_url or "unused://phase1-vibevoice-service",
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

    # Legacy CLYPT_PHASE1_AUDIO_HOST_* env aliases accepted through
    # 2026-05-17 (commit 393abaee). Drop together with the AudioHostSettings
    # alias.
    vibevoice_asr_url = _getenv_with_aliases(
        "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL",
        _VIBEVOICE_ASR_SERVICE_ENV_ALIASES["CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL"],
    )
    vibevoice_asr_token = _getenv_with_aliases(
        "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN",
        _VIBEVOICE_ASR_SERVICE_ENV_ALIASES[
            "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN"
        ],
    )
    if require_vibevoice_asr_service and not vibevoice_asr_url:
        raise ValueError(
            "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL is required. VibeVoice ASR has no "
            "in-process fallback for callers. Point this at the local Phase 1 "
            "service URL or another explicitly provisioned ASR service. The legacy "
            "CLYPT_PHASE1_AUDIO_HOST_URL alias is accepted through 2026-05-17 "
            "(commit 393abaee)."
        )
    if require_vibevoice_asr_service and not vibevoice_asr_token:
        raise ValueError(
            "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN is required for callers "
            "of the VibeVoice ASR service. The legacy CLYPT_PHASE1_AUDIO_HOST_TOKEN "
            "alias is accepted through 2026-05-17 (commit 393abaee)."
        )
    vibevoice_asr_service = VibeVoiceAsrServiceSettings(
        service_url=(vibevoice_asr_url or "").rstrip("/"),
        auth_token=vibevoice_asr_token or "",
        timeout_s=float(
            _getenv_with_aliases(
                "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_TIMEOUT_S",
                _VIBEVOICE_ASR_SERVICE_ENV_ALIASES[
                    "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_TIMEOUT_S"
                ],
            )
            or "7200"
        ),
        healthcheck_path=_getenv_with_aliases(
            "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_HEALTHCHECK_PATH",
            _VIBEVOICE_ASR_SERVICE_ENV_ALIASES[
                "CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_HEALTHCHECK_PATH"
            ],
        )
        or "/health",
    )

    node_media_prep_url = _read_env("CLYPT_PHASE24_NODE_MEDIA_PREP_URL")
    node_media_prep_token = _read_env("CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN")
    if require_node_media_prep and not node_media_prep_url:
        raise ValueError(
            "CLYPT_PHASE24_NODE_MEDIA_PREP_URL is required. Phase 2 node-media prep "
            "runs through the configured remote media-prep service; there is no "
            "local fallback."
        )
    if require_node_media_prep and not node_media_prep_token:
        raise ValueError(
            "CLYPT_PHASE24_NODE_MEDIA_PREP_TOKEN is required for the remote "
            "node-media-prep endpoint."
        )
    node_media_prep = NodeMediaPrepSettings(
        service_url=(node_media_prep_url or "").rstrip("/"),
        auth_token=node_media_prep_token or "",
        timeout_s=float(_read_env("CLYPT_PHASE24_NODE_MEDIA_PREP_TIMEOUT_S") or "3600"),
        max_concurrency=max(
            1,
            _read_int_env("CLYPT_PHASE24_NODE_MEDIA_PREP_MAX_CONCURRENCY", default=16),
        ),
    )

    phase1_visual_service: Phase1VisualServiceSettings | None = None
    phase1_visual_service_url = _read_env("CLYPT_PHASE1_VISUAL_SERVICE_URL")
    phase1_visual_service_token = _read_env("CLYPT_PHASE1_VISUAL_SERVICE_AUTH_TOKEN")
    if require_phase1_visual_service and not phase1_visual_service_url:
        raise ValueError(
            "CLYPT_PHASE1_VISUAL_SERVICE_URL is required. Phase 1 visual extraction "
            "runs through the persistent local Phase 1 visual service."
        )
    if require_phase1_visual_service and not phase1_visual_service_token:
        raise ValueError(
            "CLYPT_PHASE1_VISUAL_SERVICE_AUTH_TOKEN is required for the Phase 1 "
            "visual service."
        )
    if phase1_visual_service_url or phase1_visual_service_token:
        phase1_visual_service = Phase1VisualServiceSettings(
            service_url=(phase1_visual_service_url or "").rstrip("/"),
            auth_token=phase1_visual_service_token or "",
            timeout_s=float(_read_env("CLYPT_PHASE1_VISUAL_SERVICE_TIMEOUT_S") or "3600"),
            healthcheck_path=_read_env("CLYPT_PHASE1_VISUAL_SERVICE_HEALTHCHECK_PATH") or "/health",
        )

    phase26_dispatch_service: Phase26DispatchServiceSettings | None = None
    phase26_dispatch_url = _read_env("CLYPT_PHASE24_DISPATCH_URL")
    phase26_dispatch_token = _read_env("CLYPT_PHASE24_DISPATCH_AUTH_TOKEN")
    if require_phase26_dispatch_service and not phase26_dispatch_url:
        raise ValueError(
            "CLYPT_PHASE24_DISPATCH_URL is required. Phase 1 hands off completed "
            "artifacts to the remote Phase26 enqueue service."
        )
    if require_phase26_dispatch_service and not phase26_dispatch_token:
        raise ValueError(
            "CLYPT_PHASE24_DISPATCH_AUTH_TOKEN is required for the remote Phase26 "
            "enqueue service."
        )
    if phase26_dispatch_url or phase26_dispatch_token:
        phase26_dispatch_service = Phase26DispatchServiceSettings(
            service_url=(phase26_dispatch_url or "").rstrip("/"),
            auth_token=phase26_dispatch_token or "",
            timeout_s=float(_read_env("CLYPT_PHASE24_DISPATCH_TIMEOUT_S") or "30"),
            healthcheck_path=_read_env("CLYPT_PHASE24_DISPATCH_HEALTHCHECK_PATH") or "/health",
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
        temperature=float(_read_env("CLYPT_LOCAL_LLM_TEMPERATURE") or "0.0"),
        top_p=float(_read_env("CLYPT_LOCAL_LLM_TOP_P") or "1.0"),
        top_k=max(1, int(_read_env("CLYPT_LOCAL_LLM_TOP_K") or "40")),
        min_p=max(0.0, float(_read_env("CLYPT_LOCAL_LLM_MIN_P") or "0.0")),
        presence_penalty=float(_read_env("CLYPT_LOCAL_LLM_PRESENCE_PENALTY") or "0.0"),
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
            generation_model=_read_env("GENAI_GENERATION_MODEL") or "Qwen/Qwen3.6-35B-A3B",
            embedding_model=_read_env("VERTEX_EMBEDDING_MODEL") or "gemini-embedding-2-preview",
            flash_model=_read_env("GENAI_FLASH_MODEL") or "Qwen/Qwen3.6-35B-A3B",
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
        vibevoice_asr_service=vibevoice_asr_service,
        node_media_prep=node_media_prep,
        phase1_visual_service=phase1_visual_service,
        phase26_dispatch_service=phase26_dispatch_service,
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
        phase1_runtime=Phase1RuntimeSettings(
            working_root=Path(
                _read_env("CLYPT_PHASE1_WORK_ROOT") or "backend/outputs/v3_1_phase1_work"
            ),
            run_yamnet_on_gpu=(_read_env("CLYPT_PHASE1_YAMNET_DEVICE") or "cpu").lower()
            == "gpu",
            input_mode=(_read_env("CLYPT_PHASE1_INPUT_MODE") or "test_bank").strip().lower(),
            test_bank_path=_read_env("CLYPT_PHASE1_TEST_BANK_PATH"),
            test_bank_strict=(_read_env("CLYPT_PHASE1_TEST_BANK_STRICT") or "1") == "1",
        ),
    )


def load_audio_host_settings() -> AudioHostProcessSettings:
    """Load the subset of provider settings that the Phase 1 VibeVoice service needs.

    Unlike ``load_provider_settings``, this does NOT require the runner-caller
    envs (``CLYPT_PHASE1_AUDIO_HOST_URL``, ``CLYPT_PHASE24_NODE_MEDIA_PREP_URL``,
    their tokens, Vertex/Spanner/Phase 2-4 worker settings, etc.) because the
    service process is the endpoint being called — it does not need to configure
    itself as its own remote client.
    """
    _load_local_env_files()
    _raise_if_removed_local_generation_env_present()

    gcs_bucket = _read_env("GCS_BUCKET", "CLYPT_GCS_BUCKET")
    if not gcs_bucket:
        raise ValueError(
            "GCS_BUCKET or CLYPT_GCS_BUCKET is required on the Phase 1 VibeVoice service "
            "for Phase 1 asset URL resolution."
        )

    hotwords_context = _normalize_hotwords_context(_read_env("VIBEVOICE_HOTWORDS_CONTEXT"))
    backend = (_read_env("VIBEVOICE_BACKEND") or "vllm").lower()
    if backend != "vllm":
        raise ValueError(
            f"Unsupported VIBEVOICE_BACKEND={backend!r}; only 'vllm' is supported."
        )

    vllm_base_url = _read_env("VIBEVOICE_VLLM_BASE_URL")
    if not vllm_base_url:
        raise ValueError(
            "VIBEVOICE_VLLM_BASE_URL is required on the Phase 1 VibeVoice service "
            "(co-located vLLM sidecar, typically http://127.0.0.1:8000)."
        )
    vllm_settings = VibeVoiceVLLMSettings(
        base_url=vllm_base_url,
        model=_read_env("VIBEVOICE_VLLM_MODEL") or "vibevoice",
        timeout_s=float(_read_env("VIBEVOICE_VLLM_TIMEOUT_S") or "7200"),
        healthcheck_path=_read_env("VIBEVOICE_VLLM_HEALTHCHECK_PATH") or "/health",
        max_retries=int(_read_env("VIBEVOICE_VLLM_MAX_RETRIES") or "1"),
        audio_mode=_read_env("VIBEVOICE_VLLM_AUDIO_MODE") or "url",
    )

    vibevoice_settings = VibeVoiceSettings(
        hotwords_context=hotwords_context,
        max_new_tokens=int(_read_env("VIBEVOICE_MAX_NEW_TOKENS") or "32768"),
        do_sample=_read_bool_env("VIBEVOICE_DO_SAMPLE", default=False),
        temperature=float(_read_env("VIBEVOICE_TEMPERATURE") or "0"),
        top_p=float(_read_env("VIBEVOICE_TOP_P") or "1.0"),
        repetition_penalty=float(_read_env("VIBEVOICE_REPETITION_PENALTY") or "1.03"),
        num_beams=_read_int_env("VIBEVOICE_NUM_BEAMS", default=1),
    )

    phase1_runtime = Phase1RuntimeSettings(
        working_root=Path(
            _read_env("CLYPT_PHASE1_WORK_ROOT") or "backend/outputs/v3_1_phase1_work"
        ),
        run_yamnet_on_gpu=(_read_env("CLYPT_PHASE1_YAMNET_DEVICE") or "cpu").lower()
        == "gpu",
        input_mode=(_read_env("CLYPT_PHASE1_INPUT_MODE") or "test_bank").strip().lower(),
        test_bank_path=_read_env("CLYPT_PHASE1_TEST_BANK_PATH"),
        test_bank_strict=(_read_env("CLYPT_PHASE1_TEST_BANK_STRICT") or "1") == "1",
    )

    return AudioHostProcessSettings(
        vibevoice=vibevoice_settings,
        vllm_vibevoice=vllm_settings,
        storage=StorageSettings(gcs_bucket=gcs_bucket),
        phase1_runtime=phase1_runtime,
    )


def load_phase1_host_settings() -> ProviderSettings:
    return load_provider_settings(
        require_vibevoice_asr_service=True,
        require_node_media_prep=False,
        require_phase1_visual_service=True,
        require_phase26_dispatch_service=True,
    )


def load_phase26_host_settings() -> ProviderSettings:
    return load_provider_settings(
        require_vibevoice_asr_service=False,
        require_node_media_prep=True,
        require_phase1_visual_service=False,
        require_phase26_dispatch_service=False,
    )


__all__ = [
    "AudioHostProcessSettings",
    "AudioHostSettings",  # deprecated alias of VibeVoiceAsrServiceSettings
    "LocalGenerationSettings",
    "NodeMediaPrepSettings",
    "Phase1VisualServiceSettings",
    "Phase26DispatchServiceSettings",
    "Phase24LocalQueueSettings",
    "Phase1RuntimeSettings",
    "Phase1ASRSettings",
    "ProviderSettings",
    "StorageSettings",
    "SpannerSettings",
    "Phase24WorkerSettings",
    "VertexSettings",
    "VibeVoiceAsrServiceSettings",
    "VibeVoiceSettings",
    "VibeVoiceVLLMSettings",
    "load_audio_host_settings",
    "load_phase1_host_settings",
    "load_phase26_host_settings",
    "load_provider_settings",
]
