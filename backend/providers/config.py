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
    """VibeVoice ASR: native ~7B subprocess (Gradio parity) or in-process HF 9B."""

    backend: str = "native"
    native_venv_python: str = ""
    model_id: str = "microsoft/VibeVoice-ASR"
    flash_attention: bool = True
    liger_kernel: bool = True
    hotwords_context: str = _DEFAULT_HOTWORDS
    system_prompt: str = ""
    max_new_tokens: int = 32768
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    num_beams: int = 1
    attn_implementation: str = "flash_attention_2"
    subprocess_timeout_s: int = 7200


@dataclass(slots=True)
class VibeVoiceVLLMSettings:
    """Settings for the persistent vLLM VibeVoice ASR sidecar service."""

    base_url: str
    model: str = "microsoft/VibeVoice-ASR"
    timeout_s: float = 7200.0
    healthcheck_path: str = "/health"
    max_retries: int = 1
    audio_mode: str = "base64"


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
    vibevoice: VibeVoiceSettings
    vertex: VertexSettings
    storage: StorageSettings
    phase1_runtime: Phase1RuntimeSettings = field(default_factory=Phase1RuntimeSettings)
    vllm_vibevoice: VibeVoiceVLLMSettings | None = None


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
    backend = (_read_env("VIBEVOICE_BACKEND") or "native").lower()
    if backend not in ("native", "hf", "vllm"):
        raise ValueError(
            f"Unknown VIBEVOICE_BACKEND={backend!r}; valid values: native, hf, vllm"
        )

    model_default = (
        "microsoft/VibeVoice-ASR-HF" if backend == "hf" else "microsoft/VibeVoice-ASR"
    )

    # vLLM backend: build VibeVoiceVLLMSettings and skip native-subprocess validation.
    vllm_settings: VibeVoiceVLLMSettings | None = None
    if backend == "vllm":
        vllm_base_url = _read_env("VIBEVOICE_VLLM_BASE_URL")
        if not vllm_base_url:
            raise ValueError(
                "VIBEVOICE_VLLM_BASE_URL is required when VIBEVOICE_BACKEND=vllm."
            )
        vllm_settings = VibeVoiceVLLMSettings(
            base_url=vllm_base_url,
            model=_read_env("VIBEVOICE_VLLM_MODEL") or model_default,
            timeout_s=float(_read_env("VIBEVOICE_VLLM_TIMEOUT_S") or "7200"),
            healthcheck_path=_read_env("VIBEVOICE_VLLM_HEALTHCHECK_PATH") or "/health",
            max_retries=int(_read_env("VIBEVOICE_VLLM_MAX_RETRIES") or "1"),
            audio_mode=_read_env("VIBEVOICE_VLLM_AUDIO_MODE") or "base64",
        )

    return ProviderSettings(
        vibevoice=VibeVoiceSettings(
            backend=backend,
            # native_venv_python not required when backend=vllm
            native_venv_python=_read_env("VIBEVOICE_NATIVE_VENV_PYTHON") or "",
            model_id=_read_env("VIBEVOICE_MODEL_ID") or model_default,
            flash_attention=(_read_env("VIBEVOICE_FLASH_ATTN") or "1") == "1",
            liger_kernel=(_read_env("VIBEVOICE_LIGER") or "1") == "1",
            hotwords_context=hotwords_context,
            system_prompt=_read_env("VIBEVOICE_SYSTEM_PROMPT") or "",
            max_new_tokens=int(_read_env("VIBEVOICE_MAX_NEW_TOKENS") or "32768"),
            do_sample=(_read_env("VIBEVOICE_DO_SAMPLE") or "0") == "1",
            temperature=float(_read_env("VIBEVOICE_TEMPERATURE") or "0"),
            top_p=float(_read_env("VIBEVOICE_TOP_P") or "1"),
            repetition_penalty=float(_read_env("VIBEVOICE_REPETITION_PENALTY") or "1"),
            num_beams=int(_read_env("VIBEVOICE_NUM_BEAMS") or "1"),
            attn_implementation=_read_env("VIBEVOICE_ATTN_IMPLEMENTATION")
            or "flash_attention_2",
            subprocess_timeout_s=int(_read_env("VIBEVOICE_SUBPROCESS_TIMEOUT_S") or "7200"),
        ),
        vllm_vibevoice=vllm_settings,
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
    "StorageSettings",
    "VertexSettings",
    "VibeVoiceSettings",
    "VibeVoiceVLLMSettings",
    "load_provider_settings",
]
