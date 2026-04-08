from __future__ import annotations


def validate_torchaudio_runtime() -> dict[str, str]:
    """Validate that torchaudio.info is available in the worker environment."""
    try:
        import torchaudio
    except ImportError as exc:
        raise RuntimeError("torchaudio is required in the main worker environment.") from exc

    info_fn = getattr(torchaudio, "info", None)
    if not callable(info_fn):
        version = getattr(torchaudio, "__version__", "unknown")
        raise RuntimeError(
            "torchaudio.info is required in the main worker environment; "
            f"found torchaudio {version!s} without a callable info() API."
        )

    return {"torchaudio_version": str(getattr(torchaudio, "__version__", "unknown"))}


__all__ = ["validate_torchaudio_runtime"]
