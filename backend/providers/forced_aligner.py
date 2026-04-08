from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_NFA_DEFAULT_MODEL = "stt_en_fastconformer_hybrid_large_pc"
_DEFAULT_PHASE1_CACHE_HOME = "/opt/clypt-phase1/.cache"
_DEFAULT_MODEL_LOAD_RETRIES = 2
_DEFAULT_MODEL_LOAD_RETRY_BACKOFF_S = 5.0


def _ensure_cache_env() -> None:
    """
    Ensure model caches resolve to a deterministic path across prewarm, API,
    and worker runtime contexts.
    """
    cache_home = (
        os.getenv("CLYPT_PHASE1_CACHE_HOME")
        or os.getenv("XDG_CACHE_HOME")
        or (
            os.path.join(os.getenv("HOME", ""), ".cache")
            if os.getenv("HOME")
            else None
        )
        or _DEFAULT_PHASE1_CACHE_HOME
    )

    os.environ.setdefault("CLYPT_PHASE1_CACHE_HOME", cache_home)
    os.environ.setdefault("XDG_CACHE_HOME", cache_home)
    os.environ.setdefault("TORCH_HOME", os.path.join(cache_home, "torch"))
    os.environ.setdefault("HF_HOME", os.path.join(cache_home, "huggingface"))
    os.environ.setdefault("MODELSCOPE_CACHE", os.path.join(cache_home, "modelscope"))

    for path in (
        os.environ["XDG_CACHE_HOME"],
        os.environ["TORCH_HOME"],
        os.environ["HF_HOME"],
        os.environ["MODELSCOPE_CACHE"],
    ):
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            # Best-effort only: if dir creation fails, NeMo/modelscope will still
            # raise explicit errors later that include the actual root cause.
            pass


def _patch_hf_hub_compat() -> None:
    """
    NeMo 1.x imports legacy symbols from `huggingface_hub` that are removed in
    newer releases. Add minimal shims so NeMo imports remain stable.
    """
    try:
        import huggingface_hub as hf_hub
    except Exception:
        return

    if not hasattr(hf_hub, "HfFolder"):
        class _HfFolder:
            @staticmethod
            def get_token():
                getter = getattr(hf_hub, "get_token", None)
                return getter() if callable(getter) else None

        hf_hub.HfFolder = _HfFolder

    if not hasattr(hf_hub, "ModelFilter"):
        class _ModelFilter:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        hf_hub.ModelFilter = _ModelFilter


def _patch_numpy_compat() -> None:
    """
    NeMo 1.x (or transitive deps) may still reference `np.sctypes`, removed in
    NumPy 2. Recreate the minimal mapping expected by legacy code.
    """
    try:
        import numpy as np
    except Exception:
        return

    if hasattr(np, "sctypes"):
        return

    np.sctypes = {  # type: ignore[attr-defined]
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, np.object_, np.str_, np.bytes_],
    }


class ForcedAlignmentProvider:
    """
    Uses NVIDIA NeMo Forced Aligner (NFA) to produce word-level timestamps
    for each VibeVoice turn.

    NFA uses a FastConformer CTC model with Viterbi decoding — production-grade
    alignment that is both accurate and stable on NVIDIA GPUs.

    Each word entry:
        {"word_id": str, "text": str, "start_ms": int, "end_ms": int, "speaker_id": str}

    If nemo_toolkit is not installed, logs a warning and returns an empty list
    so the pipeline degrades gracefully.
    """

    def __init__(
        self,
        *,
        model_name: str = _NFA_DEFAULT_MODEL,
        device: str = "auto",
        model_load_retries: int = _DEFAULT_MODEL_LOAD_RETRIES,
        model_load_retry_backoff_s: float = _DEFAULT_MODEL_LOAD_RETRY_BACKOFF_S,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._available: bool | None = None
        self._model_load_retries = max(1, int(model_load_retries))
        self._model_load_retry_backoff_s = max(0.0, float(model_load_retry_backoff_s))

    def _check_available(self) -> bool:
        if self._available is None:
            try:
                _ensure_cache_env()
                _patch_hf_hub_compat()
                _patch_numpy_compat()
                import soundfile  # noqa: F401
                from nemo.collections.asr.models.hybrid_rnnt_ctc_models import (  # noqa: F401
                    EncDecHybridRNNTCTCModel,
                )
                from nemo.collections.asr.parts.utils.transcribe_utils import (  # noqa: F401
                    setup_model,
                )
                from backend.providers.nfa_viterbi import (  # noqa: F401
                    add_t_start_end_to_utt_obj,
                    get_single_sample_batch_variables,
                    viterbi_decoding,
                )

                self._available = True
            except Exception as exc:
                logger.warning(
                    "[forced_aligner] NeMo aligner import failed (%s: %s) — "
                    "word-level timestamps will be empty.",
                    type(exc).__name__,
                    exc,
                )
                logger.warning(
                    "[forced_aligner] Ensure nemo-toolkit[asr] is installed and "
                    "its transitive deps (especially protobuf/wandb) are compatible."
                )
                self._available = False
        return self._available

    def _resolve_device(self) -> str:
        if self._device != "auto":
            return self._device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _ensure_model(self, device: str):
        """Lazy-load the NeMo ASR model on first use."""
        if self._model is not None:
            return

        _ensure_cache_env()

        last_exc: Exception | None = None
        for attempt in range(1, self._model_load_retries + 1):
            try:
                _patch_hf_hub_compat()
                _patch_numpy_compat()
                import torch
                from nemo.collections.asr.models.hybrid_rnnt_ctc_models import (
                    EncDecHybridRNNTCTCModel,
                )
                from nemo.collections.asr.parts.utils.transcribe_utils import setup_model
                from omegaconf import OmegaConf

                cfg = OmegaConf.create(
                    {"pretrained_name": self._model_name, "model_path": None}
                )
                map_location = torch.device(device)

                logger.info(
                    "[forced_aligner] loading NFA model '%s' on %s (attempt %d/%d) ...",
                    self._model_name,
                    device,
                    attempt,
                    self._model_load_retries,
                )
                t0 = time.perf_counter()
                model, _ = setup_model(cfg, map_location)
                model.eval()
                if isinstance(model, EncDecHybridRNNTCTCModel):
                    model.change_decoding_strategy(decoder_type="ctc")
                logger.info(
                    "[forced_aligner] model loaded in %.1f s",
                    time.perf_counter() - t0,
                )
                self._model = model
                return
            except Exception as exc:  # pragma: no cover - network/runtime dependent
                last_exc = exc
                if attempt >= self._model_load_retries:
                    break
                logger.warning(
                    "[forced_aligner] model load attempt %d/%d failed (%s: %s); retrying in %.1f s ...",
                    attempt,
                    self._model_load_retries,
                    type(exc).__name__,
                    exc,
                    self._model_load_retry_backoff_s,
                )
                time.sleep(self._model_load_retry_backoff_s)

        if last_exc is None:
            raise RuntimeError("NFA model load failed for unknown reason")
        raise last_exc

    def run(
        self,
        audio_path: str | Path,
        turns: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Align words to timing for each turn in *turns*.

        Args:
            audio_path: Path to the full audio file (WAV, 16kHz recommended).
            turns: List of turn dicts from vibevoice_merge.  Each turn must have
                   start_ms, end_ms, transcript_text, turn_id, speaker_id.

        Returns:
            List of word dicts:
                [{"word_id": str, "text": str, "start_ms": int, "end_ms": int, "speaker_id": str}]
        """
        if not turns:
            return []

        if not self._check_available():
            return []

        _ensure_cache_env()
        audio_path = Path(audio_path)
        device = self._resolve_device()

        logger.info(
            "[forced_aligner] aligning %d turns on %s (model=%s) ...",
            len(turns),
            device,
            self._model_name,
        )
        t0 = time.perf_counter()

        try:
            self._ensure_model(device)
        except Exception as e:
            logger.warning(
                "[forced_aligner] failed to load NFA model: %s — skipping", e
            )
            return []

        all_words: list[dict[str, Any]] = []
        global_word_idx = 1
        total_turns = len(turns)
        _log_every = max(1, total_turns // 10)

        try:
            import torch
            import torchaudio
            _patch_numpy_compat()
            from backend.providers.nfa_viterbi import (
                Segment,
                Word,
                add_t_start_end_to_utt_obj,
                get_single_sample_batch_variables,
                viterbi_decoding,
            )

            waveform, sr = torchaudio.load(str(audio_path))
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                sr = 16000
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            tmpdir = tempfile.mkdtemp(prefix="nfa_slices_")
            output_timestep_duration: float | None = None

            for turn_idx, turn in enumerate(turns, start=1):
                turn_id = turn.get("turn_id", "")
                speaker_id = str(turn.get("speaker_id") or "UNKNOWN")
                start_ms = int(turn.get("start_ms") or 0)
                end_ms = int(turn.get("end_ms") or 0)
                text = str(turn.get("transcript_text") or "").strip()

                if not text or end_ms <= start_ms:
                    pass
                else:
                    try:
                        start_sample = int(start_ms / 1000 * sr)
                        end_sample = int(end_ms / 1000 * sr)
                        segment_wav = waveform[:, start_sample:end_sample]

                        if segment_wav.shape[1] < 400:
                            pass
                        else:
                            slice_path = os.path.join(tmpdir, f"{turn_id}.wav")
                            torchaudio.save(slice_path, segment_wav, sr)

                            (
                                log_probs_batch,
                                y_batch,
                                T_batch,
                                U_batch,
                                utt_obj_batch,
                                output_timestep_duration,
                            ) = get_single_sample_batch_variables(
                                audio_filepath=slice_path,
                                text=text,
                                model=self._model,
                                output_timestep_duration=output_timestep_duration,
                            )

                            alignments_batch = viterbi_decoding(
                                log_probs_batch,
                                y_batch,
                                T_batch,
                                U_batch,
                                viterbi_device=torch.device(device),
                            )

                            utt_obj = add_t_start_end_to_utt_obj(
                                utt_obj_batch[0],
                                alignments_batch[0],
                                output_timestep_duration,
                            )

                            for seg_or_tok in utt_obj.segments_and_tokens:
                                if not isinstance(seg_or_tok, Segment):
                                    continue
                                for wt in seg_or_tok.words_and_tokens:
                                    if not isinstance(wt, Word):
                                        continue
                                    word_text = (wt.text or "").strip()
                                    if (
                                        not word_text
                                        or wt.t_start is None
                                        or wt.t_start < 0
                                    ):
                                        continue
                                    word_start_ms = start_ms + int(
                                        round(wt.t_start * 1000)
                                    )
                                    word_end_ms = start_ms + int(
                                        round(wt.t_end * 1000)
                                    )
                                    all_words.append(
                                        {
                                            "word_id": f"w_{global_word_idx:06d}",
                                            "text": word_text,
                                            "start_ms": word_start_ms,
                                            "end_ms": word_end_ms,
                                            "speaker_id": speaker_id,
                                        }
                                    )
                                    global_word_idx += 1

                            try:
                                os.unlink(slice_path)
                            except OSError:
                                pass

                    except Exception as e:
                        logger.warning(
                            "[forced_aligner] failed to align turn %s ('%s...'): %s",
                            turn_id,
                            text[:40],
                            e,
                        )

                if (
                    turn_idx == 1
                    or turn_idx % _log_every == 0
                    or turn_idx == total_turns
                ):
                    elapsed = time.perf_counter() - t0
                    logger.info(
                        "[forced_aligner] %d/%d turns  (%d words so far, %.1f s elapsed)",
                        turn_idx,
                        total_turns,
                        len(all_words),
                        elapsed,
                    )

            try:
                os.rmdir(tmpdir)
            except OSError:
                pass

        except Exception as e:
            logger.warning("[forced_aligner] alignment failed: %s — returning partial results", e)

        logger.info(
            "[forced_aligner] alignment done in %.1f s — %d words across %d turns",
            time.perf_counter() - t0,
            len(all_words),
            len(turns),
        )
        return all_words


__all__ = ["ForcedAlignmentProvider"]
