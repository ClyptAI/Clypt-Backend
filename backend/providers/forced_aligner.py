from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_NFA_DEFAULT_MODEL = "stt_en_fastconformer_hybrid_large_pc"


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
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._available: bool | None = None

    def _check_available(self) -> bool:
        if self._available is None:
            try:
                from nemo.collections.asr.parts.utils.aligner_utils import (  # noqa: F401
                    get_batch_variables,
                )

                self._available = True
            except ImportError:
                logger.warning(
                    "[forced_aligner] nemo_toolkit[asr] not installed — "
                    "word-level timestamps will be empty. "
                    "Install with: pip install 'nemo-toolkit[asr]'"
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
            "[forced_aligner] loading NFA model '%s' on %s ...",
            self._model_name,
            device,
        )
        t0 = time.perf_counter()
        model, _ = setup_model(cfg, map_location)
        model.eval()
        if isinstance(model, EncDecHybridRNNTCTCModel):
            model.change_decoding_strategy(decoder_type="ctc")
        logger.info(
            "[forced_aligner] model loaded in %.1f s", time.perf_counter() - t0
        )
        self._model = model

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
            from nemo.collections.asr.parts.utils.aligner_utils import (
                Segment,
                Word,
                add_t_start_end_to_utt_obj,
                get_batch_variables,
                viterbi_decoding,
            )

            waveform, sr = torchaudio.load(str(audio_path))
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                sr = 16000
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            tmpdir = tempfile.mkdtemp(prefix="nfa_slices_")

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

                            import inspect
                            _gbv_sig = inspect.signature(get_batch_variables)
                            _gbv_kwargs: dict[str, Any] = {
                                "audio": [slice_path],
                                "model": self._model,
                                "gt_text_batch": [text],
                                "align_using_pred_text": False,
                            }
                            if "verbose" in _gbv_sig.parameters:
                                _gbv_kwargs["verbose"] = False

                            (
                                log_probs_batch,
                                y_batch,
                                T_batch,
                                U_batch,
                                utt_obj_batch,
                                output_timestep_duration,
                            ) = get_batch_variables(**_gbv_kwargs)

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
