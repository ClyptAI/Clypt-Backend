from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def validate_torchaudio_runtime() -> dict[str, str]:
    """Validate the main-worker torchaudio install before Phase 1 starts.

    The native VibeVoice model runs in its own venv, but the main worker still
    probes WAV metadata via ``torchaudio.info`` before launching the subprocess.
    Fail fast during deploy if that API is missing from the worker venv.
    """
    try:
        import torchaudio
    except ImportError as exc:
        raise RuntimeError(
            "torchaudio is required in the main worker environment."
        ) from exc

    info_fn = getattr(torchaudio, "info", None)
    if not callable(info_fn):
        version = getattr(torchaudio, "__version__", "unknown")
        raise RuntimeError(
            f"torchaudio.info is required in the main worker environment; "
            f"found torchaudio {version!s} without a callable info() API."
        )

    return {
        "torchaudio_version": str(getattr(torchaudio, "__version__", "unknown")),
    }


def _probe_audio_duration_s(audio_path: Path) -> float:
    """
    Wall-clock duration for logging / RTF using ``torchaudio`` only.

    We intentionally do not fall back to :mod:`wave`: on some WAV variants it can report
    garbage ``nframes`` metadata (we observed ``2**30 - 1`` on `joeroganxflagrant.wav`,
    which logged as ~6.8 h even though `torchaudio` reported ~13.1 min). We want the log
    probe to match the actual decode stack or fail fast.
    """
    path_str = str(audio_path)
    try:
        import torchaudio
    except ImportError as exc:
        raise RuntimeError(
            "torchaudio is required to probe audio duration for VibeVoice input."
        ) from exc

    try:
        info = torchaudio.info(path_str)
    except Exception as exc:
        raise RuntimeError(
            f"[vibevoice] torchaudio.info failed for {audio_path}: {exc}"
        ) from exc

    sr = int(info.sample_rate)
    nframes = int(info.num_frames)
    ch = int(getattr(info, "num_channels", 0) or 0)
    if sr <= 0 or nframes < 0:
        raise RuntimeError(
            f"[vibevoice] invalid torchaudio metadata for {audio_path}: "
            f"sample_rate={sr} num_frames={nframes}"
        )

    duration = float(nframes) / float(sr)
    logger.debug(
        "[vibevoice] torchaudio.info: %s sr=%d frames=%d ch=%d -> %.3f s",
        audio_path.name,
        sr,
        nframes,
        ch,
        duration,
    )
    if duration > 48 * 3600:
        raise RuntimeError(
            f"[vibevoice] refusing implausible torchaudio duration for {audio_path}: "
            f"{duration / 3600.0:.1f} h"
        )
    return duration


class VibeVoiceASRProvider:
    """
    VibeVoice-ASR for diarization + transcription.

    - ``backend=native`` (default): subprocess using ``VIBEVOICE_NATIVE_VENV_PYTHON`` and
      :mod:`backend.runtime.vibevoice_native_worker` (Microsoft ``vibevoice`` package,
      ``microsoft/VibeVoice-ASR`` ~7B, Gradio-parity inference).
    - ``backend=hf``: in-process Hugging Face ``microsoft/VibeVoice-ASR-HF`` (~9B).

    Outputs: ``[{"Start": float, "End": float, "Speaker": int, "Content": str}, ...]``
    """

    def __init__(
        self,
        *,
        backend: str = "native",
        native_venv_python: str | None = None,
        model_id: str = "microsoft/VibeVoice-ASR",
        flash_attention: bool = True,
        liger_kernel: bool = True,
        hotwords_context: str | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int = 32768,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_beams: int = 1,
        attn_implementation: str = "flash_attention_2",
        subprocess_timeout_s: int = 7200,
    ) -> None:
        self.backend = (backend or "native").lower()
        if self.backend not in ("native", "hf"):
            self.backend = "native"
        self.native_venv_python = (native_venv_python or "").strip() or None
        self.model_id = model_id
        self.flash_attention = flash_attention
        self.liger_kernel = liger_kernel
        self.hotwords_context = hotwords_context if hotwords_context is not None else _DEFAULT_HOTWORDS
        self._system_prompt = system_prompt  # HF only; None = chat template default
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.num_beams = num_beams
        self.attn_implementation = attn_implementation
        self.subprocess_timeout_s = subprocess_timeout_s

        self._model = None
        self._processor = None
        self._attn_impl: str | None = None

    def load(self) -> None:
        if self.backend == "native":
            self._ensure_native_venv()
            return
        self._load_hf_model()

    def _ensure_native_venv(self) -> None:
        if not self.native_venv_python:
            raise RuntimeError(
                "VIBEVOICE_NATIVE_VENV_PYTHON is required when VIBEVOICE_BACKEND=native. "
                "Create the venv per requirements-vibevoice-native.txt and set the path to "
                "that venv's python executable."
            )
        p = Path(self.native_venv_python)
        if not p.is_file():
            raise RuntimeError(
                f"VIBEVOICE_NATIVE_VENV_PYTHON does not exist: {self.native_venv_python}"
            )
        logger.info("[vibevoice] native backend — venv python=%s", self.native_venv_python)

    def _maybe_apply_liger_kernel(self) -> None:
        if not self.liger_kernel:
            return
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2

            apply_liger_kernel_to_qwen2(
                rope=True,
                rms_norm=True,
                swiglu=True,
                cross_entropy=False,
            )
            logger.info("[vibevoice] Liger kernel applied to Qwen2 (RoPE, RMSNorm, SwiGLU)")
        except Exception as exc:
            logger.warning("[vibevoice] Liger kernel not applied: %s", exc)

    def _load_hf_model(self) -> None:
        self._maybe_apply_liger_kernel()

        logger.info("[vibevoice] loading HF model %s ...", self.model_id)
        t0 = time.perf_counter()
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoProcessor,
                VibeVoiceAsrForConditionalGeneration,
            )
        except ImportError as e:
            raise RuntimeError(
                "transformers>=5.3.0 is required for VibeVoice HF. "
                "Run: pip install 'transformers>=5.3.0'"
            ) from e

        self._processor = AutoProcessor.from_pretrained(self.model_id)

        attn_impl = "flash_attention_2" if (self.flash_attention and torch.cuda.is_available()) else "sdpa"
        config = AutoConfig.from_pretrained(self.model_id)

        def _load(text_attn: str):
            config.text_config._attn_implementation_internal = text_attn
            return VibeVoiceAsrForConditionalGeneration.from_pretrained(
                self.model_id,
                config=config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        try:
            self._model = _load(attn_impl)
        except Exception as exc:
            if attn_impl == "flash_attention_2":
                logger.warning(
                    "[vibevoice] flash_attention_2 failed (%s); falling back to sdpa", exc,
                )
                attn_impl = "sdpa"
                self._model = _load(attn_impl)
            else:
                raise

        self._model.eval()
        self._attn_impl = attn_impl
        logger.info(
            "[vibevoice] HF model loaded in %.1f s (attn=%s, device=%s)",
            time.perf_counter() - t0,
            attn_impl,
            getattr(self._model, "device", "auto"),
        )

    def run(
        self,
        audio_path: str | Path,
        context_info: str | None = None,
    ) -> list[dict[str, Any]]:
        context = context_info if context_info is not None else self.hotwords_context
        audio_path = Path(audio_path)

        audio_duration_s = _probe_audio_duration_s(audio_path)

        backend_tag = f"native-subprocess:{self.model_id}" if self.backend == "native" else "hf-transformers"
        logger.info(
            "[vibevoice] running ASR (%s) on %s (%.1f s audio, context=%d chars) ...",
            backend_tag, audio_path.name, audio_duration_s, len(context),
        )
        t0 = time.perf_counter()

        if self.backend == "native":
            turns = self._run_native_subprocess(audio_path, context)
        else:
            turns = self._run_hf(audio_path, context)

        elapsed = time.perf_counter() - t0
        if audio_duration_s > 0:
            rtf = elapsed / audio_duration_s
            logger.info(
                "[vibevoice] done in %.1f s — %d turns  (RTF %.2fx)",
                elapsed, len(turns), rtf,
            )
        else:
            logger.info("[vibevoice] done in %.1f s — %d turns", elapsed, len(turns))
        return turns

    def _run_native_subprocess(self, audio_path: Path, context: str) -> list[dict[str, Any]]:
        self._ensure_native_venv()
        root = _repo_root()
        job = {
            "audio_path": str(audio_path.resolve()),
            "model_path": self.model_id,
            "context_info": context or None,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "num_beams": self.num_beams,
            "attn_implementation": self.attn_implementation,
        }
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        cmd = [self.native_venv_python, "-m", "backend.runtime.vibevoice_native_worker"]
        logger.info("[vibevoice] spawning native worker: %s", " ".join(cmd))

        stdout_chunks: list[str] = []
        stderr_lines: list[str] = []

        def _drain_stdout(pipe: Any) -> None:
            if pipe is None:
                return
            data = pipe.read()
            if data:
                stdout_chunks.append(data)

        def _drain_stderr(pipe: Any) -> None:
            if pipe is None:
                return
            for line in pipe:
                cleaned = line.rstrip()
                if not cleaned:
                    continue
                stderr_lines.append(cleaned)
                logger.info("[vibevoice-native] %s", cleaned)

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=str(root),
                env=env,
            )
        except Exception as exc:
            raise RuntimeError(f"[vibevoice] failed to spawn native worker: {exc}") from exc

        stdout_thread = threading.Thread(
            target=_drain_stdout,
            args=(proc.stdout,),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_drain_stderr,
            args=(proc.stderr,),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        try:
            if proc.stdin is None:
                raise RuntimeError("[vibevoice] native worker missing stdin pipe")
            proc.stdin.write(json.dumps(job))
            proc.stdin.close()
            proc.wait(timeout=float(self.subprocess_timeout_s))
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            raise RuntimeError(
                f"[vibevoice] native worker exceeded {self.subprocess_timeout_s}s timeout"
            ) from None
        finally:
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)

        stdout_text = "".join(stdout_chunks)
        stderr_text = "\n".join(stderr_lines)

        if proc.returncode not in (0, 2):
            err = stderr_text + "\n" + stdout_text
            raise RuntimeError(
                f"[vibevoice] native worker failed (code={proc.returncode}): {err[-4000:]}"
            )

        try:
            payload = json.loads(stdout_text.strip())
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"[vibevoice] invalid JSON from native worker: {e}\n{stdout_text[:2000]}"
            ) from e

        err = payload.get("error")
        if err:
            raise RuntimeError(f"[vibevoice] native worker error: {err}")

        turns = payload.get("turns") or []
        return self._normalize_turns(turns)

    def _run_hf(self, audio_path: Path, context: str) -> list[dict[str, Any]]:
        if self._model is None or self._processor is None:
            self._load_hf_model()

        import torch
        import torchaudio

        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        audio_np = waveform.squeeze(0).numpy()

        prompt = context or None
        sp = self._system_prompt
        inputs = self._processor.apply_transcription_request(
            audio=audio_np,
            prompt=prompt,
            system_prompt=sp if sp else None,
        )
        inputs = inputs.to(self._model.device, self._model.dtype)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs)

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        raw = self._processor.decode(generated_ids, return_format="parsed")
        turns_raw = raw[0] if raw else []
        return self._normalize_turns(turns_raw)

    def _normalize_turns(self, raw_turns: list[Any]) -> list[dict[str, Any]]:
        normalized = []
        for item in raw_turns:
            if not isinstance(item, dict):
                continue
            sp = item.get("Speaker")
            if sp is None:
                sp = item.get("speaker") or item.get("speaker_id")
            normalized.append(
                {
                    "Start": float(
                        item.get("Start")
                        or item.get("start")
                        or item.get("start_time")
                        or 0.0
                    ),
                    "End": float(
                        item.get("End")
                        or item.get("end")
                        or item.get("end_time")
                        or 0.0
                    ),
                    "Speaker": int(sp or 0),
                    "Content": str(
                        item.get("Content")
                        or item.get("content")
                        or item.get("text")
                        or ""
                    ).strip(),
                }
            )
        return normalized

    def teardown(self) -> None:
        self._model = None
        self._processor = None


__all__ = ["VibeVoiceASRProvider"]
