"""
Run inside the **native VibeVoice venv only** (transformers<5 + vibevoice package).

Invoked as: python -m backend.runtime.vibevoice_native_worker
Reads one JSON object from stdin; writes one JSON object to stdout.

Job schema:
{
  "audio_path": "/abs/path.wav",
  "model_path": "microsoft/VibeVoice-ASR",
  "context_info": "optional hotwords string",
  "max_new_tokens": 32768,
  "do_sample": false,
  "temperature": 0.0,
  "top_p": 1.0,
  "repetition_penalty": 1.0,
  "num_beams": 1,
  "attn_implementation": "flash_attention_2"
}
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [vibevoice-native] %(message)s",
        stream=sys.stderr,
    )


def _maybe_apply_liger_kernel() -> None:
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2

        apply_liger_kernel_to_qwen2(
            rope=True,
            rms_norm=True,
            swiglu=True,
            cross_entropy=False,
        )
        logger.info("Liger kernel applied to Qwen2 (RoPE, RMSNorm, SwiGLU)")
    except Exception as exc:
        logger.warning("Liger kernel not applied: %s", exc)


def _load_model(
    model_path: str,
    device: str,
    attn_implementation: str,
):
    import torch
    from vibevoice.modular.modeling_vibevoice_asr import (
        VibeVoiceASRForConditionalGeneration,
    )
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

    processor = VibeVoiceASRProcessor.from_pretrained(model_path)

    def _load(attn: str):
        return VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=None,
            attn_implementation=attn,
            trust_remote_code=True,
        )

    try:
        model = _load(attn_implementation)
    except Exception as exc:
        if attn_implementation == "flash_attention_2":
            logger.warning("flash_attention_2 failed (%s); retrying sdpa", exc)
            model = _load("sdpa")
            attn_implementation = "sdpa"
        else:
            raise

    if device != "auto":
        model = model.to(device)
    model.eval()
    dev = next(model.parameters()).device
    logger.info("Model loaded (attn=%s, device=%s)", attn_implementation, dev)
    return processor, model, attn_implementation, dev


def _transcribe(
    *,
    processor,
    model,
    device,
    audio_path: str,
    context_info: str | None,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    num_beams: int,
) -> tuple[list[dict[str, Any]], str]:
    import torch

    inputs = processor(
        audio=audio_path,
        sampling_rate=None,
        return_tensors="pt",
        add_generation_prompt=True,
        context_info=context_info or None,
    )
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    generation_config: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p if do_sample else None,
        "do_sample": do_sample,
        "num_beams": num_beams,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": processor.pad_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }
    generation_config = {k: v for k, v in generation_config.items() if v is not None}

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_config)
    logger.info("generate() finished in %.1f s", time.perf_counter() - t0)

    in_len = int(inputs["input_ids"].shape[1])
    generated_ids = output_ids[0, in_len:]
    generated_text = processor.decode(generated_ids, skip_special_tokens=True)

    try:
        segments = processor.post_process_transcription(generated_text)
    except Exception as exc:
        logger.warning("post_process_transcription failed: %s", exc)
        segments = []

    turns = _segments_to_turns(segments)
    return turns, generated_text


def _segments_to_turns(segments: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        st = seg.get("start_time")
        if st is None:
            st = seg.get("Start time", seg.get("Start", 0.0))
        en = seg.get("end_time")
        if en is None:
            en = seg.get("End time", seg.get("End", 0.0))
        sp = seg.get("speaker_id")
        if sp is None:
            sp = seg.get("Speaker ID", seg.get("Speaker", 0))
        tx = seg.get("text")
        if tx is None:
            tx = seg.get("Content", "")
        out.append(
            {
                "Start": float(st or 0.0),
                "End": float(en or 0.0),
                "Speaker": _coerce_speaker_id(sp),
                "Content": str(tx or "").strip(),
            }
        )
    return out


def _coerce_speaker_id(val: Any) -> int:
    if val is None:
        return 0
    if isinstance(val, int):
        return val
    s = str(val).strip()
    up = s.upper()
    if up.startswith("SPEAKER_"):
        try:
            return int(s.split("_", 1)[1])
        except (ValueError, IndexError):
            return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    import torch

    audio_path = job.get("audio_path")
    if not audio_path:
        return {"turns": [], "error": "missing audio_path", "raw_text": ""}
    p = Path(audio_path)
    if not p.is_file():
        return {"turns": [], "error": f"audio not found: {audio_path}", "raw_text": ""}

    model_path = job.get("model_path") or "microsoft/VibeVoice-ASR"
    context_info = job.get("context_info")
    max_new_tokens = int(job.get("max_new_tokens") or 32768)
    do_sample = bool(job.get("do_sample", False))
    temperature = float(job.get("temperature", 0.0))
    top_p = float(job.get("top_p", 1.0))
    repetition_penalty = float(job.get("repetition_penalty", 1.0))
    num_beams = int(job.get("num_beams") or 1)
    attn_implementation = str(job.get("attn_implementation") or "flash_attention_2")

    device_s = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "loading model %s (attn=%s, device=%s) ...",
        model_path, attn_implementation, device_s,
    )
    t0 = time.perf_counter()
    _maybe_apply_liger_kernel()
    processor, model, attn_used, _ = _load_model(
        model_path, device_s, attn_implementation,
    )
    logger.info("load complete in %.1f s (attn=%s)", time.perf_counter() - t0, attn_used)

    device = next(model.parameters()).device
    turns, raw_text = _transcribe(
        processor=processor,
        model=model,
        device=device,
        audio_path=str(p.resolve()),
        context_info=context_info if isinstance(context_info, str) else None,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        num_beams=num_beams,
    )
    return {
        "turns": turns,
        "error": None,
        "raw_text": raw_text,
        "attn_implementation": attn_used,
    }


def main() -> None:
    _configure_logging()
    raw = sys.stdin.read()
    if not raw.strip():
        print(json.dumps({"turns": [], "error": "empty stdin"}))
        sys.exit(1)
    try:
        job = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"turns": [], "error": f"invalid json: {e}"}))
        sys.exit(1)

    try:
        result = run_job(job)
    except Exception as e:
        logger.exception("native worker failed")
        print(json.dumps({"turns": [], "error": str(e), "raw_text": ""}))
        sys.exit(1)

    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result.get("error") is None else 2)


if __name__ == "__main__":
    main()
