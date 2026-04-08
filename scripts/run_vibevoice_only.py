#!/usr/bin/env python3
"""
Run VibeVoice ASR only (no visual / NFA / emotion / YAMNet).

Uses the same env as Phase 1: load_provider_settings() + VibeVoiceVLLMProvider.

Examples (GPU droplet, repo at /opt/clypt-phase1/repo):

  cd /opt/clypt-phase1/repo
  source .venv/bin/activate
  export PYTHONPATH=.
  python scripts/run_vibevoice_only.py --audio /tmp/joeroganxflagrant.wav

Local one-liner after scp:

  PYTHONPATH=. .venv/bin/python scripts/run_vibevoice_only.py --audio /path/to/file.wav
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clypt.vibevoice_only")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VibeVoice ASR only — reads .env.local / env for VIBEVOICE_* settings.",
    )
    parser.add_argument(
        "--audio",
        required=True,
        type=Path,
        help="Path to WAV/MP3/MP4/audio-video input accepted by the vLLM path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Optional path to write VibeVoice JSON. "
            "Shape depends on VIBEVOICE_OUTPUT_MODE "
            "(turns: Start/End/Speaker/Content, words: start_ms/end_ms/speaker_id/word)."
        ),
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent for stdout (default: 2).",
    )
    args = parser.parse_args()

    audio = args.audio.resolve()
    if not audio.is_file():
        logger.error("Audio file not found: %s", audio)
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from backend.providers.config import load_provider_settings
    from backend.providers.vibevoice_vllm import VibeVoiceVLLMProvider

    settings = load_provider_settings()
    v = settings.vibevoice
    vv = settings.vllm_vibevoice
    provider = VibeVoiceVLLMProvider(
        base_url=vv.base_url,
        model=vv.model,
        timeout_s=vv.timeout_s,
        healthcheck_path=vv.healthcheck_path,
        max_retries=vv.max_retries,
        audio_mode=vv.audio_mode,
        hotwords_context=v.hotwords_context,
        output_mode=v.output_mode,
        word_turn_gap_ms=v.word_turn_gap_ms,
        max_new_tokens=v.max_new_tokens,
        do_sample=v.do_sample,
        temperature=v.temperature,
        top_p=v.top_p,
        repetition_penalty=v.repetition_penalty,
        num_beams=v.num_beams,
    )
    logger.info(
        "backend=vllm model=%s url=%s output_mode=%s audio=%s",
        vv.model,
        vv.base_url,
        v.output_mode,
        audio,
    )
    t0 = time.perf_counter()
    provider.load()
    outputs = provider.run(audio_path=audio)
    elapsed = time.perf_counter() - t0
    item_kind = "words" if v.output_mode == "words" else "turns"
    logger.info("Done in %.1f s — %d %s", elapsed, len(outputs), item_kind)

    text = json.dumps(outputs, indent=args.indent if args.indent > 0 else None, ensure_ascii=True)
    print(text)
    if args.output_json:
        args.output_json.write_text(text + "\n", encoding="utf-8")
        logger.info("Wrote %s", args.output_json.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
