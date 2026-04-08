from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from backend.pipeline.timeline.vibevoice_merge import merge_vibevoice_outputs

from .models import Phase1SidecarOutputs, Phase1Workspace

logger = logging.getLogger(__name__)


def _is_forced_alignment_required() -> bool:
    return (os.getenv("CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT") or "1") == "1"


def _speaker_int_from_value(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        raw = value.strip()
        if raw.upper().startswith("SPEAKER_"):
            raw = raw.split("_", 1)[1]
        try:
            return int(raw)
        except ValueError:
            return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _normalize_vibevoice_words(vibevoice_words: list[dict]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(vibevoice_words, start=1):
        text = str(item.get("word") or item.get("text") or "").strip()
        if not text:
            continue
        try:
            start_ms = int(round(float(item.get("start_ms"))))
            end_ms = int(round(float(item.get("end_ms"))))
        except (TypeError, ValueError):
            continue
        if end_ms <= start_ms:
            continue
        speaker_int = _speaker_int_from_value(item.get("speaker_id"))
        normalized.append(
            {
                "word_id": f"w_{idx:06d}",
                "text": text,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "speaker_id": f"SPEAKER_{speaker_int}",
            }
        )
    normalized.sort(key=lambda w: (w["start_ms"], w["end_ms"]))
    return normalized


def _build_turns_from_words(
    *,
    word_alignments: list[dict[str, Any]],
    turn_gap_ms: int,
) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    if not word_alignments:
        return turns

    current_words: list[dict[str, Any]] = []
    current_speaker = ""

    def _flush() -> None:
        nonlocal current_words, current_speaker
        if not current_words:
            return
        start_ms = int(current_words[0]["start_ms"])
        end_ms = int(current_words[-1]["end_ms"])
        content = " ".join(str(w.get("text") or "").strip() for w in current_words).strip()
        turns.append(
            {
                "Start": start_ms / 1000.0,
                "End": end_ms / 1000.0,
                "Speaker": _speaker_int_from_value(current_speaker),
                "Content": content,
            }
        )
        current_words = []
        current_speaker = ""

    for word in sorted(word_alignments, key=lambda w: (w["start_ms"], w["end_ms"])):
        word_speaker = str(word.get("speaker_id") or "SPEAKER_0")
        if not current_words:
            current_words = [word]
            current_speaker = word_speaker
            continue

        prev_end = int(current_words[-1]["end_ms"])
        this_start = int(word["start_ms"])
        same_speaker = word_speaker == current_speaker
        close_gap = (this_start - prev_end) <= turn_gap_ms
        if same_speaker and close_gap:
            current_words.append(word)
            continue

        _flush()
        current_words = [word]
        current_speaker = word_speaker

    _flush()
    return turns


def _run_audio_chain(
    *,
    workspace: Phase1Workspace,
    vibevoice_output: list[dict],
    vibevoice_output_mode: str,
    vibevoice_word_turn_gap_ms: int,
    forced_aligner: Any,
    emotion_provider: Any,
    yamnet_provider: Any,
) -> tuple[dict, Any, Any]:
    """
    NFA → emotion2vec+ → YAMNet, always serial with each other to avoid
    CUDA graph conflicts between them.  Returns (diarization_payload,
    emotion2vec_payload, yamnet_payload).

    On the vLLM path this runs in a ThreadPoolExecutor worker thread while
    RF-DETR visual extraction is still running on the main GPU stream.
    """
    output_mode = (vibevoice_output_mode or "turns").lower()
    if output_mode == "words":
        word_alignments = _normalize_vibevoice_words(vibevoice_output)
        if not word_alignments:
            raise RuntimeError(
                "[extract] VibeVoice output_mode=words produced 0 usable words."
            )
        vibevoice_turns = _build_turns_from_words(
            word_alignments=word_alignments,
            turn_gap_ms=max(0, int(vibevoice_word_turn_gap_ms)),
        )
        logger.info(
            "[extract] VibeVoice output_mode=words — synthesized %d turns from %d words (gap=%d ms)",
            len(vibevoice_turns),
            len(word_alignments),
            max(0, int(vibevoice_word_turn_gap_ms)),
        )
        logger.info("[extract] skipping forced-alignment because VIBEVOICE_OUTPUT_MODE=words")
    else:
        vibevoice_turns = list(vibevoice_output)
        # Build prelim turns for the aligner
        prelim_turns = []
        for idx, t in enumerate(vibevoice_turns, start=1):
            prelim_turns.append(
                {
                    "turn_id": f"t_{idx:06d}",
                    "speaker_id": f"SPEAKER_{int(t.get('Speaker') or 0)}",
                    "start_ms": int(round(float(t.get("Start") or 0) * 1000)),
                    "end_ms": int(round(float(t.get("End") or 0) * 1000)),
                    "transcript_text": str(t.get("Content") or "").strip(),
                }
            )

        logger.info("[extract] starting forced-alignment on %d turns ...", len(prelim_turns))
        t_fa = time.perf_counter()
        word_alignments = forced_aligner.run(
            audio_path=workspace.audio_path,
            turns=prelim_turns,
        )
        logger.info(
            "[extract] forced-alignment done in %.1f s — %d words",
            time.perf_counter() - t_fa,
            len(word_alignments),
        )
        alignable_turns = [
            t
            for t in prelim_turns
            if int(t.get("end_ms") or 0) > int(t.get("start_ms") or 0)
            and str(t.get("transcript_text") or "").strip()
        ]
        if alignable_turns and not word_alignments:
            msg = (
                "[extract] forced-alignment produced 0 words for non-empty ASR turns; "
                "this indicates NFA is unhealthy."
            )
            if _is_forced_alignment_required():
                raise RuntimeError(
                    f"{msg} Set CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT=0 only for temporary debugging bypass."
                )
            logger.warning(
                "%s Continuing because CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT=0.", msg
            )

    merged = merge_vibevoice_outputs(
        vibevoice_turns=vibevoice_turns,
        word_alignments=word_alignments,
    )
    turns = list(merged.get("turns") or [])

    diarization_payload: dict[str, Any] = {
        "turns": [
            {
                "turn_id": t["turn_id"],
                "speaker_id": t["speaker_id"],
                "start_ms": t["start_ms"],
                "end_ms": t["end_ms"],
                "transcript_text": t["transcript_text"],
                "word_ids": t.get("word_ids", []),
                "identification_match": None,
            }
            for t in turns
        ],
        "words": merged.get("words", []),
    }

    logger.info("[extract] starting emotion2vec+ on %d turns ...", len(turns))
    t_emotion = time.perf_counter()
    emotion2vec_payload = emotion_provider.run(
        audio_path=workspace.audio_path,
        turns=turns,
    )
    logger.info("[extract] emotion2vec+ done in %.1f s", time.perf_counter() - t_emotion)

    logger.info("[extract] starting YAMNet ...")
    t_yamnet = time.perf_counter()
    yamnet_payload = yamnet_provider.run(audio_path=workspace.audio_path)
    yamnet_event_count = len((yamnet_payload or {}).get("events") or [])
    logger.info(
        "[extract] YAMNet done in %.1f s — %d events",
        time.perf_counter() - t_yamnet,
        yamnet_event_count,
    )

    return diarization_payload, emotion2vec_payload, yamnet_payload


def run_phase1_sidecars(
    *,
    source_url: str,
    video_gcs_uri: str,
    workspace: Phase1Workspace,
    vibevoice_provider: Any,
    forced_aligner: Any,
    visual_extractor: Any,
    emotion_provider: Any,
    yamnet_provider: Any,
    on_audio_chain_complete: Any | None = None,
) -> Phase1SidecarOutputs:
    """
    Run Phase 1 sidecar tasks for the vLLM-only pipeline.

    Execution order:
      1a. Visual extraction  ─┐ concurrent via ThreadPoolExecutor(max_workers=3)
      1b. vLLM ASR           ─┘
          ↓ ASR done — audio chain starts immediately, visual still running
      2.  NeMo Forced Aligner  ─┐
      3.  emotion2vec+          ├─ concurrent with RF-DETR, serial with each other
      4.  YAMNet               ─┘

    Phases 2–4 only need the audio chain outputs (canonical_timeline,
    speech_emotion_timeline, audio_event_timeline).  RF-DETR output is only
    needed by Phase 5.  Starting the audio chain as soon as ASR finishes
    means Phases 2–4 artifacts are ready ~230 s earlier on a ~13-min clip.
    """
    t_total = time.perf_counter()

    if not getattr(vibevoice_provider, "supports_concurrent_visual", False):
        raise RuntimeError(
            "V3.1 main pipeline requires a concurrent-capable vLLM ASR provider."
        )

    logger.info("[extract] starting visual extraction + vLLM ASR in parallel ...")
    t_overlap = time.perf_counter()

    with ThreadPoolExecutor(max_workers=3) as pool:
        visual_future = pool.submit(
            visual_extractor.extract,
            video_path=workspace.video_path,
            workspace=workspace,
        )
        asr_future = pool.submit(
            vibevoice_provider.run,
            audio_path=workspace.audio_path,
        )

        vibevoice_output = asr_future.result()
        vibevoice_output_mode = str(getattr(vibevoice_provider, "output_mode", "turns"))
        item_kind = "words" if vibevoice_output_mode == "words" else "turns"
        logger.info(
            "[extract] ASR done (%d %s) — starting audio chain while RF-DETR still running ...",
            len(vibevoice_output),
            item_kind,
        )
        audio_chain_future = pool.submit(
            _run_audio_chain,
            workspace=workspace,
            vibevoice_output=vibevoice_output,
            vibevoice_output_mode=vibevoice_output_mode,
            vibevoice_word_turn_gap_ms=int(
                getattr(vibevoice_provider, "word_turn_gap_ms", 900)
            ),
            forced_aligner=forced_aligner,
            emotion_provider=emotion_provider,
            yamnet_provider=yamnet_provider,
        )

        diarization_payload, emotion2vec_payload, yamnet_payload = audio_chain_future.result()

        if on_audio_chain_complete is not None:
            _partial = Phase1SidecarOutputs(
                phase1_audio={
                    "source_audio": source_url,
                    "video_gcs_uri": video_gcs_uri,
                    "local_video_path": str(workspace.video_path),
                    "local_audio_path": str(workspace.audio_path),
                },
                diarization_payload=diarization_payload,
                phase1_visual={},
                emotion2vec_payload=emotion2vec_payload,
                yamnet_payload=yamnet_payload,
            )
            on_audio_chain_complete(_partial)
            logger.info(
                "[extract] audio-chain callback fired — Phases 2-4 starting while RF-DETR finishes"
            )

        phase1_visual = visual_future.result()

    logger.info(
        "[extract] visual + audio chain both done in %.1f s",
        time.perf_counter() - t_overlap,
    )

    logger.info("[extract] all sidecars done in %.1f s", time.perf_counter() - t_total)

    return Phase1SidecarOutputs(
        phase1_audio={
            "source_audio": source_url,
            "video_gcs_uri": video_gcs_uri,
            "local_video_path": str(workspace.video_path),
            "local_audio_path": str(workspace.audio_path),
        },
        diarization_payload=diarization_payload,
        phase1_visual=phase1_visual,
        emotion2vec_payload=emotion2vec_payload,
        yamnet_payload=yamnet_payload,
    )


__all__ = ["run_phase1_sidecars"]
