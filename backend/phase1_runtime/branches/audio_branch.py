from __future__ import annotations

from typing import Any

from backend.pipeline.timeline.vibevoice_merge import merge_vibevoice_outputs

from ..models import Phase1Workspace


def _build_preliminary_turns(*, vibevoice_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prelim_turns: list[dict[str, Any]] = []
    for idx, turn in enumerate(vibevoice_turns, start=1):
        prelim_turns.append(
            {
                "turn_id": f"t_{idx:06d}",
                "speaker_id": f"SPEAKER_{int(turn.get('Speaker') or 0)}",
                "start_ms": int(round(float(turn.get("Start") or 0) * 1000)),
                "end_ms": int(round(float(turn.get("End") or 0) * 1000)),
                "transcript_text": str(turn.get("Content") or "").strip(),
            }
        )
    return prelim_turns


def run_audio_branch(
    *,
    request: Any | None = None,
    workspace: Phase1Workspace,
    vibevoice_provider: Any,
    forced_aligner: Any,
    emotion_provider: Any,
) -> dict[str, Any]:
    vibevoice_turns = vibevoice_provider.run(audio_path=workspace.audio_path)
    prelim_turns = _build_preliminary_turns(vibevoice_turns=vibevoice_turns)
    word_alignments = forced_aligner.run(
        audio_path=workspace.audio_path,
        turns=prelim_turns,
    )
    merged = merge_vibevoice_outputs(
        vibevoice_turns=vibevoice_turns,
        word_alignments=word_alignments,
    )
    turns = list(merged.get("turns") or [])
    diarization_payload: dict[str, Any] = {
        "turns": [
            {
                "turn_id": turn["turn_id"],
                "speaker_id": turn["speaker_id"],
                "start_ms": turn["start_ms"],
                "end_ms": turn["end_ms"],
                "transcript_text": turn["transcript_text"],
                "word_ids": turn.get("word_ids", []),
                "identification_match": None,
            }
            for turn in turns
        ],
        "words": merged.get("words", []),
    }
    emotion2vec_payload = emotion_provider.run(
        audio_path=workspace.audio_path,
        turns=turns,
    )
    return {
        "diarization_payload": diarization_payload,
        "emotion2vec_payload": emotion2vec_payload,
    }


__all__ = ["run_audio_branch"]
