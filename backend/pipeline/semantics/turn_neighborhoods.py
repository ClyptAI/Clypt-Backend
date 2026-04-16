from __future__ import annotations

from ..contracts import AudioEventTimeline, CanonicalTimeline, SpeechEmotionTimeline


def build_turn_neighborhoods(
    *,
    canonical_timeline: CanonicalTimeline,
    speech_emotion_timeline: SpeechEmotionTimeline | None = None,
    audio_event_timeline: AudioEventTimeline | None = None,
    target_turn_count: int = 8,
    halo_turn_count: int = 2,
) -> list[dict]:
    """Build overlapping local turn neighborhoods for Qwen merge/classify calls."""
    turns = list(canonical_timeline.turns)
    if target_turn_count <= 0:
        raise ValueError("target_turn_count must be positive")
    if halo_turn_count < 0:
        raise ValueError("halo_turn_count must be non-negative")
    if not turns:
        return []

    emotion_by_turn: dict[str, list[str]] = {}
    for event in (speech_emotion_timeline.events if speech_emotion_timeline else []):
        emotion_by_turn.setdefault(event.turn_id, []).append(event.primary_emotion_label)

    audio_events = list(audio_event_timeline.events) if audio_event_timeline else []

    neighborhoods: list[dict] = []
    batch_number = 1
    for start_idx in range(0, len(turns), target_turn_count):
        target_turns = turns[start_idx : start_idx + target_turn_count]
        left_halo_turns = turns[max(0, start_idx - halo_turn_count) : start_idx]
        right_halo_turns = turns[start_idx + len(target_turns) : start_idx + len(target_turns) + halo_turn_count]
        ordered_turns = [*left_halo_turns, *target_turns, *right_halo_turns]

        serialized_turns = []
        for turn in ordered_turns:
            audio_labels = [
                event.event_label
                for event in audio_events
                if not (event.end_ms < turn.start_ms or event.start_ms > turn.end_ms)
            ]
            serialized_turns.append(
                {
                    "turn_id": turn.turn_id,
                    "speaker_id": turn.speaker_id,
                    "start_ms": turn.start_ms,
                    "end_ms": turn.end_ms,
                    "transcript_text": turn.transcript_text,
                    "emotion_labels": emotion_by_turn.get(turn.turn_id, []),
                    "audio_events": audio_labels,
                    "role": "target" if turn in target_turns else "halo",
                }
            )

        neighborhoods.append(
            {
                "batch_id": f"nb_{batch_number:04d}",
                "target_turn_ids": [turn.turn_id for turn in target_turns],
                "turns": serialized_turns,
            }
        )
        batch_number += 1

    return neighborhoods
