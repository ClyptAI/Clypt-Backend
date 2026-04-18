from __future__ import annotations

from typing import Any

from ..contracts import SpeechEmotionEvent, SpeechEmotionTimeline
from .payload_utils import payload_to_dict


def build_speech_emotion_timeline(*, emotion2vec_payload: Any) -> SpeechEmotionTimeline:
    """Convert turn-level emotion2vec+ outputs into the V3.1 artifact."""
    payload = payload_to_dict(emotion2vec_payload)
    raw_segments = payload.get("segments") or payload.get("events") or []
    events: list[SpeechEmotionEvent] = []
    for raw_segment in raw_segments:
        labels = raw_segment.get("labels") or []
        scores = raw_segment.get("scores") or []
        primary_label = raw_segment.get("primary_emotion_label") or (labels[0] if labels else None)
        primary_score = raw_segment.get("primary_emotion_score")
        if primary_score is None and scores:
            primary_score = scores[0]
        per_class_scores = dict(raw_segment.get("per_class_scores") or {})
        if primary_label is None:
            continue
        if primary_label not in per_class_scores:
            per_class_scores[primary_label] = float(primary_score or 0.0)
        events.append(
            SpeechEmotionEvent(
                turn_id=str(raw_segment["turn_id"]),
                primary_emotion_label=primary_label,
                primary_emotion_score=float(primary_score or 0.0),
                per_class_scores=per_class_scores,
            )
        )
    return SpeechEmotionTimeline(events=events)
