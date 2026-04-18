from __future__ import annotations

from typing import Any

from ..contracts import AudioEvent, AudioEventTimeline
from .payload_utils import payload_to_dict


def build_audio_event_timeline(*, yamnet_payload: Any) -> AudioEventTimeline:
    """Convert raw YAMNet detections into merged event spans."""
    payload = payload_to_dict(yamnet_payload)
    raw_events = payload.get("events") or payload.get("detections") or []
    sorted_events = sorted(
        raw_events,
        key=lambda item: (int(item.get("start_ms", 0)), int(item.get("end_ms", 0))),
    )
    merged: list[AudioEvent] = []
    for raw_event in sorted_events:
        label = str(raw_event.get("event_label") or raw_event.get("label") or "").strip()
        if not label:
            continue
        start_ms = int(raw_event.get("start_ms", 0))
        end_ms = int(raw_event.get("end_ms", start_ms))
        confidence = raw_event.get("confidence")
        if merged and merged[-1].event_label == label and start_ms <= merged[-1].end_ms:
            merged[-1] = AudioEvent(
                event_label=label,
                start_ms=merged[-1].start_ms,
                end_ms=max(merged[-1].end_ms, end_ms),
                confidence=max(
                    float(merged[-1].confidence or 0.0),
                    float(confidence or 0.0),
                ),
            )
            continue
        merged.append(
            AudioEvent(
                event_label=label,
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=float(confidence) if confidence is not None else None,
            )
        )
    return AudioEventTimeline(events=merged)
