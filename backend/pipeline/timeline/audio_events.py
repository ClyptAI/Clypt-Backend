from __future__ import annotations

from ..contracts import AudioEvent, AudioEventTimeline

from ..contracts import AudioEventTimeline


def build_audio_event_timeline(*, yamnet_payload: dict) -> AudioEventTimeline:
    """Convert raw YAMNet detections into merged event spans."""
    raw_events = yamnet_payload.get("events") or yamnet_payload.get("detections") or []
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
