from __future__ import annotations

import json
import os
from typing import Any


DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
OVERLAP_FOLLOW_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "camera_target_local_track_id": {"type": ["string", "null"]},
        "camera_target_track_id": {"type": ["string", "null"]},
        "stay_wide": {"type": "boolean"},
        "decision_source": {"type": "string"},
        "confidence": {"type": ["number", "null"]},
    },
    "required": [
        "camera_target_local_track_id",
        "camera_target_track_id",
        "stay_wide",
        "decision_source",
        "confidence",
    ],
}


def overlap_follow_enabled() -> bool:
    raw = str(os.getenv("CLYPT_OVERLAP_FOLLOW_ENABLE", "1")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def overlap_follow_model_name() -> str:
    raw = str(os.getenv("CLYPT_OVERLAP_FOLLOW_MODEL", DEFAULT_GEMINI_MODEL)).strip()
    return raw or DEFAULT_GEMINI_MODEL


def _visible_track_map(span: dict) -> dict[str, str]:
    visible_local = [str(track_id) for track_id in (span.get("visible_local_track_ids") or []) if str(track_id)]
    visible_global = [str(track_id) for track_id in (span.get("visible_track_ids") or []) if str(track_id)]
    mapping: dict[str, str] = {}
    for index, local_track_id in enumerate(visible_local):
        if index < len(visible_global):
            mapping[local_track_id] = visible_global[index]
    return mapping


def build_deterministic_overlap_follow_decisions(active_speakers_local: list[dict] | None) -> list[dict]:
    decisions: list[dict] = []
    for span in active_speakers_local or []:
        if not bool(span.get("overlap", False)):
            continue
        visible_local_track_ids = [
            str(track_id)
            for track_id in (span.get("visible_local_track_ids") or [])
            if str(track_id)
        ]
        offscreen_audio_speaker_ids = [
            str(speaker_id)
            for speaker_id in (span.get("offscreen_audio_speaker_ids") or [])
            if str(speaker_id)
        ]
        visible_track_map = _visible_track_map(span)
        camera_target_local_track_id = None
        camera_target_track_id = None
        stay_wide = True
        if len(visible_local_track_ids) == 1 and not offscreen_audio_speaker_ids:
            camera_target_local_track_id = visible_local_track_ids[0]
            camera_target_track_id = visible_track_map.get(camera_target_local_track_id)
            stay_wide = False
        decisions.append(
            {
                "start_time_ms": int(span.get("start_time_ms", 0) or 0),
                "end_time_ms": int(span.get("end_time_ms", 0) or 0),
                "camera_target_local_track_id": camera_target_local_track_id,
                "camera_target_track_id": camera_target_track_id,
                "stay_wide": stay_wide,
                "visible_local_track_ids": visible_local_track_ids,
                "offscreen_audio_speaker_ids": offscreen_audio_speaker_ids,
                "decision_model": None,
                "decision_source": "deterministic",
                "confidence": span.get("confidence"),
            }
        )
    return decisions


def _words_for_span(words: list[dict] | None, span: dict, *, context_ms: int = 1200) -> list[dict]:
    selected: list[dict] = []
    start_ms = int(span.get("start_time_ms", 0) or 0) - context_ms
    end_ms = int(span.get("end_time_ms", 0) or 0) + context_ms
    for word in words or []:
        word_start_ms = int(word.get("start_time_ms", 0) or 0)
        word_end_ms = int(word.get("end_time_ms", word_start_ms) or word_start_ms)
        if word_end_ms < start_ms or word_start_ms > end_ms:
            continue
        selected.append(
            {
                "word": str(word.get("word") or word.get("text") or ""),
                "start_time_ms": word_start_ms,
                "end_time_ms": word_end_ms,
            }
        )
    return selected


def _candidate_debug_for_span(speaker_candidate_debug: list[dict] | None, span: dict) -> list[dict]:
    selected: list[dict] = []
    start_ms = int(span.get("start_time_ms", 0) or 0)
    end_ms = int(span.get("end_time_ms", 0) or 0)
    for entry in speaker_candidate_debug or []:
        entry_start_ms = int(entry.get("start_time_ms", 0) or 0)
        entry_end_ms = int(entry.get("end_time_ms", entry_start_ms) or entry_start_ms)
        if entry_end_ms < start_ms or entry_start_ms > end_ms:
            continue
        selected.append(dict(entry))
    return selected


def _neighbor_context(
    active_speakers_local: list[dict] | None,
    *,
    overlap_index: int,
) -> tuple[dict | None, dict | None]:
    overlap_spans = [span for span in (active_speakers_local or []) if bool(span.get("overlap", False))]
    all_spans = list(active_speakers_local or [])
    if overlap_index < 0 or overlap_index >= len(overlap_spans):
        return None, None
    current_span = overlap_spans[overlap_index]
    try:
        current_index = all_spans.index(current_span)
    except ValueError:
        return None, None

    previous_context = None
    for span in reversed(all_spans[:current_index]):
        previous_context = dict(span)
        break

    next_context = None
    for span in all_spans[current_index + 1 :]:
        next_context = dict(span)
        break

    return previous_context, next_context


def _build_overlap_prompt(
    span: dict,
    *,
    words: list[dict],
    speaker_candidate_debug: list[dict],
    previous_context: dict | None,
    next_context: dict | None,
) -> str:
    payload = {
        "instruction": (
            "Choose the single best visible camera target for this overlap window. "
            "Return JSON only with keys: camera_target_local_track_id, camera_target_track_id, "
            "stay_wide, decision_source, confidence. If no confident target exists, set stay_wide=true "
            "and the track ids to null."
        ),
        "overlap_span": span,
        "previous_context": previous_context,
        "next_context": next_context,
        "context_words": words,
        "speaker_candidate_debug": speaker_candidate_debug,
    }
    return json.dumps(payload, ensure_ascii=True)


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty Gemini response")
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(raw[start : end + 1])


def _normalize_decision(span: dict, payload: dict[str, Any], *, model_name: str) -> dict[str, Any]:
    visible_local_track_ids = [
        str(track_id)
        for track_id in (span.get("visible_local_track_ids") or [])
        if str(track_id)
    ]
    offscreen_audio_speaker_ids = [
        str(speaker_id)
        for speaker_id in (span.get("offscreen_audio_speaker_ids") or [])
        if str(speaker_id)
    ]
    visible_track_map = _visible_track_map(span)

    local_track_id = payload.get("camera_target_local_track_id")
    if local_track_id in (None, ""):
        local_track_id = None
    else:
        local_track_id = str(local_track_id)
    if local_track_id is not None and local_track_id not in visible_local_track_ids:
        local_track_id = None

    stay_wide = bool(payload.get("stay_wide", local_track_id is None))
    if stay_wide:
        local_track_id = None
    track_id = payload.get("camera_target_track_id")
    if track_id in (None, ""):
        track_id = None
    else:
        track_id = str(track_id)
    if local_track_id and track_id is None:
        track_id = visible_track_map.get(local_track_id)
    if local_track_id is None:
        track_id = None

    confidence = payload.get("confidence", span.get("confidence"))
    if confidence is not None:
        try:
            confidence = float(confidence)
        except Exception:
            confidence = span.get("confidence")

    return {
        "start_time_ms": int(span.get("start_time_ms", 0) or 0),
        "end_time_ms": int(span.get("end_time_ms", 0) or 0),
        "camera_target_local_track_id": local_track_id,
        "camera_target_track_id": track_id,
        "stay_wide": stay_wide,
        "visible_local_track_ids": visible_local_track_ids,
        "offscreen_audio_speaker_ids": offscreen_audio_speaker_ids,
        "decision_model": model_name,
        "decision_source": str(payload.get("decision_source") or "gemini"),
        "confidence": confidence,
    }


def _load_gemini_client(client=None):
    if client is not None:
        return client
    api_key = (
        str(os.getenv("GEMINI_API_KEY", "")).strip()
        or str(os.getenv("GOOGLE_API_KEY", "")).strip()
    )
    if not api_key:
        return None
    try:
        from google import genai
    except Exception:
        return None
    return genai.Client(api_key=api_key)


def maybe_adjudicate_overlap_follow_decisions(
    *,
    active_speakers_local: list[dict] | None,
    words: list[dict] | None,
    speaker_candidate_debug: list[dict] | None,
    client=None,
    model_name: str | None = None,
    enabled: bool | None = None,
) -> list[dict]:
    deterministic = build_deterministic_overlap_follow_decisions(active_speakers_local)
    if enabled is None:
        enabled = overlap_follow_enabled()
    if not enabled or not deterministic:
        return deterministic

    model_name = model_name or overlap_follow_model_name()
    api_client = _load_gemini_client(client=client)
    if api_client is None:
        return deterministic

    decisions: list[dict] = []
    overlap_spans = [span for span in (active_speakers_local or []) if bool(span.get("overlap", False))]
    for overlap_index, (span, fallback) in enumerate(zip(overlap_spans, deterministic)):
        if not span.get("visible_local_track_ids"):
            decisions.append(fallback)
            continue
        try:
            previous_context, next_context = _neighbor_context(
                active_speakers_local,
                overlap_index=overlap_index,
            )
            prompt = _build_overlap_prompt(
                span,
                words=_words_for_span(words, span),
                speaker_candidate_debug=_candidate_debug_for_span(speaker_candidate_debug, span),
                previous_context=previous_context,
                next_context=next_context,
            )
            response = api_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": OVERLAP_FOLLOW_RESPONSE_SCHEMA,
                },
            )
            payload = _extract_json_object(getattr(response, "text", ""))
            decisions.append(_normalize_decision(span, payload, model_name=model_name))
        except Exception:
            decisions.append(fallback)
    return decisions
