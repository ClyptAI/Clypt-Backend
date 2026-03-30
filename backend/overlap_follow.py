from __future__ import annotations

import copy
import json
import os
from typing import Any

from backend.pipeline.phase1.config import get_phase1_config


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


def overlap_follow_min_confidence() -> float:
    return float(get_phase1_config().overlap_follow_min_confidence)


def overlap_follow_min_evidence_score() -> int:
    return int(get_phase1_config().overlap_follow_min_evidence_score)


def _merge_log_context(evidence_context: dict[str, Any], log_context: dict[str, Any] | None) -> None:
    if not log_context:
        return
    jid = log_context.get("job_id")
    if jid:
        evidence_context["job_id"] = jid
    wid = log_context.get("worker_id")
    if wid:
        evidence_context["worker_id"] = wid


def _overlap_low_evidence_gate(
    span: dict,
    *,
    words: list[dict] | None,
    speaker_candidate_debug: list[dict] | None,
) -> tuple[bool, dict[str, Any]]:
    cfg = get_phase1_config()
    span_words = _words_for_span(words, span)
    span_debug = _candidate_debug_for_span(speaker_candidate_debug, span)
    word_count = len(span_words)
    debug_count = len(span_debug)
    score = word_count + debug_count
    min_score = int(cfg.overlap_follow_min_evidence_score)
    min_conf = float(cfg.overlap_follow_min_confidence)
    raw_conf = span.get("confidence")
    conf_val: float | None
    try:
        conf_val = float(raw_conf) if raw_conf is not None else None
    except (TypeError, ValueError):
        conf_val = None

    visible_track_count = len([t for t in (span.get("visible_local_track_ids") or []) if str(t)])
    offscreen_audio_count = len([s for s in (span.get("offscreen_audio_speaker_ids") or []) if str(s)])
    diarization_overlap_strength = 0.0
    for key in ("diarization_overlap_confidence", "diarization_overlap_score", "overlap_confidence"):
        try:
            raw = span.get(key)
            if raw is not None:
                diarization_overlap_strength = max(diarization_overlap_strength, float(raw))
        except (TypeError, ValueError):
            continue
    if diarization_overlap_strength <= 0.0 and offscreen_audio_count > 0:
        # Offscreen audio speakers are a weak overlap prior when explicit diarization score is absent.
        diarization_overlap_strength = 0.25

    top_margin = None
    competition_strength = 0.0
    for entry in span_debug:
        try:
            margin = float(entry.get("top_1_top_2_margin"))
        except (TypeError, ValueError):
            continue
        if top_margin is None or margin < top_margin:
            top_margin = margin
    if top_margin is not None:
        # Smaller margin => higher candidate competition.
        competition_strength = max(0.0, min(1.0, 1.0 - top_margin))

    base_conf = conf_val if conf_val is not None else 0.0
    overlap_confidence = max(
        0.0,
        min(
            1.0,
            0.5 * base_conf
            + 0.2 * (1.0 if visible_track_count >= 2 else 0.0)
            + 0.2 * max(0.0, min(1.0, diarization_overlap_strength))
            + 0.1 * competition_strength,
        ),
    )

    gated_low_evidence = score < min_score
    gated_low_confidence = False
    if min_conf > 0.0:
        if overlap_confidence < min_conf:
            gated_low_confidence = True

    gated = gated_low_evidence or gated_low_confidence
    detail: dict[str, Any] = {
        "evidence_word_count": word_count,
        "evidence_debug_count": debug_count,
        "evidence_score": score,
        "min_evidence_score": min_score,
        "overlap_span_confidence": conf_val,
        "overlap_confidence": overlap_confidence,
        "min_overlap_confidence": min_conf,
        "visible_track_count": visible_track_count,
        "offscreen_audio_count": offscreen_audio_count,
        "diarization_overlap_strength": diarization_overlap_strength,
        "candidate_competition": competition_strength,
        "top_1_top_2_margin_min": top_margin,
        "gated_low_evidence": gated_low_evidence,
        "gated_low_confidence": gated_low_confidence,
    }
    return gated, detail


def _bump_run_metadata(
    run_metadata_out: dict[str, Any] | None,
    *,
    reason_code: str,
    path: str,
) -> None:
    if run_metadata_out is None:
        return
    rc = run_metadata_out.setdefault("reason_code_counts", {})
    rc[reason_code] = int(rc.get(reason_code, 0)) + 1
    pc = run_metadata_out.setdefault("adjudication_path_counts", {})
    pc[path] = int(pc.get(path, 0)) + 1
    fc = run_metadata_out.setdefault("fallback_category_counts", {})
    if path == "gemini":
        fc["gemini_success"] = int(fc.get("gemini_success", 0)) + 1
    else:
        fc["deterministic_fallback"] = int(fc.get("deterministic_fallback", 0)) + 1


def _finalize_run_metadata_header(
    run_metadata_out: dict[str, Any] | None,
    *,
    log_context: dict[str, Any] | None,
) -> None:
    if run_metadata_out is None:
        return
    _merge_log_context(run_metadata_out, log_context)


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
        if len(visible_local_track_ids) == 1:
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
                "decision_code": "deterministic_selected",
                "confidence": span.get("confidence"),
                "evidence_context": {
                    "visible_track_count": len(visible_local_track_ids),
                    "overlap_span_confidence": span.get("confidence"),
                },
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


def _normalize_decision(
    span: dict,
    payload: dict[str, Any],
    *,
    model_name: str,
    evidence_word_count: int = 0,
    evidence_debug_count: int = 0,
    log_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
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

    if stay_wide or local_track_id is None:
        decision_code = "overlap_stay_wide"
    else:
        decision_code = "follow_primary_speaker"

    evidence_context: dict[str, Any] = {
        "visible_track_count": len(visible_local_track_ids),
        "overlap_span_confidence": span.get("confidence"),
        "evidence_word_count": evidence_word_count,
        "evidence_debug_count": evidence_debug_count,
        "adjudication_reason": "gemini",
    }
    _merge_log_context(evidence_context, log_context)

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
        "decision_code": decision_code,
        "confidence": confidence,
        "evidence_context": evidence_context,
    }


def _stamp_deterministic_overlap_decision(
    fallback: dict[str, Any],
    *,
    reason_code: str,
    log_context: dict[str, Any] | None,
    evidence_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    d = copy.deepcopy(fallback)
    d["decision_code"] = reason_code
    ec: dict[str, Any] = dict(d.get("evidence_context") or {})
    if evidence_extra:
        ec.update(evidence_extra)
    ec["adjudication_reason"] = reason_code
    _merge_log_context(ec, log_context)
    d["evidence_context"] = ec
    return d


def _load_gemini_client(client=None):
    if client is not None:
        return client
    vertex_ai_raw = str(os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "")).strip().lower()
    vertex_ai_enabled = vertex_ai_raw in {"1", "true", "yes", "on"}
    vertex_project = str(os.getenv("GOOGLE_CLOUD_PROJECT", "")).strip()
    vertex_location = str(os.getenv("GOOGLE_CLOUD_LOCATION", "")).strip() or "global"
    api_key = (
        str(os.getenv("GEMINI_API_KEY", "")).strip()
        or str(os.getenv("GOOGLE_API_KEY", "")).strip()
    )
    try:
        from google import genai
    except Exception:
        return None
    if vertex_ai_enabled and vertex_project:
        return genai.Client(
            vertexai=True,
            project=vertex_project,
            location=vertex_location,
        )
    if not api_key:
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
    log_context: dict[str, Any] | None = None,
    run_metadata_out: dict[str, Any] | None = None,
) -> list[dict]:
    deterministic = build_deterministic_overlap_follow_decisions(active_speakers_local)
    if enabled is None:
        enabled = overlap_follow_enabled()

    _finalize_run_metadata_header(run_metadata_out, log_context=log_context)

    if not deterministic:
        return deterministic

    if not enabled:
        for _fb in deterministic:
            _bump_run_metadata(
                run_metadata_out,
                reason_code="deterministic_selected",
                path="deterministic",
            )
        return deterministic

    model_name = model_name or overlap_follow_model_name()
    api_client = _load_gemini_client(client=client)
    overlap_spans = [span for span in (active_speakers_local or []) if bool(span.get("overlap", False))]

    if api_client is None:
        decisions: list[dict] = []
        for _span, fb in zip(overlap_spans, deterministic):
            decisions.append(
                _stamp_deterministic_overlap_decision(
                    fb,
                    reason_code="gemini_unavailable",
                    log_context=log_context,
                    evidence_extra={"api_status": "no_client"},
                )
            )
            _bump_run_metadata(
                run_metadata_out,
                reason_code="gemini_unavailable",
                path="deterministic",
            )
        return decisions

    # Authority boundary: adjudication reads transcript words for evidence only.
    # Use a private deep copy so future edits cannot mutate caller-owned word dicts
    # (including any `speaker_*` labels).
    words_for_evidence = copy.deepcopy(words) if words else None

    decisions = []
    for overlap_index, (span, fallback) in enumerate(zip(overlap_spans, deterministic)):
        if not span.get("visible_local_track_ids"):
            extra = {"skip_gemini": "no_visible_local_track_ids"}
            decisions.append(
                _stamp_deterministic_overlap_decision(
                    fallback,
                    reason_code="deterministic_selected",
                    log_context=log_context,
                    evidence_extra=extra,
                )
            )
            _bump_run_metadata(
                run_metadata_out,
                reason_code="deterministic_selected",
                path="deterministic",
            )
            continue

        gated, gate_detail = _overlap_low_evidence_gate(
            span,
            words=words_for_evidence,
            speaker_candidate_debug=speaker_candidate_debug,
        )
        if gated:
            decisions.append(
                _stamp_deterministic_overlap_decision(
                    fallback,
                    reason_code="low_overlap_evidence",
                    log_context=log_context,
                    evidence_extra=gate_detail,
                )
            )
            _bump_run_metadata(
                run_metadata_out,
                reason_code="low_overlap_evidence",
                path="deterministic",
            )
            continue

        word_ctx = _words_for_span(words_for_evidence, span)
        debug_ctx = _candidate_debug_for_span(speaker_candidate_debug, span)
        try:
            previous_context, next_context = _neighbor_context(
                active_speakers_local,
                overlap_index=overlap_index,
            )
            prompt = _build_overlap_prompt(
                span,
                words=word_ctx,
                speaker_candidate_debug=debug_ctx,
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
            try:
                payload = _extract_json_object(getattr(response, "text", ""))
            except ValueError:
                decisions.append(
                    _stamp_deterministic_overlap_decision(
                        fallback,
                        reason_code="gemini_invalid_response",
                        log_context=log_context,
                        evidence_extra={"parse_error": True},
                    )
                )
                _bump_run_metadata(
                    run_metadata_out,
                    reason_code="gemini_invalid_response",
                    path="deterministic",
                )
                continue
            decisions.append(
                _normalize_decision(
                    span,
                    payload,
                    model_name=model_name,
                    evidence_word_count=len(word_ctx),
                    evidence_debug_count=len(debug_ctx),
                    log_context=log_context,
                )
            )
            code = str(decisions[-1].get("decision_code") or "gemini")
            _bump_run_metadata(run_metadata_out, reason_code=code, path="gemini")
        except Exception:
            decisions.append(
                _stamp_deterministic_overlap_decision(
                    fallback,
                    reason_code="gemini_unavailable",
                    log_context=log_context,
                    evidence_extra={"api_status": "generate_content_failed"},
                )
            )
            _bump_run_metadata(
                run_metadata_out,
                reason_code="gemini_unavailable",
                path="deterministic",
            )
    return decisions
