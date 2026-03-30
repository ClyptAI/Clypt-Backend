import json
from unittest.mock import patch

from backend.pipeline.phase1.stage_log import (
    build_stage_log_payload,
    emit_overlap_follow_postpass_summary,
    emit_stage_log,
    resolve_job_and_worker_ids,
)


REQUIRED_KEYS = frozenset(
    {"job_id", "stage", "event", "decision_source", "reason_code", "elapsed_ms", "worker_id"}
)


def test_build_stage_log_payload_has_required_fields():
    p = build_stage_log_payload(
        job_id="abc",
        stage="tracking",
        event="stage_end",
        decision_source="pipeline",
        reason_code="ok",
        elapsed_ms=42,
        worker_id="w1",
    )
    assert REQUIRED_KEYS.issubset(p.keys())
    assert p["job_id"] == "abc"
    assert p["stage"] == "tracking"
    assert p["event"] == "stage_end"
    assert p["decision_source"] == "pipeline"
    assert p["reason_code"] == "ok"
    assert p["elapsed_ms"] == 42
    assert p["worker_id"] == "w1"


def test_emit_stage_log_prints_compact_json_with_required_keys():
    with patch("backend.pipeline.phase1.stage_log.print") as mock_print:
        emit_stage_log(
            job_id="j",
            stage="clustering",
            event="stage_start",
            decision_source="pipeline",
            reason_code="ok",
            elapsed_ms=0,
            worker_id="wid",
        )
    mock_print.assert_called_once()
    line = mock_print.call_args[0][0]
    obj = json.loads(line)
    assert REQUIRED_KEYS.issubset(obj.keys())


def test_overlap_postpass_summary_includes_reason_codes_and_counters():
    with patch("backend.pipeline.phase1.stage_log.print") as mock_print:
        emit_overlap_follow_postpass_summary(
            job_id="job-1",
            worker_id="w99",
            run_metadata={
                "reason_code_counts": {
                    "gemini_unavailable": 1,
                    "deterministic_selected": 2,
                },
                "adjudication_path_counts": {"deterministic": 2, "gemini": 0},
                "fallback_category_counts": {"deterministic_fallback": 2},
            },
        )
    obj = json.loads(mock_print.call_args[0][0])
    assert REQUIRED_KEYS.issubset(obj.keys())
    assert obj["stage"] == "overlap_follow"
    assert obj["event"] == "postpass_summary"
    assert obj["decision_source"] == "overlap_adjudication"
    assert obj["reason_code"] == "gemini_unavailable"
    assert obj["reason_code_counts"]["deterministic_selected"] == 2
    assert obj["fallback_category_counts"]["deterministic_fallback"] == 2


def test_resolve_job_and_worker_ids_prefers_log_context():
    jid, wid = resolve_job_and_worker_ids({"job_id": "x", "worker_id": "y"})
    assert jid == "x"
    assert wid == "y"
