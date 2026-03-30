import pytest

from backend.overlap_follow import (
    build_deterministic_overlap_follow_decisions,
    maybe_adjudicate_overlap_follow_decisions,
)


def _det_base_evidence(visible_n: int, conf: float):
    return {
        "visible_track_count": visible_n,
        "overlap_span_confidence": pytest.approx(conf, abs=1e-6),
    }


def test_build_deterministic_overlap_follow_decisions_stays_wide_for_multi_visible_overlap():
    decisions = build_deterministic_overlap_follow_decisions(
        [
            {
                "start_time_ms": 400,
                "end_time_ms": 1100,
                "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
                "visible_local_track_ids": ["track_1", "track_2"],
                "visible_track_ids": ["Global_Person_0", "Global_Person_1"],
                "offscreen_audio_speaker_ids": [],
                "overlap": True,
                "confidence": 0.71,
                "decision_source": "turn_binding",
            }
        ]
    )

    assert decisions == [
        {
            "start_time_ms": 400,
            "end_time_ms": 1100,
            "camera_target_local_track_id": None,
            "camera_target_track_id": None,
            "stay_wide": True,
            "visible_local_track_ids": ["track_1", "track_2"],
            "offscreen_audio_speaker_ids": [],
            "decision_model": None,
            "decision_source": "deterministic",
            "decision_code": "deterministic_selected",
            "confidence": pytest.approx(0.71, abs=1e-6),
            "evidence_context": _det_base_evidence(2, 0.71),
        }
    ]


def test_build_deterministic_overlap_follow_decisions_follows_single_visible_track_even_with_offscreen_audio():
    decisions = build_deterministic_overlap_follow_decisions(
        [
            {
                "start_time_ms": 400,
                "end_time_ms": 1100,
                "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
                "visible_local_track_ids": ["track_1"],
                "visible_track_ids": ["Global_Person_0"],
                "offscreen_audio_speaker_ids": ["SPEAKER_01"],
                "overlap": True,
                "confidence": 0.71,
                "decision_source": "turn_binding",
            }
        ]
    )

    assert decisions == [
        {
            "start_time_ms": 400,
            "end_time_ms": 1100,
            "camera_target_local_track_id": "track_1",
            "camera_target_track_id": "Global_Person_0",
            "stay_wide": False,
            "visible_local_track_ids": ["track_1"],
            "offscreen_audio_speaker_ids": ["SPEAKER_01"],
            "decision_model": None,
            "decision_source": "deterministic",
            "decision_code": "deterministic_selected",
            "confidence": pytest.approx(0.71, abs=1e-6),
            "evidence_context": _det_base_evidence(1, 0.71),
        }
    ]


def test_maybe_adjudicate_overlap_follow_decisions_uses_client_response():
    class _FakeResponse:
        text = (
            '{"camera_target_local_track_id":"track_1","camera_target_track_id":"Global_Person_0",'
            '"stay_wide":false,"decision_source":"gemini","confidence":0.84}'
        )

    class _FakeModels:
        def generate_content(self, *, model, contents, config=None):
            assert model == "gemini-3-flash-preview"
            assert "SPEAKER_00" in contents
            assert config is not None
            return _FakeResponse()

    class _FakeClient:
        models = _FakeModels()

    decisions = maybe_adjudicate_overlap_follow_decisions(
        active_speakers_local=[
            {
                "start_time_ms": 400,
                "end_time_ms": 1100,
                "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
                "visible_local_track_ids": ["track_1"],
                "visible_track_ids": ["Global_Person_0"],
                "offscreen_audio_speaker_ids": ["SPEAKER_01"],
                "overlap": True,
                "confidence": 0.71,
                "decision_source": "turn_binding",
            }
        ],
        words=[{"word": "hello", "start_time_ms": 450, "end_time_ms": 700}],
        speaker_candidate_debug=[],
        client=_FakeClient(),
        model_name="gemini-3-flash-preview",
        enabled=True,
    )

    assert decisions[0]["camera_target_local_track_id"] == "track_1"
    assert decisions[0]["camera_target_track_id"] == "Global_Person_0"
    assert decisions[0]["decision_model"] == "gemini-3-flash-preview"
    assert decisions[0]["decision_source"] == "gemini"
    assert decisions[0]["decision_code"] == "follow_primary_speaker"
    assert decisions[0]["evidence_context"]["evidence_word_count"] == 1
    assert decisions[0]["evidence_context"]["adjudication_reason"] == "gemini"


def test_maybe_adjudicate_overlap_follow_decisions_falls_back_when_client_raises():
    class _BoomModels:
        def generate_content(self, *, model, contents, config=None):
            raise RuntimeError("boom")

    class _BoomClient:
        models = _BoomModels()

    decisions = maybe_adjudicate_overlap_follow_decisions(
        active_speakers_local=[
            {
                "start_time_ms": 400,
                "end_time_ms": 1100,
                "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
                "visible_local_track_ids": ["track_1", "track_2"],
                "visible_track_ids": ["Global_Person_0", "Global_Person_1"],
                "offscreen_audio_speaker_ids": [],
                "overlap": True,
                "confidence": 0.71,
                "decision_source": "turn_binding",
            }
        ],
        words=[],
        speaker_candidate_debug=[],
        client=_BoomClient(),
        model_name="gemini-3-flash-preview",
        enabled=True,
    )

    assert decisions[0]["camera_target_local_track_id"] is None
    assert decisions[0]["stay_wide"] is True
    assert decisions[0]["decision_source"] == "deterministic"
    assert decisions[0]["decision_code"] == "gemini_unavailable"
    assert decisions[0]["evidence_context"]["api_status"] == "generate_content_failed"


def test_maybe_adjudicate_skips_gemini_when_low_overlap_confidence(monkeypatch):
    calls = []

    class _FakeModels:
        def generate_content(self, *, model, contents, config=None):
            calls.append(contents)
            raise AssertionError("Gemini should not be called when gated")

    class _FakeClient:
        models = _FakeModels()

    monkeypatch.setenv("CLYPT_OVERLAP_FOLLOW_MIN_CONFIDENCE", "0.9")

    meta: dict = {}
    decisions = maybe_adjudicate_overlap_follow_decisions(
        active_speakers_local=[
            {
                "start_time_ms": 400,
                "end_time_ms": 1100,
                "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
                "visible_local_track_ids": ["track_1"],
                "visible_track_ids": ["Global_Person_0"],
                "offscreen_audio_speaker_ids": ["SPEAKER_01"],
                "overlap": True,
                "confidence": 0.71,
                "decision_source": "turn_binding",
            }
        ],
        words=[{"word": "x", "start_time_ms": 450, "end_time_ms": 700}],
        speaker_candidate_debug=[],
        client=_FakeClient(),
        model_name="gemini-3-flash-preview",
        enabled=True,
        run_metadata_out=meta,
    )
    assert calls == []
    assert decisions[0]["decision_code"] == "low_overlap_evidence"
    assert meta["reason_code_counts"]["low_overlap_evidence"] == 1


def test_maybe_adjudicate_skips_gemini_when_low_evidence_score(monkeypatch):
    calls = []

    class _FakeModels:
        def generate_content(self, *, model, contents, config=None):
            calls.append(contents)
            raise AssertionError("Gemini should not be called when evidence is too low")

    class _FakeClient:
        models = _FakeModels()

    monkeypatch.setenv("CLYPT_OVERLAP_FOLLOW_MIN_CONFIDENCE", "0.0")
    monkeypatch.setenv("CLYPT_OVERLAP_FOLLOW_MIN_EVIDENCE_SCORE", "5")

    meta: dict = {}
    decisions = maybe_adjudicate_overlap_follow_decisions(
        active_speakers_local=[
            {
                "start_time_ms": 400,
                "end_time_ms": 1100,
                "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
                "visible_local_track_ids": ["track_1"],
                "visible_track_ids": ["Global_Person_0"],
                "offscreen_audio_speaker_ids": ["SPEAKER_01"],
                "overlap": True,
                "confidence": 0.71,
                "decision_source": "turn_binding",
            }
        ],
        words=[{"word": "x", "start_time_ms": 450, "end_time_ms": 700}],
        speaker_candidate_debug=[],
        client=_FakeClient(),
        model_name="gemini-3-flash-preview",
        enabled=True,
        run_metadata_out=meta,
    )
    assert calls == []
    assert decisions[0]["decision_code"] == "low_overlap_evidence"
    assert decisions[0]["evidence_context"]["gated_low_evidence"] is True
    assert meta["reason_code_counts"]["low_overlap_evidence"] == 1


def test_maybe_adjudicate_gemini_unavailable_without_client(monkeypatch):
    monkeypatch.setattr(
        "backend.overlap_follow._load_gemini_client",
        lambda client=None: None,
    )

    meta: dict = {}
    decisions = maybe_adjudicate_overlap_follow_decisions(
        active_speakers_local=[
            {
                "start_time_ms": 400,
                "end_time_ms": 1100,
                "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
                "visible_local_track_ids": ["track_1"],
                "visible_track_ids": ["Global_Person_0"],
                "offscreen_audio_speaker_ids": ["SPEAKER_01"],
                "overlap": True,
                "confidence": 0.71,
                "decision_source": "turn_binding",
            }
        ],
        words=[],
        speaker_candidate_debug=[],
        client=None,
        enabled=True,
        run_metadata_out=meta,
    )
    assert decisions[0]["decision_code"] == "gemini_unavailable"
    assert meta["reason_code_counts"]["gemini_unavailable"] == 1


def test_maybe_adjudicate_gemini_invalid_response():
    class _BadResponse:
        text = "not-json-at-all"

    class _FakeModels:
        def generate_content(self, *, model, contents, config=None):
            return _BadResponse()

    class _FakeClient:
        models = _FakeModels()

    meta: dict = {}
    decisions = maybe_adjudicate_overlap_follow_decisions(
        active_speakers_local=[
            {
                "start_time_ms": 400,
                "end_time_ms": 1100,
                "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
                "visible_local_track_ids": ["track_1"],
                "visible_track_ids": ["Global_Person_0"],
                "offscreen_audio_speaker_ids": ["SPEAKER_01"],
                "overlap": True,
                "confidence": 0.71,
                "decision_source": "turn_binding",
            }
        ],
        words=[{"word": "hello", "start_time_ms": 450, "end_time_ms": 700}],
        speaker_candidate_debug=[],
        client=_FakeClient(),
        model_name="gemini-3-flash-preview",
        enabled=True,
        run_metadata_out=meta,
    )
    assert decisions[0]["decision_code"] == "gemini_invalid_response"
    assert decisions[0]["evidence_context"]["parse_error"] is True
    assert meta["reason_code_counts"]["gemini_invalid_response"] == 1
    assert meta["fallback_category_counts"]["deterministic_fallback"] == 1


def test_maybe_adjudicate_deterministic_default_reason_and_counters():
    meta: dict = {}
    decisions = maybe_adjudicate_overlap_follow_decisions(
        active_speakers_local=[
            {
                "start_time_ms": 400,
                "end_time_ms": 1100,
                "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
                "visible_local_track_ids": ["track_1", "track_2"],
                "visible_track_ids": ["Global_Person_0", "Global_Person_1"],
                "offscreen_audio_speaker_ids": [],
                "overlap": True,
                "confidence": 0.71,
                "decision_source": "turn_binding",
            }
        ],
        words=[],
        speaker_candidate_debug=[],
        client=None,
        enabled=False,
        run_metadata_out=meta,
        log_context={"job_id": "job-abc", "worker_id": "w-1"},
    )
    assert decisions[0]["decision_code"] == "deterministic_selected"
    assert meta["reason_code_counts"]["deterministic_selected"] == 1
    assert meta["adjudication_path_counts"]["deterministic"] == 1
    assert meta["fallback_category_counts"]["deterministic_fallback"] == 1
    assert meta["job_id"] == "job-abc"
    assert meta["worker_id"] == "w-1"


def test_maybe_adjudicate_overlap_follow_decisions_passes_structured_config_and_context():
    captured = {}

    class _FakeResponse:
        text = (
            '{"camera_target_local_track_id":"track_1","camera_target_track_id":"Global_Person_0",'
            '"stay_wide":false,"decision_source":"gemini","confidence":0.88}'
        )

    class _FakeModels:
        def generate_content(self, *, model, contents, config=None):
            captured["model"] = model
            captured["contents"] = contents
            captured["config"] = config
            return _FakeResponse()

    class _FakeClient:
        models = _FakeModels()

    decisions = maybe_adjudicate_overlap_follow_decisions(
        active_speakers_local=[
            {
                "start_time_ms": 0,
                "end_time_ms": 300,
                "audio_speaker_ids": ["SPEAKER_00"],
                "visible_local_track_ids": ["track_1"],
                "visible_track_ids": ["Global_Person_0"],
                "offscreen_audio_speaker_ids": [],
                "overlap": False,
                "confidence": 0.93,
                "decision_source": "turn_binding",
            },
            {
                "start_time_ms": 300,
                "end_time_ms": 600,
                "audio_speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
                "visible_local_track_ids": ["track_1"],
                "visible_track_ids": ["Global_Person_0"],
                "offscreen_audio_speaker_ids": ["SPEAKER_01"],
                "overlap": True,
                "confidence": 0.71,
                "decision_source": "turn_binding",
            },
            {
                "start_time_ms": 600,
                "end_time_ms": 900,
                "audio_speaker_ids": ["SPEAKER_00"],
                "visible_local_track_ids": ["track_1"],
                "visible_track_ids": ["Global_Person_0"],
                "offscreen_audio_speaker_ids": [],
                "overlap": False,
                "confidence": 0.89,
                "decision_source": "turn_binding",
            },
        ],
        words=[{"word": "hello", "start_time_ms": 350, "end_time_ms": 500}],
        speaker_candidate_debug=[],
        client=_FakeClient(),
        model_name="gemini-3-flash-preview",
        enabled=True,
    )

    assert decisions[0]["camera_target_local_track_id"] == "track_1"
    assert captured["model"] == "gemini-3-flash-preview"
    assert '"previous_context"' in captured["contents"]
    assert '"next_context"' in captured["contents"]
    assert '"SPEAKER_01"' in captured["contents"]
    assert captured["config"] is not None
    assert captured["config"]["response_mime_type"] == "application/json"


def test_load_gemini_client_uses_vertex_ai_when_configured(monkeypatch):
    import sys
    import types

    from backend.overlap_follow import _load_gemini_client

    calls = []

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

    fake_google = types.ModuleType("google")
    fake_genai = types.SimpleNamespace(Client=_FakeClient)
    fake_google.genai = fake_genai

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "clypt-v2")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")

    client = _load_gemini_client()

    assert client is not None
    assert calls == [{"vertexai": True, "project": "clypt-v2", "location": "global"}]
