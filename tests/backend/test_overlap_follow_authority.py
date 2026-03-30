"""Authority boundary: overlap-follow adjudication must not mutate transcript words or speaker_* fields."""

from __future__ import annotations

import copy
import json

from backend.do_phase1_worker import ClyptWorker
from backend.overlap_follow import maybe_adjudicate_overlap_follow_decisions


def _words_canonical_snapshot(words: list[dict]) -> bytes:
    return json.dumps(words, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")


def _speaker_projection(words: list[dict]) -> list[dict]:
    """Per-word map of keys starting with speaker_ (stable for equality checks)."""
    out: list[dict] = []
    for w in words:
        out.append({k: v for k, v in w.items() if str(k).startswith("speaker_")})
    return out


def _overlap_active_span():
    return {
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


def _rich_words_for_overlap():
    return [
        {
            "word": "hello",
            "start_time_ms": 450,
            "end_time_ms": 700,
            "speaker_track_id": "Global_Person_0",
            "speaker_local_track_id": "track_1",
            "speaker_tag": "SPK_A",
            "speaker_local_tag": "loc_a",
            "speaker_confidence": 0.88,
            "meta": {"nested": [1, 2]},
        }
    ]


def test_maybe_adjudicate_overlap_follow_authority_preserves_words_disabled():
    words = _rich_words_for_overlap()
    words_before = copy.deepcopy(words)
    snap_before = _words_canonical_snapshot(words)
    spk_before = _speaker_projection(words)

    decisions = maybe_adjudicate_overlap_follow_decisions(
        active_speakers_local=[_overlap_active_span()],
        words=words,
        speaker_candidate_debug=[],
        client=None,
        enabled=False,
    )

    assert decisions
    assert words == words_before
    assert _words_canonical_snapshot(words) == snap_before
    assert _speaker_projection(words) == spk_before


def test_maybe_adjudicate_overlap_follow_authority_preserves_words_gemini_path(monkeypatch):
    class _FakeResponse:
        text = (
            '{"camera_target_local_track_id":"track_1","camera_target_track_id":"Global_Person_0",'
            '"stay_wide":false,"decision_source":"gemini","confidence":0.84}'
        )

    class _FakeModels:
        def generate_content(self, *, model, contents, config=None):
            return _FakeResponse()

    class _FakeClient:
        models = _FakeModels()

    monkeypatch.setenv("CLYPT_OVERLAP_FOLLOW_MIN_CONFIDENCE", "0")
    monkeypatch.setenv("CLYPT_OVERLAP_FOLLOW_MIN_EVIDENCE_SCORE", "0")

    words = _rich_words_for_overlap()
    words_before = copy.deepcopy(words)
    snap_before = _words_canonical_snapshot(words)
    spk_before = _speaker_projection(words)

    decisions = maybe_adjudicate_overlap_follow_decisions(
        active_speakers_local=[_overlap_active_span()],
        words=words,
        speaker_candidate_debug=[],
        client=_FakeClient(),
        model_name="gemini-3-flash-preview",
        enabled=True,
    )

    assert decisions[0]["decision_source"] == "gemini"
    assert words == words_before
    assert _words_canonical_snapshot(words) == snap_before
    assert _speaker_projection(words) == spk_before


def test_maybe_adjudicate_overlap_follow_authority_preserves_words_no_client(monkeypatch):
    monkeypatch.setattr(
        "backend.overlap_follow._load_gemini_client",
        lambda client=None: None,
    )
    words = _rich_words_for_overlap()
    snap_before = _words_canonical_snapshot(words)

    maybe_adjudicate_overlap_follow_decisions(
        active_speakers_local=[_overlap_active_span()],
        words=words,
        speaker_candidate_debug=[],
        client=None,
        enabled=True,
    )

    assert _words_canonical_snapshot(words) == snap_before


def test_maybe_adjudicate_overlap_follow_authority_preserves_words_on_gemini_failure(monkeypatch):
    class _BoomModels:
        def generate_content(self, *, model, contents, config=None):
            raise RuntimeError("boom")

    class _BoomClient:
        models = _BoomModels()

    monkeypatch.setenv("CLYPT_OVERLAP_FOLLOW_MIN_CONFIDENCE", "0")
    monkeypatch.setenv("CLYPT_OVERLAP_FOLLOW_MIN_EVIDENCE_SCORE", "0")

    words = _rich_words_for_overlap()
    snap_before = _words_canonical_snapshot(words)

    maybe_adjudicate_overlap_follow_decisions(
        active_speakers_local=[_overlap_active_span()],
        words=words,
        speaker_candidate_debug=[],
        client=_BoomClient(),
        model_name="gemini-3-flash-preview",
        enabled=True,
    )

    assert _words_canonical_snapshot(words) == snap_before


def test_maybe_adjudicate_overlap_follow_authority_preserves_speaker_candidate_debug(monkeypatch):
    """Debug rows are read-only input; overlap-follow must not mutate their speaker-related fields."""

    class _FakeModels:
        def generate_content(self, *, model, contents, config=None):
            raise AssertionError("gated before Gemini")

    class _FakeClient:
        models = _FakeModels()

    monkeypatch.setenv("CLYPT_OVERLAP_FOLLOW_MIN_EVIDENCE_SCORE", "99")

    debug = [
        {
            "start_time_ms": 500,
            "end_time_ms": 900,
            "speaker_id": "SPEAKER_00",
            "speaker_track_hint": "Global_Person_0",
            "top_1_top_2_margin": 0.05,
        }
    ]
    debug_before = copy.deepcopy(debug)

    maybe_adjudicate_overlap_follow_decisions(
        active_speakers_local=[_overlap_active_span()],
        words=_rich_words_for_overlap(),
        speaker_candidate_debug=debug,
        client=_FakeClient(),
        model_name="gemini-3-flash-preview",
        enabled=True,
    )

    assert debug == debug_before


def test_finalize_overlap_follow_authority_worker_words_unchanged(monkeypatch):
    """Integration: real overlap postpass attaches decisions without changing word speaker labels."""
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)

    tracks = [
        {
            "frame_idx": 0,
            "track_id": "local-1",
            "class_id": 0,
            "label": "person",
            "geometry_type": "aabb",
            "x1": 10.0,
            "y1": 20.0,
            "x2": 110.0,
            "y2": 220.0,
            "x_center": 60.0,
            "y_center": 120.0,
            "width": 100.0,
            "height": 200.0,
            "confidence": 0.9,
        }
    ]
    words = [
        {
            "text": "hello",
            "start_time_ms": 0,
            "end_time_ms": 100,
            "speaker_track_id": "Global_Person_0",
            "speaker_tag": "Global_Person_0",
            "speaker_local_track_id": "local-1",
            "speaker_local_tag": "local-1",
        }
    ]
    audio_turns = [
        {"speaker_id": "SPEAKER_00", "start_time_ms": 0, "end_time_ms": 600, "exclusive": True},
        {
            "speaker_id": "SPEAKER_01",
            "start_time_ms": 300,
            "end_time_ms": 800,
            "exclusive": False,
            "overlap": True,
        },
    ]

    class _FakeResponse:
        text = (
            '{"camera_target_local_track_id":"local-1","camera_target_track_id":"Global_Person_0",'
            '"stay_wide":false,"decision_source":"gemini","confidence":0.81}'
        )

    class _FakeModels:
        def generate_content(self, *, model, contents, config=None):
            return _FakeResponse()

    class _FakeClient:
        models = _FakeModels()

    monkeypatch.setenv("CLYPT_EXPERIMENT_LOCAL_CLIP_BINDINGS", "1")
    monkeypatch.setenv("CLYPT_OVERLAP_FOLLOW_MIN_CONFIDENCE", "0")
    monkeypatch.setenv("CLYPT_OVERLAP_FOLLOW_MIN_EVIDENCE_SCORE", "0")
    monkeypatch.setattr("backend.overlap_follow._load_gemini_client", lambda client=None: _FakeClient())

    monkeypatch.setattr(worker, "_tracking_contract_pass_rate", lambda tracks: 1.0)
    monkeypatch.setattr(worker, "_validate_tracking_contract", lambda tracks: None)
    monkeypatch.setattr(worker, "_enforce_rollout_gates", lambda metrics: None)
    monkeypatch.setattr(worker, "_build_track_indexes", lambda tracks: ({0: tracks}, {"local-1": tracks}))
    monkeypatch.setattr(worker, "_build_visual_detection_ledgers", lambda *args, **kwargs: ([], [], {}))
    monkeypatch.setattr(worker, "_run_audio_diarization", lambda audio_path: audio_turns)

    def _fake_run_speaker_binding(*args, **kwargs):
        worker._last_audio_turn_bindings = [
            {
                "speaker_id": "SPEAKER_00",
                "start_time_ms": 0,
                "end_time_ms": 600,
                "local_track_id": "local-1",
                "ambiguous": False,
                "winning_score": 0.93,
                "winning_margin": 0.28,
                "support_ratio": 0.96,
            },
            {
                "speaker_id": "SPEAKER_01",
                "start_time_ms": 300,
                "end_time_ms": 800,
                "local_track_id": None,
                "ambiguous": True,
                "winning_score": 0.41,
                "winning_margin": 0.02,
                "support_ratio": 0.32,
            },
        ]
        return [
            {"track_id": "Global_Person_0", "start_time_ms": 0, "end_time_ms": 100, "word_count": 1}
        ]

    def _fake_cluster(video_path, tracks, **kwargs):
        worker._last_cluster_id_map = {"local-1": "Global_Person_0"}
        worker._last_track_identity_features_after_clustering = None
        return [{**track, "track_id": "Global_Person_0"} for track in tracks]

    monkeypatch.setattr(worker, "_cluster_tracklets", _fake_cluster)
    monkeypatch.setattr(worker, "_run_speaker_binding", _fake_run_speaker_binding)
    monkeypatch.setattr(worker, "_build_speaker_follow_bindings", lambda bindings: list(bindings))
    monkeypatch.setattr(worker, "_speaker_remap_collision_metrics", lambda w: {})
    monkeypatch.setattr(worker, "_local_clip_bindings_enabled", lambda: True)

    words_snap_before = _words_canonical_snapshot(words)
    spk_before = _speaker_projection(words)

    result = worker._finalize_from_words_tracks(
        video_path="video.mp4",
        audio_path="audio.wav",
        youtube_url="https://youtube.com/watch?v=overlap-authority",
        words=words,
        tracks=tracks,
        tracking_metrics={"schema_pass_rate": 1.0},
    )

    assert result["phase_1_audio"]["overlap_follow_decisions"]
    assert any(
        d.get("decision_source") == "gemini" for d in result["phase_1_audio"]["overlap_follow_decisions"]
    )
    assert _words_canonical_snapshot(words) == words_snap_before
    assert _speaker_projection(words) == spk_before
