"""Wave 2 phase1 seams: shared decode context + post-track ReID merge."""

from __future__ import annotations

import numpy as np

from backend.pipeline.phase1.clustering import post_track_reid_merge
from backend.pipeline.phase1.decode_cache import Phase1AnalysisContext
from backend.pipeline.phase1.tracking import run_tracking_stage


def test_phase1_analysis_context_reuses_single_prepare() -> None:
    calls = {"n": 0}

    class _W:
        def _prepare_direct_analysis_context(self, path: str) -> dict:
            calls["n"] += 1
            return {
                "source_video_path": path,
                "prepared_video_path": path,
                "analysis_video_path": path,
                "source_meta": {"width": 1920, "height": 1080},
                "analysis_meta": {"width": 1920, "height": 1080, "fps": 25.0},
                "scale_x": 1.0,
                "scale_y": 1.0,
            }

        def _prepare_analysis_video(self, path: str) -> dict:
            raise AssertionError("should not be called in direct mode")

    ctx = Phase1AnalysisContext("/tmp/fake_video.mp4")
    w = _W()
    first = ctx.ensure_prepared(w, tracking_mode="direct")
    second = ctx.ensure_prepared(w, tracking_mode="direct")
    assert first is second
    assert calls["n"] == 1
    assert ctx.prepare_invocations == 1
    assert ctx.cache_hits >= 1


def test_run_tracking_stage_reuses_context_across_internal_calls() -> None:
    """``run_tracking_stage`` must attach one :class:`Phase1AnalysisContext` to metrics."""

    class _Worker:
        def _select_tracking_mode(self) -> str:
            return "direct"

        def _run_tracking_direct(self, video_path: str, analysis_context=None):
            assert analysis_context is not None
            assert analysis_context["analysis_video_path"] == video_path
            return [], {"schema_pass_rate": 1.0, "analysis_context": analysis_context}

    w = _Worker()
    ctx = Phase1AnalysisContext("/tmp/x.mp4")
    ctx.bind_payload(
        {
            "source_video_path": "/tmp/x.mp4",
            "prepared_video_path": "/tmp/x.mp4",
            "analysis_video_path": "/tmp/x.mp4",
            "source_meta": {},
            "analysis_meta": {"fps": 25.0},
            "scale_x": 1.0,
            "scale_y": 1.0,
        }
    )
    tracks, metrics = run_tracking_stage(w, "/tmp/x.mp4", ctx)
    assert tracks == []
    assert metrics.get("phase1_decode_context") is ctx
    assert metrics.get("analysis_context") == ctx.ensure_prepared(w, tracking_mode="direct")


def _emb() -> list[float]:
    v = np.ones(128, dtype=np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def test_post_track_reid_merge_merges_same_shot_strong_evidence() -> None:
    fps = 25.0
    emb = _emb()
    tracks: list[dict] = []
    for fi in range(10, 21):
        tracks.append({"track_id": "track_1", "frame_idx": fi})
    for fi in range(100, 111):
        tracks.append({"track_id": "track_2", "frame_idx": fi})

    features = {
        "1": {"embedding": emb, "embedding_count": 3, "face_observations": []},
        "2": {"embedding": emb, "embedding_count": 3, "face_observations": []},
    }
    shots = [{"start_time_ms": 0, "end_time_ms": 30_000}, {"start_time_ms": 30_000, "end_time_ms": 60_000}]

    out, out_f, m = post_track_reid_merge(
        tracks,
        features,
        shot_timeline_ms=shots,
        video_fps=fps,
        min_cos_sim=0.85,
    )
    assert m["post_track_reid_merge_merged_pairs"] >= 1
    assert m.get("post_track_reid_merge_components_merged", 0) >= 1
    tids = {str(t["track_id"]) for t in out}
    assert tids == {"track_1"}, "fragmented ids should collapse to one canonical id"
    assert out_f is not None
    assert len(out_f) == 1


def test_post_track_reid_merge_skips_cross_shot_even_when_similar() -> None:
    fps = 25.0
    emb_a = _emb()
    emb_b = np.array(emb_a, dtype=np.float32) * 0.99 + 0.01 / 128.0
    emb_b = (emb_b / np.linalg.norm(emb_b)).tolist()

    tracks: list[dict] = []
    for fi in range(10, 21):
        tracks.append({"track_id": "track_1", "frame_idx": fi})
    # 400 frames -> center ~16s at 25fps -> 16000ms in second shot (below)
    for fi in range(390, 401):
        tracks.append({"track_id": "track_2", "frame_idx": fi})

    features = {
        "1": {"embedding": emb_a, "embedding_count": 2, "face_observations": []},
        "2": {"embedding": emb_b, "embedding_count": 2, "face_observations": []},
    }
    shots = [{"start_time_ms": 0, "end_time_ms": 12_000}, {"start_time_ms": 12_000, "end_time_ms": 60_000}]

    out, out_f, m = post_track_reid_merge(
        tracks,
        features,
        shot_timeline_ms=shots,
        video_fps=fps,
        min_cos_sim=0.80,
    )
    assert m["post_track_reid_merge_merged_pairs"] == 0
    assert m["post_track_reid_merge_skipped_shot_incompatible"] >= 1
    tids = {str(t["track_id"]) for t in out}
    assert "track_1" in tids and "track_2" in tids
    assert out_f is not None
    assert len(out_f) == 2


def test_post_track_reid_merge_supports_numeric_track_ids() -> None:
    fps = 25.0
    emb = _emb()
    tracks = [{"track_id": 1, "frame_idx": i} for i in range(10, 21)] + [
        {"track_id": 2, "frame_idx": i} for i in range(100, 111)
    ]
    features = {
        "1": {"embedding": emb, "embedding_count": 2, "face_observations": []},
        "2": {"embedding": emb, "embedding_count": 2, "face_observations": []},
    }
    shots = [{"start_time_ms": 0, "end_time_ms": 60_000}]
    out, out_f, m = post_track_reid_merge(
        tracks,
        features,
        shot_timeline_ms=shots,
        video_fps=fps,
        min_cos_sim=0.85,
    )
    assert m["post_track_reid_merge_merged_pairs"] >= 1
    assert {str(t["track_id"]) for t in out} == {"track_1"}
    assert out_f is not None and len(out_f) == 1


def test_post_track_reid_merge_supports_chunk_prefixed_track_ids() -> None:
    fps = 25.0
    emb = _emb()
    tracks = [{"track_id": "chunk_0_track_1", "frame_idx": i} for i in range(10, 21)] + [
        {"track_id": "chunk_1_track_1", "frame_idx": i} for i in range(100, 111)
    ]
    features = {
        "chunk_0_track_1": {"embedding": emb, "embedding_count": 2, "face_observations": []},
        "chunk_1_track_1": {"embedding": emb, "embedding_count": 2, "face_observations": []},
    }
    shots = [{"start_time_ms": 0, "end_time_ms": 60_000}]
    out, out_f, m = post_track_reid_merge(
        tracks,
        features,
        shot_timeline_ms=shots,
        video_fps=fps,
        min_cos_sim=0.85,
    )
    assert m["post_track_reid_merge_merged_pairs"] >= 1
    assert len({str(t["track_id"]) for t in out}) == 1
    assert out_f is not None and len(out_f) == 1


def test_post_track_reid_merge_tracks_missing_embedding_metric() -> None:
    fps = 25.0
    emb = _emb()
    tracks = [{"track_id": "track_1", "frame_idx": i} for i in range(10, 21)] + [
        {"track_id": "track_2", "frame_idx": i} for i in range(100, 111)
    ]
    features = {
        "1": {"embedding": emb, "embedding_count": 2, "face_observations": []},
    }
    shots = [{"start_time_ms": 0, "end_time_ms": 60_000}]
    _, _, m = post_track_reid_merge(
        tracks,
        features,
        shot_timeline_ms=shots,
        video_fps=fps,
        min_cos_sim=0.85,
    )
    assert m["post_track_reid_merge_skipped_missing_embedding"] >= 1
