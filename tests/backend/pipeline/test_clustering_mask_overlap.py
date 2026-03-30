"""Mask-overlap / stability-aware histogram attachment (Wave 4 clustering)."""

from __future__ import annotations

import pytest

from backend.do_phase1_worker import ClyptWorker
from backend.pipeline.phase1.clustering import (
    build_mask_overlap_clustering_signals,
    mask_overlap_attachment_cost_reduction,
)


def test_build_mask_overlap_signals_inactive_without_pairs_or_face_overlap():
    tracklets = {
        "only": [
            {
                "frame_idx": 5,
                "x_center": 0.5,
                "y_center": 0.5,
                "width": 0.2,
                "height": 0.5,
                "confidence": 0.9,
            },
        ],
    }
    ctx = build_mask_overlap_clustering_signals(
        tracklets,
        None,
        video_fps=25.0,
        duration_ms=10_000,
    )
    assert ctx["active"] is False
    assert ctx["per_track"]["only"]["stab"] == 0.0


def test_build_mask_overlap_signals_active_from_consecutive_iou_without_face_obs():
    tracklets = {
        "t": [
            {
                "frame_idx": 10,
                "x_center": 320.0,
                "y_center": 240.0,
                "width": 120.0,
                "height": 200.0,
                "confidence": 0.9,
            },
            {
                "frame_idx": 11,
                "x_center": 321.0,
                "y_center": 240.0,
                "width": 120.0,
                "height": 200.0,
                "confidence": 0.9,
            },
        ],
    }
    ctx = build_mask_overlap_clustering_signals(
        tracklets,
        None,
        video_fps=25.0,
        duration_ms=60_000,
    )
    assert ctx["active"] is True
    assert ctx["per_track"]["t"]["mean_consecutive_iou"] > 0.85
    assert mask_overlap_attachment_cost_reduction(["t"], ["t"], ctx) == 0.0


def test_mask_overlap_reduction_positive_when_face_and_iou_proxies_present(monkeypatch):
    monkeypatch.setenv("CLYPT_CLUSTER_MASK_OVERLAP_ATTACH_WEIGHT", "1.0")
    tracklets = {
        "h0": [
            {
                "frame_idx": 14,
                "x_center": 399.0,
                "y_center": 200.0,
                "width": 80.0,
                "height": 160.0,
                "confidence": 0.9,
            },
            {
                "frame_idx": 15,
                "x_center": 400.0,
                "y_center": 200.0,
                "width": 80.0,
                "height": 160.0,
                "confidence": 0.9,
            },
        ],
        "face_a": [
            {
                "frame_idx": 14,
                "x_center": 396.0,
                "y_center": 200.0,
                "width": 80.0,
                "height": 160.0,
                "confidence": 0.95,
            },
            {
                "frame_idx": 15,
                "x_center": 397.0,
                "y_center": 200.0,
                "width": 80.0,
                "height": 160.0,
                "confidence": 0.95,
            },
        ],
    }
    tidf = {
        "h0": {"face_observations": [{"frame_idx": 14}, {"frame_idx": 15}]},
        "face_a": {"face_observations": [{"frame_idx": 14}, {"frame_idx": 15}]},
    }
    ctx = build_mask_overlap_clustering_signals(
        tracklets,
        tidf,
        video_fps=25.0,
        duration_ms=60_000,
    )
    assert ctx["active"] is True
    red = mask_overlap_attachment_cost_reduction(["h0"], ["face_a"], ctx)
    assert red > 0.05


def test_hist_groups_hungarian_aux_metrics_when_mask_term_applies(monkeypatch):
    monkeypatch.setenv("CLYPT_CLUSTER_MASK_OVERLAP_ATTACH_WEIGHT", "8.0")
    worker = ClyptWorker.__new__(ClyptWorker)
    tracklets = {
        "face_a": [
            {
                "frame_idx": 14,
                "x_center": 396.0,
                "y_center": 200.0,
                "width": 80.0,
                "height": 160.0,
                "confidence": 0.95,
            },
            {
                "frame_idx": 15,
                "x_center": 397.0,
                "y_center": 200.0,
                "width": 80.0,
                "height": 160.0,
                "confidence": 0.95,
            },
        ],
        "face_b": [
            {
                "frame_idx": 14,
                "x_center": 402.0,
                "y_center": 200.0,
                "width": 80.0,
                "height": 160.0,
                "confidence": 0.95,
            },
            {
                "frame_idx": 15,
                "x_center": 403.0,
                "y_center": 200.0,
                "width": 80.0,
                "height": 160.0,
                "confidence": 0.95,
            },
        ],
        "h0": [
            {
                "frame_idx": 14,
                "x_center": 399.0,
                "y_center": 200.0,
                "width": 80.0,
                "height": 160.0,
                "confidence": 0.9,
            },
            {
                "frame_idx": 15,
                "x_center": 400.0,
                "y_center": 200.0,
                "width": 80.0,
                "height": 160.0,
                "confidence": 0.9,
            },
        ],
    }
    tidf = {
        "face_a": {"face_observations": [{"frame_idx": 14}, {"frame_idx": 15}]},
        "face_b": {"face_observations": [{"frame_idx": 14}]},
        "h0": {"face_observations": [{"frame_idx": 14}, {"frame_idx": 15}]},
    }
    aux: dict[str, int | float] = {}
    worker._hist_groups_hungarian_signature_assign(
        hist_groups=[["h0"]],
        tracklets=tracklets,
        face_label_by_tid={"face_a": 0, "face_b": 1},
        histogram_attach_max_sig=1.15,
        shot_segments=[{"start_time_ms": 0, "end_time_ms": 60_000}],
        video_fps=25.0,
        duration_ms=60_000,
        track_identity_features=tidf,
        aux_metrics=aux,
    )
    assert aux.get("cluster_mask_overlap_term_active") == 1
    assert int(aux.get("cluster_mask_overlap_assignment_cells_adjusted", 0)) >= 1
    assert float(aux.get("cluster_mask_overlap_cost_reduction_sum", 0.0)) > 1.0


def test_hist_groups_hungarian_no_mask_metrics_when_signals_inactive(monkeypatch):
    monkeypatch.setenv("CLYPT_CLUSTER_MASK_OVERLAP_ATTACH_WEIGHT", "8.0")
    worker = ClyptWorker.__new__(ClyptWorker)
    tracklets = {
        "face_a": [
            {
                "frame_idx": 10,
                "x_center": 100.0,
                "y_center": 200.0,
                "width": 80.0,
                "height": 160.0,
                "confidence": 0.95,
            },
        ],
        "face_b": [
            {
                "frame_idx": 10,
                "x_center": 400.0,
                "y_center": 200.0,
                "width": 80.0,
                "height": 160.0,
                "confidence": 0.95,
            },
        ],
        "h0": [
            {
                "frame_idx": 12,
                "x_center": 402.0,
                "y_center": 200.0,
                "width": 80.0,
                "height": 160.0,
                "confidence": 0.9,
            },
        ],
        "h1": [
            {
                "frame_idx": 12,
                "x_center": 98.0,
                "y_center": 200.0,
                "width": 80.0,
                "height": 160.0,
                "confidence": 0.9,
            },
        ],
    }
    aux: dict[str, int | float] = {}
    worker._hist_groups_hungarian_signature_assign(
        hist_groups=[["h0"], ["h1"]],
        tracklets=tracklets,
        face_label_by_tid={"face_a": 0, "face_b": 1},
        histogram_attach_max_sig=1.15,
        shot_segments=[{"start_time_ms": 0, "end_time_ms": 60_000}],
        video_fps=25.0,
        duration_ms=60_000,
        track_identity_features=None,
        aux_metrics=aux,
    )
    assert aux.get("cluster_mask_overlap_term_active") == 0
    assert int(aux.get("cluster_mask_overlap_assignment_cells_adjusted", 0)) == 0
    assert float(aux.get("cluster_mask_overlap_cost_reduction_sum", 0.0)) == 0.0


def test_hist_groups_hungarian_cross_assignment_unchanged_when_mask_inactive(monkeypatch):
    """Regression: cross-cluster Hungarian case has no iou pairs / face overlap → no mask term."""
    monkeypatch.setenv("CLYPT_CLUSTER_MASK_OVERLAP_ATTACH_WEIGHT", "0.5")
    worker_cls = ClyptWorker._get_user_cls()
    worker = worker_cls.__new__(worker_cls)
    tracklets = {
        "face_a": [
            {
                "frame_idx": 10,
                "x_center": 100.0,
                "y_center": 200.0,
                "width": 80.0,
                "height": 160.0,
                "confidence": 0.95,
            },
        ],
        "face_b": [
            {
                "frame_idx": 10,
                "x_center": 400.0,
                "y_center": 200.0,
                "width": 80.0,
                "height": 160.0,
                "confidence": 0.95,
            },
        ],
        "h0": [
            {
                "frame_idx": 12,
                "x_center": 402.0,
                "y_center": 200.0,
                "width": 80.0,
                "height": 160.0,
                "confidence": 0.9,
            },
        ],
        "h1": [
            {
                "frame_idx": 12,
                "x_center": 98.0,
                "y_center": 200.0,
                "width": 80.0,
                "height": 160.0,
                "confidence": 0.9,
            },
        ],
    }
    picks = worker._hist_groups_hungarian_signature_assign(
        hist_groups=[["h0"], ["h1"]],
        tracklets=tracklets,
        face_label_by_tid={"face_a": 0, "face_b": 1},
        histogram_attach_max_sig=1.15,
        shot_segments=[{"start_time_ms": 0, "end_time_ms": 60_000}],
        video_fps=25.0,
        duration_ms=60_000,
        track_identity_features=None,
        aux_metrics=None,
    )
    assert picks[0] == 1
    assert picks[1] == 0
