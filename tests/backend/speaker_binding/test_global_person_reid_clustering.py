from __future__ import annotations

from backend.do_phase1_worker import ClyptWorker


def _make_tracklet(track_id: str, x_center: float, *, y_center: float = 120.0) -> list[dict]:
    return [
        {
            "track_id": track_id,
            "frame_idx": 0,
            "x_center": x_center,
            "y_center": y_center,
            "width": 100.0,
            "height": 200.0,
        },
        {
            "track_id": track_id,
            "frame_idx": 1,
            "x_center": x_center + 5.0,
            "y_center": y_center,
            "width": 100.0,
            "height": 200.0,
        },
    ]


def _make_covisible_tracklets() -> dict[str, list[dict]]:
    return {
        "42": _make_tracklet("42", 120.0),
        "43": _make_tracklet("43", 320.0),
        "44": _make_tracklet("44", 520.0),
    }


def test_repair_covisible_cluster_merges_splits_simultaneous_distinct_people() -> None:
    worker = object.__new__(ClyptWorker)
    tracklets = _make_covisible_tracklets()
    merged = {"42": 0, "43": 0, "44": 0}

    repaired, metrics = worker._repair_covisible_cluster_merges(
        tracklets,
        merged,
        anchored_tids={"42", "43", "44"},
    )

    assert repaired == {"42": 0, "43": 1, "44": 2}
    assert metrics == {
        "repaired_cluster_count": 1,
        "repaired_tracklet_count": 3,
        "repaired_conflict_pair_count": 3,
    }


def test_repair_skip_is_blocked_when_collision_metrics_are_severe() -> None:
    tracklets = _make_covisible_tracklets()
    merged = {"42": 0, "43": 0, "44": 0}

    collision_metrics = ClyptWorker._same_identity_frame_collision_metrics(tracklets, merged)

    assert collision_metrics == {
        "same_identity_frame_collision_pairs": 3,
        "same_identity_frame_collision_frames": 2,
        "same_identity_labels_with_collisions": 1,
    }
    assert (
        ClyptWorker._should_skip_cluster_repair(
            face_cluster_count=6,
            clusters_before_repair=6,
            visible_people_est=6,
            anchored_track_count=101,
            collision_pair_count=collision_metrics["same_identity_frame_collision_pairs"],
            collision_frame_count=collision_metrics["same_identity_frame_collision_frames"],
            collision_label_count=collision_metrics["same_identity_labels_with_collisions"],
        )
        is False
    )
