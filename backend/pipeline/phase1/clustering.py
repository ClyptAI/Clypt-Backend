"""Track clustering stage + conservative post-track ReID merge (before global clustering).

v3 wave order (see docs/superpowers/specs/clypt_v3_refactor_spec.md):
Wave 3 — Hybrid clustering uses face-primary + same-shot signature fallback with shot ID and
temporal proximity; no mask-overlap term in base signature scoring.
Wave 4 — After segmentation masks are integrated on the execution path, optional mask/bbox
stability signals may activate (e.g. ``build_mask_overlap_clustering_signals`` →
``mask_overlap_attachment_cost_reduction``); callers gate on ``mask_ctx["active"]``.
"""

from __future__ import annotations

import os
from collections import defaultdict
from statistics import fmean
from typing import Any

import numpy as np

# Conservative defaults — high similarity, same-shot only, no temporal overlap.
_POST_REID_MIN_COS_SIM = 0.88
_POST_REID_MAX_CANDIDATE_PAIRS = 800


def _norm_track_id(tid: str) -> str:
    s = str(tid).strip()
    if s.startswith("track_"):
        return s
    return f"track_{s}"


def _feature_key_for_track(track_identity_features: dict[str, dict], tid: str) -> str | None:
    """Resolve storage key for a logical track id."""
    n = _norm_track_id(tid)
    for k in track_identity_features.keys():
        if _norm_track_id(k) == n:
            return str(k)
    return None


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _track_frame_span(tracks: list[dict], tid: str) -> tuple[int, int] | None:
    frames: list[int] = []
    for t in tracks:
        if _norm_track_id(str(t.get("track_id", ""))) != _norm_track_id(tid):
            continue
        fi = int(t.get("frame_idx", -1))
        if fi >= 0:
            frames.append(fi)
    if not frames:
        return None
    return min(frames), max(frames)


def _covisible(span_a: tuple[int, int], span_b: tuple[int, int]) -> bool:
    a0, a1 = span_a
    b0, b1 = span_b
    return max(a0, b0) <= min(a1, b1)


def _bbox_iou_norm_xywh(det_a: dict, det_b: dict) -> float:
    """IoU in normalized x_center/y_center/width/height space (matches worker `_detection_iou`)."""
    ax1 = float(det_a.get("x_center", 0.0)) - 0.5 * float(det_a.get("width", 0.0))
    ay1 = float(det_a.get("y_center", 0.0)) - 0.5 * float(det_a.get("height", 0.0))
    ax2 = ax1 + float(det_a.get("width", 0.0))
    ay2 = ay1 + float(det_a.get("height", 0.0))
    bx1 = float(det_b.get("x_center", 0.0)) - 0.5 * float(det_b.get("width", 0.0))
    by1 = float(det_b.get("y_center", 0.0)) - 0.5 * float(det_b.get("height", 0.0))
    bx2 = bx1 + float(det_b.get("width", 0.0))
    by2 = by1 + float(det_b.get("height", 0.0))
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(1.0, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return float(inter / union)


def build_mask_overlap_clustering_signals(
    track_to_dets: dict[str, list[dict]],
    track_identity_features: dict[str, dict] | None,
    *,
    video_fps: float,
    duration_ms: int,
) -> dict[str, Any]:
    """Per-track proxies aligned with worker `mask_stability_signals` / bbox continuity (Wave 4).

    Used only for optional histogram→face attachment cost shaping. When ``active`` is False,
    callers must not apply mask-overlap terms.
    """
    fps = max(1e-6, float(video_fps or 25.0))
    duration_ms = max(0, int(duration_ms))
    max_frame_gap = 5

    max_det_frame = max(
        (int(d.get("frame_idx", -1)) for dets in track_to_dets.values() for d in dets),
        default=-1,
    )
    if max_det_frame < 0:
        max_det_frame = int(min(1_000_000, round((duration_ms / 1000.0) * fps)))

    window_span = max(30, min(300, int(round(fps * 3.0))))
    win_end = max_det_frame
    win_start = max(0, win_end - window_span + 1)

    person_keys: set[tuple[str, int]] = set()
    for tid in sorted(track_to_dets.keys()):
        for d in track_to_dets[tid]:
            fi = int(d.get("frame_idx", -1))
            if fi < 0:
                continue
            if win_start <= fi <= win_end:
                person_keys.add((str(tid), fi))

    face_keys: set[tuple[str, int]] = set()
    if isinstance(track_identity_features, dict) and track_identity_features:
        for tid, feat in track_identity_features.items():
            tid_s = str(tid)
            if not tid_s:
                continue
            for obs in feat.get("face_observations") or []:
                if not isinstance(obs, dict):
                    continue
                fi = int(obs.get("frame_idx", -1))
                if fi < 0:
                    continue
                if win_start <= fi <= win_end:
                    face_keys.add((tid_s, fi))

    overlap_pf = person_keys & face_keys

    iou_values: list[float] = []
    per_track_iou_vals: dict[str, list[float]] = defaultdict(list)
    for tid in sorted(track_to_dets.keys()):
        window_dets = [
            d
            for d in track_to_dets[tid]
            if win_start <= int(d.get("frame_idx", -1)) <= win_end
        ]
        window_dets.sort(key=lambda x: int(x["frame_idx"]))
        for i in range(len(window_dets) - 1):
            a, b = window_dets[i], window_dets[i + 1]
            fa, fb = int(a.get("frame_idx", -1)), int(b.get("frame_idx", -1))
            gap = fb - fa
            if gap <= 0 or gap > max_frame_gap:
                continue
            iou_v = float(_bbox_iou_norm_xywh(a, b))
            iou_values.append(iou_v)
            per_track_iou_vals[str(tid)].append(iou_v)

    person_frames_by_tid: dict[str, set[int]] = defaultdict(set)
    for tid, fi in person_keys:
        person_frames_by_tid[tid].add(fi)
    overlap_frames_by_tid: dict[str, set[int]] = defaultdict(set)
    for tid, fi in overlap_pf:
        overlap_frames_by_tid[tid].add(fi)

    per_track: dict[str, dict[str, float]] = {}
    for tid in person_frames_by_tid:
        pcount = len(person_frames_by_tid[tid])
        fp_ratio = len(overlap_frames_by_tid[tid]) / max(1, pcount)
        ious = per_track_iou_vals.get(tid) or []
        mean_iou = float(fmean(ious)) if ious else 0.0
        stab = float(0.55 * fp_ratio + 0.45 * mean_iou)
        per_track[tid] = {
            "face_person_overlap_ratio": round(fp_ratio, 6),
            "mean_consecutive_iou": round(mean_iou, 6),
            "stab": round(stab, 6),
        }

    active = bool(person_keys) and (bool(overlap_pf) or bool(iou_values))
    return {
        "active": active,
        "per_track": per_track,
        "signal_version": "cluster_attach_bbox_v3",
        "short_term_stability_memory": {
            "window_start_frame": win_start,
            "window_end_frame": win_end,
            "face_person_frame_overlap": len(overlap_pf),
            "iou_consecutive_pair_count": len(iou_values),
            "person_track_frames_in_window": len(person_keys),
        },
    }


def mask_overlap_attachment_cost_reduction(
    group_tids: list[str],
    cluster_tids: list[str],
    mask_ctx: dict[str, Any],
) -> float:
    """Non-negative reduction subtracted from Hungarian cost when mask/stability context is active."""
    if not mask_ctx.get("active"):
        return 0.0
    ga = {str(t) for t in group_tids if str(t)}
    ca = {str(t) for t in cluster_tids if str(t)}
    if ga == ca:
        return 0.0
    per_track: dict[str, dict[str, float]] = mask_ctx.get("per_track") or {}
    weight = float(os.getenv("CLYPT_CLUSTER_MASK_OVERLAP_ATTACH_WEIGHT", "0.12"))
    eps = 1e-9

    def _mean_stab(tids: list[str]) -> float:
        vals = [float(per_track[t]["stab"]) for t in tids if t in per_track]
        return float(sum(vals) / max(1, len(vals))) if vals else 0.0

    mg = _mean_stab([str(t) for t in group_tids])
    mc = _mean_stab([str(t) for t in cluster_tids])
    if mg < eps and mc < eps:
        return 0.0
    return float(weight * mg * mc)


def _shot_index_for_time_ms(shot_timeline_ms: list[dict], t_ms: float) -> int | None:
    for i, seg in enumerate(shot_timeline_ms):
        a = float(seg.get("start_time_ms", 0))
        b = float(seg.get("end_time_ms", a))
        if a <= t_ms <= b:
            return i
    return None


def _center_time_ms(span: tuple[int, int], video_fps: float) -> float:
    lo, hi = span
    mid_f = 0.5 * float(lo + hi)
    return float(mid_f / max(video_fps, 1e-6) * 1000.0)


class _UnionFind:
    def __init__(self, ids: list[str]):
        self._p = {i: i for i in ids}

    def find(self, x: str) -> str:
        while self._p[x] != x:
            self._p[x] = self._p[self._p[x]]
            x = self._p[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if ra < rb:
            self._p[rb] = ra
        else:
            self._p[ra] = rb
        return True


def _merge_identity_feature_group(
    features: dict[str, dict],
    keys: list[str],
) -> dict:
    vecs: list[np.ndarray] = []
    observations: list[dict] = []
    face_track_ids: set[str] = set()
    for k in keys:
        f = features.get(k) or {}
        emb = f.get("embedding")
        if emb is not None:
            vecs.append(np.asarray(emb, dtype=np.float32))
        observations.extend(list(f.get("face_observations", [])))
        for ftid in f.get("face_track_ids", []) or []:
            face_track_ids.add(str(ftid))
    emb_out = None
    if vecs:
        emb_out = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32).tolist()
    return {
        "embedding": emb_out,
        "embedding_source": "face" if emb_out is not None else "none",
        "embedding_count": int(sum(int((features.get(k) or {}).get("embedding_count", 0) or 0) for k in keys)),
        "face_observations": observations,
        "face_observation_count": len(observations),
        "face_track_ids": sorted(face_track_ids),
    }


def post_track_reid_merge(
    tracks: list[dict],
    track_identity_features: dict[str, dict] | None,
    *,
    shot_timeline_ms: list[dict] | None,
    video_fps: float,
    min_cos_sim: float = _POST_REID_MIN_COS_SIM,
) -> tuple[list[dict], dict[str, dict] | None, dict[str, int]]:
    """Merge fragmented raw track IDs using face embeddings + shot gates (pre-DBSCAN).

    Conservative: no-op when embeddings or shot evidence are insufficient.
    """
    metrics = {
        "post_track_reid_merge_attempted_pairs": 0,
        "post_track_reid_merge_merged_pairs": 0,
        "post_track_reid_merge_skipped_covisible": 0,
        "post_track_reid_merge_skipped_shot_incompatible": 0,
        "post_track_reid_merge_skipped_low_similarity": 0,
        "post_track_reid_merge_skipped_missing_embedding": 0,
        "post_track_reid_merge_skipped_missing_span": 0,
    }
    if not tracks or not isinstance(track_identity_features, dict) or not track_identity_features:
        metrics["post_track_reid_merge_components_merged"] = 0
        return tracks, track_identity_features, metrics

    timeline = [dict(s) for s in (shot_timeline_ms or [])]
    if not timeline:
        metrics["post_track_reid_merge_components_merged"] = 0
        return tracks, track_identity_features, metrics

    ids = sorted({_norm_track_id(str(t.get("track_id", ""))) for t in tracks if str(t.get("track_id", ""))})
    if len(ids) < 2:
        metrics["post_track_reid_merge_components_merged"] = 0
        return tracks, track_identity_features, metrics

    emb_keys: dict[str, str] = {}
    embeddings: dict[str, np.ndarray] = {}
    for tid in ids:
        fk = _feature_key_for_track(track_identity_features, tid)
        if fk is None:
            metrics["post_track_reid_merge_skipped_missing_embedding"] += 1
            continue
        feat = track_identity_features.get(fk) or {}
        emb = feat.get("embedding")
        if emb is None:
            metrics["post_track_reid_merge_skipped_missing_embedding"] += 1
            continue
        arr = np.asarray(emb, dtype=np.float32)
        if arr.size == 0:
            metrics["post_track_reid_merge_skipped_missing_embedding"] += 1
            continue
        emb_keys[tid] = fk
        embeddings[tid] = arr

    id_list = sorted(embeddings.keys())
    if len(id_list) < 2:
        metrics["post_track_reid_merge_components_merged"] = 0
        return tracks, track_identity_features, metrics

    uf = _UnionFind(id_list)
    pair_budget = max(0, int(_POST_REID_MAX_CANDIDATE_PAIRS))
    considered = 0
    stop_pairs = False

    for i in range(len(id_list)):
        if stop_pairs:
            break
        for j in range(i + 1, len(id_list)):
            if considered >= pair_budget:
                stop_pairs = True
                break
            a, b = id_list[i], id_list[j]
            metrics["post_track_reid_merge_attempted_pairs"] += 1
            considered += 1

            span_a = _track_frame_span(tracks, a)
            span_b = _track_frame_span(tracks, b)
            if span_a is None or span_b is None:
                metrics["post_track_reid_merge_skipped_missing_span"] += 1
                continue
            if _covisible(span_a, span_b):
                metrics["post_track_reid_merge_skipped_covisible"] += 1
                continue

            ca = _center_time_ms(span_a, video_fps)
            cb = _center_time_ms(span_b, video_fps)
            sa = _shot_index_for_time_ms(timeline, ca)
            sb = _shot_index_for_time_ms(timeline, cb)
            if sa is None or sb is None or sa != sb:
                metrics["post_track_reid_merge_skipped_shot_incompatible"] += 1
                continue

            sim = _cosine_sim(embeddings[a], embeddings[b])
            if sim < float(min_cos_sim):
                metrics["post_track_reid_merge_skipped_low_similarity"] += 1
                continue

            if uf.union(a, b):
                metrics["post_track_reid_merge_merged_pairs"] += 1

    components: dict[str, list[str]] = {}
    for tid in id_list:
        r = uf.find(tid)
        components.setdefault(r, []).append(tid)

    remap: dict[str, str] = {}
    components_merged = 0
    for _root, members in components.items():
        canonical = min(members)
        if len(members) > 1:
            components_merged += 1
        for m in members:
            remap[m] = canonical

    if components_merged == 0:
        metrics["post_track_reid_merge_components_merged"] = 0
        return tracks, track_identity_features, metrics

    for t in tracks:
        tid = _norm_track_id(str(t.get("track_id", "")))
        if tid in remap:
            t["track_id"] = remap[tid]
        if str(t.get("track_id", "")).startswith("track_"):
            try:
                t["local_track_id"] = int(str(t["track_id"]).split("_", 1)[1])
            except (ValueError, IndexError):
                pass

    new_features: dict[str, dict] = dict(track_identity_features)
    for _root, members in components.items():
        if len(members) <= 1:
            continue
        keys = [emb_keys[m] for m in members if m in emb_keys]
        if not keys:
            continue
        canonical_tid = min(members)
        canon_key = emb_keys.get(canonical_tid) or keys[0]
        merged_f = _merge_identity_feature_group(track_identity_features, keys)
        for m in members:
            fk = emb_keys.get(m)
            if fk and fk in new_features:
                del new_features[fk]
        new_features[canon_key] = merged_f

    metrics["post_track_reid_merge_components_merged"] = int(components_merged)
    return tracks, new_features, metrics


def run_cluster_tracklets_stage(
    worker: Any,
    video_path: str,
    tracks: list[dict],
    *,
    track_to_dets: dict[str, list[dict]] | None = None,
    track_identity_features: dict[str, dict] | None = None,
    face_track_features: dict[str, dict] | None = None,
    shot_timeline_ms: list[dict] | None = None,
    cluster_video_fps: float | None = None,
    cluster_duration_ms: int | None = None,
) -> list[dict]:
    """Delegate global tracklet clustering to the worker implementation."""
    return worker._cluster_tracklets(
        video_path,
        tracks,
        track_to_dets=track_to_dets,
        track_identity_features=track_identity_features,
        face_track_features=face_track_features,
        shot_timeline_ms=shot_timeline_ms,
        cluster_video_fps=cluster_video_fps,
        cluster_duration_ms=cluster_duration_ms,
    )


__all__ = [
    "build_mask_overlap_clustering_signals",
    "mask_overlap_attachment_cost_reduction",
    "post_track_reid_merge",
    "run_cluster_tracklets_stage",
    "_norm_track_id",
]
