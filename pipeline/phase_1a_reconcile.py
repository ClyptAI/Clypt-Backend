#!/usr/bin/env python3
"""
Phase 1A-R: Speaker-to-Face Reconciliation
============================================
Correlates STT speaker_tags (acoustic identity) with Video Intelligence
face tracks (spatial identity) by analysing temporal co-occurrence and
bounding box activity.

Inputs:
  - phase_1a_visual.json  (face_detections with per-frame bounding boxes)
  - phase_1a_audio.json   (words with speaker_tag + timestamps)

Output:
  - phase_1a_speaker_map.json  (speaker_tag -> face_track_index mapping)
"""

import json
import logging
import os
import statistics
from collections import defaultdict
from pathlib import Path

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
VISUAL_PATH = ROOT / "outputs" / "phase_1a_visual.json"
AUDIO_PATH = ROOT / "outputs" / "phase_1a_audio.json"
OUTPUT_PATH = ROOT / "outputs" / "phase_1a_speaker_map.json"

# Tolerance window (ms) for matching a word timestamp to a face frame
TEMPORAL_TOLERANCE_MS = 150
# Window (ms) around a word to compute bounding box height variance
ACTIVITY_WINDOW_MS = 500
# Max horizontal distance (normalized 0-1) for assigning additional tracks
# to the same speaker after primary reconciliation.
SPATIAL_GROUP_THRESHOLD = float(os.getenv("RECONCILE_SPATIAL_GROUP_THRESHOLD", "0.06"))
MAX_EXPANDED_TRACKS_PER_SPEAKER = int(os.getenv("RECONCILE_MAX_TRACKS_PER_SPEAKER", "3"))
MIN_TRACK_FRAMES_FOR_EXPANSION = int(os.getenv("RECONCILE_MIN_TRACK_FRAMES", "12"))

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase_1a_reconcile")


def _bbox_center_x(bbox: dict) -> float:
    return (bbox["left"] + bbox["right"]) / 2


def _bbox_height(bbox: dict) -> float:
    return bbox["bottom"] - bbox["top"]


def _compute_median_center_x(face_track: dict) -> float:
    """Compute the median horizontal center of a face track."""
    centers = [
        _bbox_center_x(obj["bounding_box"])
        for obj in face_track["timestamped_objects"]
        if "bounding_box" in obj
    ]
    if not centers:
        return 0.5
    return statistics.median(centers)


def _compute_activity_score(face_track: dict, center_ms: int) -> float:
    """
    Compute face bbox height variance in a window around center_ms.
    Higher variance → more likely speaking (jaw movement changes bbox height).
    """
    window_start = center_ms - ACTIVITY_WINDOW_MS
    window_end = center_ms + ACTIVITY_WINDOW_MS

    heights = [
        _bbox_height(obj["bounding_box"])
        for obj in face_track["timestamped_objects"]
        if "bounding_box" in obj
        and window_start <= obj["time_ms"] <= window_end
    ]

    if len(heights) < 2:
        return 0.0
    return statistics.variance(heights)


def _find_active_tracks_at(face_tracks: list, time_ms: int) -> list[int]:
    """Return indices of face tracks that have a frame near time_ms."""
    active = []
    for i, track in enumerate(face_tracks):
        # Check if any timestamped object is within tolerance
        for obj in track["timestamped_objects"]:
            if abs(obj["time_ms"] - time_ms) <= TEMPORAL_TOLERANCE_MS:
                active.append(i)
                break
    return active


def main():
    log.info("=" * 60)
    log.info("PHASE 1A-R — Speaker-to-Face Reconciliation")
    log.info("=" * 60)

    # ── Load data ──
    visual = json.loads(VISUAL_PATH.read_text())
    audio = json.loads(AUDIO_PATH.read_text())

    face_tracks = visual.get("face_detections", [])
    words = audio.get("words", [])

    log.info(f"Face tracks: {len(face_tracks)}")
    log.info(f"Words: {len(words)}")

    # ── Collect unique speaker tags ──
    speaker_tags = sorted({
        w["speaker_tag"] for w in words
        if w.get("speaker_tag") and w["speaker_tag"] != "unknown"
    })
    log.info(f"Speaker tags: {speaker_tags}")

    if not face_tracks:
        log.warning("No face tracks found — falling back to positional heuristic")
        # Assign speakers left-to-right based on tag order
        speaker_map = {tag: i for i, tag in enumerate(speaker_tags)}
        speaker_to_tracks = {tag: [idx] for tag, idx in speaker_map.items()}
        _save_output(speaker_map, face_tracks, speaker_to_tracks)
        return

    if not speaker_tags:
        log.warning("No speaker tags found in audio — cannot reconcile")
        _save_output({}, face_tracks, {})
        return

    # ── Compute spatial identity for each face track ──
    for i, track in enumerate(face_tracks):
        track["_median_cx"] = _compute_median_center_x(track)
        log.info(f"  Face track {i}: median_cx={track['_median_cx']:.3f}, "
                 f"frames={len(track['timestamped_objects'])}, "
                 f"segment={track['segment_start_ms']}–{track['segment_end_ms']}ms")

    # ── Build voting matrix: speaker_tag × face_track_index ──
    # For each word, find which face tracks are active and score them
    # by bounding box activity (proxy for lip movement).
    votes: dict[str, dict[int, float]] = {
        tag: defaultdict(float) for tag in speaker_tags
    }

    for word in words:
        tag = word.get("speaker_tag")
        if not tag or tag == "unknown":
            continue

        word_mid_ms = (word["start_time_ms"] + word["end_time_ms"]) // 2
        active_indices = _find_active_tracks_at(face_tracks, word_mid_ms)

        if not active_indices:
            continue

        if len(active_indices) == 1:
            # Only one face visible — it's the speaker
            votes[tag][active_indices[0]] += 1.0
        else:
            # Multiple faces visible — weight by activity score
            activities = {
                idx: _compute_activity_score(face_tracks[idx], word_mid_ms)
                for idx in active_indices
            }
            total_activity = sum(activities.values())
            if total_activity > 0:
                for idx, activity in activities.items():
                    votes[tag][idx] += activity / total_activity
            else:
                # Equal activity — split vote
                for idx in active_indices:
                    votes[tag][idx] += 1.0 / len(active_indices)

    # ── Assign speaker_tag → face_track_index via greedy argmax ──
    speaker_map: dict[str, int] = {}
    assigned_tracks: set[int] = set()

    # Sort speakers by total vote count (highest first) for greedy assignment
    tag_totals = {
        tag: sum(track_votes.values())
        for tag, track_votes in votes.items()
    }

    for tag in sorted(tag_totals, key=lambda t: -tag_totals[t]):
        track_votes = votes[tag]
        if not track_votes:
            continue

        # Find best unassigned track
        for track_idx in sorted(track_votes, key=lambda i: -track_votes[i]):
            if track_idx not in assigned_tracks:
                speaker_map[tag] = track_idx
                assigned_tracks.add(track_idx)
                log.info(f"  Speaker '{tag}' → face track {track_idx} "
                         f"(votes={track_votes[track_idx]:.1f}, "
                         f"cx={face_tracks[track_idx]['_median_cx']:.3f})")
                break

    # ── Fallback: assign any unmatched speakers by spatial position ──
    unmatched_tags = [t for t in speaker_tags if t not in speaker_map]
    if unmatched_tags:
        log.warning(f"Unmatched speakers (spatial fallback): {unmatched_tags}")
        available_tracks = [
            i for i in range(len(face_tracks))
            if i not in assigned_tracks
        ]
        # Sort available tracks by horizontal position
        available_tracks.sort(key=lambda i: face_tracks[i]["_median_cx"])
        for tag, track_idx in zip(unmatched_tags, available_tracks):
            speaker_map[tag] = track_idx
            log.info(f"  Speaker '{tag}' → face track {track_idx} (spatial fallback)")

    # ── Expand each speaker to nearby face tracks (cuts / fragmentation) ──
    speaker_to_tracks: dict[str, list[int]] = {
        str(tag): [int(track_idx)]
        for tag, track_idx in speaker_map.items()
    }
    if speaker_map and len(speaker_tags) >= 2:
        primary_centers = {
            str(tag): face_tracks[int(track_idx)]["_median_cx"]
            for tag, track_idx in speaker_map.items()
        }
        primary_tracks = {int(idx) for idx in speaker_map.values()}
        candidates_by_tag: dict[str, list[tuple[float, int, int]]] = defaultdict(list)
        for track_idx, track in enumerate(face_tracks):
            if track_idx in primary_tracks:
                continue
            frame_count = len(track.get("timestamped_objects", []))
            if frame_count < MIN_TRACK_FRAMES_FOR_EXPANSION:
                continue
            cx = track["_median_cx"]
            nearest_tag = min(
                primary_centers,
                key=lambda t: abs(cx - primary_centers[t]),
            )
            dist = abs(cx - primary_centers[nearest_tag])
            if dist <= SPATIAL_GROUP_THRESHOLD:
                candidates_by_tag[nearest_tag].append((dist, -frame_count, track_idx))

        for tag, candidates in candidates_by_tag.items():
            base = speaker_to_tracks.setdefault(tag, [])
            room = max(0, MAX_EXPANDED_TRACKS_PER_SPEAKER - len(base))
            if room <= 0:
                continue
            for _, _, track_idx in sorted(candidates)[:room]:
                base.append(track_idx)
    elif len(speaker_tags) < 2:
        log.info("Single diarized speaker detected — keeping primary track-only mapping.")

    for tag, tracks in speaker_to_tracks.items():
        deduped = sorted(set(int(t) for t in tracks))
        speaker_to_tracks[tag] = deduped
        if deduped:
            log.info(
                f"  Speaker '{tag}' track set: {len(deduped)} "
                f"(primary={speaker_map.get(tag)})"
            )

    _save_output(speaker_map, face_tracks, speaker_to_tracks)


def _save_output(speaker_map: dict, face_tracks: list, speaker_to_tracks: dict | None = None):
    """Save the speaker map with metadata."""
    if speaker_to_tracks is None:
        speaker_to_tracks = {
            str(tag): [int(track_idx)]
            for tag, track_idx in speaker_map.items()
        }

    output = {
        "speaker_to_track": speaker_map,
        "speaker_to_tracks": speaker_to_tracks,
        "track_positions": {
            str(i): round(track.get("_median_cx", 0.5), 4)
            for i, track in enumerate(face_tracks)
        },
    }

    # Clean up transient keys
    for track in face_tracks:
        track.pop("_median_cx", None)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"Output saved → {OUTPUT_PATH}")
    log.info("=" * 60)
    log.info("PHASE 1A-R COMPLETE")
    log.info(f"  Mappings: {len(output['speaker_to_track'])}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
