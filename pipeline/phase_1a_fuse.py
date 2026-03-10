#!/usr/bin/env python3
"""
Phase 1A-FUSE: Active Speaker Fusion (ASD + STT diarization)
=============================================================
Builds a frame-aware speaker timeline by fusing ASD segments with STT
word-level diarization. If ASD is unavailable, falls back to word-only
segments.

Inputs:
  - outputs/phase_1a_active_speaker_timeline.json      (optional)
  - outputs/phase_1a_audio.json
  - outputs/phase_1a_speaker_map.json                  (optional)

Output:
  - outputs/phase_1a_active_speaker_timeline_v2.json
"""

from __future__ import annotations

import json
import logging
import os
from bisect import bisect_left
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASD_PATH = ROOT / "outputs" / "phase_1a_active_speaker_timeline.json"
AUDIO_PATH = ROOT / "outputs" / "phase_1a_audio.json"
SPEAKER_MAP_PATH = ROOT / "outputs" / "phase_1a_speaker_map.json"
OUTPUT_PATH = ROOT / "outputs" / "phase_1a_active_speaker_timeline_v2.json"

WORD_MERGE_GAP_MS = int(os.getenv("FUSE_WORD_MERGE_GAP_MS", "120"))
ASD_MERGE_GAP_MS = int(os.getenv("FUSE_ASD_MERGE_GAP_MS", "160"))
MIN_SEGMENT_MS = int(os.getenv("FUSE_MIN_SEGMENT_MS", "180"))
MIN_ASD_CONFIDENCE = float(os.getenv("FUSE_MIN_ASD_CONFIDENCE", "0.58"))
MIN_OVERLAP_CONFIRM_MS = int(os.getenv("FUSE_MIN_OVERLAP_CONFIRM_MS", "120"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase_1a_fuse")


def _normalize_segments(raw_segments: list[dict]) -> list[dict]:
    out: list[dict] = []
    for seg in raw_segments:
        try:
            start_ms = int(seg.get("start_ms", 0))
            end_ms = int(seg.get("end_ms", 0))
        except (TypeError, ValueError):
            continue
        if end_ms <= start_ms:
            continue
        speaker_tag = seg.get("speaker_tag")
        if speaker_tag is not None and str(speaker_tag).strip() == "":
            speaker_tag = None
        track_id = seg.get("track_id")
        confidence = seg.get("confidence")
        try:
            confidence = None if confidence is None else float(confidence)
        except (TypeError, ValueError):
            confidence = None
        out.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "speaker_tag": None if speaker_tag is None else str(speaker_tag),
                "track_id": track_id,
                "confidence": confidence,
                "source": str(seg.get("source", "unknown")),
            }
        )
    out.sort(key=lambda s: (s["start_ms"], s["end_ms"]))
    return out


def _merge_segments(segments: list[dict], gap_ms: int) -> list[dict]:
    if not segments:
        return []
    merged: list[dict] = []
    for seg in segments:
        if not merged:
            merged.append(dict(seg))
            continue
        prev = merged[-1]
        same_speaker = prev.get("speaker_tag") == seg.get("speaker_tag")
        same_track = (prev.get("track_id") or None) == (seg.get("track_id") or None)
        if same_speaker and same_track and seg["start_ms"] <= prev["end_ms"] + gap_ms:
            prev["end_ms"] = max(prev["end_ms"], seg["end_ms"])
            p_conf = prev.get("confidence")
            s_conf = seg.get("confidence")
            if isinstance(p_conf, (float, int)) and isinstance(s_conf, (float, int)):
                prev["confidence"] = max(float(p_conf), float(s_conf))
            elif isinstance(s_conf, (float, int)):
                prev["confidence"] = float(s_conf)
        else:
            merged.append(dict(seg))

    return [s for s in merged if s["end_ms"] - s["start_ms"] >= MIN_SEGMENT_MS]


def _build_word_segments(audio: dict) -> list[dict]:
    words = audio.get("words", [])
    raw: list[dict] = []
    for w in words:
        tag = w.get("speaker_tag")
        if not tag or str(tag) == "unknown":
            continue
        try:
            start_ms = int(w.get("start_time_ms", 0))
            end_ms = int(w.get("end_time_ms", start_ms))
        except (TypeError, ValueError):
            continue
        if end_ms < start_ms:
            end_ms = start_ms
        if end_ms == start_ms:
            end_ms = start_ms + 1
        raw.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "speaker_tag": str(tag),
                "track_id": None,
                "confidence": 0.62,
                "source": "stt_word",
            }
        )
    return _merge_segments(_normalize_segments(raw), WORD_MERGE_GAP_MS)


def _load_track_to_speaker() -> dict[str, str]:
    if not SPEAKER_MAP_PATH.exists():
        return {}
    data = json.loads(SPEAKER_MAP_PATH.read_text())
    out: dict[str, str] = {}
    for speaker_tag, tracks in (data.get("speaker_to_tracks") or {}).items():
        if isinstance(tracks, list):
            for track_id in tracks:
                out[str(track_id)] = str(speaker_tag)
    for speaker_tag, track_id in (data.get("speaker_to_track") or {}).items():
        out.setdefault(str(track_id), str(speaker_tag))
    return out


def _load_asd_segments(track_to_speaker: dict[str, str]) -> list[dict]:
    if not ASD_PATH.exists():
        return []

    raw_obj = json.loads(ASD_PATH.read_text())
    raw_segments = raw_obj.get("segments", raw_obj) if isinstance(raw_obj, dict) else raw_obj
    if not isinstance(raw_segments, list):
        return []

    segments = _normalize_segments(raw_segments)
    gated: list[dict] = []
    for seg in segments:
        conf = seg.get("confidence")
        if isinstance(conf, (float, int)) and float(conf) < MIN_ASD_CONFIDENCE:
            continue
        if seg.get("speaker_tag") is None and seg.get("track_id") is not None:
            mapped = track_to_speaker.get(str(seg["track_id"]))
            if mapped:
                seg["speaker_tag"] = mapped
        seg["source"] = "asd"
        gated.append(seg)
    return _merge_segments(gated, ASD_MERGE_GAP_MS)


def _overlap_ms(a: dict, b: dict) -> int:
    return max(0, min(a["end_ms"], b["end_ms"]) - max(a["start_ms"], b["start_ms"]))


def _build_word_index(word_segments: list[dict]) -> tuple[list[int], list[dict]]:
    starts = [s["start_ms"] for s in word_segments]
    return starts, word_segments


def _find_word_support(seg: dict, starts: list[int], words: list[dict]) -> tuple[bool, str | None]:
    if not words:
        return False, None
    i = bisect_left(starts, seg["start_ms"])
    candidates = []
    for idx in (i - 2, i - 1, i, i + 1, i + 2):
        if 0 <= idx < len(words):
            candidates.append(words[idx])

    best_overlap = 0
    best_speaker = None
    for w in candidates:
        ov = _overlap_ms(seg, w)
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = w.get("speaker_tag")
    return best_overlap >= MIN_OVERLAP_CONFIRM_MS, best_speaker


def _fuse(asd_segments: list[dict], word_segments: list[dict]) -> list[dict]:
    if not asd_segments:
        return word_segments

    starts, words = _build_word_index(word_segments)
    fused: list[dict] = []

    for seg in asd_segments:
        supported, suggested_speaker = _find_word_support(seg, starts, words)
        out = dict(seg)
        if out.get("speaker_tag") is None and suggested_speaker:
            out["speaker_tag"] = suggested_speaker
        if not supported:
            conf = out.get("confidence")
            if isinstance(conf, (float, int)):
                out["confidence"] = max(0.0, float(conf) - 0.08)
            out["source"] = "asd_weak"
        fused.append(out)

    # Backfill with STT-only gaps where ASD is missing coverage.
    for w in words:
        covered = any(_overlap_ms(w, a) >= MIN_OVERLAP_CONFIRM_MS for a in asd_segments)
        if not covered:
            backfill = dict(w)
            backfill["confidence"] = min(0.7, float(backfill.get("confidence") or 0.62))
            backfill["source"] = "stt_backfill"
            fused.append(backfill)

    fused = _merge_segments(_normalize_segments(fused), ASD_MERGE_GAP_MS)
    for seg in fused:
        if seg.get("speaker_tag") is None:
            seg["source"] = "unknown"
    return fused


def main() -> None:
    log.info("=" * 60)
    log.info("PHASE 1A-FUSE — Active Speaker Fusion")
    log.info("=" * 60)

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(f"Missing audio ledger: {AUDIO_PATH}")

    audio = json.loads(AUDIO_PATH.read_text())
    track_to_speaker = _load_track_to_speaker()

    word_segments = _build_word_segments(audio)
    asd_segments = _load_asd_segments(track_to_speaker)
    fused_segments = _fuse(asd_segments, word_segments)

    payload = {
        "source": "asd+stt" if asd_segments else "stt_only",
        "has_true_asd": ASD_PATH.exists(),
        "segment_count": len(fused_segments),
        "segments": fused_segments,
    }

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))

    log.info(f"Word segments: {len(word_segments)}")
    log.info(f"ASD segments: {len(asd_segments)}")
    log.info(f"Fused segments: {len(fused_segments)}")
    log.info(f"Saved → {OUTPUT_PATH}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
