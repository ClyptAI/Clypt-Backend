#!/usr/bin/env python3
"""
Crowd Clip Stage 4: convert crowd candidates into Remotion payloads.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from pipeline.audience.crowd_transcript import load_transcript_words
from pipeline.audience.crowd_utils import OUTPUTS_DIR, extract_video_id

INPUT_PATH = OUTPUTS_DIR / "crowd_3_clip_candidates.json"
OUTPUT_PATH = OUTPUTS_DIR / "crowd_remotion_payloads_array.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("crowd_4")


def _build_speaker_timeline(words: list[dict], start_ms: int, end_ms: int) -> list[dict]:
    timeline: list[dict] = []
    current: dict | None = None
    for word in words:
        w_start = int(word.get("start_time_ms", 0) or 0)
        w_end = int(word.get("end_time_ms", w_start) or w_start)
        if not (w_start < end_ms and w_end > start_ms):
            continue
        speaker = word.get("speaker_tag") or word.get("speaker_track_id")
        if not speaker:
            continue
        tag = str(speaker)
        if current and current["speaker_tag"] == tag and w_start <= current["end_ms"] + 1200:
            current["end_ms"] = max(current["end_ms"], w_end)
            continue
        current = {"start_ms": w_start, "end_ms": w_end, "speaker_tag": tag}
        timeline.append(current)
    return timeline


def main(youtube_url: str | None = None) -> dict:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing Crowd Clip candidates artifact: {INPUT_PATH}")

    payload = json.loads(INPUT_PATH.read_text(encoding="utf-8-sig"))
    source_url = youtube_url or str(payload.get("source_url", "") or "")
    video_id = str(payload.get("video_id", "") or "")
    if not video_id:
        video_id = extract_video_id(source_url)

    words, _ = load_transcript_words(video_id)
    candidates = payload.get("candidates", [])

    remotion_payloads = []
    for candidate in candidates:
        start_ms = int(candidate.get("clip_start_ms", 0) or 0)
        end_ms = int(candidate.get("clip_end_ms", start_ms) or start_ms)
        remotion_payloads.append(
            {
                "clip_start_ms": start_ms,
                "clip_end_ms": end_ms,
                "final_score": float(candidate.get("final_score", 0.0) or 0.0),
                "justification": str(candidate.get("justification", "") or ""),
                "combined_transcript": str(candidate.get("transcript_excerpt", "") or ""),
                "tracking_uris": [],
                "included_node_ids": candidate.get("aligned_node_indices", []),
                "active_speaker_timeline": _build_speaker_timeline(words, start_ms, end_ms),
                "crowd_metrics": {
                    "rank": candidate.get("rank"),
                    "raw_score": candidate.get("raw_score"),
                    "explicit_timestamp_count": candidate.get("explicit_timestamp_count"),
                    "quote_match_count": candidate.get("quote_match_count"),
                    "keyword_overlap_count": candidate.get("keyword_overlap_count"),
                    "unique_comment_count": candidate.get("unique_comment_count"),
                    "unique_author_count": candidate.get("unique_author_count"),
                    "total_like_count": candidate.get("total_like_count"),
                    "total_reply_count": candidate.get("total_reply_count"),
                    "evidence_comment_ids": candidate.get("evidence_comment_ids", []),
                    "sample_comments": candidate.get("sample_comments", []),
                    "aligned_node_indices": candidate.get("aligned_node_indices", []),
                },
            }
        )

    OUTPUT_PATH.write_text(json.dumps(remotion_payloads, indent=2), encoding="utf-8")
    log.info("=" * 60)
    log.info("CROWD CLIP — STAGE 4 REMOTION PAYLOADS")
    log.info("=" * 60)
    log.info("Payloads generated: %d", len(remotion_payloads))
    log.info("Output saved → %s", OUTPUT_PATH)
    return {"payload_count": len(remotion_payloads), "output_path": str(OUTPUT_PATH)}


if __name__ == "__main__":
    main()
