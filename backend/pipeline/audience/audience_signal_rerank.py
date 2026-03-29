#!/usr/bin/env python3
"""
Audience Signal Layer
=====================
Treats Content Clip as the primary engine and uses crowd signals as a
post-publish reranking / annotation layer over existing content payloads.

Inputs:
  - outputs/remotion_payloads_array.json
  - outputs/crowd_3_clip_candidates.json

Output:
  - outputs/remotion_payloads_array_audience.json
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from pipeline.audience.crowd_1_ingest_youtube import main as crowd_ingest_main
from pipeline.audience.crowd_2_resolve_signals import main as crowd_resolve_main
from pipeline.audience.crowd_3_score_windows import main as crowd_score_main
from pipeline.audience.crowd_4_make_payloads import main as crowd_payload_main
from pipeline.audience.crowd_utils import OUTPUTS_DIR, clamp, overlap_ratio

CONTENT_PAYLOADS_PATH = OUTPUTS_DIR / "remotion_payloads_array.json"
CROWD_CANDIDATES_PATH = OUTPUTS_DIR / "crowd_3_clip_candidates.json"
OUTPUT_PATH = OUTPUTS_DIR / "remotion_payloads_array_audience.json"

MAX_AUDIENCE_BOOST = 8.0
MIN_MATCH_OVERLAP = 0.18

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("audience_rerank")


def _truthy_env(name: str) -> bool:
    return str(os.getenv(name, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def _crowd_strength(candidate: dict) -> float:
    explicit = float(candidate.get("explicit_timestamp_count", 0) or 0)
    quote = float(candidate.get("quote_match_count", 0) or 0)
    keyword = float(candidate.get("keyword_overlap_count", 0) or 0)
    unique_comments = float(candidate.get("unique_comment_count", 0) or 0)
    like_count = float(candidate.get("total_like_count", 0) or 0)
    base = float(candidate.get("final_score", 0.0) or 0.0) / 100.0

    signal_bonus = (
        min(1.0, explicit / 20.0) * 0.35
        + min(1.0, quote / 10.0) * 0.2
        + min(1.0, keyword / 12.0) * 0.1
        + min(1.0, unique_comments / 20.0) * 0.2
        + min(1.0, like_count / 15000.0) * 0.15
    )
    return clamp((0.55 * base) + (0.45 * signal_bonus), 0.0, 1.0)


def _interval_match_score(content_payload: dict, crowd_candidate: dict) -> float:
    content_start = int(content_payload.get("clip_start_ms", 0) or 0)
    content_end = int(content_payload.get("clip_end_ms", content_start) or content_start)
    crowd_start = int(crowd_candidate.get("clip_start_ms", 0) or 0)
    crowd_end = int(crowd_candidate.get("clip_end_ms", crowd_start) or crowd_start)

    inter = max(0, min(content_end, crowd_end) - max(content_start, crowd_start))
    if inter <= 0:
        return 0.0

    content_len = max(1, content_end - content_start)
    crowd_len = max(1, crowd_end - crowd_start)
    coverage = inter / content_len
    containment = inter / crowd_len
    jaccard = overlap_ratio(content_start, content_end, crowd_start, crowd_end)

    return clamp((0.45 * coverage) + (0.35 * containment) + (0.2 * jaccard), 0.0, 1.0)


def apply_audience_signals_to_payloads(content_payloads: list[dict], crowd_candidates: list[dict]) -> list[dict]:
    reranked: list[dict] = []

    for payload in content_payloads:
        matched_candidates: list[dict] = []
        best_support = 0.0

        for candidate in crowd_candidates:
            overlap = _interval_match_score(payload, candidate)
            if overlap < MIN_MATCH_OVERLAP:
                continue
            support = overlap * _crowd_strength(candidate)
            best_support = max(best_support, support)
            matched_candidates.append(
                {
                    "rank": candidate.get("rank"),
                    "clip_start_ms": candidate.get("clip_start_ms"),
                    "clip_end_ms": candidate.get("clip_end_ms"),
                    "overlap_score": round(overlap, 3),
                    "audience_support_score": round(support, 3),
                    "explicit_timestamp_count": candidate.get("explicit_timestamp_count"),
                    "quote_match_count": candidate.get("quote_match_count"),
                    "keyword_overlap_count": candidate.get("keyword_overlap_count"),
                    "unique_comment_count": candidate.get("unique_comment_count"),
                    "sample_comments": candidate.get("sample_comments", [])[:2],
                }
            )

        matched_candidates.sort(key=lambda item: item["audience_support_score"], reverse=True)
        matched_candidates = matched_candidates[:3]

        base_score = float(payload.get("final_score", 0.0) or 0.0)
        audience_boost = round(min(MAX_AUDIENCE_BOOST, best_support * 10.0), 1)
        audience_score = round(best_support * 100.0, 1)
        boosted_score = round(min(100.0, base_score + audience_boost), 1)

        audience_signals = {
            "enabled": True,
            "viewer_backed": best_support >= 0.35,
            "audience_support_score": audience_score,
            "audience_boost": audience_boost,
            "matched_candidate_count": len(matched_candidates),
            "matched_crowd_candidates": matched_candidates,
        }

        justification = str(payload.get("justification", "") or "").strip()
        if matched_candidates:
            strongest = matched_candidates[0]
            support_note = (
                f" Audience-backed: crowd overlap {strongest['overlap_score']}, "
                f"{strongest['explicit_timestamp_count']} timestamp refs, "
                f"{strongest['quote_match_count']} quote refs."
            )
            justification = (justification + support_note).strip()

        reranked.append(
            {
                **payload,
                "base_final_score": base_score,
                "final_score": boosted_score,
                "justification": justification,
                "audience_signals": audience_signals,
            }
        )

    reranked.sort(
        key=lambda payload: (
            -float(payload.get("final_score", 0.0) or 0.0),
            -float(payload.get("audience_signals", {}).get("audience_support_score", 0.0) or 0.0),
        )
    )
    return reranked


def main(youtube_url: str | None = None, refresh: bool | None = None) -> dict:
    if refresh is None:
        refresh = _truthy_env("REFRESH_AUDIENCE_SIGNALS")

    if refresh:
        if not youtube_url:
            raise RuntimeError("Refreshing audience signals requires the YouTube URL.")
        if not str(os.getenv("YOUTUBE_API_KEY", "") or "").strip():
            raise RuntimeError("Audience signals refresh requires YOUTUBE_API_KEY.")
        crowd_ingest_main(youtube_url)
        crowd_resolve_main(youtube_url)
        crowd_score_main(youtube_url)
        crowd_payload_main(youtube_url)

    if not CONTENT_PAYLOADS_PATH.exists():
        raise FileNotFoundError(f"Missing content payloads: {CONTENT_PAYLOADS_PATH}")
    if not CROWD_CANDIDATES_PATH.exists():
        raise FileNotFoundError(f"Missing crowd candidates: {CROWD_CANDIDATES_PATH}")

    content_payloads = json.loads(CONTENT_PAYLOADS_PATH.read_text(encoding="utf-8-sig"))
    crowd_payload = json.loads(CROWD_CANDIDATES_PATH.read_text(encoding="utf-8-sig"))
    crowd_candidates = crowd_payload.get("candidates", []) if isinstance(crowd_payload, dict) else []

    if not isinstance(content_payloads, list):
        content_payloads = [content_payloads]

    reranked = apply_audience_signals_to_payloads(content_payloads, crowd_candidates)
    OUTPUT_PATH.write_text(json.dumps(reranked, indent=2), encoding="utf-8")

    viewer_backed = sum(1 for payload in reranked if payload.get("audience_signals", {}).get("viewer_backed"))
    log.info("=" * 60)
    log.info("AUDIENCE SIGNAL RERANK COMPLETE")
    log.info("=" * 60)
    log.info("Content payloads: %d", len(content_payloads))
    log.info("Crowd candidates: %d", len(crowd_candidates))
    log.info("Viewer-backed clips: %d", viewer_backed)
    log.info("Output saved → %s", OUTPUT_PATH)
    return {
        "content_payload_count": len(content_payloads),
        "crowd_candidate_count": len(crowd_candidates),
        "viewer_backed_count": viewer_backed,
        "output_path": str(OUTPUT_PATH),
    }


if __name__ == "__main__":
    main()
