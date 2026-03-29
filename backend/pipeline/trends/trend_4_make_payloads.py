#!/usr/bin/env python3
"""
Trend Trim Stage 4: turn resurfacing candidates into render-friendly payloads.
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

from pipeline.trends.trend_utils import OUTPUTS_DIR, clamp, utc_now_iso

INPUT_PATH = OUTPUTS_DIR / "trend_3_resurfacing_candidates.json"
OUTPUT_PATH = OUTPUTS_DIR / "trend_remotion_payloads_array.json"
MAX_TREND_BOOST = 6.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s \u2013 %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("trend_4")


def main() -> list[dict]:
    log.info("=" * 60)
    log.info("TREND TRIM STAGE 4 \u2013 Payloads")
    log.info("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing trend candidates: {INPUT_PATH}")

    payload = json.loads(INPUT_PATH.read_text(encoding="utf-8-sig"))
    candidates = payload.get("candidates", []) if isinstance(payload, dict) else []

    trend_payloads: list[dict] = []
    for candidate in candidates:
        base_payload = dict(candidate.get("payload", {}) or {})
        if not base_payload:
            continue
        match_score = float(candidate.get("trend_match_score", 0.0) or 0.0)
        boost = round(min(MAX_TREND_BOOST, max(0.0, (match_score - 55.0) / 8.0)), 1)
        base_score = float(base_payload.get("final_score", 0.0) or 0.0)
        boosted_score = round(clamp(base_score + boost, 0.0, 100.0), 1)
        trend_note = (
            f" Trend-backed: {candidate.get('trend_title')} "
            f"({candidate.get('trend_source')}, match {match_score})."
        )

        trend_payloads.append(
            {
                **base_payload,
                "base_final_score": base_score,
                "final_score": boosted_score,
                "justification": (str(base_payload.get("justification", "") or "").strip() + trend_note).strip(),
                "catalog_id": candidate.get("catalog_id"),
                "source_url": candidate.get("source_url"),
                "source_video_id": candidate.get("video_id"),
                "source_video_gcs_uri": candidate.get("video_gcs_uri"),
                "source_video_title": candidate.get("video_title"),
                "trend_signals": {
                    "enabled": True,
                    "trend_title": candidate.get("trend_title"),
                    "trend_source": candidate.get("trend_source"),
                    "trend_query_text": candidate.get("trend_query_text"),
                    "trend_match_score": match_score,
                    "trend_signal_strength": candidate.get("trend_signal_strength"),
                    "overlap_terms": candidate.get("overlap_terms", []),
                    "reason": candidate.get("reason"),
                    "generated_at": utc_now_iso(),
                },
            }
        )

    OUTPUT_PATH.write_text(json.dumps(trend_payloads, indent=2), encoding="utf-8")
    log.info("Trend payloads written: %d", len(trend_payloads))
    log.info("Output saved \u2192 %s", OUTPUT_PATH)
    return trend_payloads


if __name__ == "__main__":
    main()
