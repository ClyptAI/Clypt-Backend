#!/usr/bin/env python3
"""
Standalone Trend Trim runner.

Trend Trim scans external trend sources, indexes the local catalog under
backend/videos, and resurfaces existing clips when current topics align.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from pipeline.trends.trend_1_ingest_external import main as trend_ingest_main
from pipeline.trends.trend_2_catalog_index import main as trend_catalog_main
from pipeline.trends.trend_3_resurface_catalog import main as trend_resurface_main
from pipeline.trends.trend_4_make_payloads import main as trend_payload_main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_trend_trim")


def banner(text: str):
    log.info("=" * 60)
    log.info(text)
    log.info("=" * 60)


def main():
    banner("CLYPT TREND TRIM")
    trend_ingest_main()
    trend_catalog_main()
    trend_resurface_main()
    trend_payload_main()
    banner("TREND TRIM COMPLETE")
    log.info("Artifacts:")
    log.info("  backend/outputs/trend_1_external_signals.json")
    log.info("  backend/outputs/trend_2_catalog_index.json")
    log.info("  backend/outputs/trend_3_resurfacing_candidates.json")
    log.info("  backend/outputs/trend_remotion_payloads_array.json")


if __name__ == "__main__":
    main()
