#!/usr/bin/env python3
"""
Standalone Crowd Clip runner.

This pipeline is intentionally separate from Content Clip. It ranks clips from
audience signals only: comments, basic engagement, and transcript-aligned crowd
references.
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

from pipeline.audience.crowd_1_ingest_youtube import main as crowd_ingest_main
from pipeline.audience.crowd_2_resolve_signals import main as crowd_resolve_main
from pipeline.audience.crowd_3_score_windows import main as crowd_score_main
from pipeline.audience.crowd_4_make_payloads import main as crowd_payload_main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_crowd_clip")


def banner(text: str):
    log.info("=" * 60)
    log.info(text)
    log.info("=" * 60)


def main():
    banner("CLYPT CROWD CLIP")

    url = ""
    if len(sys.argv) > 1:
        url = sys.argv[1].strip()
    if not url:
        url = input("\nEnter YouTube URL: ").strip()
    if not url:
        raise SystemExit("No YouTube URL provided.")

    log.info("Target: %s", url)
    crowd_ingest_main(url)
    crowd_resolve_main(url)
    crowd_score_main(url)
    crowd_payload_main(url)

    banner("CROWD CLIP COMPLETE")
    log.info("Artifacts:")
    log.info("  backend/outputs/crowd_1_youtube_signals.json")
    log.info("  backend/outputs/crowd_2_resolved_signals.json")
    log.info("  backend/outputs/crowd_3_clip_candidates.json")
    log.info("  backend/outputs/crowd_remotion_payloads_array.json")


if __name__ == "__main__":
    main()
