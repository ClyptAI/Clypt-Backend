#!/usr/bin/env python3
"""
Deduplicate remotion_payloads_array.json using non-maximum suppression.
Reads the existing payload file and writes a deduplicated version in-place.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD_PATH = ROOT / "outputs" / "remotion_payloads_array.json"

NMS_OVERLAP_THRESHOLD = 0.5
MAX_CLIPS_OUTPUT = 10


def overlap_ratio(a: dict, b: dict) -> float:
    start = max(a["clip_start_ms"], b["clip_start_ms"])
    end = min(a["clip_end_ms"], b["clip_end_ms"])
    if end <= start:
        return 0.0
    intersection = end - start
    len_a = a["clip_end_ms"] - a["clip_start_ms"]
    len_b = b["clip_end_ms"] - b["clip_start_ms"]
    return intersection / min(len_a, len_b)


def main():
    clips = json.loads(PAYLOAD_PATH.read_text())
    print(f"Loaded {len(clips)} clips")

    clips.sort(key=lambda c: c["final_score"], reverse=True)

    kept = []
    for candidate in clips:
        suppressed = any(
            overlap_ratio(candidate, k) > NMS_OVERLAP_THRESHOLD
            for k in kept
        )
        if not suppressed:
            kept.append(candidate)
        if len(kept) >= MAX_CLIPS_OUTPUT:
            break

    print(f"After deduplication: {len(kept)} unique clips")
    for i, c in enumerate(kept, 1):
        duration = (c["clip_end_ms"] - c["clip_start_ms"]) / 1000
        print(f"  #{i}: {c['final_score']}/100  {c['clip_start_ms']}ms–{c['clip_end_ms']}ms  ({duration:.1f}s)")

    PAYLOAD_PATH.write_text(json.dumps(kept, indent=2))
    print(f"Saved → {PAYLOAD_PATH}")


if __name__ == "__main__":
    main()
