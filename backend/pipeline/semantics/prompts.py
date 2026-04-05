from __future__ import annotations

import json


def build_merge_and_classify_prompt(*, neighborhood_payload: dict) -> str:
    """Build the Gemini prompt for local merged-unit construction and classification."""
    return (
        "Merge contiguous target turns into semantic units and classify each merged unit.\n"
        "Only use target turns, never split turns, and return strict JSON.\n\n"
        f"Neighborhood payload:\n{json.dumps(neighborhood_payload, ensure_ascii=True, indent=2)}"
    )


def build_boundary_reconciliation_prompt(*, overlap_payload: dict) -> str:
    """Build the Gemini prompt for edge-batch boundary reconciliation."""
    return (
        "Review the adjacent batch-boundary node proposals and decide whether to keep both or merge.\n"
        "Return strict JSON only.\n\n"
        f"Boundary payload:\n{json.dumps(overlap_payload, ensure_ascii=True, indent=2)}"
    )
