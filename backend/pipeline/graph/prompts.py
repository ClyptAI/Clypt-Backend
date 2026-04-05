from __future__ import annotations

import json


def build_local_semantic_edge_prompt(*, batch_payload: dict) -> str:
    return (
        "Draw only local semantic graph edges for the target nodes in this batch.\n"
        "Use halo/context nodes only as context. Return strict JSON.\n\n"
        f"Batch payload:\n{json.dumps(batch_payload, ensure_ascii=True, indent=2)}"
    )


def build_long_range_edge_prompt(*, pair_payload: dict) -> str:
    return (
        "Adjudicate only callback_to and topic_recurrence edges for these shortlisted later-earlier node pairs.\n"
        "Return strict JSON.\n\n"
        f"Pair payload:\n{json.dumps(pair_payload, ensure_ascii=True, indent=2)}"
    )


__all__ = [
    "build_local_semantic_edge_prompt",
    "build_long_range_edge_prompt",
]
