from __future__ import annotations

import json


def build_subgraph_review_prompt(*, subgraph_payload: dict) -> str:
    return (
        "Review this local semantic subgraph and return up to 3 contiguous clip candidates in strict JSON.\n"
        "Do not invent timestamps or node ids. Candidates must use whole-node boundaries only.\n\n"
        f"Subgraph payload:\n{json.dumps(subgraph_payload, ensure_ascii=True, indent=2)}"
    )


def build_pooled_candidate_review_prompt(*, candidate_payload: dict) -> str:
    return (
        "Review this candidate pool, decide which candidates to keep or drop, and rank the kept candidates in strict JSON.\n"
        "Do not change candidate boundaries.\n\n"
        f"Candidate pool payload:\n{json.dumps(candidate_payload, ensure_ascii=True, indent=2)}"
    )


__all__ = [
    "build_pooled_candidate_review_prompt",
    "build_subgraph_review_prompt",
]
