from __future__ import annotations

import json


def build_subgraph_review_prompt(*, subgraph_payload: dict) -> str:
    return (
        "You are selecting clip candidates from a local semantic subgraph of a long-form video or podcast.\n\n"
        "TASK: review the subgraph and propose up to 3 contiguous clip candidates that would make "
        "strong standalone short-form clips.\n\n"
        "RULES:\n"
        "- Return the subgraph_id and seed_node_id from the input EXACTLY as given — do not modify them.\n"
        "- Candidates MUST use whole nodes only — do NOT invent timestamps.\n"
        "- start_ms must match the first chosen node's start_ms EXACTLY.\n"
        "- end_ms must match the last chosen node's end_ms EXACTLY.\n"
        "- node_ids must be a contiguous chronological span taken from the subgraph's node list.\n"
        "- Maximum 3 candidates. Return fewer if fewer qualify.\n"
        "- If no candidate is strong enough, set reject_all=true and explain in reject_reason.\n\n"
        "OUTPUT: Return ONLY this JSON object, no other text:\n"
        "{\n"
        '  "subgraph_id": "<copy subgraph_id from input exactly>",\n'
        '  "seed_node_id": "<copy seed_node_id from input exactly>",\n'
        '  "reject_all": false,\n'
        '  "reject_reason": "",\n'
        '  "candidates": [\n'
        "    {\n"
        '      "node_ids": ["<node_id_1>", "<node_id_2>"],\n'
        '      "start_ms": <start_ms of first node — copy exactly from node data>,\n'
        '      "end_ms": <end_ms of last node — copy exactly from node data>,\n'
        '      "score": 0.85,\n'
        '      "rationale": "<why this makes a strong standalone clip>"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Or if rejecting all candidates:\n"
        "{\n"
        '  "subgraph_id": "<copy from input>",\n'
        '  "seed_node_id": "<copy from input>",\n'
        '  "reject_all": true,\n'
        '  "reject_reason": "<why no good clip was found>",\n'
        '  "candidates": []\n'
        "}\n\n"
        f"Subgraph payload:\n{json.dumps(subgraph_payload, ensure_ascii=True, indent=2)}"
    )


def build_pooled_candidate_review_prompt(*, candidate_payload: dict) -> str:
    return (
        "You are performing a final quality review of a pool of clip candidates from a long-form video or podcast.\n\n"
        "TASK: keep the best candidates, drop weaker ones, and rank the kept candidates.\n\n"
        "RULES:\n"
        "- You must account for EVERY candidate — either keep or drop each one, no exceptions.\n"
        "- Do NOT change any candidate's boundaries (node_ids, start_ms, end_ms).\n"
        "- ranked_candidates: list of kept candidates with pool_rank starting at 1 (1 = best).\n"
        "- dropped_candidate_temp_ids: list of clip_ids for candidates you are dropping.\n"
        "- Every kept candidate must have a unique pool_rank (1, 2, 3, ...).\n"
        "- Use each candidate's clip_id exactly as the candidate_temp_id.\n\n"
        "OUTPUT: Return ONLY this JSON object, no other text:\n"
        "{\n"
        '  "ranked_candidates": [\n'
        "    {\n"
        '      "candidate_temp_id": "<clip_id from input — copy exactly>",\n'
        '      "keep": true,\n'
        '      "pool_rank": 1,\n'
        '      "score": 0.90,\n'
        '      "score_breakdown": {"virality": 0.9, "coherence": 0.9, "engagement": 0.9},\n'
        '      "rationale": "<why this is a strong clip>"\n'
        "    }\n"
        "  ],\n"
        '  "dropped_candidate_temp_ids": ["<clip_id>", "<clip_id>"]\n'
        "}\n\n"
        "Note: dropped_candidate_temp_ids must be an empty list [] if you keep all candidates.\n\n"
        f"Candidate pool:\n{json.dumps(candidate_payload, ensure_ascii=True, indent=2)}"
    )


__all__ = [
    "build_pooled_candidate_review_prompt",
    "build_subgraph_review_prompt",
]
