from __future__ import annotations

import json


META_PROMPT_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "prompts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["prompts"],
}

SUBGRAPH_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "subgraph_id": {"type": "string"},
        "seed_node_id": {"type": "string"},
        "reject_all": {"type": "boolean"},
        "reject_reason": {"type": "string"},
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "node_ids": {"type": "array", "items": {"type": "string"}},
                    "start_ms": {"type": "integer"},
                    "end_ms": {"type": "integer"},
                    "score": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": ["node_ids", "start_ms", "end_ms", "score", "rationale"],
            },
        },
    },
    "required": ["subgraph_id", "seed_node_id", "reject_all", "candidates"],
}

POOL_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "ranked_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_temp_id": {"type": "string"},
                    "keep": {"type": "boolean"},
                    "pool_rank": {"type": "integer"},
                    "score": {"type": "number"},
                    "score_breakdown": {
                        "type": "object",
                        "properties": {
                            "virality": {"type": "number"},
                            "coherence": {"type": "number"},
                            "engagement": {"type": "number"},
                        },
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["candidate_temp_id", "keep", "pool_rank", "score", "rationale"],
            },
        },
        "dropped_candidate_temp_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["ranked_candidates", "dropped_candidate_temp_ids"],
}


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


def build_meta_prompt_generation_prompt(*, node_summaries: list[dict], target_count: int) -> str:
    return (
        "You are designing retrieval queries to find the best short-form clip candidates in a long-form video.\n\n"
        "You have analyzed the video's semantic structure. Each node below represents a coherent semantic unit "
        "(a claim, anecdote, reveal, qa_exchange, etc.) with its type, flags, and a one-sentence summary.\n\n"
        f"TASK: generate exactly {target_count} targeted retrieval prompts specific to THIS video's content "
        "that would surface the strongest, most shareable clip moments. Each prompt will be used as a "
        "semantic similarity query against node embeddings.\n\n"
        "RULES:\n"
        "- Write prompts that reflect what you can actually see in the nodes (topics, themes, moment types present)\n"
        "- Do NOT write prompts so generic they could apply to any video\n"
        "- Only generate prompts for moment types you see clear evidence of\n"
        "- Each prompt must be a single sentence starting with \"Find\"\n"
        f"- Return EXACTLY {target_count} prompts — no more, no fewer\n\n"
        "OUTPUT: Return ONLY this JSON object, no other text:\n"
        "{\n"
        '  "prompts": [\n'
        '    "Find the moment where ...",\n'
        '    "Find the strongest ...",\n'
        "    ...\n"
        "  ]\n"
        "}\n\n"
        f"Semantic node summaries:\n{json.dumps(node_summaries, ensure_ascii=True, indent=2)}"
    )


__all__ = [
    "META_PROMPT_GENERATION_SCHEMA",
    "POOL_REVIEW_SCHEMA",
    "SUBGRAPH_REVIEW_SCHEMA",
    "build_meta_prompt_generation_prompt",
    "build_pooled_candidate_review_prompt",
    "build_subgraph_review_prompt",
]
