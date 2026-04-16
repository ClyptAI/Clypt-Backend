from __future__ import annotations

import json


def _compact_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


MERGE_AND_CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "merged_nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_turn_ids": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "node_type": {
                        "type": "string",
                        "enum": ["claim", "explanation", "example", "anecdote",
                                 "reaction_beat", "qa_exchange", "challenge_exchange",
                                 "setup_payoff", "reveal", "transition"],
                    },
                    "node_flags": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "topic_pivot",
                                "callback_candidate",
                                "high_resonance_candidate",
                                "backchannel_dense",
                                "interruption_heavy",
                                "overlap_heavy",
                                "resumed_topic",
                            ],
                        },
                    },
                    "summary": {"type": "string", "minLength": 1},
                },
                "required": ["source_turn_ids", "node_type", "node_flags", "summary"],
            },
        },
    },
    "required": ["merged_nodes"],
}

BOUNDARY_RECONCILIATION_SCHEMA = {
    "type": "object",
    "properties": {
        "resolution": {"type": "string", "enum": ["keep_both", "merge"]},
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "existing_node_id": {"type": "string", "minLength": 1},
                    "source_turn_ids": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "node_type": {
                        "type": "string",
                        "enum": ["claim", "explanation", "example", "anecdote",
                                 "reaction_beat", "qa_exchange", "challenge_exchange",
                                 "setup_payoff", "reveal", "transition"],
                    },
                    "node_flags": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "topic_pivot",
                                "callback_candidate",
                                "high_resonance_candidate",
                                "backchannel_dense",
                                "interruption_heavy",
                                "overlap_heavy",
                                "resumed_topic",
                            ],
                        },
                    },
                    "summary": {"type": "string", "minLength": 1},
                },
                "required": ["existing_node_id", "source_turn_ids", "node_type", "node_flags", "summary"],
            },
        },
        "merged_node": {
            "type": "object",
            "properties": {
                "source_turn_ids": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
                "node_type": {
                    "type": "string",
                    "enum": ["claim", "explanation", "example", "anecdote",
                             "reaction_beat", "qa_exchange", "challenge_exchange",
                             "setup_payoff", "reveal", "transition"],
                },
                "node_flags": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "topic_pivot",
                            "callback_candidate",
                            "high_resonance_candidate",
                            "backchannel_dense",
                            "interruption_heavy",
                            "overlap_heavy",
                            "resumed_topic",
                        ],
                    },
                },
                "summary": {"type": "string", "minLength": 1},
            },
            "required": ["source_turn_ids", "node_type", "node_flags", "summary"],
        },
    },
    "required": ["resolution"],
}


_NODE_TYPES = (
    '"claim"          — a factual assertion or opinion stated directly\n'
    '"explanation"    — elaboration or reasoning supporting a prior point\n'
    '"example"        — a concrete instance used to illustrate a point\n'
    '"anecdote"       — a personal story or narrative that stands on its own\n'
    '"reaction_beat"  — a reaction, brief interjection, or listener response\n'
    '"qa_exchange"    — a question-answer exchange between speakers\n'
    '"challenge_exchange" — a debate, pushback, or challenge between speakers\n'
    '"setup_payoff"   — a setup followed by its payoff within the same unit\n'
    '"reveal"         — a surprising disclosure, confession, or unexpected fact\n'
    '"transition"     — a topic shift, wrap-up, or conversational bridge'
)

_NODE_FLAGS = (
    '"topic_pivot"             — introduces a significant topic change\n'
    '"callback_candidate"      — references a topic discussed earlier\n'
    '"high_resonance_candidate"— likely to resonate strongly with audiences\n'
    '"backchannel_dense"       — contains many backchannels (yeah, uh-huh, right)\n'
    '"interruption_heavy"      — contains interruptions or crosstalk\n'
    '"overlap_heavy"           — contains significant speaker overlap\n'
    '"resumed_topic"           — returns to a topic that was dropped earlier'
)


def build_merge_and_classify_prompt(*, neighborhood_payload: dict) -> str:
    """Build the Qwen prompt for local merged-unit construction and classification."""
    return (
        "You are analyzing a neighborhood of speaker turns from a long-form audio/video conversation.\n\n"
        "TASK: merge contiguous TARGET turns into semantic units (nodes) and classify each unit.\n\n"
        "RULES:\n"
        '- Only use turns with role="target". Ignore halo turns (role="halo") — they are context only.\n'
        "- Every target turn must appear in exactly one merged node (complete partition, no gaps, no skipped turns).\n"
        "- Each node's source_turn_ids must be contiguous within the target sequence.\n"
        "- NEVER group non-adjacent target turns by semantic similarity while skipping intervening target turns.\n"
        "- If a story/explanation is interleaved with reactions, you must still output contiguous segments: either include the intervening reaction turns in the same node, or end the current node and start a new contiguous node after the interruption.\n"
        "- Invalid pattern example: [t_10, t_12, t_14] while omitting [t_11, t_13]. Do not do this.\n"
        "- Do NOT invent timestamps. Do NOT split individual turns.\n\n"
        "OUTPUT: Return ONLY this JSON object, no other text:\n"
        "{\n"
        '  "merged_nodes": [\n'
        "    {\n"
        '      "source_turn_ids": ["t_000001", "t_000002"],\n'
        '      "node_type": "<exactly one from the list below>",\n'
        '      "node_flags": ["<zero or more from the list below>"],\n'
        '      "summary": "One sentence describing what this semantic unit is about."\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"VALID node_type values (pick exactly one):\n{_NODE_TYPES}\n\n"
        f"VALID node_flags (include any that apply, or empty list []):\n{_NODE_FLAGS}\n\n"
        f"Neighborhood payload:\n{_compact_json(neighborhood_payload)}"
    )


def build_boundary_reconciliation_prompt(*, overlap_payload: dict) -> str:
    """Build the Qwen prompt for edge-batch boundary reconciliation."""
    return (
        "You are reviewing two adjacent node proposals at the boundary between two semantic batches.\n"
        "Decide whether to KEEP BOTH nodes as distinct semantic units, or MERGE them into one.\n\n"
        "OUTPUT: Return ONLY one of these two JSON structures, no other text:\n\n"
        "Option A — keep both (if they are clearly distinct semantic units):\n"
        "{\n"
        '  "resolution": "keep_both",\n'
        '  "nodes": [\n'
        "    {\n"
        '      "existing_node_id": "<node_id from input — copy exactly>",\n'
        '      "source_turn_ids": ["<turn_ids belonging to this node>"],\n'
        '      "node_type": "<node_type>",\n'
        '      "node_flags": [],\n'
        '      "summary": "<one sentence summary>"\n'
        "    },\n"
        "    {\n"
        '      "existing_node_id": "<node_id from input — copy exactly>",\n'
        '      "source_turn_ids": ["<turn_ids belonging to this node>"],\n'
        '      "node_type": "<node_type>",\n'
        '      "node_flags": [],\n'
        '      "summary": "<one sentence summary>"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Option B — merge (if the two form one cohesive semantic unit):\n"
        "{\n"
        '  "resolution": "merge",\n'
        '  "merged_node": {\n'
        '    "source_turn_ids": ["<all turn_ids from both nodes in chronological order>"],\n'
        '    "node_type": "<node_type>",\n'
        '    "node_flags": [],\n'
        '    "summary": "<one sentence summary>"\n'
        "  }\n"
        "}\n\n"
        f"Valid node_type values: claim, explanation, example, anecdote, reaction_beat, qa_exchange, challenge_exchange, setup_payoff, reveal, transition\n"
        f"Valid node_flags: topic_pivot, callback_candidate, high_resonance_candidate, backchannel_dense, interruption_heavy, overlap_heavy, resumed_topic\n\n"
        f"Boundary payload:\n{_compact_json(overlap_payload)}"
    )


__all__ = [
    "BOUNDARY_RECONCILIATION_SCHEMA",
    "MERGE_AND_CLASSIFY_SCHEMA",
    "build_boundary_reconciliation_prompt",
    "build_merge_and_classify_prompt",
]
