from __future__ import annotations

import json


def _compact_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


LOCAL_SEMANTIC_EDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_node_id": {"type": "string", "minLength": 1},
                    "target_node_id": {"type": "string", "minLength": 1},
                    "edge_type": {
                        "type": "string",
                        "enum": ["answers", "challenges", "contradicts", "supports",
                                 "elaborates", "setup_for", "payoff_of", "reaction_to", "escalates"],
                    },
                    "rationale": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["source_node_id", "target_node_id", "edge_type"],
            },
        },
    },
    "required": ["edges"],
}

LONG_RANGE_EDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_node_id": {"type": "string", "minLength": 1},
                    "target_node_id": {"type": "string", "minLength": 1},
                    "edge_type": {
                        "type": "string",
                        "enum": ["callback_to", "topic_recurrence"],
                    },
                    "rationale": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["source_node_id", "target_node_id", "edge_type"],
            },
        },
    },
    "required": ["edges"],
}


def build_local_semantic_edge_prompt(*, batch_payload: dict) -> str:
    return (
        "You are analyzing a batch of semantic graph nodes from a long-form conversation.\n\n"
        "TASK: identify meaningful semantic edges ONLY between TARGET nodes.\n"
        "Context/halo nodes (in context_node_ids but not target_node_ids) are provided for reference only.\n\n"
        "RULES:\n"
        "- source_node_id MUST be one of the target_node_ids.\n"
        "- target_node_id MUST be one of the context_node_ids (which includes target nodes).\n"
        "- Only emit edges for clear, strong semantic relationships.\n"
        "- Do NOT emit edges just because two nodes are adjacent — structural adjacency is handled separately.\n"
        "- Omit rationale and confidence if you have no meaningful observation.\n\n"
        "OUTPUT: Return ONLY this JSON object, no other text:\n"
        "{\n"
        '  "edges": [\n'
        "    {\n"
        '      "source_node_id": "<node_id of the source — must be a target node>",\n'
        '      "target_node_id": "<node_id of the target — must be a context node>",\n'
        '      "edge_type": "<exactly one from the list below>",\n'
        '      "rationale": "<one sentence explaining why>",\n'
        '      "confidence": 0.85\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        'If no meaningful edges exist, return: {"edges": []}\n\n'
        "VALID edge_type values:\n"
        '"answers"     — source directly answers a question posed in target\n'
        '"challenges"  — source disputes or pushes back on target\n'
        '"contradicts" — source contradicts a claim made in target\n'
        '"supports"    — source provides evidence or agreement supporting target\n'
        '"elaborates"  — source expands on or deepens the idea in target\n'
        '"setup_for"   — source creates anticipation or sets up target as the payoff\n'
        '"payoff_of"   — source is the payoff/resolution of a setup in target\n'
        '"reaction_to" — source is an emotional or conversational reaction to target\n'
        '"escalates"   — source raises the stakes, intensity, or severity relative to target\n\n'
        f"Batch payload:\n{_compact_json(batch_payload)}"
    )


def build_long_range_edge_prompt(*, pair_payload: dict) -> str:
    return (
        "You are adjudicating potential long-range semantic edges in a conversation graph.\n\n"
        "TASK: for each (later_node, earlier_node) pair in candidate_pairs, decide whether "
        "a meaningful long-range edge should be drawn.\n\n"
        "RULES:\n"
        "- Only emit edges for pairs listed in candidate_pairs.\n"
        "- source_node_id must be the later_node_id; target_node_id must be the earlier_node_id.\n"
        "- NEVER reverse direction: invalid example is source=earlier_node_id and target=later_node_id.\n"
        "- Treat candidate_pairs as a strict allowlist of exact tuples. For every emitted edge, "
        "the (source_node_id, target_node_id) tuple must exactly match one candidate_pairs entry.\n"
        "- You must copy source_node_id and target_node_id verbatim from the same candidate_pairs row. "
        "Do not synthesize IDs or combine IDs from different rows.\n"
        "- Do NOT emit \"nearby\" or \"overlapping\" alternatives that are not explicitly listed, even if "
        "they look semantically related.\n"
        "- Invalid examples (do not do this):\n"
        "  * choosing a pair with the correct source but a different earlier target that is not in candidate_pairs\n"
        "  * choosing a pair with the correct target but a different later source that is not in candidate_pairs\n"
        "  * emitting any tuple not present in candidate_pairs, even when edge_type is valid\n"
        "- Valid edge types: 'callback_to' and 'topic_recurrence' ONLY.\n"
        "  'callback_to'      — the later node explicitly refers back to the earlier node's specific moment or claim.\n"
        "  'topic_recurrence' — the later node revisits the same general topic, but is not an explicit reference.\n"
        "- Only emit an edge if the relationship is genuinely meaningful. Omit pairs with no real connection.\n\n"
        "OUTPUT: Return ONLY this JSON object, no other text:\n"
        "{\n"
        '  "edges": [\n'
        "    {\n"
        '      "source_node_id": "<later_node_id — copy from candidate_pairs>",\n'
        '      "target_node_id": "<earlier_node_id — copy from candidate_pairs>",\n'
        '      "edge_type": "callback_to or topic_recurrence",\n'
        '      "rationale": "<one sentence explaining the relationship>",\n'
        '      "confidence": 0.80\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        'If no meaningful long-range edges exist, return: {"edges": []}\n\n'
        f"Pair payload:\n{_compact_json(pair_payload)}"
    )


__all__ = [
    "LOCAL_SEMANTIC_EDGE_SCHEMA",
    "LONG_RANGE_EDGE_SCHEMA",
    "build_local_semantic_edge_prompt",
    "build_long_range_edge_prompt",
]
