from __future__ import annotations

import re

from ..contracts import SemanticGraphNode, SemanticNodeEvidence


def _tokenize_text(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) >= 3
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _turn_ordinals(turn_ids: list[str]) -> list[int]:
    ordinals: list[int] = []
    for turn_id in turn_ids:
        match = re.search(r"(\d+)$", str(turn_id))
        if match is None:
            continue
        ordinals.append(int(match.group(1)))
    return ordinals


def should_skip_boundary_reconciliation(
    *,
    left_node: SemanticGraphNode,
    right_node: SemanticGraphNode,
) -> dict[str, object]:
    overlap_turns = sorted(set(left_node.source_turn_ids) & set(right_node.source_turn_ids))
    shared_flags = sorted(set(left_node.node_flags) & set(right_node.node_flags))
    summary_similarity = _jaccard(_tokenize_text(left_node.summary), _tokenize_text(right_node.summary))
    transcript_similarity = _jaccard(
        _tokenize_text(left_node.transcript_text),
        _tokenize_text(right_node.transcript_text),
    )
    time_gap_ms = max(0, int(right_node.start_ms) - int(left_node.end_ms))
    left_ordinals = _turn_ordinals(list(left_node.source_turn_ids))
    right_ordinals = _turn_ordinals(list(right_node.source_turn_ids))
    turn_gap: int | None = None
    if left_ordinals and right_ordinals:
        turn_gap = max(0, min(right_ordinals) - max(left_ordinals) - 1)

    reason = "ambiguous_default"
    skip_llm = False
    if overlap_turns:
        reason = "overlapping_turns"
    elif time_gap_ms >= 5000:
        skip_llm = True
        reason = "large_time_gap"
    elif turn_gap is not None and turn_gap >= 2:
        skip_llm = True
        reason = "non_adjacent_turn_gap"
    elif (
        time_gap_ms >= 1200
        and summary_similarity < 0.12
        and transcript_similarity < 0.12
        and not shared_flags
        and left_node.node_type != right_node.node_type
    ):
        skip_llm = True
        reason = "clear_semantic_split"

    return {
        "skip_llm": skip_llm,
        "reason": reason,
        "time_gap_ms": time_gap_ms,
        "turn_gap": turn_gap,
        "summary_similarity": summary_similarity,
        "transcript_similarity": transcript_similarity,
        "shared_flag_count": len(shared_flags),
        "shared_flags": shared_flags,
        "overlap_turn_count": len(overlap_turns),
        "same_node_type": left_node.node_type == right_node.node_type,
    }


def reconcile_boundary_nodes(*, left_batch_nodes: list[SemanticGraphNode], right_batch_nodes: list[SemanticGraphNode], llm_response: dict | None = None) -> list[SemanticGraphNode]:
    """Resolve overlapping edge-node proposals between adjacent semantic batches."""
    if llm_response is None:
        raise ValueError("llm_response is required")

    known_nodes = {
        node.node_id: node
        for node in [*left_batch_nodes, *right_batch_nodes]
    }
    known_turns: dict[str, tuple[int, int, list[str], str]] = {}
    for node in known_nodes.values():
        for turn_id in node.source_turn_ids:
            known_turns[turn_id] = (
                node.start_ms,
                node.end_ms,
                list(node.word_ids),
                node.transcript_text,
            )

    resolution = llm_response.get("resolution")
    if resolution == "keep_both":
        output_nodes: list[SemanticGraphNode] = []
        for item in llm_response.get("nodes") or []:
            existing_node = known_nodes[item["existing_node_id"]]
            output_nodes.append(
                SemanticGraphNode(
                    node_id=existing_node.node_id,
                    node_type=item["node_type"],
                    start_ms=existing_node.start_ms,
                    end_ms=existing_node.end_ms,
                    source_turn_ids=list(item["source_turn_ids"]),
                    word_ids=list(existing_node.word_ids),
                    transcript_text=existing_node.transcript_text,
                    node_flags=list(item.get("node_flags") or []),
                    summary=str(item.get("summary") or "").strip(),
                    evidence=existing_node.evidence,
                    semantic_embedding=existing_node.semantic_embedding,
                    multimodal_embedding=existing_node.multimodal_embedding,
                )
            )
        return output_nodes

    if resolution == "merge":
        merged_node = llm_response.get("merged_node") or {}
        source_turn_ids = list(merged_node.get("source_turn_ids") or [])
        if not source_turn_ids:
            raise ValueError("merged boundary node must include source_turn_ids")

        candidate_nodes = [
            node for node in known_nodes.values() if set(node.source_turn_ids) & set(source_turn_ids)
        ]
        if not candidate_nodes:
            raise ValueError("merged boundary node must reference known boundary turns")

        start_ms = min(node.start_ms for node in candidate_nodes)
        end_ms = max(node.end_ms for node in candidate_nodes)
        word_ids: list[str] = []
        transcript_parts: list[str] = []
        emotion_labels: list[str] = []
        audio_events: list[str] = []
        for node in sorted(candidate_nodes, key=lambda item: item.start_ms):
            word_ids.extend(node.word_ids)
            transcript_parts.append(node.transcript_text)
            for emotion_label in node.evidence.emotion_labels:
                if emotion_label not in emotion_labels:
                    emotion_labels.append(emotion_label)
            for audio_label in node.evidence.audio_events:
                if audio_label not in audio_events:
                    audio_events.append(audio_label)

        return [
            SemanticGraphNode(
                node_id=f"node_{source_turn_ids[0]}__{source_turn_ids[-1]}",
                node_type=merged_node["node_type"],
                start_ms=start_ms,
                end_ms=end_ms,
                source_turn_ids=source_turn_ids,
                word_ids=word_ids,
                transcript_text=" ".join(part.strip() for part in transcript_parts if part.strip()),
                node_flags=list(merged_node.get("node_flags") or []),
                summary=str(merged_node.get("summary") or "").strip(),
                evidence=SemanticNodeEvidence(
                    emotion_labels=emotion_labels,
                    audio_events=audio_events,
                ),
            )
        ]

    raise ValueError(f"unsupported reconciliation resolution: {resolution!r}")


__all__ = ["reconcile_boundary_nodes", "should_skip_boundary_reconciliation"]
