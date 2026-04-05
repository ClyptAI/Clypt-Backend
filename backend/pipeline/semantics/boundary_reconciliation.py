from __future__ import annotations

from ..contracts import SemanticGraphNode, SemanticNodeEvidence


def reconcile_boundary_nodes(*, left_batch_nodes: list[SemanticGraphNode], right_batch_nodes: list[SemanticGraphNode], gemini_response: dict | None = None) -> list[SemanticGraphNode]:
    """Resolve overlapping edge-node proposals between adjacent semantic batches."""
    if gemini_response is None:
        raise ValueError("gemini_response is required")

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

    resolution = gemini_response.get("resolution")
    if resolution == "keep_both":
        output_nodes: list[SemanticGraphNode] = []
        for item in gemini_response.get("nodes") or []:
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
        merged_node = gemini_response.get("merged_node") or {}
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
