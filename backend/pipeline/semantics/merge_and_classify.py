from __future__ import annotations

from ..contracts import SemanticGraphNode, SemanticNodeEvidence


def merge_and_classify_neighborhood(
    *,
    neighborhood_payload: dict,
    llm_response: dict | None = None,
    turn_word_ids_by_turn_id: dict[str, list[str]] | None = None,
) -> list[SemanticGraphNode]:
    """Convert one Qwen neighborhood response into proposed merged semantic nodes."""
    if llm_response is None:
        raise ValueError("llm_response is required")

    turns = list(neighborhood_payload.get("turns") or [])
    turn_map = {turn["turn_id"]: turn for turn in turns}
    target_turn_ids = list(neighborhood_payload.get("target_turn_ids") or [])
    target_turn_id_set = set(target_turn_ids)
    merged_nodes = list(llm_response.get("merged_nodes") or [])

    seen_target_turn_ids: list[str] = []
    results: list[SemanticGraphNode] = []

    for raw_node in merged_nodes:
        source_turn_ids = list(raw_node.get("source_turn_ids") or [])
        if not source_turn_ids:
            raise ValueError("each merged node must include source_turn_ids")
        if any(turn_id not in turn_map for turn_id in source_turn_ids):
            raise ValueError("merged node references unknown turn_id")
        if not set(source_turn_ids).issubset(target_turn_id_set):
            raise ValueError("merged node source_turn_ids must stay within the target partition")

        target_positions = [target_turn_ids.index(turn_id) for turn_id in source_turn_ids]
        contiguous_span = list(range(min(target_positions), max(target_positions) + 1))
        if target_positions != contiguous_span:
            raise ValueError("merged node source_turn_ids must form a contiguous target partition")

        ordered_turns = [turn_map[turn_id] for turn_id in source_turn_ids]
        seen_target_turn_ids.extend(source_turn_ids)

        word_ids: list[str] = []
        transcript_segments: list[str] = []
        emotion_labels: list[str] = []
        audio_events: list[str] = []
        for turn in ordered_turns:
            turn_id = str(turn.get("turn_id") or "")
            if turn_word_ids_by_turn_id is not None:
                word_ids.extend(turn_word_ids_by_turn_id.get(turn_id, []))
            else:
                word_ids.extend(turn.get("word_ids") or [])
            transcript_segments.append(str(turn.get("transcript_text") or "").strip())
            for emotion_label in turn.get("emotion_labels") or []:
                if emotion_label not in emotion_labels:
                    emotion_labels.append(emotion_label)
            for audio_label in turn.get("audio_events") or []:
                if audio_label not in audio_events:
                    audio_events.append(audio_label)

        first_turn = ordered_turns[0]
        last_turn = ordered_turns[-1]
        results.append(
            SemanticGraphNode(
                node_id=f"node_{source_turn_ids[0]}__{source_turn_ids[-1]}",
                node_type=raw_node["node_type"],
                start_ms=first_turn["start_ms"],
                end_ms=last_turn["end_ms"],
                source_turn_ids=source_turn_ids,
                word_ids=word_ids,
                transcript_text=" ".join(segment for segment in transcript_segments if segment).strip(),
                node_flags=list(raw_node.get("node_flags") or []),
                summary=str(raw_node.get("summary") or "").strip(),
                evidence=SemanticNodeEvidence(
                    emotion_labels=emotion_labels,
                    audio_events=audio_events,
                ),
            )
        )

    if sorted(seen_target_turn_ids, key=target_turn_ids.index) != target_turn_ids:
        raise ValueError("merged node proposals must partition the target turns completely")

    return results
