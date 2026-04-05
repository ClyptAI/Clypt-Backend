from __future__ import annotations

from ..contracts import SemanticGraphNode
from .._embedding_utils import embed_media_descriptor, embed_text


def _compact_list(values: list[str]) -> str:
    deduped: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return ", ".join(deduped)


def build_semantic_embedding_payload(*, node: SemanticGraphNode) -> str:
    parts = [
        f"Summary: {node.summary.strip()}",
        f"Type: {node.node_type}",
    ]
    flags = _compact_list(node.node_flags)
    if flags:
        parts.append(f"Flags: {flags}")
    emotions = _compact_list(node.evidence.emotion_labels)
    if emotions:
        parts.append(f"Emotion: {emotions}")
    audio_events = _compact_list(node.evidence.audio_events)
    if audio_events:
        parts.append(f"Audio events: {audio_events}")
    transcript = " ".join(node.transcript_text.split())
    if transcript:
        parts.append(f"Transcript: {transcript}")
    return "\n".join(parts)


def build_multimodal_proxy_payload(*, node: SemanticGraphNode) -> str:
    parts = [
        "Local media span for semantic node.",
        f"Summary: {node.summary.strip()}",
        f"Transcript: {' '.join(node.transcript_text.split())}",
    ]
    emotions = _compact_list(node.evidence.emotion_labels)
    if emotions:
        parts.append(f"Spoken emotion cues: {emotions}")
    audio_events = _compact_list(node.evidence.audio_events)
    if audio_events:
        parts.append(f"Audio events: {audio_events}")
    return "\n".join(parts)


def embed_semantic_nodes(*, nodes: list[SemanticGraphNode]) -> list[SemanticGraphNode]:
    """Attach durable embeddings to canonical semantic graph nodes."""
    embedded_nodes: list[SemanticGraphNode] = []
    for node in nodes:
        embedded_nodes.append(
            node.model_copy(
                update={
                    "semantic_embedding": embed_text(
                        text=build_semantic_embedding_payload(node=node)
                    ),
                    "multimodal_embedding": embed_media_descriptor(
                        descriptor=build_multimodal_proxy_payload(node=node)
                    ),
                }
            )
        )
    return embedded_nodes


__all__ = [
    "build_multimodal_proxy_payload",
    "build_semantic_embedding_payload",
    "embed_semantic_nodes",
]
