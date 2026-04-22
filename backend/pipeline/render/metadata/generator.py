from __future__ import annotations

import re
from typing import Any

from ..contracts import PublishMetadata, PublishMetadataClip, SourceContext


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _title_case(text: str, *, max_words: int = 8) -> str:
    words = _WORD_RE.findall(text)
    clipped = words[:max_words]
    if not clipped:
        return "Clip Highlight"
    return " ".join(word.capitalize() for word in clipped)


def _sentence(text: str, *, max_words: int) -> str:
    words = text.split()
    if not words:
        return ""
    clipped = words[:max_words]
    joined = " ".join(clipped).strip()
    if joined and joined[-1] not in ".!?":
        joined = f"{joined}."
    return joined


def _topic_tags(source_context: SourceContext, finalist: dict[str, Any]) -> list[str]:
    raw = [
        *source_context.tags,
        *finalist.get("semantic_node_summaries", []),
        finalist.get("transcript_excerpt", ""),
        finalist.get("rationale", ""),
    ]
    seen: set[str] = set()
    tags: list[str] = []
    for item in raw:
        for token in _WORD_RE.findall(str(item).lower()):
            if len(token) < 4:
                continue
            if token in seen:
                continue
            seen.add(token)
            tags.append(token)
    if len(tags) < 3:
        tags.extend(token for token in ["clips", "creator", "story"] if token not in seen)
    return tags[:8]


def _hashtags(topic_tags: list[str], source_context: SourceContext) -> list[str]:
    base = [f"#{tag.replace('-', '')}" for tag in topic_tags[:4]]
    channel_token = "".join(part.capitalize() for part in _WORD_RE.findall(source_context.channel_title))
    if channel_token:
        base.append(f"#{channel_token}")
    base.append("#shorts")
    deduped: list[str] = []
    seen: set[str] = set()
    for tag in base:
        lowered = tag.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(tag)
    return deduped[:6]


def build_publish_metadata(
    *,
    run_id: str,
    source_context: SourceContext | dict[str, Any],
    finalists: list[dict[str, Any]],
) -> dict[str, Any]:
    context = (
        source_context
        if isinstance(source_context, SourceContext)
        else SourceContext.model_validate(source_context)
    )

    clips: list[PublishMetadataClip] = []
    for finalist in finalists:
        excerpt = str(finalist.get("transcript_excerpt", "")).strip()
        rationale = str(finalist.get("rationale", "")).strip()
        title_primary = _title_case(excerpt or rationale or context.source_title)
        alternates = [
            _title_case(rationale or context.source_title),
            _title_case(" ".join(finalist.get("semantic_node_summaries", [])) or context.channel_title),
            _title_case(f"{context.source_title} {excerpt}"),
        ]
        title_alternates = [item for item in dict.fromkeys(alternates) if item and item != title_primary][:4]
        if len(title_alternates) < 2:
            title_alternates.append(_title_case(context.channel_title))

        description_parts = [part for part in [excerpt, rationale, context.source_description] if part]
        description_short = _sentence(" ".join(description_parts), max_words=32)
        thumbnail_text = " ".join(_WORD_RE.findall((excerpt or title_primary).upper())[:4]) or "CLIP"
        topic_tags = _topic_tags(context, finalist)
        hashtags = _hashtags(topic_tags, context)

        finalist_summary = {
            "clip_id": finalist.get("clip_id"),
            "rationale": rationale,
            "transcript_excerpt": excerpt,
        }
        generation_inputs_summary = {
            "youtube_video_id": context.youtube_video_id,
            "source_title": context.source_title,
            "source_context": context.model_dump(mode="json"),
            "finalist": finalist_summary,
        }

        clips.append(
            PublishMetadataClip(
                clip_id=str(finalist["clip_id"]),
                title_primary=title_primary,
                title_alternates=title_alternates[:4],
                description_short=description_short or _sentence(context.source_description, max_words=24),
                thumbnail_text=thumbnail_text,
                topic_tags=topic_tags[:8],
                hashtags=hashtags[:6],
                generation_inputs_summary=generation_inputs_summary,
            )
        )

    return PublishMetadata(run_id=run_id, clips=clips).model_dump(mode="json")
