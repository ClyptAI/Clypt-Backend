from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from backend.integrations.senso_client import (
    SensoAPIError,
    SensoClient,
    SensoContentRecord,
    SensoGenerateResponse,
    SensoSearchResponse,
)

_log = logging.getLogger("creator_knowledge")


@dataclass(frozen=True)
class CreatorChannel:
    channel_id: str
    channel_name: str
    description: str = ""


@dataclass(frozen=True)
class CreatorVideoDocument:
    title: str
    text: str
    summary: str = ""
    video_id: str = ""


@dataclass(frozen=True)
class CreatorWorkspace:
    kb_folder_node_id: str
    kb_folder_name: str
    creator_label: str


def creator_topic_name(channel: CreatorChannel) -> str:
    return f"{channel.channel_name} ({channel.channel_id})"


class CreatorKnowledgeService:
    def __init__(self, client: SensoClient) -> None:
        self.client = client

    def ensure_creator_workspace(self, channel: CreatorChannel) -> CreatorWorkspace:
        root = self.client.get_kb_root()
        return CreatorWorkspace(
            kb_folder_node_id=str(root.kb_node_id or root.id or "").strip(),
            kb_folder_name=root.name or "Root",
            creator_label=creator_topic_name(channel),
        )

    def ingest_video_documents(
        self,
        workspace: CreatorWorkspace,
        documents: list[CreatorVideoDocument],
    ) -> list[SensoContentRecord]:
        created: list[SensoContentRecord] = []
        for document in documents:
            title = document.title.strip() or f"Untitled Creator Document {len(created) + 1}"
            namespaced_title = f"{workspace.creator_label} :: {title}"
            try:
                record = self.client.create_raw_content(
                    title=namespaced_title,
                    summary=document.summary or None,
                    text=document.text,
                    kb_folder_node_id=workspace.kb_folder_node_id,
                )
                created.append(record)
            except SensoAPIError as exc:
                if exc.status_code == 409:
                    _log.info("Skipping duplicate content: %s", namespaced_title)
                    continue
                raise
        return created

    def build_creator_profile(
        self,
        workspace: CreatorWorkspace,
        *,
        prompt_id: str,
        content_ids: list[str],
        template_id: str | None = None,
        max_results: int = 8,
        save: bool = False,
    ) -> SensoGenerateResponse:
        del template_id, save
        prompt = self.client.get_prompt(prompt_id)
        query = self._build_profile_query(prompt.text, workspace.creator_label)
        response = self.client.wait_for_search_results(
            query=query,
            content_ids=content_ids,
            max_results=max_results,
            include_answer=True,
            require_scoped_ids=bool(content_ids),
        )
        return SensoGenerateResponse(
            generated_text=response.answer,
            processing_time_ms=response.processing_time_ms,
            sources=response.results,
        )

    def search_creator_context(
        self,
        *,
        query: str,
        content_ids: list[str],
        max_results: int = 5,
    ) -> SensoSearchResponse:
        return self.client.search(
            query=query,
            max_results=max_results,
            content_ids=content_ids,
            require_scoped_ids=True,
            include_answer=False,
        )

    @staticmethod
    def _build_profile_query(prompt_text: str, creator_label: str) -> str:
        schema = {
            "creator_archetype": "string",
            "creator_summary": "string",
            "category": "string — e.g. Podcast, Gaming, Education, Comedy, Music, Tech, Vlog, Sports, News, Fitness, Cooking, Finance, Beauty, Travel, DIY, Science, Lifestyle, Entertainment",
            "tone": "string or array of strings",
            "pacing": "string",
            "hook_style": "string",
            "payoff_style": "string",
            "recurring_topics": ["string"],
            "audience": "string",
            "brand_voice": ["string"],
            "dominant_mechanisms": {
                "humor": {"intensity": "0_to_1_float", "style": "string"},
                "emotion": {"intensity": "0_to_1_float", "style": "string"},
                "social": {"intensity": "0_to_1_float", "style": "string"},
                "expertise": {"intensity": "0_to_1_float", "style": "string"},
            },
        }
        return (
            f"{prompt_text.strip()}\n\n"
            f"Creator label: {creator_label}\n\n"
            "Based only on the provided creator documents, return valid JSON matching this schema exactly:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Rules:\n"
            "- No markdown.\n"
            "- No prose before or after the JSON.\n"
            "- Keep recurring_topics and brand_voice concise.\n"
            "- category must be a single short label (1-2 words) describing the content genre.\n"
            "- If evidence is weak, still fill the field with the best grounded summary.\n"
            "- dominant_mechanisms intensities must be numbers between 0 and 1.\n"
        )
