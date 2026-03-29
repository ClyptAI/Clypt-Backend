from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable
from backend.services.creator_knowledge import (
    CreatorChannel,
    CreatorKnowledgeService,
    CreatorVideoDocument,
)
from backend.services.youtube_channel_service import (
    ChannelResolveResult,
    ChannelVideo,
    YouTubeChannelService,
)


ProgressCallback = Callable[[str, int, str], None]


@dataclass(frozen=True)
class CreatorProfileAnalysis:
    creator_id: str
    channel: dict
    profile: dict
    workspace: dict
    sources: list[dict]


class CreatorOnboardingService:
    def __init__(
        self,
        *,
        youtube_service: YouTubeChannelService,
        creator_knowledge_service: CreatorKnowledgeService,
        prompt_id: str,
        template_id: str | None = None,
        max_videos: int = 6,
        max_results: int = 8,
    ) -> None:
        if not prompt_id.strip():
            raise ValueError("Senso creator profile prompt id must not be empty.")
        self.youtube_service = youtube_service
        self.creator_knowledge_service = creator_knowledge_service
        self.prompt_id = prompt_id.strip()
        self.template_id = template_id.strip() if template_id else None
        self.max_videos = max(1, max_videos)
        self.max_results = max(1, max_results)

    @classmethod
    def from_env(cls) -> "CreatorOnboardingService":
        from backend.integrations.senso_client import SensoClient

        prompt_id = str(os.getenv("SENSO_CREATOR_PROFILE_PROMPT_ID", "") or "").strip()
        if not prompt_id:
            raise RuntimeError("Missing SENSO_CREATOR_PROFILE_PROMPT_ID.")
        template_id = str(os.getenv("SENSO_CREATOR_PROFILE_TEMPLATE_ID", "") or "").strip() or None
        max_videos = int(os.getenv("SENSO_CREATOR_PROFILE_MAX_VIDEOS", "6") or 6)
        max_results = int(os.getenv("SENSO_CREATOR_PROFILE_MAX_RESULTS", "8") or 8)
        return cls(
            youtube_service=YouTubeChannelService.from_env(),
            creator_knowledge_service=CreatorKnowledgeService(SensoClient.from_env()),
            prompt_id=prompt_id,
            template_id=template_id,
            max_videos=max_videos,
            max_results=max_results,
        )

    def resolve_channel(self, query: str) -> ChannelResolveResult:
        return self.youtube_service.resolve_channel(query)

    def analyze_channel(
        self,
        channel_id: str,
        *,
        progress: ProgressCallback | None = None,
    ) -> CreatorProfileAnalysis:
        self._notify(progress, "resolve_channel", 10, "Resolving YouTube channel and recent uploads")
        resolved = self.youtube_service.get_channel_by_id(channel_id, recent_video_limit=max(self.max_videos, 8))
        creator_channel = CreatorChannel(
            channel_id=resolved.channel.channel_id,
            channel_name=resolved.channel.channel_name,
            description=resolved.channel.description,
        )

        self._notify(progress, "ensure_workspace", 25, "Preparing Senso creator workspace")
        workspace = self.creator_knowledge_service.ensure_creator_workspace(creator_channel)

        source_videos = self._pick_source_videos(resolved)
        self._notify(progress, "fetch_transcripts", 45, f"Collecting transcripts for {len(source_videos)} recent uploads")
        documents: list[CreatorVideoDocument] = []
        _log = logging.getLogger("creator_onboarding")
        for idx, video in enumerate(source_videos):
            if idx > 0:
                time.sleep(1.5)  # avoid YouTube 429 rate-limit on captions
            try:
                documents.append(self._build_video_document(video))
            except Exception as exc:
                _log.warning("Skipping video %s — transcript fetch failed: %s", video.video_id, exc)
                continue

        if not documents:
            raise RuntimeError("Could not fetch transcripts for any videos — YouTube may be rate-limiting requests. Please try again in a minute.")

        self._notify(progress, "ingest_senso", 70, "Ingesting creator context into Senso")
        records = self.creator_knowledge_service.ingest_video_documents(workspace, documents)

        self._notify(progress, "generate_profile", 90, "Generating creator profile from Senso prompt")
        generated = self.creator_knowledge_service.build_creator_profile(
            workspace,
            prompt_id=self.prompt_id,
            content_ids=[record.id for record in records],
            template_id=self.template_id,
            max_results=self.max_results,
            save=False,
        )
        profile = self._parse_profile_text(generated.generated_text)
        profile.setdefault("creator_id", resolved.channel.channel_id)
        profile.setdefault("channel_id", resolved.channel.channel_id)
        profile.setdefault("channel_name", resolved.channel.channel_name)
        profile.setdefault("source_video_ids", [video.video_id for video in source_videos])
        profile.setdefault("senso_content_ids", [record.id for record in records])

        self._notify(progress, "complete", 100, "Creator profile ready")
        return CreatorProfileAnalysis(
            creator_id=resolved.channel.channel_id,
            channel={
                "channel_id": resolved.channel.channel_id,
                "channel_name": resolved.channel.channel_name,
                "channel_url": resolved.channel.channel_url,
                "handle": resolved.channel.handle,
            },
            profile=profile,
            workspace={
                "kb_folder_node_id": workspace.kb_folder_node_id,
                "kb_folder_name": workspace.kb_folder_name,
                "creator_label": workspace.creator_label,
            },
            sources=[
                {
                    "video_id": video.video_id,
                    "title": video.title,
                    "published_at": video.published_at,
                    "duration_seconds": video.duration_seconds,
                }
                for video in source_videos
            ],
        )

    def _pick_source_videos(self, resolved: ChannelResolveResult) -> list[ChannelVideo]:
        ranked = [*resolved.recent_videos, *resolved.recent_shorts]
        return ranked[: self.max_videos]

    def _build_video_document(self, video: ChannelVideo) -> CreatorVideoDocument:
        from backend.pipeline.audience.crowd_transcript import load_transcript_words

        words, _payload = load_transcript_words(video.video_id)
        transcript_text = " ".join(str(word.get("word", "") or "").strip() for word in words).strip()
        metadata_lines = [
            f"Video title: {video.title}",
            f"Video id: {video.video_id}",
            f"Published at: {video.published_at}",
            f"Duration seconds: {video.duration_seconds}",
            f"Views: {video.views}",
            f"Likes: {video.likes}",
        ]
        if video.description.strip():
            metadata_lines.append(f"Description: {video.description.strip()}")
        text = "\n".join(metadata_lines) + "\n\nTranscript:\n" + transcript_text
        return CreatorVideoDocument(
            title=video.title,
            summary=f"{video.views_label} views, {video.duration_label}, published {video.published_at[:10]}",
            text=text,
            video_id=video.video_id,
        )

    @staticmethod
    def _parse_profile_text(text: str) -> dict:
        cleaned = text.strip()
        if not cleaned:
            return {"raw_profile_text": ""}
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return {"raw_profile_text": cleaned}
        return parsed if isinstance(parsed, dict) else {"raw_profile_text": cleaned, "parsed_payload": parsed}

    @staticmethod
    def _notify(progress: ProgressCallback | None, stage: str, pct: int, detail: str) -> None:
        if progress:
            progress(stage, pct, detail)