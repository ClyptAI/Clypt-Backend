from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def compose_retrieve_query(
    *,
    profile: dict[str, Any] | None,
    preferences: dict[str, Any] | None,
    current_request: dict[str, Any],
) -> str:
    lines: list[str] = []

    length_range = current_request.get("length_range") or {}
    min_seconds = length_range.get("min_seconds")
    max_seconds = length_range.get("max_seconds")
    if min_seconds is not None and max_seconds is not None:
        lines.append(f"Find a {min_seconds}-{max_seconds} second clip from this video.")
    else:
        lines.append("Find the best matching clip from this video.")

    profile_parts: list[str] = []
    if profile:
        for key in ("primary_content_type", "tone", "pacing", "hook_style", "audience"):
            value = profile.get(key)
            if isinstance(value, str) and value.strip():
                profile_parts.append(f"{key.replace('_', ' ')}: {value.strip()}")
        topics = profile.get("recurring_topics") or profile.get("topics")
        if isinstance(topics, list) and topics:
            profile_parts.append("topics: " + ", ".join(str(item) for item in topics[:6]))
    if profile_parts:
        lines.append("Creator profile: " + "; ".join(profile_parts) + ".")

    preference_parts: list[str] = []
    if preferences:
        for key in ("caption_style", "brand_safety", "speaker_focus", "framing_preference", "hook_style_preference"):
            value = preferences.get(key)
            if isinstance(value, str) and value.strip():
                preference_parts.append(f"{key.replace('_', ' ')}: {value.strip()}")
        goals = preferences.get("clip_goals") or preferences.get("tone_preferences")
        if isinstance(goals, list) and goals:
            preference_parts.append("goals: " + ", ".join(str(item) for item in goals[:6]))
    if preference_parts:
        lines.append("User preferences: " + "; ".join(preference_parts) + ".")

    current_parts: list[str] = []
    query = str(current_request.get("query", "") or "").strip()
    if query:
        current_parts.append(query)
    goal = str(current_request.get("goal", "") or "").strip()
    if goal and goal.lower() not in query.lower():
        current_parts.append(f"Goal: {goal}")
    must_include = current_request.get("must_include")
    if isinstance(must_include, list) and must_include:
        current_parts.append("Must include: " + ", ".join(str(item) for item in must_include))
    avoid = current_request.get("avoid")
    if isinstance(avoid, list) and avoid:
        current_parts.append("Avoid: " + ", ".join(str(item) for item in avoid))
    if current_parts:
        lines.append("Current ask: " + " ".join(current_parts))

    return "\n".join(lines).strip()


@dataclass(frozen=True)
class RetrieveResult:
    query: str
    anchor_node_id: str
    clip: dict[str, Any]
    retrieval_mode: str = "output_fallback"


class OutputBackedRetrieveService:
    def __init__(
        self,
        *,
        candidates_path: str | Path,
    ) -> None:
        self.candidates_path = Path(candidates_path)

    def retrieve(self, *, final_query: str, run_id: str, creator_id: str | None = None) -> RetrieveResult:
        payload = json.loads(self.candidates_path.read_text(encoding="utf-8"))
        candidates = list(payload.get("candidates", []))
        if not candidates:
            raise RuntimeError(f"No clip candidates found in {self.candidates_path}")

        query_tokens = set(_tokenize(final_query))

        def candidate_score(candidate: dict[str, Any]) -> float:
            text_parts = [
                str(candidate.get("transcript_excerpt", "") or ""),
                str(candidate.get("justification", "") or ""),
                " ".join(str(item) for item in (candidate.get("sample_comments") or [])),
            ]
            candidate_tokens = set(_tokenize(" ".join(text_parts)))
            overlap = len(query_tokens & candidate_tokens)
            return float(candidate.get("final_score", 0.0) or 0.0) + overlap * 2.5

        best = max(candidates, key=candidate_score)
        clip_start_ms = int(best.get("clip_start_ms", 0) or 0)
        clip_end_ms = int(best.get("clip_end_ms", 0) or 0)
        duration_s = max(0.0, (clip_end_ms - clip_start_ms) / 1000.0)
        title = self._build_title(best)
        clip = {
            "id": f"{run_id}-retrieve-{best.get('rank', 1)}",
            "rank": int(best.get("rank", 1) or 1),
            "title": title,
            "start_time": round(clip_start_ms / 1000.0, 3),
            "end_time": round(clip_end_ms / 1000.0, 3),
            "duration": round(duration_s, 3),
            "score": float(best.get("final_score", 0.0) or 0.0),
            "transcript": str(best.get("transcript_excerpt", "") or ""),
            "justification": str(best.get("justification", "") or ""),
            "framing_type": "single_person",
            "speaker": None,
            "scores": {
                "clip_worthiness": float(best.get("final_score", 0.0) or 0.0),
                "audience_signal_score": float(best.get("raw_score", 0.0) or 0.0),
            },
            "node_ids": [],
            "best_cut": True,
            "pinned": False,
            "evidence_comment_ids": best.get("evidence_comment_ids", []),
        }
        return RetrieveResult(
            query=final_query,
            anchor_node_id=str(best.get("rank", "1")),
            clip=clip,
        )

    @staticmethod
    def _build_title(candidate: dict[str, Any]) -> str:
        sample_comments = candidate.get("sample_comments") or []
        if sample_comments:
            text = str(sample_comments[0]).strip().strip('"')
            return text[:80] + ("..." if len(text) > 80 else "")
        transcript = str(candidate.get("transcript_excerpt", "") or "").strip()
        words = transcript.split()
        if not words:
            return "Retrieved Clip"
        preview = " ".join(words[:8]).strip()
        return preview[:80] + ("..." if len(preview) > 80 else "")
