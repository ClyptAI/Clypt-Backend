from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.onboarding_jobs import OnboardingJobStore
from backend.services.creator_store import FileCreatorStore


class StubChannel:
    def __init__(self) -> None:
        self.channel_id = "UC123"
        self.channel_name = "Theo - t3.gg"
        self.channel_url = "https://youtube.com/@t3dotgg"
        self.handle = "@t3dotgg"
        self.avatar_url = "https://example.com/avatar.jpg"
        self.banner_url = "https://example.com/banner.jpg"
        self.description = "Developer content"
        self.category = "Science & Technology"
        self.subscriber_count = 420000
        self.subscriber_count_label = "420K"
        self.total_views = 89000000
        self.total_views_label = "89M"
        self.upload_frequency_label = "~5 videos/week"
        self.joined_date_label = "2020"


class StubVideo:
    def __init__(self, video_id: str, title: str) -> None:
        self.video_id = video_id
        self.title = title
        self.views = 2100000
        self.views_label = "2.1M"
        self.duration_seconds = 58
        self.duration_label = "58s"
        self.likes = 89000
        self.likes_label = "89K"
        self.thumbnail_url = "https://example.com/thumb.jpg"
        self.published_at = "2026-03-20T00:00:00Z"
        self.description = "Description"


class StubResolveResult:
    def __init__(self) -> None:
        self.channel = StubChannel()
        self.recent_shorts = [StubVideo("short_1", "React Server Components in 60 seconds")]
        self.recent_videos = [StubVideo("video_1", "The T3 Stack in 2024")]


class StubAnalysis:
    def __init__(self) -> None:
        self.creator_id = "creator_001"
        self.channel = {"channel_id": "UC123", "channel_name": "Theo - t3.gg"}
        self.profile = {
            "creator_id": "creator_001",
            "creator_archetype": "Educator-Entertainer",
            "brand_voice": ["Opinionated", "Fast-paced"],
        }
        self.workspace = {"category_id": "cat_1", "topic_id": "topic_1"}
        self.sources = [{"video_id": "video_1", "title": "The T3 Stack in 2024"}]


class StubOnboardingService:
    def __init__(self) -> None:
        self.analyze_calls: list[str] = []

    def resolve_channel(self, query: str):
        assert query == "@t3dotgg"
        return StubResolveResult()

    def analyze_channel(self, channel_id: str, *, progress=None):
        self.analyze_calls.append(channel_id)
        if progress:
            progress("resolve_channel", 10, "Resolving")
            progress("generate_profile", 90, "Generating")
        return StubAnalysis()


class StubRetrieveService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def retrieve(self, *, final_query: str, run_id: str, creator_id: str | None = None):
        self.calls.append(
            {
                "final_query": final_query,
                "run_id": run_id,
                "creator_id": creator_id,
            }
        )
        return type(
            "RetrieveResult",
            (),
            {
                "query": final_query,
                "anchor_node_id": "anchor_1",
                "clip": {
                    "id": "clip_1",
                    "title": "Retrieved Clip",
                    "start_time": 12.0,
                    "end_time": 39.0,
                    "duration": 27.0,
                    "score": 94.0,
                    "transcript": "Clip transcript",
                    "justification": "Best match",
                    "node_ids": [],
                },
                "retrieval_mode": "stub",
            },
        )()


def _temp_root() -> Path:
    root = Path(".pytest-local-temp") / f"onboarding-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_client(root: Path, *, onboarding_service=None, retrieve_service=None) -> TestClient:
    onboarding = onboarding_service or StubOnboardingService()
    return TestClient(
        create_app(
            onboarding_service=onboarding,
            youtube_service=onboarding,
            job_store=OnboardingJobStore(root / "jobs"),
            creator_store=FileCreatorStore(root / "creators"),
            retrieve_service=retrieve_service or StubRetrieveService(),
        )
    )


def test_resolve_channel_endpoint_returns_frontend_shape():
    tmp_path = _temp_root()
    service = StubOnboardingService()
    try:
        client = _build_client(tmp_path, onboarding_service=service)

        response = client.post("/api/v1/onboarding/channel/resolve", json={"query": "@t3dotgg"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["channel"]["channel_id"] == "UC123"
        assert payload["recent_shorts"][0]["video_id"] == "short_1"
        assert payload["recent_videos"][0]["video_id"] == "video_1"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_analyze_channel_endpoint_creates_job_and_persists_result():
    tmp_path = _temp_root()
    service = StubOnboardingService()
    try:
        client = _build_client(tmp_path, onboarding_service=service)

        response = client.post("/api/v1/onboarding/channel/analyze", json={"channel_id": "UC123"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "queued"

        job_response = client.get(f"/api/v1/onboarding/channel/analyze/{payload['job_id']}")
        assert job_response.status_code == 200
        job = job_response.json()
        assert job["status"] == "succeeded"
        assert job["profile"]["creator_archetype"] == "Educator-Entertainer"
        assert job["recent_items_scanned"][0]["video_id"] == "video_1"

        profile_response = client.get("/api/v1/creators/creator_001/profile")
        assert profile_response.status_code == 200
        assert profile_response.json()["creator_archetype"] == "Educator-Entertainer"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_get_analysis_job_returns_404_for_unknown_job():
    tmp_path = _temp_root()
    try:
        client = _build_client(tmp_path)

        response = client.get("/api/v1/onboarding/channel/analyze/missing")

        assert response.status_code == 404
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_creator_preferences_round_trip():
    tmp_path = _temp_root()
    try:
        client = _build_client(tmp_path)

        save_response = client.put(
            "/api/v1/creators/creator_001/preferences",
            json={
                "preferred_duration_range": {"min_seconds": 20, "max_seconds": 35},
                "clip_goals": ["funny", "controversial"],
                "caption_style": "viral bold",
                "brand_safety": "brand safe",
                "speaker_focus": "best moment regardless",
                "framing_preference": "tight",
                "default_retrieve_queries": ["Find the funniest moment"],
            },
        )

        assert save_response.status_code == 200
        assert save_response.json()["saved"] is True

        get_response = client.get("/api/v1/creators/creator_001/preferences")
        assert get_response.status_code == 200
        payload = get_response.json()
        assert payload["caption_style"] == "viral bold"
        assert payload["preferred_duration_range"]["min_seconds"] == 20
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_retrieve_endpoint_builds_final_query_from_profile_and_preferences():
    tmp_path = _temp_root()
    retrieve_service = StubRetrieveService()
    try:
        client = _build_client(tmp_path, retrieve_service=retrieve_service)

        client.put(
            "/api/v1/creators/creator_001/preferences",
            json={
                "preferred_duration_range": {"min_seconds": 20, "max_seconds": 35},
                "clip_goals": ["funny"],
                "caption_style": "viral bold",
                "brand_safety": "brand safe",
                "speaker_focus": "best moment regardless",
                "framing_preference": "tight",
            },
        )
        creator_store = FileCreatorStore(tmp_path / "creators")
        creator_store.save_profile(
            "creator_001",
            {
                "primary_content_type": "commentary",
                "tone": "sarcastic",
                "pacing": "fast-paced",
                "hook_style": "aggressive cold open",
                "recurring_topics": ["crypto", "internet culture"],
                "audience": "young creators",
            },
        )

        response = client.post(
            "/api/v1/runs/run_001/clips/retrieve",
            json={
                "query": "Find the strongest hot take",
                "creator_id": "creator_001",
                "goal": "controversial",
                "length_range": {"min_seconds": 20, "max_seconds": 35},
                "must_include": ["clear payoff"],
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["retrieval_mode"] == "stub"
        assert payload["query"] == "Find the strongest hot take"
        assert "Creator profile:" in payload["final_query"]
        assert "User preferences:" in payload["final_query"]
        assert "Current ask:" in payload["final_query"]
        assert retrieve_service.calls[0]["creator_id"] == "creator_001"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
