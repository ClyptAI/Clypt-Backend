from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.api.onboarding_jobs import OnboardingJobStore
from backend.services.creator_onboarding import CreatorOnboardingService
from backend.services.creator_store import FileCreatorStore
from backend.services.auth_store import (
    DuplicateEmailError,
    FileAuthStore,
    InvalidCredentialsError,
    InvalidTokenError,
)
from backend.services.retrieve_bridge import OutputBackedRetrieveService, compose_retrieve_query
from backend.services.youtube_channel_service import YouTubeChannelError, YouTubeChannelService


DEFAULT_ONBOARDING_STATE_ROOT = Path(
    os.getenv("CLYPT_ONBOARDING_STATE_ROOT", Path(__file__).resolve().parent.parent / "outputs" / "onboarding_jobs")
)
DEFAULT_CREATOR_STORE_ROOT = Path(
    os.getenv("CLYPT_CREATOR_STORE_ROOT", Path(__file__).resolve().parent.parent / "outputs" / "creators")
)
DEFAULT_AUTH_STATE_ROOT = Path(
    os.getenv("CLYPT_AUTH_STATE_ROOT", Path(__file__).resolve().parent.parent / "outputs" / "auth")
)
DEFAULT_RETRIEVE_CANDIDATES_PATH = Path(
    os.getenv(
        "CLYPT_RETRIEVE_CANDIDATES_PATH",
        Path(__file__).resolve().parent.parent / "outputs" / "crowd_3_clip_candidates.json",
    )
)
DEFAULT_AUTH_COOKIE_NAME = str(os.getenv("CLYPT_AUTH_COOKIE_NAME", "clypt_session") or "clypt_session").strip()
DEFAULT_AUTH_SESSION_TTL_HOURS = int(os.getenv("CLYPT_AUTH_SESSION_TTL_HOURS", "720") or 720)


class ChannelResolveRequest(BaseModel):
    query: str = Field(min_length=1)


class ChannelAnalyzeRequest(BaseModel):
    channel_id: str = Field(min_length=1)


class PreferredDurationRange(BaseModel):
    min_seconds: int = Field(ge=0)
    max_seconds: int = Field(ge=0)


class CreatorPreferencesPayload(BaseModel):
    preferred_duration_range: PreferredDurationRange | None = None
    target_platforms: list[str] = Field(default_factory=list)
    tone_preferences: list[str] = Field(default_factory=list)
    avoid_topics: list[str] = Field(default_factory=list)
    caption_style: str = ""
    hook_importance: float | None = Field(default=None, ge=0.0, le=1.0)
    payoff_importance: float | None = Field(default=None, ge=0.0, le=1.0)
    default_retrieve_queries: list[str] = Field(default_factory=list)
    clip_goals: list[str] = Field(default_factory=list)
    brand_safety: str = ""
    speaker_focus: str = ""
    framing_preference: str = ""
    hook_style_preference: str = ""


class AuthSignupRequest(BaseModel):
    email: str = Field(min_length=3)
    password: str = Field(min_length=8)
    display_name: str = ""


class AuthLoginRequest(BaseModel):
    email: str = Field(min_length=3)
    password: str = Field(min_length=1)


class RetrieveClipRequest(BaseModel):
    query: str = ""
    creator_id: str | None = None
    goal: str = ""
    length_range: PreferredDurationRange | None = None
    must_include: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    preferences_override: dict[str, Any] = Field(default_factory=dict)


def create_app(
    *,
    onboarding_service: CreatorOnboardingService | None = None,
    youtube_service: YouTubeChannelService | None = None,
    job_store: OnboardingJobStore | None = None,
    creator_store: FileCreatorStore | None = None,
    auth_store: FileAuthStore | None = None,
    retrieve_service: Any | None = None,
) -> FastAPI:
    app = FastAPI(title="Clypt Backend API")
    frontend_origins = [
        origin.strip()
        for origin in str(
            os.getenv(
                "CLYPT_FRONTEND_ORIGINS",
                "http://localhost:8080,http://localhost:3000,http://localhost:5173",
            )
        ).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=frontend_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.job_store = job_store or OnboardingJobStore(DEFAULT_ONBOARDING_STATE_ROOT)
    app.state.onboarding_service = onboarding_service
    app.state.youtube_service = youtube_service
    app.state.creator_store = creator_store or FileCreatorStore(DEFAULT_CREATOR_STORE_ROOT)
    app.state.auth_store = auth_store or FileAuthStore(
        DEFAULT_AUTH_STATE_ROOT,
        session_ttl_hours=DEFAULT_AUTH_SESSION_TTL_HOURS,
    )
    app.state.retrieve_service = retrieve_service or OutputBackedRetrieveService(
        candidates_path=DEFAULT_RETRIEVE_CANDIDATES_PATH
    )

    def _job_store() -> OnboardingJobStore:
        return app.state.job_store

    def _creator_store() -> FileCreatorStore:
        return app.state.creator_store

    def _auth_store() -> FileAuthStore:
        return app.state.auth_store

    def _retrieve_service():
        return app.state.retrieve_service

    def _youtube_service() -> YouTubeChannelService:
        if app.state.youtube_service is None:
            app.state.youtube_service = YouTubeChannelService.from_env()
        return app.state.youtube_service

    def _onboarding_service() -> CreatorOnboardingService:
        if app.state.onboarding_service is None:
            app.state.onboarding_service = CreatorOnboardingService.from_env()
        return app.state.onboarding_service

    def _token_from_request(request: Request) -> str | None:
        header = str(request.headers.get("authorization", "") or "").strip()
        if header.lower().startswith("bearer "):
            token = header[7:].strip()
            if token:
                return token
        cookie_token = str(request.cookies.get(DEFAULT_AUTH_COOKIE_NAME, "") or "").strip()
        return cookie_token or None

    def _set_auth_cookie(response: Response, token: str) -> None:
        response.set_cookie(
            key=DEFAULT_AUTH_COOKIE_NAME,
            value=token,
            httponly=True,
            samesite="lax",
            secure=False,
            max_age=DEFAULT_AUTH_SESSION_TTL_HOURS * 3600,
        )

    def _clear_auth_cookie(response: Response) -> None:
        response.delete_cookie(key=DEFAULT_AUTH_COOKIE_NAME)

    def _build_auth_payload(user: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
        return {
            "token": session["token"],
            "expires_at": session["expires_at"],
            "user": user,
        }

    def _ensure_creator_shell(user: dict[str, Any]) -> None:
        creator_id = str(user.get("creator_id", "") or "")
        if not creator_id or _creator_store().get_profile(creator_id) is not None:
            return
        _creator_store().save_profile(
            creator_id,
            {
                "creator_id": creator_id,
                "user_id": user.get("user_id"),
                "email": user.get("email"),
                "display_name": user.get("display_name", ""),
                "created_via": "signup",
            },
        )

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "onboarding_state_root": str(_job_store().root),
            "creator_store_root": str(_creator_store().root),
            "auth_state_root": str(_auth_store().root),
            "senso_enabled": bool(str(os.getenv("SENSO_API_KEY", "") or "").strip()),
        }

    @app.post("/api/v1/auth/signup", status_code=status.HTTP_201_CREATED)
    def signup(payload: AuthSignupRequest, response: Response) -> dict:
        try:
            user = _auth_store().create_user(
                email=payload.email,
                password=payload.password,
                display_name=payload.display_name,
            )
        except DuplicateEmailError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        _ensure_creator_shell(user)
        session = _auth_store().create_session(user=user)
        _set_auth_cookie(response, session["token"])
        return _build_auth_payload(user, session)

    @app.post("/api/v1/auth/login")
    def login(payload: AuthLoginRequest, response: Response) -> dict:
        try:
            user = _auth_store().authenticate(email=payload.email, password=payload.password)
        except InvalidCredentialsError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

        _ensure_creator_shell(user)
        session = _auth_store().create_session(user=user)
        _set_auth_cookie(response, session["token"])
        return _build_auth_payload(user, session)

    @app.get("/api/v1/auth/me")
    def current_user(request: Request) -> dict:
        token = _token_from_request(request)
        if not token:
            raise HTTPException(status_code=401, detail="Authentication required.")
        try:
            user = _auth_store().get_user_for_token(token)
        except InvalidTokenError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        return {"user": user}

    @app.post("/api/v1/auth/logout")
    def logout(request: Request, response: Response) -> dict:
        token = _token_from_request(request)
        if token:
            _auth_store().revoke_session(token)
        _clear_auth_cookie(response)
        return {"logged_out": True}

    @app.post("/api/v1/onboarding/channel/resolve")
    def resolve_channel(payload: ChannelResolveRequest) -> dict:
        try:
            resolved = _youtube_service().resolve_channel(payload.query)
        except (RuntimeError, ValueError, YouTubeChannelError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "channel": resolved.channel.__dict__,
            "recent_shorts": [item.__dict__ for item in resolved.recent_shorts],
            "recent_videos": [item.__dict__ for item in resolved.recent_videos],
        }

    @app.post("/api/v1/onboarding/channel/analyze")
    def analyze_channel(payload: ChannelAnalyzeRequest, background_tasks: BackgroundTasks) -> dict:
        store = _job_store()
        creators = _creator_store()
        job = store.create_job(channel_id=payload.channel_id)

        def _run() -> None:
            try:
                analysis = _onboarding_service().analyze_channel(
                    payload.channel_id,
                    progress=lambda stage, pct, detail: store.mark_running(
                        job["job_id"],
                        stage=stage,
                        progress_pct=pct,
                        detail=detail,
                    ),
                )
            except Exception as exc:
                store.mark_failed(job["job_id"], stage="failed", detail=str(exc))
                return

            saved_profile = creators.save_profile(analysis.creator_id, analysis.profile)
            store.mark_succeeded(
                job["job_id"],
                profile=saved_profile,
                channel=analysis.channel,
                sources=analysis.sources,
            )

        background_tasks.add_task(_run)
        return {"job_id": job["job_id"], "status": job["status"]}

    @app.get("/api/v1/onboarding/channel/analyze/{job_id}")
    def get_analysis_job(job_id: str) -> dict:
        job = _job_store().get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="analysis job not found")
        return job

    @app.get("/api/v1/creators/{creator_id}/profile")
    def get_creator_profile(creator_id: str) -> dict:
        profile = _creator_store().get_profile(creator_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="creator profile not found")
        return profile

    @app.get("/api/v1/creators/{creator_id}/preferences")
    def get_creator_preferences(creator_id: str) -> dict:
        preferences = _creator_store().get_preferences(creator_id)
        if preferences is None:
            raise HTTPException(status_code=404, detail="creator preferences not found")
        return preferences

    @app.put("/api/v1/creators/{creator_id}/preferences")
    def save_creator_preferences(creator_id: str, payload: CreatorPreferencesPayload) -> dict:
        preferences = _creator_store().save_preferences(
            creator_id,
            payload.model_dump(mode="json", exclude_none=True),
        )
        return {
            "creator_id": creator_id,
            "saved": True,
            "preferences": preferences,
        }

    @app.post("/api/v1/runs/{run_id}/clips/retrieve")
    def retrieve_clip(run_id: str, payload: RetrieveClipRequest) -> dict:
        creator_id = payload.creator_id
        profile = _creator_store().get_profile(creator_id) if creator_id else None
        preferences = _creator_store().get_preferences(creator_id) if creator_id else None
        merged_preferences = _merge_preferences(preferences, payload.preferences_override)
        final_query = compose_retrieve_query(
            profile=profile,
            preferences=merged_preferences,
            current_request=payload.model_dump(mode="json", exclude_none=True),
        )
        try:
            result = _retrieve_service().retrieve(
                final_query=final_query,
                run_id=run_id,
                creator_id=creator_id,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "query": payload.query,
            "final_query": result.query,
            "anchor_node_id": result.anchor_node_id,
            "clip": result.clip,
            "retrieval_mode": result.retrieval_mode,
        }

    return app


def _merge_preferences(stored: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(stored or {})
    for key, value in (override or {}).items():
        merged[key] = value
    return merged


app = create_app()
