from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.api.onboarding_jobs import OnboardingJobStore
from backend.api.pipeline_runs import PipelineRunStore
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
DEFAULT_PIPELINE_RUN_STATE_ROOT = Path(
    os.getenv("CLYPT_PIPELINE_RUN_STATE_ROOT", Path(__file__).resolve().parent.parent / "outputs" / "pipeline_runs")
)
DEFAULT_PIPELINE_OUTPUT_ROOT = Path(
    os.getenv("CLYPT_PIPELINE_OUTPUT_ROOT", Path(__file__).resolve().parent.parent / "outputs")
)
# FFmpeg clips output directory (our renderer writes to backend/outputs/clips/)
DEFAULT_RENDERED_CLIPS_ROOT = Path(
    os.getenv("CLYPT_RENDERED_CLIPS_ROOT", Path(__file__).resolve().parent.parent / "outputs" / "clips")
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


class PipelineRunRequest(BaseModel):
    video_url: str = Field(min_length=1)
    creator_id: str | None = None


def create_app(
    *,
    onboarding_service: CreatorOnboardingService | None = None,
    youtube_service: YouTubeChannelService | None = None,
    job_store: OnboardingJobStore | None = None,
    creator_store: FileCreatorStore | None = None,
    auth_store: FileAuthStore | None = None,
    retrieve_service: Any | None = None,
    pipeline_run_store: PipelineRunStore | None = None,
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
    app.state.pipeline_run_store = pipeline_run_store or PipelineRunStore(DEFAULT_PIPELINE_RUN_STATE_ROOT)

    def _job_store() -> OnboardingJobStore:
        return app.state.job_store

    def _creator_store() -> FileCreatorStore:
        return app.state.creator_store

    def _auth_store() -> FileAuthStore:
        return app.state.auth_store

    def _retrieve_service():
        return app.state.retrieve_service

    def _pipeline_run_store() -> PipelineRunStore:
        return app.state.pipeline_run_store

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
            "pipeline_run_state_root": str(_pipeline_run_store().root),
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

    @app.post("/api/v1/runs")
    def start_pipeline_run(payload: PipelineRunRequest, background_tasks: BackgroundTasks) -> dict:
        store = _pipeline_run_store()
        run = store.create_run(video_url=payload.video_url, creator_id=payload.creator_id)

        def _run() -> None:
            repo_root = Path(__file__).resolve().parents[2]
            log_root = _pipeline_run_store().root / run["run_id"]
            log_root.mkdir(parents=True, exist_ok=True)
            stdout_path = log_root / "stdout.log"
            stderr_path = log_root / "stderr.log"
            try:
                if _full_pipeline_enabled():
                    store.mark_running(run["run_id"], phase="phase_1", progress_pct=10, detail="Starting full video pipeline run")
                    env = os.environ.copy()
                    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
                        process = subprocess.run(
                            [sys.executable, "-m", "backend.pipeline.run_pipeline"],
                            input=(payload.video_url.strip() + "\n"),
                            text=True,
                            cwd=str(repo_root),
                            env={**env, "PYTHONPATH": str(repo_root)},
                            stdout=stdout_file,
                            stderr=stderr_file,
                            check=False,
                        )
                    if process.returncode != 0:
                        detail = _tail_text(stderr_path) or _tail_text(stdout_path) or f"Pipeline failed ({process.returncode})"
                        store.mark_failed(run["run_id"], phase="pipeline", detail=detail)
                        return
                    summary = _build_pipeline_summary(run["run_id"], payload.video_url)
                else:
                    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
                        summary = _run_lightweight_video_analysis(
                            run_id=run["run_id"],
                            video_url=payload.video_url,
                            store=store,
                            stdout_file=stdout_file,
                            stderr_file=stderr_file,
                        )
            except Exception as exc:
                store.mark_failed(run["run_id"], phase="pipeline", detail=str(exc))
                return
            store.mark_succeeded(run["run_id"], summary=summary)

        background_tasks.add_task(_run)
        return {"run_id": run["run_id"], "status": run["status"], "video_url": payload.video_url}

    @app.get("/api/v1/runs/{run_id}/status")
    def get_pipeline_run_status(run_id: str) -> dict:
        run = _pipeline_run_store().get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="pipeline run not found")
        return run

    @app.get("/api/v1/runs/{run_id}")
    def get_pipeline_run(run_id: str) -> dict:
        run = _pipeline_run_store().get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="pipeline run not found")
        return {**run, "summary": run.get("summary") or _build_pipeline_summary(run_id, run.get("video_url", ""))}

    @app.get("/api/v1/runs/{run_id}/graph")
    def get_pipeline_graph(run_id: str) -> dict:
        run = _pipeline_run_store().get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="pipeline run not found")
        prefer_default_graph = str(((run.get("summary") or {}).get("mode") or "")).strip() == "video_url_lite"
        nodes, edges = _select_graph_payload(run_id, prefer_default=prefer_default_graph)
        return {"run_id": run_id, "nodes": nodes, "edges": edges}

    @app.get("/api/v1/rendered-clips/{filename}")
    def get_rendered_clip(filename: str) -> FileResponse:
        if "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="invalid rendered clip filename")
        path = DEFAULT_RENDERED_CLIPS_ROOT / filename
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="rendered clip not found")
        return FileResponse(path, media_type="video/mp4", filename=filename)

    @app.get("/api/v1/runs/{run_id}/clips")
    def get_pipeline_clips(request: Request, run_id: str) -> dict:
        run = _pipeline_run_store().get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="pipeline run not found")
        clips_payload = _load_json_file(_run_artifact_path(run_id, "crowd_3_clip_candidates.json")) or _load_json_file(DEFAULT_PIPELINE_OUTPUT_ROOT / "crowd_3_clip_candidates.json")
        remotion_payloads = _load_json_file(_run_artifact_path(run_id, "crowd_remotion_payloads_array.json")) or _load_json_file(DEFAULT_PIPELINE_OUTPUT_ROOT / "crowd_remotion_payloads_array.json")
        clips = clips_payload.get("candidates", []) if isinstance(clips_payload, dict) else (clips_payload if isinstance(clips_payload, list) else [])
        rendered_files = _rendered_clip_files()
        rendered_base = str(request.base_url).rstrip("/")
        rendered_clips = [{"filename": path.name, "url": f"{rendered_base}/api/v1/rendered-clips/{path.name}", "size_bytes": path.stat().st_size} for path in rendered_files]
        enriched_clips: list[dict[str, Any]] = []
        for index, clip in enumerate(clips if isinstance(clips, list) else []):
            item = dict(clip) if isinstance(clip, dict) else {"value": clip}
            if index < len(rendered_clips):
                item["rendered_video_url"] = rendered_clips[index]["url"]
                item["rendered_video_filename"] = rendered_clips[index]["filename"]
            enriched_clips.append(item)
        return {"run_id": run_id, "clips": enriched_clips, "remotion_payloads": remotion_payloads if isinstance(remotion_payloads, list) else [], "rendered_clips": rendered_clips}

    @app.get("/api/v1/runs/{run_id}/artifacts")
    def get_pipeline_artifacts(run_id: str) -> dict:
        run = _pipeline_run_store().get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="pipeline run not found")
        return {"run_id": run_id, "artifacts": _artifact_inventory(run_id)}

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


def _full_pipeline_enabled() -> bool:
    return bool(str(os.getenv("DO_PHASE1_BASE_URL", "") or "").strip())


def _run_artifact_root(run_id: str) -> Path:
    return DEFAULT_PIPELINE_OUTPUT_ROOT / "pipeline_runs" / run_id


def _run_artifact_path(run_id: str, name: str) -> Path:
    return _run_artifact_root(run_id) / name


def _tail_text(path: Path, max_chars: int = 2000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        text = "…" + text[-max_chars:]
    return text.strip()


def _load_json_file(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _rendered_clip_files() -> list[Path]:
    if not DEFAULT_RENDERED_CLIPS_ROOT.exists():
        return []
    files = sorted(DEFAULT_RENDERED_CLIPS_ROOT.glob("*.mp4"))
    return files


def _artifact_inventory(run_id: str) -> list[dict[str, Any]]:
    root = _run_artifact_root(run_id)
    if not root.exists():
        root = DEFAULT_PIPELINE_OUTPUT_ROOT
    items: list[dict[str, Any]] = []
    if not root.exists():
        return items
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix in (".json", ".log", ".txt", ".csv"):
            items.append({
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "suffix": path.suffix,
            })
    return items


def _select_graph_payload(run_id: str, *, prefer_default: bool = False) -> tuple[list, list]:
    candidates = [
        _run_artifact_path(run_id, "crowd_2a_nodes.json"),
        DEFAULT_PIPELINE_OUTPUT_ROOT / "crowd_2a_nodes.json",
        DEFAULT_PIPELINE_OUTPUT_ROOT / "phase_2a_content_mechanism_nodes.json",
    ]
    if prefer_default:
        candidates = list(reversed(candidates))
    nodes: list = []
    for path in candidates:
        data = _load_json_file(path)
        if isinstance(data, list) and data:
            nodes = data
            break
        if isinstance(data, dict) and data.get("nodes"):
            nodes = data["nodes"]
            break
    edge_candidates = [
        _run_artifact_path(run_id, "crowd_2b_edges.json"),
        DEFAULT_PIPELINE_OUTPUT_ROOT / "crowd_2b_edges.json",
        DEFAULT_PIPELINE_OUTPUT_ROOT / "phase_2b_narrative_edges.json",
    ]
    if prefer_default:
        edge_candidates = list(reversed(edge_candidates))
    edges: list = []
    for path in edge_candidates:
        data = _load_json_file(path)
        if isinstance(data, list) and data:
            edges = data
            break
        if isinstance(data, dict) and data.get("edges"):
            edges = data["edges"]
            break
    return nodes, edges


def _build_pipeline_summary(run_id: str, video_url: str, prefer_default_graph: bool = False) -> dict[str, Any]:
    nodes, edges = _select_graph_payload(run_id, prefer_default=prefer_default_graph)
    clips_payload = _load_json_file(_run_artifact_path(run_id, "crowd_3_clip_candidates.json")) or _load_json_file(DEFAULT_PIPELINE_OUTPUT_ROOT / "crowd_3_clip_candidates.json")
    clips = clips_payload.get("candidates", []) if isinstance(clips_payload, dict) else (clips_payload if isinstance(clips_payload, list) else [])
    rendered = _rendered_clip_files()
    return {
        "video_url": video_url,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "clip_candidate_count": len(clips) if isinstance(clips, list) else 0,
        "rendered_clip_count": len(rendered),
    }


def _parse_timestamp_to_ms(token: str) -> int | None:
    token = token.strip()
    iso_seconds = _parse_iso8601_duration_seconds(token)
    if iso_seconds is not None:
        return int(iso_seconds * 1000)
    colon_match = re.match(r"^(\d+):(\d{2}):(\d{2})(?:\.(\d+))?$", token)
    if colon_match:
        h, m, s = int(colon_match.group(1)), int(colon_match.group(2)), int(colon_match.group(3))
        frac = float(f"0.{colon_match.group(4)}") if colon_match.group(4) else 0.0
        return int((h * 3600 + m * 60 + s + frac) * 1000)
    short_match = re.match(r"^(\d+):(\d{2})(?:\.(\d+))?$", token)
    if short_match:
        m, s = int(short_match.group(1)), int(short_match.group(2))
        frac = float(f"0.{short_match.group(3)}") if short_match.group(3) else 0.0
        return int((m * 60 + s + frac) * 1000)
    plain_match = re.match(r"^(\d+(?:\.\d+)?)$", token)
    if plain_match:
        return int(float(plain_match.group(1)) * 1000)
    return None


def _parse_iso8601_duration_seconds(value: str) -> float | None:
    m = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?$", value, re.IGNORECASE)
    if not m:
        return None
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    seconds = float(m.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def _format_duration_label(seconds: float) -> str:
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _build_lightweight_candidates(stage1_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Build clip candidates from a lightweight stage-1 analysis (e.g. YouTube comments)."""
    candidates: list[dict[str, Any]] = []
    comments = stage1_payload.get("comments", [])
    for idx, comment in enumerate(comments if isinstance(comments, list) else []):
        if not isinstance(comment, dict):
            continue
        text = str(comment.get("text", "") or "").strip()
        if not text:
            continue
        ts_raw = comment.get("timestamp") or comment.get("time") or ""
        start_ms = _parse_timestamp_to_ms(str(ts_raw)) if ts_raw else None
        candidates.append({
            "index": idx,
            "text": text,
            "start_ms": start_ms,
            "source": "youtube_comment",
        })
    return candidates


def _run_lightweight_video_analysis(
    *,
    run_id: str,
    video_url: str,
    store: Any,
    stdout_file: Any,
    stderr_file: Any,
) -> dict[str, Any]:
    """Lightweight analysis when no Phase 1 service is available.

    Attempts to fetch public YouTube metadata (comments, description) and
    build basic clip candidates from timestamps found in comments.
    """
    store.mark_running(run_id, phase="lightweight", progress_pct=10, detail="Starting lightweight video analysis")

    stage1: dict[str, Any] = {"video_url": video_url, "comments": [], "description": ""}

    # Try yt-dlp for metadata
    try:
        repo_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, "-c", (
                "import json, subprocess, sys; "
                "r = subprocess.run(['yt-dlp', '--skip-download', '--dump-json', sys.argv[1]], capture_output=True, text=True, check=True); "
                "info = json.loads(r.stdout); "
                "comments = [{'text': c.get('text',''), 'timestamp': c.get('timestamp','')} for c in (info.get('comments') or [])[:200]]; "
                "print(json.dumps({'description': info.get('description',''), 'title': info.get('title',''), 'comments': comments}))"
            ), video_url],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=120,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            stage1.update(json.loads(result.stdout.strip()))
    except Exception:
        pass

    store.mark_running(run_id, phase="lightweight", progress_pct=50, detail="Building clip candidates")

    candidates = _build_lightweight_candidates(stage1)

    # Persist stage-1 and candidates
    artifact_root = _run_artifact_root(run_id)
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "lightweight_stage1.json").write_text(json.dumps(stage1, indent=2, default=str), encoding="utf-8")
    (artifact_root / "crowd_3_clip_candidates.json").write_text(
        json.dumps({"candidates": candidates}, indent=2, default=str), encoding="utf-8",
    )

    store.mark_running(run_id, phase="lightweight", progress_pct=90, detail="Finalizing")

    return {
        "video_url": video_url,
        "mode": "video_url_lite",
        "clip_candidate_count": len(candidates),
        "node_count": 0,
        "edge_count": 0,
        "rendered_clip_count": 0,
    }


app = create_app()