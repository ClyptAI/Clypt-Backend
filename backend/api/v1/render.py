"""Render endpoints — stubbed until render pipeline integration."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .schemas import RenderJobStatus, RenderPreset, RenderSubmitRequest

router = APIRouter(tags=["render"])

# Default presets — these match what the frontend mock uses
_DEFAULT_PRESETS: list[RenderPreset] = [
    RenderPreset(
        id="tiktok_9x16",
        platform="TikTok",
        label="TikTok / Reels (9:16)",
        aspect_ratio="9:16",
        width=1080,
        height=1920,
        frame_rate=30,
        max_duration_s=180,
    ),
    RenderPreset(
        id="youtube_shorts_9x16",
        platform="YouTube",
        label="YouTube Shorts (9:16)",
        aspect_ratio="9:16",
        width=1080,
        height=1920,
        frame_rate=30,
        max_duration_s=60,
    ),
    RenderPreset(
        id="instagram_1x1",
        platform="Instagram",
        label="Instagram Square (1:1)",
        aspect_ratio="1:1",
        width=1080,
        height=1080,
        frame_rate=30,
        max_duration_s=60,
    ),
    RenderPreset(
        id="youtube_16x9",
        platform="YouTube",
        label="YouTube (16:9)",
        aspect_ratio="16:9",
        width=1920,
        height=1080,
        frame_rate=30,
        max_duration_s=None,
    ),
]


@router.get("/render/presets", response_model=list[RenderPreset])
def list_presets() -> list[RenderPreset]:
    return _DEFAULT_PRESETS


@router.post(
    "/runs/{run_id}/clips/{clip_id}/render",
    response_model=RenderJobStatus,
)
def submit_render(
    run_id: str,
    clip_id: str,
    body: RenderSubmitRequest,
) -> RenderJobStatus:
    # Stub — returns queued status. Real implementation will dispatch to render worker.
    return RenderJobStatus(
        clip_id=clip_id,
        status="queued",
        progress_pct=0.0,
        output_url=None,
        error=None,
    )


@router.get(
    "/runs/{run_id}/clips/{clip_id}/render",
    response_model=RenderJobStatus,
)
def get_render_status(
    run_id: str,
    clip_id: str,
) -> RenderJobStatus:
    # Stub — no render jobs exist yet
    raise HTTPException(status_code=404, detail="no render job found for this clip")
