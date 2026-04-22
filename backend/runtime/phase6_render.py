"""Phase 6 render/export worker contract."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.pipeline.render.presets import load_caption_presets
from backend.providers.storage import parse_gcs_uri


@dataclass(slots=True)
class Phase6RenderRequest:
    run_id: str
    source_video_gcs_uri: str
    artifact_gcs_uris: dict[str, str]
    clips: list[dict[str, Any]]
    output_prefix: str
    output_fps: int = 30

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_video_gcs_uri": self.source_video_gcs_uri,
            "artifact_gcs_uris": dict(self.artifact_gcs_uris),
            "clips": [dict(item) for item in self.clips],
            "output_prefix": self.output_prefix,
            "output_fps": int(self.output_fps),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Phase6RenderRequest":
        run_id = str(payload.get("run_id") or "").strip()
        source_video_gcs_uri = str(payload.get("source_video_gcs_uri") or "").strip()
        artifact_gcs_uris = payload.get("artifact_gcs_uris") or {}
        clips = payload.get("clips") or []
        if not run_id:
            raise ValueError("run_id is required")
        parse_gcs_uri(source_video_gcs_uri)
        if not isinstance(artifact_gcs_uris, dict) or not artifact_gcs_uris.get("render_plan"):
            raise ValueError("artifact_gcs_uris.render_plan is required")
        if not isinstance(clips, list) or not clips:
            raise ValueError("clips must be a non-empty list")
        return cls(
            run_id=run_id,
            source_video_gcs_uri=source_video_gcs_uri,
            artifact_gcs_uris={str(key): str(value) for key, value in artifact_gcs_uris.items()},
            clips=[dict(item) for item in clips],
            output_prefix=str(payload.get("output_prefix") or f"phase14/{run_id}/render_outputs").strip("/"),
            output_fps=max(1, int(payload.get("output_fps") or 30)),
        )


def _ass_uri(request: Phase6RenderRequest, clip_id: str) -> str:
    key = f"captions_{clip_id}.ass"
    uri = request.artifact_gcs_uris.get(key)
    if not uri:
        raise ValueError(f"missing {key} artifact for clip {clip_id!r}")
    return uri


def _ffmpeg_filter(*, ass_path: Path, fonts_dir: Path) -> str:
    escaped_ass = str(ass_path).replace("\\", "/").replace(":", "\\:")
    escaped_fonts = str(fonts_dir).replace("\\", "/").replace(":", "\\:")
    return (
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"subtitles={escaped_ass}:fontsdir={escaped_fonts}"
    )


def _stage_font_assets(*, render_plan: dict[str, Any], clip_ids: list[str], fonts_dir: Path) -> None:
    default_asset_root = Path(__file__).resolve().parents[1] / "assets" / "fonts"
    asset_root = Path(os.environ.get("CLYPT_PHASE6_FONT_ASSET_DIR") or default_asset_root)
    presets = load_caption_presets()
    needed_assets: set[str] = set()
    clips_by_id = {str(clip["clip_id"]): dict(clip) for clip in render_plan.get("clips", [])}
    for clip_id in clip_ids:
        clip = clips_by_id.get(clip_id)
        if clip is None:
            continue
        preset_id = str(clip.get("caption_preset_id") or "").strip()
        if not preset_id:
            continue
        preset = presets[preset_id]
        needed_assets.add(preset.font_asset_id)

    for font_asset_id in sorted(needed_assets):
        candidates = [
            asset_root / f"{font_asset_id}.ttf",
            asset_root / f"{font_asset_id}.otf",
            *sorted(asset_root.glob(f"{font_asset_id}.*")),
        ]
        source = next((path for path in candidates if path.exists() and path.is_file()), None)
        if source is None:
            raise ValueError(
                f"missing pinned font asset {font_asset_id!r} in {asset_root}"
            )
        shutil.copy2(source, fonts_dir / source.name)


def run_phase6_render(
    *,
    request: Phase6RenderRequest,
    storage_client: Any,
    scratch_root: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix=f"phase6-render-{request.run_id}-",
        dir=str(scratch_root),
    ) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        source_video_path = storage_client.download_file(
            gcs_uri=request.source_video_gcs_uri,
            local_path=tmp_dir / "source.mp4",
        )
        render_plan_path = storage_client.download_file(
            gcs_uri=request.artifact_gcs_uris["render_plan"],
            local_path=tmp_dir / "render_plan.json",
        )
        render_plan = json.loads(render_plan_path.read_text(encoding="utf-8"))
        clips_by_id = {str(clip["clip_id"]): dict(clip) for clip in render_plan.get("clips", [])}
        fonts_dir = tmp_dir / "fonts"
        fonts_dir.mkdir(parents=True, exist_ok=True)
        _stage_font_assets(
            render_plan=render_plan,
            clip_ids=[str(clip["clip_id"]) for clip in request.clips],
            fonts_dir=fonts_dir,
        )

        outputs: list[dict[str, Any]] = []
        for clip in request.clips:
            clip_id = str(clip["clip_id"])
            compiled = clips_by_id.get(clip_id) or clip
            ass_path = storage_client.download_file(
                gcs_uri=_ass_uri(request, clip_id),
                local_path=tmp_dir / f"captions_{clip_id}.ass",
            )
            output_path = tmp_dir / f"{clip_id}.mp4"
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{float(compiled['clip_start_ms']) / 1000.0:.3f}",
                "-to",
                f"{float(compiled['clip_end_ms']) / 1000.0:.3f}",
                "-i",
                str(source_video_path),
                "-vf",
                _ffmpeg_filter(ass_path=ass_path, fonts_dir=fonts_dir),
                "-r",
                str(request.output_fps),
                "-c:v",
                "h264_nvenc",
                "-c:a",
                "aac",
                str(output_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            video_gcs_uri = storage_client.upload_file(
                local_path=output_path,
                object_name=f"{request.output_prefix}/{clip_id}.mp4",
            )
            outputs.append(
                {
                    "clip_id": clip_id,
                    "video_gcs_uri": video_gcs_uri,
                    "caption_ass_gcs_uri": request.artifact_gcs_uris[f"captions_{clip_id}.ass"],
                    "ffmpeg_command": " ".join(shlex.quote(part) for part in cmd),
                }
            )

    return {
        "run_id": request.run_id,
        "outputs": outputs,
        "render_backend": "modal_l40s_ffmpeg_libass",
        "total_ms": (time.perf_counter() - started) * 1000.0,
    }


__all__ = ["Phase6RenderRequest", "run_phase6_render"]
