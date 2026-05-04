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


_TARGET_WIDTH = 1080
_TARGET_HEIGHT = 1920


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


def _even(value: float) -> int:
    return max(2, int(round(value / 2.0) * 2))


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def _probe_video_dimensions(source_video_path: Path) -> tuple[int, int]:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            str(source_video_path),
        ],
        text=True,
    )
    width_text, height_text = output.strip().replace("x", ",").split(",", maxsplit=1)
    width = int(width_text)
    height = int(height_text)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid source video dimensions {width}x{height}")
    return width, height


def _step_expression(*, cases: list[tuple[float, float, int]], default_value: int) -> str:
    expression = str(default_value)
    for start_s, end_s, value in reversed(cases):
        expression = f"if(between(t\\,{start_s:.3f}\\,{end_s:.3f})\\,{value}\\,{expression})"
    return expression


def _linear_expression(*, keyframes: list[dict[str, Any]], axis: str, default_value: int) -> str:
    points = [
        (max(0.0, float(keyframe.get("time_ms") or 0) / 1000.0), float(keyframe.get(axis) or 0.0))
        for keyframe in keyframes
    ]
    points.sort(key=lambda item: item[0])
    if not points:
        return str(default_value)
    if len(points) == 1:
        return str(_even(points[0][1]))

    expression = f"{points[-1][1]:.3f}"
    for (start_s, start_value), (end_s, end_value) in reversed(list(zip(points, points[1:]))):
        if end_s <= start_s:
            continue
        delta = end_value - start_value
        segment = (
            f"({start_value:.3f}+({delta:.3f})*(t-{start_s:.3f})/"
            f"{(end_s - start_s):.3f})"
        )
        expression = (
            f"if(between(t\\,{start_s:.3f}\\,{end_s:.3f})\\,"
            f"{segment}\\,{expression})"
        )
    first_s, first_value = points[0]
    return f"if(lte(t\\,{first_s:.3f})\\,{first_value:.3f}\\,{expression})"


def _interpolated_keyframe_value(
    *,
    points: list[tuple[float, float]],
    timestamp_s: float,
    default_value: int,
) -> float:
    if not points:
        return float(default_value)
    if timestamp_s <= points[0][0]:
        return points[0][1]
    for (start_s, start_value), (end_s, end_value) in zip(points, points[1:]):
        if end_s <= start_s:
            continue
        if start_s <= timestamp_s <= end_s:
            progress = (timestamp_s - start_s) / (end_s - start_s)
            return start_value + ((end_value - start_value) * progress)
    return points[-1][1]


def _write_crop_sendcmd_file(
    *,
    keyframes: list[dict[str, Any]],
    command_path: Path,
    output_fps: int,
    default_x: int,
    default_y: int,
) -> None:
    x_points = [
        (max(0.0, float(keyframe.get("time_ms") or 0) / 1000.0), float(keyframe.get("x") or default_x))
        for keyframe in keyframes
    ]
    y_points = [
        (max(0.0, float(keyframe.get("time_ms") or 0) / 1000.0), float(keyframe.get("y") or default_y))
        for keyframe in keyframes
    ]
    x_points.sort(key=lambda item: item[0])
    y_points.sort(key=lambda item: item[0])
    last_time_s = max([point[0] for point in x_points + y_points], default=0.0)
    frame_step_s = 1.0 / max(1, int(output_fps))
    frame_count = max(1, int(last_time_s / frame_step_s) + 2)
    lines: list[str] = []
    previous_x: int | None = None
    previous_y: int | None = None
    for frame_index in range(frame_count + 1):
        timestamp_s = min(last_time_s, frame_index * frame_step_s)
        x = _even(_interpolated_keyframe_value(points=x_points, timestamp_s=timestamp_s, default_value=default_x))
        y = _even(_interpolated_keyframe_value(points=y_points, timestamp_s=timestamp_s, default_value=default_y))
        if previous_x == x and previous_y == y and frame_index < frame_count:
            continue
        previous_x, previous_y = x, y
        lines.append(f"{timestamp_s:.3f} crop@follow x {x}, crop@follow y {y};")
    command_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _crop_filter_from_plan(
    *,
    crop_plan: dict[str, Any],
    source_dimensions: tuple[int, int],
    crop_command_path: Path | None = None,
    output_fps: int = 30,
) -> str:
    source_width, source_height = source_dimensions
    keyframes = list(crop_plan.get("keyframes") or [])
    if keyframes:
        crop_width = _clamp(
            _even(float(crop_plan.get("crop_width") or 0)),
            2,
            source_width,
        )
        crop_height = _clamp(
            _even(float(crop_plan.get("crop_height") or 0)),
            2,
            source_height,
        )
        default_x = _even((source_width - crop_width) / 2.0)
        default_y = _even((source_height - crop_height) / 2.0)
        if crop_command_path is not None:
            first_x = _even(float(keyframes[0].get("x") or default_x))
            first_y = _even(float(keyframes[0].get("y") or default_y))
            _write_crop_sendcmd_file(
                keyframes=keyframes,
                command_path=crop_command_path,
                output_fps=output_fps,
                default_x=default_x,
                default_y=default_y,
            )
            return (
                f"sendcmd=f={crop_command_path},"
                f"crop@follow={crop_width}:{crop_height}:{first_x}:{first_y},"
                f"scale={_TARGET_WIDTH}:{_TARGET_HEIGHT}"
            )
        x_expr = _linear_expression(
            keyframes=keyframes,
            axis="x",
            default_value=default_x,
        )
        y_expr = _linear_expression(
            keyframes=keyframes,
            axis="y",
            default_value=default_y,
        )
        return f"crop={crop_width}:{crop_height}:{x_expr}:{y_expr},scale={_TARGET_WIDTH}:{_TARGET_HEIGHT}"

    target_aspect = _TARGET_WIDTH / _TARGET_HEIGHT
    source_aspect = source_width / source_height
    if source_aspect > target_aspect:
        crop_width = _even(source_height * target_aspect)
        crop_height = source_height
    else:
        crop_width = source_width
        crop_height = _even(source_width / target_aspect)

    default_x = _even((source_width - crop_width) / 2.0)
    default_y = _even((source_height - crop_height) / 2.0)
    x_cases: list[tuple[float, float, int]] = []
    y_cases: list[tuple[float, float, int]] = []
    for segment in crop_plan.get("segments", []):
        bbox = segment.get("bbox_xyxy") or []
        if len(bbox) < 4:
            continue
        start_s = max(0.0, float(segment.get("start_ms") or 0) / 1000.0)
        end_s = max(start_s, float(segment.get("end_ms") or 0) / 1000.0)
        center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
        center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
        x = _clamp(_even(center_x - (crop_width / 2.0)), 0, source_width - crop_width)
        y = _clamp(_even(center_y - (crop_height / 2.0)), 0, source_height - crop_height)
        x_cases.append((start_s, end_s, x))
        y_cases.append((start_s, end_s, y))

    if not x_cases:
        return (
            f"scale={_TARGET_WIDTH}:{_TARGET_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={_TARGET_WIDTH}:{_TARGET_HEIGHT}"
        )

    x_expr = _step_expression(cases=x_cases, default_value=default_x)
    y_expr = _step_expression(cases=y_cases, default_value=default_y)
    return f"crop={crop_width}:{crop_height}:{x_expr}:{y_expr},scale={_TARGET_WIDTH}:{_TARGET_HEIGHT}"


def _ffmpeg_filter(
    *,
    ass_path: Path,
    fonts_dir: Path,
    crop_plan: dict[str, Any] | None = None,
    source_dimensions: tuple[int, int] | None = None,
    crop_command_path: Path | None = None,
    output_fps: int = 30,
) -> str:
    escaped_ass = str(ass_path).replace("\\", "/").replace(":", "\\:")
    escaped_fonts = str(fonts_dir).replace("\\", "/").replace(":", "\\:")
    if crop_plan and (crop_plan.get("segments") or crop_plan.get("keyframes")):
        if source_dimensions is None:
            raise ValueError("source dimensions are required for tracklet crop rendering")
        video_filter = _crop_filter_from_plan(
            crop_plan=crop_plan,
            source_dimensions=source_dimensions,
            crop_command_path=crop_command_path,
            output_fps=output_fps,
        )
    else:
        video_filter = (
            f"scale={_TARGET_WIDTH}:{_TARGET_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={_TARGET_WIDTH}:{_TARGET_HEIGHT}"
        )
    return (
        f"{video_filter},"
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
            crop_plan = dict(compiled.get("crop_plan") or {})
            source_dimensions = (
                _probe_video_dimensions(source_video_path)
                if crop_plan.get("segments") or crop_plan.get("keyframes")
                else None
            )
            crop_command_path = tmp_dir / f"crop_{clip_id}.cmd" if crop_plan.get("keyframes") else None
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
                _ffmpeg_filter(
                    ass_path=ass_path,
                    fonts_dir=fonts_dir,
                    crop_plan=crop_plan,
                    source_dimensions=source_dimensions,
                    crop_command_path=crop_command_path,
                    output_fps=request.output_fps,
                ),
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
