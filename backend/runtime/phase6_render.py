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
_DYNAMIC_CROP_MODE = "tracklet_follow_9x16_pose_x_dynamic_inside_person"


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


def _validate_dynamic_crop_plan(crop_plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mode = str(crop_plan.get("mode") or "").strip()
    if mode != _DYNAMIC_CROP_MODE:
        raise ValueError(f"unknown Phase6 crop plan mode {mode!r}")
    keyframes = [dict(keyframe) for keyframe in crop_plan.get("keyframes") or []]
    runs = [dict(run) for run in crop_plan.get("runs") or []]
    if not keyframes:
        raise ValueError("dynamic Phase6 crop plan requires keyframes")
    if not runs:
        raise ValueError("dynamic Phase6 crop plan requires run boundaries")
    required_keyframe_fields = {"time_ms", "x", "y", "w", "h", "run_id", "shot_id"}
    for keyframe in keyframes:
        missing = required_keyframe_fields - set(keyframe)
        if missing:
            raise ValueError(f"dynamic Phase6 crop keyframe missing fields: {sorted(missing)}")
        if float(keyframe["w"]) <= 0 or float(keyframe["h"]) <= 0:
            raise ValueError("dynamic Phase6 crop keyframe width/height must be positive")
    required_run_fields = {"run_id", "start_ms", "end_ms"}
    for run in runs:
        missing = required_run_fields - set(run)
        if missing:
            raise ValueError(f"dynamic Phase6 crop run missing fields: {sorted(missing)}")
    return keyframes, runs


def _keyframe_points_for_axis(
    *,
    keyframes: list[dict[str, Any]],
    axis: str,
    default_value: int,
) -> list[tuple[float, float]]:
    points = [
        (
            max(0.0, float(keyframe.get("time_ms") or 0) / 1000.0),
            float(keyframe.get(axis) if keyframe.get(axis) is not None else default_value),
        )
        for keyframe in keyframes
    ]
    points.sort(key=lambda item: item[0])
    return points


def _fixed_crop_dimensions_for_keyframes(
    *,
    keyframes: list[dict[str, Any]],
    source_dimensions: tuple[int, int],
) -> tuple[int, int]:
    source_width, source_height = source_dimensions
    target_aspect = _TARGET_WIDTH / _TARGET_HEIGHT
    min_width = min(max(2.0, float(keyframe.get("w") or 0.0)) for keyframe in keyframes)
    min_height = min(max(2.0, float(keyframe.get("h") or 0.0)) for keyframe in keyframes)
    fixed_width = min(min_width, min_height * target_aspect, float(source_width))
    fixed_width_even = _even(fixed_width)
    while fixed_width_even > 2:
        fixed_height_even = _even(fixed_width_even / target_aspect)
        if fixed_width_even <= min_width and fixed_height_even <= min_height and fixed_height_even <= source_height:
            return fixed_width_even, fixed_height_even
        fixed_width_even -= 2
    raise ValueError("dynamic Phase6 crop plan cannot derive a fixed 9:16 crop size")


def _resolved_xy_for_keyframe(
    *,
    keyframe: dict[str, Any],
    fixed_width: int,
    fixed_height: int,
    source_dimensions: tuple[int, int],
) -> tuple[int, int]:
    source_width, source_height = source_dimensions
    max_x = max(0, source_width - fixed_width)
    max_y = max(0, source_height - fixed_height)
    bbox = list(keyframe.get("bbox_xyxy") or [])
    if len(bbox) >= 4:
        bbox_x1 = float(bbox[0])
        bbox_y1 = float(bbox[1])
        bbox_x2 = float(bbox[2])
        bbox_y2 = float(bbox[3])
        anchor_x = float(
            keyframe.get("anchor_x")
            if keyframe.get("anchor_x") is not None
            else (bbox_x1 + bbox_x2) / 2.0
        )
        x_lower = max(0, min(max_x, _even(bbox_x1)))
        x_upper = max(x_lower, min(max_x, _even(bbox_x2 - fixed_width)))
        desired_x = _even(anchor_x - (fixed_width / 2.0))
        x = _clamp(desired_x, x_lower, x_upper)
        y_lower = max(0, min(max_y, _even(bbox_y1)))
        y_upper = max(y_lower, min(max_y, _even(bbox_y2 - fixed_height)))
        y = _clamp(y_lower, y_lower, y_upper)
        return x, y

    x = _clamp(_even(float(keyframe.get("x") or 0.0)), 0, max_x)
    y = _clamp(_even(float(keyframe.get("y") or 0.0)), 0, max_y)
    return x, y


def _command_line_for_dynamic_crop(
    *,
    timestamp_s: float,
    keyframes: list[dict[str, Any]],
    default_x: int,
    default_y: int,
) -> str:
    x = _even(
        _interpolated_keyframe_value(
            points=_keyframe_points_for_axis(keyframes=keyframes, axis="x", default_value=default_x),
            timestamp_s=timestamp_s,
            default_value=default_x,
        )
    )
    y = _even(
        _interpolated_keyframe_value(
            points=_keyframe_points_for_axis(keyframes=keyframes, axis="y", default_value=default_y),
            timestamp_s=timestamp_s,
            default_value=default_y,
        )
    )
    return f"{timestamp_s:.3f} crop@follow x {x}, crop@follow y {y};"


def _write_crop_sendcmd_file(
    *,
    keyframes: list[dict[str, Any]],
    command_path: Path,
    output_fps: int,
    source_dimensions: tuple[int, int],
    fixed_width: int,
    fixed_height: int,
    default_x: int,
    default_y: int,
    time_offset_ms: int = 0,
) -> None:
    frame_step_s = 1.0 / max(1, int(output_fps))
    ordered_keyframes = sorted((dict(keyframe) for keyframe in keyframes), key=lambda item: int(item["time_ms"]))
    if not ordered_keyframes:
        raise ValueError("dynamic Phase6 crop sendcmd generation requires keyframes")

    resolved_keyframes = []
    for keyframe in ordered_keyframes:
        resolved_x, resolved_y = _resolved_xy_for_keyframe(
            keyframe=keyframe,
            fixed_width=fixed_width,
            fixed_height=fixed_height,
            source_dimensions=source_dimensions,
        )
        local_time_ms = max(0, int(keyframe["time_ms"]) - int(time_offset_ms))
        resolved_keyframes.append(
            {
                **keyframe,
                "time_ms": local_time_ms,
                "x": resolved_x,
                "y": resolved_y,
                "w": fixed_width,
                "h": fixed_height,
            }
        )

    start_s = max(0.0, float(resolved_keyframes[0]["time_ms"]) / 1000.0)
    end_s = max(start_s, float(resolved_keyframes[-1]["time_ms"]) / 1000.0)
    timestamps = [start_s]
    frame_index = 1
    while True:
        timestamp_s = start_s + (frame_index * frame_step_s)
        if timestamp_s >= end_s:
            break
        timestamps.append(timestamp_s)
        frame_index += 1
    if end_s not in timestamps:
        timestamps.append(end_s)

    lines: list[str] = []
    previous_line: str | None = None
    for timestamp_s in timestamps:
        line = _command_line_for_dynamic_crop(
            timestamp_s=timestamp_s,
            keyframes=resolved_keyframes,
            default_x=default_x,
            default_y=default_y,
        )
        if line == previous_line and timestamp_s < end_s:
            continue
        previous_line = line
        lines.append(line)
    command_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_center_crop_filter(*, source_dimensions: tuple[int, int]) -> str:
    source_width, source_height = source_dimensions
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
    return f"crop={crop_width}:{crop_height}:{default_x}:{default_y},scale={_TARGET_WIDTH}:{_TARGET_HEIGHT}"


def _crop_filter_from_plan(
    *,
    crop_plan: dict[str, Any],
    source_dimensions: tuple[int, int],
) -> str:
    mode = str(crop_plan.get("mode") or "").strip()
    if crop_plan.get("keyframes"):
        raise ValueError("dynamic Phase6 crop plans must use stitched per-run rendering")
    if mode and not crop_plan.get("segments"):
        raise ValueError(f"unknown Phase6 crop plan mode {mode!r}")

    source_width, source_height = source_dimensions
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


def _subtitles_filter(*, ass_path: Path, fonts_dir: Path) -> str:
    escaped_ass = str(ass_path).replace("\\", "/").replace(":", "\\:")
    escaped_fonts = str(fonts_dir).replace("\\", "/").replace(":", "\\:")
    return f"subtitles={escaped_ass}:fontsdir={escaped_fonts}"


def _ffmpeg_filter(
    *,
    ass_path: Path,
    fonts_dir: Path,
    crop_plan: dict[str, Any] | None = None,
    source_dimensions: tuple[int, int] | None = None,
) -> str:
    if crop_plan and (crop_plan.get("segments") or crop_plan.get("keyframes")):
        if source_dimensions is None:
            raise ValueError("source dimensions are required for tracklet crop rendering")
        video_filter = _crop_filter_from_plan(
            crop_plan=crop_plan,
            source_dimensions=source_dimensions,
        )
    else:
        video_filter = (
            f"scale={_TARGET_WIDTH}:{_TARGET_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={_TARGET_WIDTH}:{_TARGET_HEIGHT}"
        )
    return f"{video_filter},{_subtitles_filter(ass_path=ass_path, fonts_dir=fonts_dir)}"


def _dynamic_crop_filter_for_piece(
    *,
    keyframes: list[dict[str, Any]],
    source_dimensions: tuple[int, int],
    crop_command_path: Path,
    output_fps: int,
    time_offset_ms: int,
) -> str:
    fixed_w, fixed_h = _fixed_crop_dimensions_for_keyframes(
        keyframes=keyframes,
        source_dimensions=source_dimensions,
    )
    first_x, first_y = _resolved_xy_for_keyframe(
        keyframe=keyframes[0],
        fixed_width=fixed_w,
        fixed_height=fixed_h,
        source_dimensions=source_dimensions,
    )
    _write_crop_sendcmd_file(
        keyframes=keyframes,
        command_path=crop_command_path,
        output_fps=output_fps,
        source_dimensions=source_dimensions,
        fixed_width=fixed_w,
        fixed_height=fixed_h,
        default_x=first_x,
        default_y=first_y,
        time_offset_ms=time_offset_ms,
    )
    return (
        f"sendcmd=f={crop_command_path},"
        f"crop@follow={fixed_w}:{fixed_h}:{first_x}:{first_y},"
        f"scale={_TARGET_WIDTH}:{_TARGET_HEIGHT}"
    )


def _timeline_pieces_for_dynamic_crop(
    *,
    crop_plan: dict[str, Any],
    clip_duration_ms: int,
) -> list[dict[str, Any]]:
    keyframes, runs = _validate_dynamic_crop_plan(crop_plan)
    by_run: dict[str, list[dict[str, Any]]] = {}
    for keyframe in keyframes:
        by_run.setdefault(str(keyframe["run_id"]), []).append(dict(keyframe))

    pieces: list[dict[str, Any]] = []
    cursor_ms = 0
    for run in sorted(runs, key=lambda item: int(item["start_ms"])):
        run_id = str(run["run_id"])
        run_start_ms = max(0, int(run["start_ms"]))
        run_end_ms = min(max(run_start_ms, int(run["end_ms"])), clip_duration_ms)
        if run_end_ms <= run_start_ms:
            continue
        if run_start_ms < cursor_ms:
            raise ValueError(f"dynamic Phase6 crop runs overlap or regress at {run_id!r}")
        if run_start_ms > cursor_ms:
            pieces.append(
                {
                    "kind": "gap",
                    "start_ms": cursor_ms,
                    "end_ms": run_start_ms,
                }
            )
        run_keyframes = sorted(by_run.get(run_id, []), key=lambda item: int(item["time_ms"]))
        if not run_keyframes:
            raise ValueError(f"dynamic Phase6 crop run {run_id!r} has no keyframes")
        pieces.append(
            {
                "kind": "run",
                "run_id": run_id,
                "start_ms": run_start_ms,
                "end_ms": run_end_ms,
                "keyframes": run_keyframes,
            }
        )
        cursor_ms = run_end_ms

    if cursor_ms < clip_duration_ms:
        pieces.append(
            {
                "kind": "gap",
                "start_ms": cursor_ms,
                "end_ms": clip_duration_ms,
            }
        )
    if not pieces:
        raise ValueError("dynamic Phase6 crop plan produced no renderable timeline pieces")
    return pieces


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


def _run_ffmpeg(cmd: list[str]) -> str:
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return " ".join(shlex.quote(part) for part in cmd)


def _render_dynamic_crop_clip(
    *,
    request: Phase6RenderRequest,
    compiled: dict[str, Any],
    clip_id: str,
    source_video_path: Path,
    ass_path: Path,
    fonts_dir: Path,
    tmp_dir: Path,
    source_dimensions: tuple[int, int],
) -> tuple[Path, list[str]]:
    crop_plan = dict(compiled.get("crop_plan") or {})
    clip_start_ms = int(compiled["clip_start_ms"])
    clip_end_ms = int(compiled["clip_end_ms"])
    clip_duration_ms = max(0, clip_end_ms - clip_start_ms)
    pieces = _timeline_pieces_for_dynamic_crop(
        crop_plan=crop_plan,
        clip_duration_ms=clip_duration_ms,
    )

    commands: list[str] = []
    piece_paths: list[Path] = []
    for piece_index, piece in enumerate(pieces):
        piece_start_ms = int(piece["start_ms"])
        piece_end_ms = int(piece["end_ms"])
        piece_output_path = tmp_dir / f"{clip_id}_piece_{piece_index:03d}.mp4"
        piece_filter: str
        if piece["kind"] == "run":
            crop_command_path = tmp_dir / f"{clip_id}_piece_{piece_index:03d}.cmd"
            piece_filter = _dynamic_crop_filter_for_piece(
                keyframes=[dict(keyframe) for keyframe in piece["keyframes"]],
                source_dimensions=source_dimensions,
                crop_command_path=crop_command_path,
                output_fps=request.output_fps,
                time_offset_ms=piece_start_ms,
            )
        else:
            piece_filter = _default_center_crop_filter(source_dimensions=source_dimensions)

        piece_cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{float(clip_start_ms + piece_start_ms) / 1000.0:.3f}",
            "-to",
            f"{float(clip_start_ms + piece_end_ms) / 1000.0:.3f}",
            "-i",
            str(source_video_path),
            "-vf",
            piece_filter,
            "-an",
            "-r",
            str(request.output_fps),
            "-c:v",
            "h264_nvenc",
            str(piece_output_path),
        ]
        commands.append(_run_ffmpeg(piece_cmd))
        piece_paths.append(piece_output_path)

    if not piece_paths:
        raise ValueError(f"dynamic Phase6 crop clip {clip_id!r} produced no rendered pieces")

    stitched_video_path = piece_paths[0]
    if len(piece_paths) > 1:
        concat_input_path = tmp_dir / f"{clip_id}_concat.txt"
        concat_input_path.write_text(
            "\n".join(f"file '{path.as_posix()}'" for path in piece_paths) + "\n",
            encoding="utf-8",
        )
        stitched_video_path = tmp_dir / f"{clip_id}_stitched.mp4"
        concat_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_input_path),
            "-c",
            "copy",
            str(stitched_video_path),
        ]
        commands.append(_run_ffmpeg(concat_cmd))

    clip_audio_path = tmp_dir / f"{clip_id}.m4a"
    audio_cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{float(clip_start_ms) / 1000.0:.3f}",
        "-to",
        f"{float(clip_end_ms) / 1000.0:.3f}",
        "-i",
        str(source_video_path),
        "-vn",
        "-c:a",
        "aac",
        str(clip_audio_path),
    ]
    commands.append(_run_ffmpeg(audio_cmd))

    output_path = tmp_dir / f"{clip_id}.mp4"
    subtitle_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(stitched_video_path),
        "-i",
        str(clip_audio_path),
        "-vf",
        _subtitles_filter(ass_path=ass_path, fonts_dir=fonts_dir),
        "-c:v",
        "h264_nvenc",
        "-c:a",
        "copy",
        "-shortest",
        str(output_path),
    ]
    commands.append(_run_ffmpeg(subtitle_cmd))
    return output_path, commands


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
        source_dimensions_cache: tuple[int, int] | None = None
        for clip in request.clips:
            clip_id = str(clip["clip_id"])
            compiled = clips_by_id.get(clip_id) or clip
            ass_path = storage_client.download_file(
                gcs_uri=_ass_uri(request, clip_id),
                local_path=tmp_dir / f"captions_{clip_id}.ass",
            )
            crop_plan = dict(compiled.get("crop_plan") or {})
            needs_source_dimensions = bool(crop_plan.get("segments") or crop_plan.get("keyframes"))
            if needs_source_dimensions and source_dimensions_cache is None:
                source_dimensions_cache = _probe_video_dimensions(source_video_path)

            if crop_plan.get("keyframes"):
                output_path, command_strings = _render_dynamic_crop_clip(
                    request=request,
                    compiled=compiled,
                    clip_id=clip_id,
                    source_video_path=source_video_path,
                    ass_path=ass_path,
                    fonts_dir=fonts_dir,
                    tmp_dir=tmp_dir,
                    source_dimensions=source_dimensions_cache or _probe_video_dimensions(source_video_path),
                )
            else:
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
                        source_dimensions=source_dimensions_cache if needs_source_dimensions else None,
                    ),
                    "-r",
                    str(request.output_fps),
                    "-c:v",
                    "h264_nvenc",
                    "-c:a",
                    "aac",
                    str(output_path),
                ]
                command_strings = [_run_ffmpeg(cmd)]
            video_gcs_uri = storage_client.upload_file(
                local_path=output_path,
                object_name=f"{request.output_prefix}/{clip_id}.mp4",
            )
            outputs.append(
                {
                    "clip_id": clip_id,
                    "video_gcs_uri": video_gcs_uri,
                    "caption_ass_gcs_uri": request.artifact_gcs_uris[f"captions_{clip_id}.ass"],
                    "ffmpeg_command": " && ".join(command_strings),
                }
            )

    return {
        "run_id": request.run_id,
        "outputs": outputs,
        "render_backend": "modal_l40s_ffmpeg_libass",
        "total_ms": (time.perf_counter() - started) * 1000.0,
    }


__all__ = ["Phase6RenderRequest", "run_phase6_render"]
