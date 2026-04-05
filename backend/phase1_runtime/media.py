from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess


def _default_audio_extractor(*, video_path: Path, audio_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@dataclass(frozen=True, slots=True)
class PreparedWorkspaceMedia:
    video_path: Path
    audio_path: Path


def prepare_workspace_media(
    *,
    source_url: str | None,
    source_path: str | None,
    workspace,
    downloader,
    audio_extractor=None,
) -> PreparedWorkspaceMedia:
    if bool(source_url) == bool(source_path):
        raise ValueError("Provide exactly one of source_url or source_path")

    if source_path is not None:
        shutil.copy2(source_path, workspace.video_path)
    else:
        downloader.download(source_url=source_url, output_path=workspace.video_path)

    extractor = audio_extractor or _default_audio_extractor
    extractor(video_path=workspace.video_path, audio_path=workspace.audio_path)
    return PreparedWorkspaceMedia(
        video_path=workspace.video_path,
        audio_path=workspace.audio_path,
    )


__all__ = [
    "PreparedWorkspaceMedia",
    "prepare_workspace_media",
]
