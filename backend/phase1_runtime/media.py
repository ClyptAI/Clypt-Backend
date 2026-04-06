from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

logger = logging.getLogger(__name__)


def _default_audio_extractor(*, video_path: Path, audio_path: Path) -> None:
    logger.info("[media]  extracting 16kHz mono audio from %s ...", video_path.name)
    t0 = time.perf_counter()
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
    logger.info("[media]  audio ready: %s (%.1f s)", audio_path.name, time.perf_counter() - t0)


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
        src = Path(source_path).resolve()
        dst = Path(workspace.video_path).resolve()
        if src == dst:
            logger.info("[media]  source_path is already the workspace video — skipping copy")
        else:
            logger.info("[media]  copying local file: %s", source_path)
            shutil.copy2(src, dst)
            logger.info("[media]  copied → %s", workspace.video_path)
    else:
        logger.info("[media]  downloading: %s", source_url)
        t0 = time.perf_counter()
        downloader.download(source_url=source_url, output_path=workspace.video_path)
        size_mb = workspace.video_path.stat().st_size / 1_048_576
        logger.info(
            "[media]  download done — %.1f MB in %.1f s → %s",
            size_mb,
            time.perf_counter() - t0,
            workspace.video_path,
        )

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
