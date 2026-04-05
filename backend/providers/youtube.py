from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class YouTubeDownloader:
    def __init__(
        self,
        *,
        format_selector: str | None = None,
        cookies_file: str | Path | None = None,
    ) -> None:
        # Prefer H.264 (avc1) — AV1/VP9 often fail with OpenCV on headless servers.
        self.format_selector = (
            format_selector
            or (
                "bestvideo[height<=1080][vcodec^=avc1]+bestaudio[ext=m4a]"
                "/bestvideo[height<=1080][vcodec^=avc]+bestaudio"
                "/bestvideo[height<=1080]+bestaudio"
                "/best[height<=1080]/best"
            )
        )
        # Resolve cookies file: constructor arg > env var
        _cookies = cookies_file or os.environ.get("CLYPT_YOUTUBE_COOKIES_FILE")
        self.cookies_file: Path | None = Path(_cookies) if _cookies else None

    def download(self, *, source_url: str, output_path: Path) -> Path:
        try:
            import yt_dlp
        except ImportError as exc:
            raise RuntimeError("yt-dlp is required for V3.1 source download.") from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        options = {
            "format": self.format_selector,
            "outtmpl": str(output_path),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
        }
        if self.cookies_file is not None:
            if not self.cookies_file.exists():
                raise FileNotFoundError(
                    f"YouTube cookies file not found: {self.cookies_file}\n"
                    "Set CLYPT_YOUTUBE_COOKIES_FILE to the path of a valid Netscape cookies.txt."
                )
            options["cookiefile"] = str(self.cookies_file)
            logger.info("[yt-dlp]  using cookies file: %s", self.cookies_file)
        else:
            logger.info("[yt-dlp]  no cookies file — may fail on bot-detection")

        # JS runtime for EJS n-challenge solver.
        # Env var format: "node:/usr/bin/node" or "deno" or "bun"
        # Maps to yt-dlp's js_runtimes dict: {"node": {"path": "/usr/bin/node"}}
        _js_runtime = os.environ.get("CLYPT_YOUTUBE_JS_RUNTIMES")
        if _js_runtime:
            runtime, _, path = _js_runtime.partition(":")
            options["js_runtimes"] = {runtime.lower(): {"path": path or None}}
            logger.info("[yt-dlp]  JS runtime: %s (path=%s)", runtime, path or "auto")

        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([source_url])
        return output_path


__all__ = ["YouTubeDownloader"]
