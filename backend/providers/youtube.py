from __future__ import annotations

from pathlib import Path


class YouTubeDownloader:
    def __init__(self, *, format_selector: str | None = None) -> None:
        self.format_selector = (
            format_selector
            or "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best"
        )

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
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([source_url])
        return output_path


__all__ = ["YouTubeDownloader"]
