#!/usr/bin/env python3
"""
Transcript loading helpers for Crowd Clip.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from html import unescape
from http.cookiejar import MozillaCookieJar

import httpx
import requests
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

from pipeline.audience.crowd_utils import OUTPUTS_DIR, VIDEOS_DIR

_log = logging.getLogger("crowd_transcript")


def _get_cookies_path() -> str | None:
    """Return the path to a Netscape cookies.txt file, or None."""
    path = str(os.getenv("YOUTUBE_COOKIES_PATH", "") or "").strip()
    if path and Path(path).is_file():
        return path
    # Also check for a cookies.txt next to the repo root
    default = Path(__file__).resolve().parents[3] / "cookies.txt"
    if default.is_file():
        return str(default)
    return None


def _build_transcript_api() -> YouTubeTranscriptApi:
    """Build a YouTubeTranscriptApi instance with cookies if available."""
    cookies_path = _get_cookies_path()
    if not cookies_path:
        return YouTubeTranscriptApi()
    _log.info("Loading YouTube cookies from %s", cookies_path)
    jar = MozillaCookieJar(cookies_path)
    jar.load(ignore_discard=True, ignore_expires=True)
    session = requests.Session()
    session.cookies = jar  # type: ignore[assignment]
    return YouTubeTranscriptApi(http_client=session)


def _audio_matches_video_id(payload: dict, video_id: str) -> bool:
    for key in ("source_audio", "source_video", "uri", "video_gcs_uri"):
        value = str(payload.get(key, "") or "")
        if video_id and video_id in value:
            return True
    return False


def _synth_words_from_transcript(video_id: str) -> tuple[list[dict], dict]:
    transcript = _build_transcript_api().fetch(video_id, languages=["en"])
    words: list[dict] = []
    for snippet in transcript:
        text = str(getattr(snippet, "text", "") or "").strip()
        start_s = float(getattr(snippet, "start", 0.0) or 0.0)
        duration_s = float(getattr(snippet, "duration", 0.0) or 0.0)
        tokens = [tok for tok in text.split() if tok]
        if not tokens:
            continue
        start_ms = int(round(start_s * 1000.0))
        duration_ms = max(1, int(round(duration_s * 1000.0)))
        per_word = max(1, duration_ms // max(1, len(tokens)))
        cursor = start_ms
        for idx, token in enumerate(tokens):
            end_ms = start_ms + duration_ms if idx == len(tokens) - 1 else cursor + per_word
            words.append(
                {
                    "word": token,
                    "start_time_ms": cursor,
                    "end_time_ms": max(cursor + 1, end_ms),
                    "speaker_track_id": None,
                }
            )
            cursor = max(cursor + 1, end_ms)

    payload = {
        "schema_version": "crowd-clip-transcript-v1",
        "source_audio": f"https://www.youtube.com/watch?v={video_id}",
        "video_id": video_id,
        "words": words,
        "transcript_source": "youtube_transcript_api",
    }
    return words, payload


def _pick_caption_track(info: dict) -> dict | None:
    """Pick the best English caption track from yt-dlp extracted info."""
    tracks = info.get("subtitles") or {}
    auto_tracks = info.get("automatic_captions") or {}

    # Prefer manual English captions first
    for lang_key in ("en", "en-US", "en-GB"):
        if lang_key in tracks:
            for fmt in tracks[lang_key]:
                if fmt.get("ext") == "json3":
                    return fmt
            # Fall back to first available format
            if tracks[lang_key]:
                return tracks[lang_key][0]

    # Then try auto-generated English captions
    for lang_key in ("en", "en-US", "en-GB", "en-orig"):
        if lang_key in auto_tracks:
            for fmt in auto_tracks[lang_key]:
                if fmt.get("ext") == "json3":
                    return fmt
            if auto_tracks[lang_key]:
                return auto_tracks[lang_key][0]

    return None


def _words_from_caption_events(events: list[dict]) -> list[dict]:
    """Convert json3 caption events into word dicts."""
    words: list[dict] = []
    for event in events:
        start_ms = int(event.get("tStartMs", 0) or 0)
        duration_ms = max(1, int(event.get("dDurationMs", 0) or 0))
        segs = event.get("segs") or []
        for seg in segs:
            raw = str(seg.get("utf8", "") or "").strip()
            if not raw or raw == "\n":
                continue
            text = unescape(raw)
            offset = int(seg.get("tOffsetMs", 0) or 0)
            word_start = start_ms + offset
            words.append(
                {
                    "word": text,
                    "start_time_ms": word_start,
                    "end_time_ms": max(word_start + 1, word_start + duration_ms),
                    "speaker_track_id": None,
                }
            )
    return words


def _synth_words_from_ytdlp_captions(video_id: str) -> tuple[list[dict], dict]:
    """Full yt-dlp caption fallback: extract captions via yt-dlp."""
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts: dict = {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "json3",
    }
    cookies_path = _get_cookies_path()
    if cookies_path:
        ydl_opts["cookiefile"] = cookies_path
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)

    track = _pick_caption_track(info)
    if not track:
        raise RuntimeError(f"No English captions found via yt-dlp for {video_id}")

    track_url = track.get("url", "")
    if not track_url:
        raise RuntimeError(f"Caption track has no URL for {video_id}")

    # Load cookies for httpx request if available
    cookies_path = _get_cookies_path()
    httpx_cookies = None
    if cookies_path:
        jar = MozillaCookieJar(cookies_path)
        jar.load(ignore_discard=True, ignore_expires=True)
        httpx_cookies = {c.name: c.value for c in jar if c.value is not None}

    caption_data = None
    last_exc = None
    for attempt in range(4):
        if attempt > 0:
            wait = 2 ** attempt  # 2, 4, 8 seconds
            _log.info("Retry %d for captions of %s (waiting %ds)", attempt, video_id, wait)
            time.sleep(wait)
        try:
            resp = httpx.get(track_url, timeout=30.0, cookies=httpx_cookies)
            resp.raise_for_status()
            caption_data = resp.json()
            break
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            if exc.response.status_code != 429:
                raise
            _log.warning("429 rate-limited fetching captions for %s", video_id)
    if caption_data is None:
        raise RuntimeError(f"Failed to fetch captions for {video_id} after retries: {last_exc}") from last_exc
    events = caption_data.get("events") or []
    words = _words_from_caption_events(events)

    payload = {
        "schema_version": "crowd-clip-transcript-v1",
        "source_audio": youtube_url,
        "video_id": video_id,
        "words": words,
        "transcript_source": "yt_dlp_captions",
    }
    return words, payload


def load_transcript_words(video_id: str) -> tuple[list[dict], dict]:
    """
    Load a transcript from a matching local Phase 1 audio ledger when possible.
    Fall back to public YouTube transcript fetch otherwise.
    """
    override = str(os.getenv("CROWD_AUDIO_LEDGER_PATH", "") or "").strip()
    candidate_paths: list[Path] = []
    if override:
        candidate_paths.append(Path(override))
    candidate_paths.append(VIDEOS_DIR / video_id / "outputs" / "phase_1_audio.json")
    candidate_paths.extend(VIDEOS_DIR.glob("*/outputs/phase_1_audio.json"))
    candidate_paths.append(OUTPUTS_DIR / "phase_1_audio.json")

    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if path.name == "phase_1_audio.json" and path.parent.parent.name != video_id:
            if not _audio_matches_video_id(payload, video_id):
                continue
        words = payload.get("words", [])
        if isinstance(words, list) and words:
            payload.setdefault("transcript_source", str(path))
            return words, payload

    try:
        return _synth_words_from_transcript(video_id)
    except Exception:
        return _synth_words_from_ytdlp_captions(video_id)


def transcript_duration_ms(words: list[dict]) -> int:
    if not words:
        return 0
    return int(max(int(w.get("end_time_ms", 0) or 0) for w in words))


def transcript_text_in_window(words: list[dict], start_ms: int, end_ms: int) -> str:
    tokens = [
        str(word.get("word", "")).strip()
        for word in words
        if start_ms <= int(word.get("start_time_ms", 0) or 0) < end_ms
    ]
    return " ".join(tok for tok in tokens if tok).strip()
