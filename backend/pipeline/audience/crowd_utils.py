#!/usr/bin/env python3
"""
Shared helpers for the Crowd Clip pipeline.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"
VIDEOS_DIR = ROOT / "videos"

_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_SPACE_RE = re.compile(r"\s+")


def extract_video_id(url_or_id: str) -> str:
    """Accept a YouTube URL or raw ID and return the canonical video ID."""
    value = (url_or_id or "").strip()
    if not value:
        raise ValueError("Missing YouTube URL or video ID")

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value

    parsed = urlparse(value)
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""

    if host in {"youtu.be", "www.youtu.be"}:
        candidate = path.strip("/").split("/")[0]
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
            return candidate

    if "youtube.com" in host:
        query = parse_qs(parsed.query)
        candidate = (query.get("v") or [""])[0]
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
            return candidate
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live"}:
            candidate = parts[1]
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
                return candidate

    raise ValueError(f"Could not extract YouTube video ID from: {url_or_id}")


def normalize_text(text: str) -> str:
    lowered = (text or "").lower().replace("\u2019", "'").replace("`", "'")
    lowered = _NON_ALNUM_RE.sub(" ", lowered)
    lowered = _SPACE_RE.sub(" ", lowered).strip()
    return lowered


def tokenize(text: str) -> list[str]:
    return [tok for tok in normalize_text(text).split(" ") if tok]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def overlap_ratio(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    inter = max(0, min(end_a, end_b) - max(start_a, start_b))
    if inter <= 0:
        return 0.0
    union = max(end_a, end_b) - min(start_a, start_b)
    return inter / max(1, union)


def log_weight(value: int | float, scale: float = 1.0) -> float:
    return math.log1p(max(0.0, float(value))) * scale
