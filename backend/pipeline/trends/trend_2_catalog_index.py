#!/usr/bin/env python3
"""
Trend Trim Stage 2: build a searchable index over the local Clypt video catalog.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from pipeline.audience.crowd_utils import extract_video_id
from pipeline.trends.trend_utils import (
    OUTPUTS_DIR,
    VIDEOS_DIR,
    combined_terms,
    fetch_youtube_video_metadata,
    keyword_terms,
    mean_vector,
    phrase_terms,
    youtube_api_key,
    utc_now_iso,
)

OUTPUT_PATH = OUTPUTS_DIR / "trend_2_catalog_index.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s \u2013 %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("trend_2")


def _load_json(path: Path, *, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def _transcript_text(audio_payload: dict) -> str:
    words = audio_payload.get("words", []) if isinstance(audio_payload, dict) else []
    return " ".join(str(word.get("word", "") or "").strip() for word in words if str(word.get("word", "") or "").strip())


def _clip_node_indices(clip: dict, embeddings: list[dict]) -> list[int]:
    start_ms = int(clip.get("clip_start_ms", 0) or 0)
    end_ms = int(clip.get("clip_end_ms", start_ms) or start_ms)
    indices: list[int] = []
    for idx, node in enumerate(embeddings):
        start_s = float(node.get("start_time", 0.0) or 0.0)
        end_s = float(node.get("end_time", start_s) or start_s)
        node_start = int(round(start_s * 1000.0))
        node_end = int(round(end_s * 1000.0))
        overlap = max(0, min(end_ms, node_end) - max(start_ms, node_start))
        if overlap > 0:
            indices.append(idx)
    return indices


def _clip_embedding(node_indices: list[int], embeddings: list[dict]) -> list[float] | None:
    vectors = []
    for idx in node_indices:
        if idx < 0 or idx >= len(embeddings):
            continue
        values = embeddings[idx].get("multimodal_embedding")
        if isinstance(values, list) and values:
            vectors.append([float(value) for value in values])
    return mean_vector(vectors)


def _video_record(video_dir: Path, metadata: dict[str, dict]) -> dict | None:
    outputs_dir = video_dir / "outputs"
    audio_path = outputs_dir / "phase_1_audio.json"
    nodes_path = outputs_dir / "phase_2a_nodes.json"
    embeddings_path = outputs_dir / "phase_3_embeddings.json"
    payloads_path = outputs_dir / "remotion_payloads_array.json"

    audio_payload = _load_json(audio_path, default={})
    nodes_payload = _load_json(nodes_path, default=[])
    embeddings_payload = _load_json(embeddings_path, default=[])
    payloads = _load_json(payloads_path, default=[])

    if not audio_payload or not isinstance(payloads, list) or not payloads:
        return None

    source_url = str(audio_payload.get("source_audio", "") or "").strip()
    video_id = ""
    try:
        video_id = extract_video_id(source_url)
    except Exception:
        video_id = ""

    meta = metadata.get(video_id, {}) if video_id else {}
    snippet = meta.get("snippet", {}) if isinstance(meta, dict) else {}
    statistics = meta.get("statistics", {}) if isinstance(meta, dict) else {}
    transcript_text = _transcript_text(audio_payload)
    folder_name = video_dir.name.replace("_", " ").replace("-", " ")
    video_title = str(snippet.get("title", "") or "").strip() or folder_name.title()
    channel_title = str(snippet.get("channelTitle", "") or "").strip()

    video_keywords = combined_terms(
        video_title,
        transcript_text,
        " ".join(str(item.get("justification", "") or "") for item in payloads),
        limit=20,
    )
    watchlist_terms = []
    watchlist_terms.extend(keyword_terms(video_title, limit=8, min_len=3))
    watchlist_terms.extend(phrase_terms(video_title, limit=4))
    watchlist_terms.extend(video_keywords[:6])

    nodes = []
    for idx, node in enumerate(nodes_payload if isinstance(nodes_payload, list) else []):
        transcript_segment = str(node.get("transcript_segment", "") or "").strip()
        start_ms = int(round(float(node.get("start_time", 0.0) or 0.0) * 1000.0))
        end_ms = int(round(float(node.get("end_time", 0.0) or 0.0) * 1000.0))
        nodes.append(
            {
                "index": idx,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "transcript_segment": transcript_segment,
                "keyword_terms": combined_terms(transcript_segment, str(node.get("vocal_delivery", "") or ""), limit=10),
            }
        )


    clip_records = []
    for idx, clip in enumerate(payloads):
        node_indices = _clip_node_indices(clip, embeddings_payload if isinstance(embeddings_payload, list) else [])
        clip_text = " | ".join(
            part for part in [
                str(clip.get("combined_transcript", "") or "").strip(),
                str(clip.get("justification", "") or "").strip(),
            ] if part
        )
        clip_records.append(
            {
                "clip_index": idx,
                "clip_start_ms": int(clip.get("clip_start_ms", 0) or 0),
                "clip_end_ms": int(clip.get("clip_end_ms", 0) or 0),
                "duration_ms": max(0, int(clip.get("clip_end_ms", 0) or 0) - int(clip.get("clip_start_ms", 0) or 0)),
                "base_final_score": float(clip.get("final_score", 0.0) or 0.0),
                "combined_transcript": str(clip.get("combined_transcript", "") or "").strip(),
                "justification": str(clip.get("justification", "") or "").strip(),
                "keyword_terms": combined_terms(clip_text, video_title, limit=16),
                "node_indices": node_indices,
                "payload": clip,
                "embedding": _clip_embedding(node_indices, embeddings_payload if isinstance(embeddings_payload, list) else []),
            }
        )

    return {
        "catalog_id": video_dir.name,
        "video_id": video_id,
        "source_url": source_url,
        "video_gcs_uri": str(audio_payload.get("video_gcs_uri", "") or audio_payload.get("uri", "") or "").strip(),
        "video_title": video_title,
        "channel_title": channel_title,
        "video_keywords": video_keywords,
        "watchlist_terms": list(dict.fromkeys(term for term in watchlist_terms if term))[:16],
        "statistics": {
            "view_count": int(statistics.get("viewCount", 0) or 0),
            "like_count": int(statistics.get("likeCount", 0) or 0),
            "comment_count": int(statistics.get("commentCount", 0) or 0),
        },
        "phase_paths": {
            "phase_1_audio": str(audio_path),
            "phase_2a_nodes": str(nodes_path),
            "phase_3_embeddings": str(embeddings_path),
            "remotion_payloads": str(payloads_path),
        },
        "transcript_word_count": len(transcript_text.split()),
        "nodes": nodes,
        "clips": clip_records,
    }


def main() -> dict:
    log.info("=" * 60)
    log.info("TREND TRIM STAGE 2 \u2013 Catalog Index")
    log.info("=" * 60)

    video_dirs = [path for path in sorted(VIDEOS_DIR.iterdir()) if path.is_dir()]
    source_urls = []
    temp_records = []
    for video_dir in video_dirs:
        audio_payload = _load_json(video_dir / "outputs" / "phase_1_audio.json", default={})
        source_url = str(audio_payload.get("source_audio", "") or "").strip()
        if source_url:
            temp_records.append((video_dir, source_url))
            source_urls.append(source_url)

    metadata = {}
    api_key = youtube_api_key()
    if api_key:
        video_ids = []
        for _, source_url in temp_records:
            try:
                video_ids.append(extract_video_id(source_url))
            except Exception:
                continue
        try:
            metadata = fetch_youtube_video_metadata(video_ids, api_key=api_key)
        except Exception as exc:
            log.warning("Catalog YouTube metadata fetch failed: %s", exc)

    videos = []
    for video_dir in video_dirs:
        record = _video_record(video_dir, metadata)
        if record:
            videos.append(record)

    watchlist_terms = []
    for video in videos:
        watchlist_terms.extend(video.get("watchlist_terms", []))
    deduped_watchlist = list(dict.fromkeys(term for term in watchlist_terms if term))[:40]

    payload = {
        "generated_at": utc_now_iso(),
        "video_count": len(videos),
        "watchlist_terms": deduped_watchlist,
        "videos": videos,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    log.info("Catalog videos indexed: %d", len(videos))
    for video in videos:
        log.info("  %s \u2013 clips=%d nodes=%d", video["catalog_id"], len(video["clips"]), len(video["nodes"]))
    log.info("Output saved \u2192 %s", OUTPUT_PATH)
    return payload


if __name__ == "__main__":
    main()