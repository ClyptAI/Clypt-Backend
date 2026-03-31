#!/usr/bin/env python3
"""
Crowd Clip Stage 3: cluster resolved audience signals into clip candidates.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys
from difflib import SequenceMatcher

from google import genai

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from pipeline.audience.crowd_transcript import load_transcript_words, transcript_duration_ms, transcript_text_in_window
from pipeline.audience.crowd_types import CrowdClipCandidate, ResolvedCrowdComment
from pipeline.audience.crowd_utils import OUTPUTS_DIR, VIDEOS_DIR, clamp, extract_video_id, log_weight, normalize_text, overlap_ratio

INPUT_PATH = OUTPUTS_DIR / "crowd_2_resolved_signals.json"
OUTPUT_PATH = OUTPUTS_DIR / "crowd_3_clip_candidates.json"
PROJECT_ID = "clypt-v3"
EMBEDDING_LOCATION = "us-central1"
EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIM = 768

STRONG_CLUSTER_GAP_MS = 8_000
WEAK_ATTACH_GAP_MS = 7_500
PRE_PADDING_MS = 4_000
POST_PADDING_MS = 7_000
MIN_CLIP_MS = 12_000
MAX_CLIP_MS = 45_000
TOP_N = 10
MAX_FULL_WEIGHT_KEYWORD_REFS = 8
MAX_KEYWORD_REFS_TOTAL = 16
MIN_FALLBACK_KEYWORD_CONFIDENCE = 0.63
BOUNDARY_SNAP_BACK_MS = 2_800
BOUNDARY_SNAP_FORWARD_MS = 3_600
PAUSE_BOUNDARY_MS = 650
NODE_MATCH_THRESHOLD = 0.44
MIN_ATTACH_PROXIMITY_SCORE = 0.28
COMFORT_LEAD_IN_MS = 1_600
COMFORT_TAIL_OUT_MS = 2_000
COMFORT_MAX_EXPAND_MS = 3_200
COHERENCE_STOPWORDS = {
    "that", "this", "with", "have", "from", "your", "they", "them", "just", "really", "there", "would",
    "could", "should", "about", "because", "which", "video", "thing", "stuff", "comment", "comments",
    "mrbeast", "jimmy", "karl", "alex",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s \u2014 %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("crowd_3")
_EMBED_CLIENT = None
_EMBEDDING_DISABLED = False
_EMBED_CACHE: dict[str, list[float] | None] = {}


def _reference_weight(kind: str) -> float:
    return {
        "explicit_timestamp": 4.2,
        "quote_match": 2.6,
        "keyword_overlap": 0.45,
    }.get(kind, 0.4)


def _is_strong_reference(ref) -> bool:
    return ref.kind in {"explicit_timestamp", "quote_match"}


def _align_window(words: list[dict], start_ms: int, end_ms: int, duration_ms: int) -> tuple[int, int]:
    start_ms = max(0, start_ms)
    end_ms = min(duration_ms or end_ms, end_ms)
    if end_ms - start_ms < MIN_CLIP_MS:
        end_ms = min(duration_ms or (start_ms + MIN_CLIP_MS), start_ms + MIN_CLIP_MS)
    if end_ms - start_ms > MAX_CLIP_MS:
        end_ms = start_ms + MAX_CLIP_MS

    nearest_start = start_ms
    nearest_end = end_ms
    for word in words:
        w_start = int(word.get("start_time_ms", 0) or 0)
        w_end = int(word.get("end_time_ms", w_start) or w_start)
        if w_start <= start_ms:
            nearest_start = w_start
        if w_end <= end_ms:
            nearest_end = w_end
        else:
            break

    if nearest_end - nearest_start < MIN_CLIP_MS:
        nearest_end = min(duration_ms or (nearest_start + MIN_CLIP_MS), nearest_start + MIN_CLIP_MS)
    return nearest_start, nearest_end


def _build_transcript_boundaries(words: list[dict], duration_ms: int) -> list[int]:
    boundaries = {0, max(0, duration_ms)}
    previous_word: dict | None = None
    for word in words:
        start_ms = int(word.get("start_time_ms", 0) or 0)
        end_ms = int(word.get("end_time_ms", start_ms) or start_ms)
        token = str(word.get("word", "") or "").strip()
        if previous_word is not None:
            prev_end = int(previous_word.get("end_time_ms", 0) or 0)
            prev_speaker = previous_word.get("speaker_tag") or previous_word.get("speaker_track_id")
            current_speaker = word.get("speaker_tag") or word.get("speaker_track_id")
            if start_ms - prev_end >= PAUSE_BOUNDARY_MS:
                boundaries.add(prev_end)
                boundaries.add(start_ms)
            if prev_speaker and current_speaker and prev_speaker != current_speaker:
                boundaries.add(prev_end)
                boundaries.add(start_ms)
        if token.endswith((".", "!", "?")):
            boundaries.add(end_ms)
        previous_word = word
    return sorted(boundaries)



def _snap_window_to_boundaries(
    start_ms: int,
    end_ms: int,
    boundaries: list[int],
    duration_ms: int,
) -> tuple[int, int, bool]:
    snapped = False
    candidate_starts = [value for value in boundaries if value <= start_ms and start_ms - value <= BOUNDARY_SNAP_BACK_MS]
    candidate_ends = [value for value in boundaries if value >= end_ms and value - end_ms <= BOUNDARY_SNAP_FORWARD_MS]

    if candidate_starts:
        start_ms = max(candidate_starts)
        snapped = True
    if candidate_ends:
        end_ms = min(candidate_ends)
        snapped = True

    start_ms = max(0, start_ms)
    end_ms = min(duration_ms or end_ms, end_ms)
    if end_ms - start_ms < MIN_CLIP_MS:
        target_end = min(duration_ms or (start_ms + MIN_CLIP_MS), start_ms + MIN_CLIP_MS)
        extension_candidates = [value for value in boundaries if value >= target_end]
        end_ms = extension_candidates[0] if extension_candidates else target_end
    if end_ms - start_ms > MAX_CLIP_MS:
        target_end = start_ms + MAX_CLIP_MS
        trim_candidates = [value for value in boundaries if start_ms < value <= target_end]
        end_ms = trim_candidates[-1] if trim_candidates else target_end
    return start_ms, end_ms, snapped


def _candidate_node_paths() -> list[Path]:
    paths: list[Path] = []
    override_value = str(os.getenv("CROWD_NODES_PATH", "") or "").strip()
    override = Path(override_value) if override_value else None
    if override:
        paths.append(override)
    paths.extend(VIDEOS_DIR.glob("*/outputs/phase_2a_nodes.json"))
    paths.append(OUTPUTS_DIR / "phase_2a_nodes.json")
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def _node_match_score(nodes: list[dict], words: list[dict]) -> float:
    scored: list[float] = []
    for node in nodes[:6]:
        if "start_ms" in node and "end_ms" in node:
            start_ms = int(node.get("start_ms", 0) or 0)
            end_ms = int(node.get("end_ms", 0) or 0)
        else:
            start_ms = int(round(float(node.get("start_time", 0.0) or 0.0) * 1000.0))
            end_ms = int(round(float(node.get("end_time", 0.0) or 0.0) * 1000.0))
        excerpt = transcript_text_in_window(words, start_ms, end_ms)
        transcript = str(node.get("transcript_segment", "") or "")
        if not excerpt or not transcript:
            continue
        scored.append(SequenceMatcher(None, normalize_text(excerpt), normalize_text(transcript)).ratio())
    if not scored:
        return 0.0
    return sum(scored) / len(scored)


def _load_matching_nodes(words: list[dict], duration_ms: int) -> tuple[list[dict], float, str]:
    best_nodes: list[dict] = []
    best_score = 0.0
    best_path = ""
    for path in _candidate_node_paths():
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if not isinstance(payload, list) or not payload:
            continue
        plausible = []
        for idx, node in enumerate(payload):
            try:
                start_ms = int(round(float(node.get("start_time", 0.0) or 0.0) * 1000.0))
                end_ms = int(round(float(node.get("end_time", 0.0) or 0.0) * 1000.0))
            except Exception:
                continue
            if end_ms <= start_ms or start_ms > duration_ms + 5_000:
                continue
            plausible.append(
                {
                    "index": idx,
                    "start_ms": start_ms,
                    "end_ms": min(duration_ms or end_ms, end_ms),
                    "transcript_segment": str(node.get("transcript_segment", "") or ""),
                }
            )
        if not plausible:
            continue
        score = _node_match_score(plausible, words)
        if score > best_score:
            best_nodes = plausible
            best_score = score
            best_path = str(path)
    if best_score < NODE_MATCH_THRESHOLD:
        return [], best_score, ""
    return best_nodes, best_score, best_path


def _find_supporting_nodes(cluster: list[dict], nodes: list[dict], clip_start: int, clip_end: int) -> tuple[list[dict], float]:
    if not nodes:
        return [], 0.0
    ref_anchors = [item["ref"].anchor_ms for item in cluster]
    supporting: list[dict] = []
    covered = 0
    for node in nodes:
        hit_count = sum(1 for anchor in ref_anchors if node["start_ms"] <= anchor <= node["end_ms"])
        overlap = overlap_ratio(clip_start, clip_end, node["start_ms"], node["end_ms"])
        if hit_count > 0 or overlap >= 0.34:
            supporting.append({**node, "hit_count": hit_count, "overlap": overlap})
            covered += hit_count
    alignment_score = covered / max(1, len(ref_anchors))
    return supporting, clamp(alignment_score, 0.0, 1.0)


def _apply_node_alignment(
    clip_start: int,
    clip_end: int,
    supporting_nodes: list[dict],
    boundaries: list[int],
    duration_ms: int,
) -> tuple[int, int, list[int], bool]:
    if not supporting_nodes:
        return clip_start, clip_end, [], False
    node_start = min(node["start_ms"] for node in supporting_nodes)
    node_end = max(node["end_ms"] for node in supporting_nodes)
    if node_end - node_start > MAX_CLIP_MS + 6_000:
        return clip_start, clip_end, [], False
    clip_start = min(clip_start, node_start)
    clip_end = max(clip_end, node_end)
    clip_start, clip_end, snapped = _snap_window_to_boundaries(clip_start, clip_end, boundaries, duration_ms)
    aligned_indices = [int(node["index"]) for node in supporting_nodes]
    return clip_start, clip_end, aligned_indices, snapped


def _expand_to_comfort_boundary(
    clip_start: int,
    clip_end: int,
    first_anchor: int,
    last_anchor: int,
    boundaries: list[int],
    duration_ms: int,
) -> tuple[int, int, bool]:
    changed = False

    lead_gap = max(0, first_anchor - clip_start)
    if lead_gap < COMFORT_LEAD_IN_MS:
        target_start = max(0, first_anchor - COMFORT_LEAD_IN_MS)
        start_candidates = [
            value for value in boundaries
            if value <= target_start and target_start - value <= COMFORT_MAX_EXPAND_MS
        ]
        if start_candidates:
            new_start = max(start_candidates)
            if new_start < clip_start:
                clip_start = new_start
                changed = True

    tail_gap = max(0, clip_end - last_anchor)
    if tail_gap < COMFORT_TAIL_OUT_MS:
        target_end = min(duration_ms or (last_anchor + COMFORT_TAIL_OUT_MS), last_anchor + COMFORT_TAIL_OUT_MS)
        end_candidates = [
            value for value in boundaries
            if value >= target_end and value - target_end <= COMFORT_MAX_EXPAND_MS
        ]
        if end_candidates:
            new_end = min(end_candidates)
            if new_end > clip_end:
                clip_end = new_end
                changed = True

    clip_start = max(0, clip_start)
    clip_end = min(duration_ms or clip_end, clip_end)
    if clip_end - clip_start > MAX_CLIP_MS:
        clip_end = clip_start + MAX_CLIP_MS
    return clip_start, clip_end, changed


def _build_speaker_timeline(words: list[dict], start_ms: int, end_ms: int) -> list[dict]:
    timeline: list[dict] = []
    current: dict | None = None
    for word in words:
        w_start = int(word.get("start_time_ms", 0) or 0)
        w_end = int(word.get("end_time_ms", w_start) or w_start)
        if not (w_start < end_ms and w_end > start_ms):
            continue
        speaker = word.get("speaker_tag") or word.get("speaker_track_id")
        if not speaker:
            continue
        tag = str(speaker)
        if current and current["speaker_tag"] == tag and w_start <= current["end_ms"] + 1200:
            current["end_ms"] = max(current["end_ms"], w_end)
            continue
        current = {"start_ms": w_start, "end_ms": w_end, "speaker_tag": tag}
        timeline.append(current)
    return timeline


def _nms(candidates: list[CrowdClipCandidate]) -> list[CrowdClipCandidate]:
    kept: list[CrowdClipCandidate] = []
    for candidate in sorted(candidates, key=lambda c: c.raw_score, reverse=True):
        overlaps_existing = False
        for existing in kept:
            overlap = overlap_ratio(candidate.clip_start_ms, candidate.clip_end_ms, existing.clip_start_ms, existing.clip_end_ms)
            if overlap >= 0.55:
                overlaps_existing = True
                break
        if not overlaps_existing:
            kept.append(candidate)
    return kept


def _comment_group_key(comment: ResolvedCrowdComment) -> str:
    normalized = normalize_text(comment.text)
    tokens = [tok for tok in normalized.split() if tok]
    if not tokens:
        return comment.comment_id
    tokens = tokens[:12]
    return " ".join(tokens)


def _collapse_similar_comments(cluster: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in cluster:
        grouped[_comment_group_key(item["comment"])].append(item)

    collapsed: list[dict] = []
    for items in grouped.values():
        items.sort(
            key=lambda item: (
                -item["comment"].like_count,
                -item["comment"].reply_count,
                -item["ref"].confidence,
            )
        )
        collapsed.append(items[0])
    return collapsed


def _author_key(comment: ResolvedCrowdComment) -> str:
    normalized = normalize_text(comment.author_name)
    return normalized or comment.comment_id


def _make_embed_client():
    return genai.Client(vertexai=True, project=PROJECT_ID, location=EMBEDDING_LOCATION)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = sum(a * a for a in left) ** 0.5
    norm_right = sum(b * b for b in right) ** 0.5
    if norm_left <= 1e-9 or norm_right <= 1e-9:
        return 0.0
    return dot / (norm_left * norm_right)


def _comment_semantic_text(item: dict) -> str:
    comment = item["comment"]
    ref = item["ref"]
    matched = str(ref.matched_text or "").strip()
    text = str(comment.text or "").strip()
    if matched:
        return f"{ref.kind}: {matched} | {text[:220]}"
    return text[:240]


def _embed_text(text: str) -> list[float] | None:
    global _EMBED_CLIENT, _EMBEDDING_DISABLED
    normalized = normalize_text(text)
    if not normalized:
        return None
    if normalized in _EMBED_CACHE:
        return _EMBED_CACHE[normalized]
    if _EMBEDDING_DISABLED:
        _EMBED_CACHE[normalized] = None
        return None
    try:
        if _EMBED_CLIENT is None:
            _EMBED_CLIENT = _make_embed_client()
        response = _EMBED_CLIENT.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[normalized],
            config=genai.types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
        )
        embeddings = getattr(response, "embeddings", None) or []
        values = getattr(embeddings[0], "values", None) if embeddings else None
        result = list(values) if values else None
        _EMBED_CACHE[normalized] = result
        return result
    except Exception:
        _EMBEDDING_DISABLED = True
        _EMBED_CACHE[normalized] = None
        return None


def _signature_tokens(item: dict) -> tuple[str, ...]:
    ref = item["ref"]
    comment = item["comment"]
    matched = [
        token for token in normalize_text(ref.matched_text).split()
        if len(token) >= 4 and token not in COHERENCE_STOPWORDS and not token.isdigit()
    ]
    if ref.kind != "explicit_timestamp" and len(matched) >= 2:
        return tuple(matched[:4])

    fallback = [
        token for token in normalize_text(comment.text).split()
        if len(token) >= 4 and token not in COHERENCE_STOPWORDS and not token.isdigit()
    ]
    if matched:
        combined = []
        seen = set()
        for token in [*matched, *fallback]:
            if token not in seen:
                seen.add(token)
                combined.append(token)
        return tuple(combined[:4])
    return tuple(fallback[:4])


def _pairwise_signature_overlap(signatures: list[tuple[str, ...]]) -> float:
    if len(signatures) <= 1:
        return 1.0
    total = 0.0
    count = 0
    for idx in range(len(signatures)):
        left = set(signatures[idx])
        if not left:
            continue
        for jdx in range(idx + 1, len(signatures)):
            right = set(signatures[jdx])
            if not right:
                continue
            total += len(left & right) / max(1, len(left | right))
            count += 1
    if count == 0:
        return 0.0
    return total / count


def _cluster_coherence(cluster: list[dict]) -> tuple[float, list[str]]:
    if len(cluster) <= 2:
        signatures = [sig for sig in (_signature_tokens(item) for item in cluster) if sig]
        top_terms = list(signatures[0][:3]) if signatures else []
        return 1.0, top_terms

    signatures = [sig for sig in (_signature_tokens(item) for item in cluster) if sig]
    if not signatures:
        return 0.72, []

    phrase_counts: Counter[str] = Counter(" ".join(sig[:3]) for sig in signatures if sig)
    token_counts: Counter[str] = Counter()
    for sig in signatures:
        for token in set(sig):
            token_counts[token] += 1

    dominant_phrase_share = max(phrase_counts.values()) / max(1, len(signatures))
    top_terms = [token for token, _ in token_counts.most_common(3)]
    top_term_share = (
        sum(token_counts[token] for token in top_terms)
        / max(1, len(signatures) * max(1, len(top_terms)))
    )
    pairwise_overlap = _pairwise_signature_overlap(signatures)

    coherence = clamp(
        (0.4 * dominant_phrase_share) + (0.3 * top_term_share) + (0.3 * pairwise_overlap),
        0.0,
        1.0,
    )
    return coherence, top_terms


def _semantic_cluster_coherence(cluster: list[dict]) -> tuple[float | None, list[str]]:
    texts: list[str] = []
    keywords: Counter[str] = Counter()
    seen: set[str] = set()
    for item in cluster:
        text = _comment_semantic_text(item)
        key = normalize_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        texts.append(text)
        for token in _signature_tokens(item):
            keywords[token] += 1
        if len(texts) >= 8:
            break
    if len(texts) <= 1:
        return None, [token for token, _ in keywords.most_common(3)]

    vectors = [_embed_text(text) for text in texts]
    usable = [vector for vector in vectors if vector]
    if len(usable) <= 1:
        return None, [token for token, _ in keywords.most_common(3)]

    total = 0.0
    count = 0
    for idx in range(len(usable)):
        for jdx in range(idx + 1, len(usable)):
            total += _cosine_similarity(usable[idx], usable[jdx])
            count += 1
    if count == 0:
        return None, [token for token, _ in keywords.most_common(3)]

    similarity = total / count
    semantic = clamp((similarity + 1.0) / 2.0, 0.0, 1.0)
    return semantic, [token for token, _ in keywords.most_common(3)]


def _cluster_strong_refs(flattened: list[dict]) -> list[list[dict]]:
    strong_refs = [item for item in flattened if _is_strong_reference(item["ref"])]
    if not strong_refs:
        return []

    strong_refs.sort(key=lambda item: item["ref"].anchor_ms)
    clusters: list[list[dict]] = []
    current: list[dict] = []
    for item in strong_refs:
        if not current:
            current = [item]
            continue
        if item["ref"].anchor_ms - current[-1]["ref"].anchor_ms <= STRONG_CLUSTER_GAP_MS:
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
    if current:
        clusters.append(current)
    return clusters


def _fallback_keyword_clusters(flattened: list[dict]) -> list[list[dict]]:
    fallback_refs = [
        item for item in flattened
        if item["ref"].kind == "keyword_overlap" and item["ref"].confidence >= MIN_FALLBACK_KEYWORD_CONFIDENCE
    ]
    if not fallback_refs:
        return []
    fallback_refs.sort(key=lambda item: item["ref"].anchor_ms)
    clusters: list[list[dict]] = []
    current: list[dict] = []
    for item in fallback_refs:
        if not current:
            current = [item]
            continue
        if item["ref"].anchor_ms - current[-1]["ref"].anchor_ms <= WEAK_ATTACH_GAP_MS:
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
    if current:
        clusters.append(current)
    return clusters[:3]


def _attach_keyword_refs(clusters: list[list[dict]], flattened: list[dict]) -> list[list[dict]]:
    weak_refs = [item for item in flattened if item["ref"].kind == "keyword_overlap"]
    if not clusters:
        return clusters

    cluster_bounds = []
    for cluster in clusters:
        anchors = [item["ref"].anchor_ms for item in cluster]
        cluster_bounds.append([min(anchors), max(anchors)])

    for item in weak_refs:
        anchor = item["ref"].anchor_ms
        best_idx = None
        best_score = None
        for idx, (start_anchor, end_anchor) in enumerate(cluster_bounds):
            if start_anchor - WEAK_ATTACH_GAP_MS <= anchor <= end_anchor + WEAK_ATTACH_GAP_MS:
                dist = 0 if start_anchor <= anchor <= end_anchor else min(abs(anchor - start_anchor), abs(anchor - end_anchor))
                proximity = max(0.0, 1.0 - (dist / max(1, WEAK_ATTACH_GAP_MS)))
                confidence = float(item["ref"].confidence)
                score = (0.72 * proximity) + (0.28 * confidence)
                if score < MIN_ATTACH_PROXIMITY_SCORE:
                    continue
                if best_score is None or score > best_score:
                    best_idx = idx
                    best_score = score
        if best_idx is not None:
            clusters[best_idx].append(item)
    return clusters


def main(youtube_url: str | None = None) -> dict:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing Crowd Clip resolved artifact: {INPUT_PATH}")

    payload = json.loads(INPUT_PATH.read_text(encoding="utf-8-sig"))
    source_url = youtube_url or str(payload.get("source_url", "") or "")
    video_id = str(payload.get("video_id", "") or "")
    if not video_id:
        video_id = extract_video_id(source_url)

    words, _ = load_transcript_words(video_id)
    duration_ms = transcript_duration_ms(words)
    boundaries = _build_transcript_boundaries(words, duration_ms)
    nodes, node_match_score, node_source = _load_matching_nodes(words, duration_ms)
    comments = [ResolvedCrowdComment.model_validate(comment) for comment in payload.get("comments", [])]

    flattened: list[dict] = []
    for comment in comments:
        for ref in comment.references:
            flattened.append({"comment": comment, "ref": ref})
    flattened.sort(key=lambda item: item["ref"].anchor_ms)
    clusters = _cluster_strong_refs(flattened)
    if clusters:
        clusters = _attach_keyword_refs(clusters, flattened)
    else:
        clusters = _fallback_keyword_clusters(flattened)

    candidates: list[CrowdClipCandidate] = []
    for cluster in clusters:
        cluster = _collapse_similar_comments(cluster)
        strong_items = [item for item in cluster if _is_strong_reference(item["ref"])]
        weak_items = [item for item in cluster if item["ref"].kind == "keyword_overlap"]
        weak_items.sort(key=lambda item: item["ref"].confidence, reverse=True)
        weak_items = weak_items[:MAX_KEYWORD_REFS_TOTAL]
        cluster = [*strong_items, *weak_items]

        refs = [item["ref"] for item in cluster]
        comments_in_cluster = [item["comment"] for item in cluster]
        unique_comment_ids = sorted({comment.comment_id for comment in comments_in_cluster})
        unique_authors = sorted({comment.author_name for comment in comments_in_cluster if comment.author_name})

        anchor_refs = [item["ref"] for item in strong_items] or refs
        anchor_start = min(ref.anchor_ms for ref in anchor_refs)
        anchor_end = max(ref.anchor_ms for ref in anchor_refs)
        clip_start, clip_end = _align_window(words, anchor_start - PRE_PADDING_MS, anchor_end + POST_PADDING_MS, duration_ms)
        supporting_nodes, node_alignment_score = _find_supporting_nodes(cluster, nodes, clip_start, clip_end)
        aligned_node_indices: list[int] = []
        node_snapped = False
        if supporting_nodes:
            clip_start, clip_end, aligned_node_indices, node_snapped = _apply_node_alignment(
                clip_start,
                clip_end,
                supporting_nodes,
                boundaries,
                duration_ms,
            )
        clip_start, clip_end, comfort_adjusted = _expand_to_comfort_boundary(
            clip_start,
            clip_end,
            anchor_start,
            anchor_end,
            boundaries,
            duration_ms,
        )
        clip_start, clip_end, boundary_snapped = _snap_window_to_boundaries(clip_start, clip_end, boundaries, duration_ms)
        clip_start, clip_end = _align_window(words, clip_start, clip_end, duration_ms)

        explicit = sum(1 for ref in refs if ref.kind == "explicit_timestamp")
        quote = sum(1 for ref in refs if ref.kind == "quote_match")
        keyword = sum(1 for ref in refs if ref.kind == "keyword_overlap")
        like_sum = sum(comment.like_count for comment in comments_in_cluster)
        reply_sum = sum(comment.reply_count for comment in comments_in_cluster)
        author_counts = Counter(_author_key(comment) for comment in comments_in_cluster)
        duplicate_group_counts = Counter(_comment_group_key(comment) for comment in comments_in_cluster)
        excitement_avg = (
            sum(comment.excitement_score for comment in comments_in_cluster) / max(1, len(comments_in_cluster))
        )
        lexical_coherence, lexical_terms = _cluster_coherence(cluster)
        semantic_coherence, semantic_terms = _semantic_cluster_coherence(cluster)
        if semantic_coherence is None:
            cluster_coherence = lexical_coherence
            coherence_terms = lexical_terms
            coherence_mode = "lexical"
        else:
            cluster_coherence = clamp((0.45 * lexical_coherence) + (0.55 * semantic_coherence), 0.0, 1.0)
            coherence_terms = semantic_terms or lexical_terms
            coherence_mode = "blended_semantic"

        raw_score = 0.0
        keyword_counted = 0
        for item in strong_items:
            comment = item["comment"]
            ref = item["ref"]
            raw_score += (
                (_reference_weight(ref.kind) * ref.confidence)
                + (0.16 * log_weight(comment.like_count))
                + (0.09 * comment.excitement_score)
                + (0.05 * min(8, comment.reply_count))
            )
        for item in weak_items:
            comment = item["comment"]
            ref = item["ref"]
            keyword_counted += 1
            if keyword_counted > MAX_FULL_WEIGHT_KEYWORD_REFS:
                diminishing = 0.35
            else:
                diminishing = 1.0
            raw_score += diminishing * (
                (_reference_weight(ref.kind) * ref.confidence)
                + (0.05 * log_weight(comment.like_count))
                + (0.03 * comment.excitement_score)
                + (0.02 * min(6, comment.reply_count))
            )

        duration_s = max(1.0, (clip_end - clip_start) / 1000.0)
        strong_count = len(strong_items)
        strong_ratio = strong_count / max(1, len(cluster))
        signal_density = (strong_count + 0.35 * min(len(weak_items), MAX_FULL_WEIGHT_KEYWORD_REFS)) / duration_s
        width_penalty = max(0.65, 1.0 - max(0.0, ((clip_end - clip_start) - 22000) / 50000.0))
        weak_penalty = max(0.55, 1.0 - max(0.0, (len(weak_items) - max(2, strong_count * 2)) * 0.035))
        strong_ratio_bonus = 0.6 + (0.8 * strong_ratio)
        top_author_share = (max(author_counts.values()) / max(1, len(comments_in_cluster))) if author_counts else 1.0
        duplicate_group_share = (
            max(duplicate_group_counts.values()) / max(1, len(comments_in_cluster))
            if duplicate_group_counts else 1.0
        )
        reply_ratio = sum(1 for comment in comments_in_cluster if comment.is_reply) / max(1, len(comments_in_cluster))
        author_diversity_ratio = len(unique_authors) / max(1, len(unique_comment_ids))
        consensus_multiplier = clamp(
            0.72
            + (0.34 * author_diversity_ratio)
            + (0.22 * (1.0 - top_author_share))
            + (0.18 * (1.0 - duplicate_group_share))
            + (0.16 * (1.0 - reply_ratio)),
            0.72,
            1.3,
        )
        node_alignment_multiplier = 1.0 + (0.1 * node_alignment_score)
        coherence_multiplier = clamp(0.68 + (0.62 * cluster_coherence), 0.68, 1.22)

        raw_score += (0.95 * len(unique_comment_ids))
        raw_score += (0.75 * len(unique_authors))
        raw_score += (2.5 * explicit) + (1.4 * quote)
        raw_score *= width_penalty * weak_penalty * strong_ratio_bonus
        raw_score *= clamp(0.8 + (signal_density * 0.9), 0.8, 1.55)
        raw_score *= consensus_multiplier * node_alignment_multiplier * coherence_multiplier
        if cluster_coherence < 0.42 and keyword >= max(3, strong_count):
            raw_score *= 0.82

        sample_comments = [
            comment.text.strip()
            for comment in sorted(
                {comment.comment_id: comment for comment in comments_in_cluster}.values(),
                key=lambda c: (-c.like_count, -len(c.references), c.comment_id),
            )[:3]
            if comment.text.strip()
        ]

        transcript_excerpt = transcript_text_in_window(words, clip_start, clip_end)
        justification = (
            f"Audience cluster with {len(unique_comment_ids)} comments from {len(unique_authors)} viewers, "
            f"{explicit} explicit timestamps, {quote} quote matches, {keyword} semantic matches, "
            f"{like_sum} total likes, and {reply_sum} total replies."
        )

        signal_breakdown = {
            "avg_excitement": round(excitement_avg, 3),
            "speaker_segments_detected": len(_build_speaker_timeline(words, clip_start, clip_end)),
            "strong_reference_count": strong_count,
            "strong_reference_ratio": round(strong_ratio, 3),
            "signal_density_per_sec": round(signal_density, 3),
            "width_penalty": round(width_penalty, 3),
            "weak_penalty": round(weak_penalty, 3),
            "consensus_multiplier": round(consensus_multiplier, 3),
            "coherence_multiplier": round(coherence_multiplier, 3),
            "cluster_coherence": round(cluster_coherence, 3),
            "lexical_coherence": round(lexical_coherence, 3),
            "coherence_mode": coherence_mode,
            "coherence_terms": ", ".join(coherence_terms),
            "author_diversity_ratio": round(author_diversity_ratio, 3),
            "top_author_share": round(top_author_share, 3),
            "duplicate_group_share": round(duplicate_group_share, 3),
            "reply_ratio": round(reply_ratio, 3),
            "boundary_snapped": int(boundary_snapped or node_snapped),
            "comfort_adjusted": int(comfort_adjusted),
            "aligned_node_count": len(aligned_node_indices),
            "node_alignment_score": round(node_alignment_score, 3),
            "node_match_score": round(node_match_score, 3),
            "node_source": node_source or "none",
        }
        if semantic_coherence is not None:
            signal_breakdown["semantic_coherence"] = round(semantic_coherence, 3)

        candidates.append(
            CrowdClipCandidate(
                rank=1,
                clip_start_ms=clip_start,
                clip_end_ms=clip_end,
                raw_score=round(raw_score, 4),
                final_score=0.0,
                anchor_start_ms=anchor_start,
                anchor_end_ms=anchor_end,
                explicit_timestamp_count=explicit,
                quote_match_count=quote,
                keyword_overlap_count=keyword,
                total_reference_count=len(refs),
                unique_comment_count=len(unique_comment_ids),
                unique_author_count=len(unique_authors),
                total_like_count=like_sum,
                total_reply_count=reply_sum,
                transcript_excerpt=transcript_excerpt,
                justification=justification,
                evidence_comment_ids=unique_comment_ids,
                sample_comments=sample_comments,
                aligned_node_indices=aligned_node_indices,
                signal_breakdown=signal_breakdown,
            )
        )

    candidates = _nms(candidates)
    candidates.sort(key=lambda c: c.raw_score, reverse=True)
    candidates = candidates[:TOP_N]

    if candidates:
        max_raw = max(candidate.raw_score for candidate in candidates)
        min_raw = min(candidate.raw_score for candidate in candidates)
        for idx, candidate in enumerate(candidates, start=1):
            candidate.rank = idx
            if max_raw == min_raw:
                candidate.final_score = 85.0
            else:
                normalized = (candidate.raw_score - min_raw) / max(1e-6, (max_raw - min_raw))
                candidate.final_score = round(clamp(72.0 + (normalized * 26.0), 72.0, 98.0), 1)

    output = {
        "schema_version": "crowd-clip-candidates-v1",
        "source_url": source_url,
        "video_id": video_id,
        "candidate_count": len(candidates),
        "candidates": [candidate.model_dump() for candidate in candidates],
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    log.info("=" * 60)
    log.info("CROWD CLIP \u2014 STAGE 3 CROWD SCORING")
    log.info("=" * 60)
    log.info("Clusters scored: %d", len(candidates))
    log.info("Output saved \u2192 %s", OUTPUT_PATH)
    return output


if __name__ == "__main__":
    main()