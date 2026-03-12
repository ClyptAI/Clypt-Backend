#!/usr/bin/env python3
"""
Phase 5 (Auto-Curator): Full-Graph Sweep & Viral Clip Detection
================================================================
Mimics OpusClip by sweeping the entire semantic graph instead of relying
on vector search. Groups nodes into narrative chapters using edge topology,
scores each chapter via Gemini 3.1 Pro, and outputs a ranked array of
Remotion render payloads.

Pipeline:
  1. Full-Graph Sweep  → pull all SemanticClipNodes chronologically
  2. Narrative Chunking → group nodes into chapters via NarrativeEdge links
  3. AI Batch Evaluator → score each chapter with Gemini ClipScoringAgent
  4. Global Ranking     → filter to 85+ (failsafe: Top 3)
  5. Remotion Output    → remotion_payloads_array.json
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from google import genai
from google.genai.types import HttpOptions
from google.cloud import spanner
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
PROJECT_ID = "clypt-v2"
SPANNER_INSTANCE = "clypt-spanner-v2"
SPANNER_DATABASE = "clypt-graph-db-v2"
GEMINI_LOCATION = "global"
GEMINI_MODEL = "gemini-3.1-pro-preview"

MIN_SCORE = 85
FALLBACK_TOP_N = 3
SPEAKER_GAP_MERGE_MS = 1500

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "outputs" / "remotion_payloads_array.json"

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase_5_auto")
# Suppress Spanner SDK internal metrics export errors (missing instance_id / rate limit)
for _name in ("opentelemetry.sdk.metrics._internal.export", "opentelemetry.sdk.metrics"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)

# ──────────────────────────────────────────────
# Pydantic schema for ClipScoringAgent response
# ──────────────────────────────────────────────
CLIP_SCORING_SYSTEM_INSTRUCTION = """\
You are an elite Executive Producer and Viral Auto-Curator for platforms like TikTok, YouTube Shorts, and Reels.

Your task is to analyze the provided chronological JSON array of 'Semantic Nodes'. You must identify the single best, most engaging, and self-contained short-form clip within this chapter.

Evaluation Criteria:

The Hook: The clip must start with immediate high retention. Look for a node that begins with a controversial statement, a good concept, a high-energy vocal_delivery, or an arresting visual_description. Do not OVERcompensate for this and start somewhere that does not make sense, however.

The Payoff: The clip must have a logical flow leading to a clear punchline, profound statement, or emotional spike. You must explicitly look at the content_mechanisms (e.g., high scores in Humor, Subversion, Emotion, or Social Dynamics) to locate the climax.

Pacing & Length: The ideal viral clip is between 15 and 60 seconds long. Do not include meandering setup nodes or trailing filler nodes, unless necessary for context.

Scoring:
If the chapter contains a perfect viral moment, score it 85-100.
If the chapter is mildly interesting but lacks a strong hook/payoff, score it 50-84.
If the chapter is entirely boring or administrative, score it below 50.

Strict Constraints:
You must return a valid JSON object matching the requested schema.
recommended_start_ms MUST exactly match the start_time_ms of the first node you select.
recommended_end_ms MUST exactly match the end_time_ms of the last node you select.
Your justification must explicitly mention the content_mechanisms or vocal_delivery that made you choose these nodes."""


class ClipScore(BaseModel):
    final_score: float = Field(ge=0.0, le=100.0)
    justification: str
    recommended_start_ms: int
    recommended_end_ms: int
    included_node_ids: list[str]


# ──────────────────────────────────────────────
# Step 1: Full-Graph Sweep
# ──────────────────────────────────────────────
def fetch_all_nodes(database) -> list[dict]:
    """Pull every SemanticClipNode for the video, ordered chronologically."""
    sql = """
        SELECT
            node_id,
            transcript_text,
            vocal_delivery,
            start_time_ms,
            end_time_ms,
            spatial_tracking_uri,
            content_mechanisms,
            speakers
        FROM SemanticClipNode
        ORDER BY start_time_ms ASC
    """

    with database.snapshot() as snapshot:
        results = list(snapshot.execute_sql(sql))

    def _normalize_speakers(raw) -> list[str]:
        """Normalize Spanner JSON speakers into a plain list[str]."""
        if raw is None:
            return []

        parsed = raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                parsed = [raw] if raw else []
        elif hasattr(raw, "serialize"):
            # Spanner JSON columns can come back as JsonObject wrappers.
            try:
                parsed = json.loads(raw.serialize())
            except Exception:
                parsed = []

        if isinstance(parsed, (list, tuple, set)):
            return [str(s) for s in parsed if s is not None and str(s) != ""]

        return []

    nodes = []
    for row in results:
        cm_raw = row[6]
        if isinstance(cm_raw, str):
            content_mechanisms = json.loads(cm_raw)
        elif cm_raw is not None:
            content_mechanisms = cm_raw
        else:
            content_mechanisms = {}

        nodes.append({
            "node_id": row[0],
            "transcript_text": row[1] or "",
            "vocal_delivery": row[2] or "",
            "start_time_ms": row[3],
            "end_time_ms": row[4],
            "spatial_tracking_uri": row[5] or "",
            "content_mechanisms": content_mechanisms,
            "speakers": _normalize_speakers(row[7]),
        })

    return nodes


def fetch_all_edges(database) -> list[dict]:
    """Pull every NarrativeEdge, used for chapter boundary detection."""
    sql = """
        SELECT
            from_node_id,
            to_node_id,
            label,
            narrative_classification,
            confidence_score
        FROM NarrativeEdge
    """

    with database.snapshot() as snapshot:
        results = list(snapshot.execute_sql(sql))

    edges = []
    for row in results:
        edges.append({
            "from_node_id": row[0],
            "to_node_id": row[1],
            "label": row[2],
            "narrative_classification": row[3],
            "confidence_score": row[4],
        })

    return edges


# ──────────────────────────────────────────────
# Step 2: Narrative Chunking
# ──────────────────────────────────────────────
def build_narrative_chapters(
    nodes: list[dict],
    edges: list[dict],
) -> list[list[dict]]:
    """Group sequential nodes into 'Narrative Chapters' using edge connectivity.

    Strategy: Walk through nodes chronologically. Two consecutive nodes belong
    to the same chapter if there is at least one NarrativeEdge linking them
    (in either direction) or if they're within the same narrative cluster.
    When a node has NO edge connection to the previous node, a new chapter
    begins.
    """
    if not nodes:
        return []

    node_ids = [n["node_id"] for n in nodes]
    node_index = {nid: i for i, nid in enumerate(node_ids)}

    adjacency: dict[str, set[str]] = {nid: set() for nid in node_ids}
    for edge in edges:
        fid, tid = edge["from_node_id"], edge["to_node_id"]
        if fid in adjacency and tid in adjacency:
            adjacency[fid].add(tid)
            adjacency[tid].add(fid)

    chapters: list[list[dict]] = []
    current_chapter: list[dict] = [nodes[0]]

    for i in range(1, len(nodes)):
        prev_node = nodes[i - 1]
        curr_node = nodes[i]

        connected = curr_node["node_id"] in adjacency.get(prev_node["node_id"], set())

        if not connected:
            any_link = False
            for ch_node in current_chapter:
                if curr_node["node_id"] in adjacency.get(ch_node["node_id"], set()):
                    any_link = True
                    break
            connected = any_link

        if connected:
            current_chapter.append(curr_node)
        else:
            chapters.append(current_chapter)
            current_chapter = [curr_node]

    if current_chapter:
        chapters.append(current_chapter)

    return chapters


# ──────────────────────────────────────────────
# Step 3: AI Batch Evaluator
# ──────────────────────────────────────────────
def score_chapter(chapter: list[dict], chapter_idx: int, total: int) -> ClipScore | None:
    """Send one narrative chapter to Gemini for clip scoring."""
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
    os.environ["GOOGLE_CLOUD_LOCATION"] = GEMINI_LOCATION
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    serializable_chapter = []
    for n in chapter:
        serializable_chapter.append({
            "node_id": n["node_id"],
            "transcript_text": n["transcript_text"],
            "vocal_delivery": n["vocal_delivery"],
            "start_time_ms": n["start_time_ms"],
            "end_time_ms": n["end_time_ms"],
            "content_mechanisms": n["content_mechanisms"],
        })

    prompt = (
        f"=== Chapter {chapter_idx + 1}/{total} — Semantic Nodes ===\n"
        f"{json.dumps(serializable_chapter, indent=2)}\n\n"
        "---\n\n"
        "Identify the single best viral clip within the chapter above. "
        "Output your scoring as a JSON object with final_score, justification, "
        "recommended_start_ms, recommended_end_ms, and included_node_ids."
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt],
            config=genai.types.GenerateContentConfig(
                system_instruction=CLIP_SCORING_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=ClipScore,
                temperature=0.2,
                thinking_config=genai.types.ThinkingConfig(
                    thinking_level="MEDIUM",
                ),
            ),
        )
        raw = json.loads(response.text)
        return ClipScore.model_validate(raw)
    except Exception as e:
        log.error(f"  Chapter {chapter_idx + 1} scoring failed: {e}")
        return None


# ──────────────────────────────────────────────
# Step 4: Global Ranking & Filtering
# ──────────────────────────────────────────────
def rank_and_filter(
    scored_clips: list[tuple[ClipScore, list[dict]]],
) -> list[tuple[ClipScore, list[dict]]]:
    """Sort by score descending, apply 85+ threshold with Top-3 failsafe."""
    scored_clips.sort(key=lambda x: x[0].final_score, reverse=True)

    elite = [(cs, ch) for cs, ch in scored_clips if cs.final_score >= MIN_SCORE]

    if len(elite) >= FALLBACK_TOP_N:
        return elite

    log.warning(
        f"Only {len(elite)} clips scored {MIN_SCORE}+. "
        f"Falling back to Top {FALLBACK_TOP_N}."
    )
    return scored_clips[:FALLBACK_TOP_N]


# ──────────────────────────────────────────────
# Step 5: Remotion Output
# ──────────────────────────────────────────────
def build_payloads(
    finalized: list[tuple[ClipScore, list[dict]]],
) -> list[dict]:
    """Construct Remotion-compatible payloads for each finalized clip."""
    payloads = []
    for clip_score, chapter in finalized:
        included_set = set(clip_score.included_node_ids)
        included_nodes = [n for n in chapter if n["node_id"] in included_set]
        included_nodes.sort(key=lambda n: n["start_time_ms"])

        combined_transcript = " ".join(
            n.get("transcript_text", "") for n in included_nodes
        ).strip()

        tracking_uris = [
            n["spatial_tracking_uri"]
            for n in included_nodes
            if n.get("spatial_tracking_uri")
        ]

        # Build active speaker timeline from node speaker data.
        # Merge short gaps for the same speaker to avoid lock/drop jitter.
        raw_speaker_timeline = []
        for n in included_nodes:
            speakers = n.get("speakers", [])
            primary_speaker = speakers[0] if speakers else None
            if primary_speaker:
                raw_speaker_timeline.append({
                    "start_ms": n["start_time_ms"],
                    "end_ms": n["end_time_ms"],
                    "speaker_tag": str(primary_speaker),
                })

        active_speaker_timeline = []
        for seg in raw_speaker_timeline:
            if not active_speaker_timeline:
                active_speaker_timeline.append(seg)
                continue
            prev = active_speaker_timeline[-1]
            same_speaker = prev["speaker_tag"] == seg["speaker_tag"]
            small_gap = seg["start_ms"] <= prev["end_ms"] + SPEAKER_GAP_MERGE_MS
            if same_speaker and small_gap:
                prev["end_ms"] = max(prev["end_ms"], seg["end_ms"])
            else:
                active_speaker_timeline.append(seg)

        payloads.append({
            "clip_start_ms": clip_score.recommended_start_ms,
            "clip_end_ms": clip_score.recommended_end_ms,
            "final_score": clip_score.final_score,
            "justification": clip_score.justification,
            "combined_transcript": combined_transcript,
            "tracking_uris": tracking_uris,
            "included_node_ids": clip_score.included_node_ids,
            "active_speaker_timeline": active_speaker_timeline,
        })

    return payloads


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("PHASE 4 — Auto-Curator (Full-Graph Sweep)")
    log.info("=" * 60)

    # ── Step 1: Full-Graph Sweep ──
    log.info(f"Connecting to Spanner: {SPANNER_INSTANCE}/{SPANNER_DATABASE}")
    spanner_client = spanner.Client(project=PROJECT_ID)
    instance = spanner_client.instance(SPANNER_INSTANCE)
    database = instance.database(SPANNER_DATABASE)

    log.info("Fetching all SemanticClipNodes…")
    nodes = fetch_all_nodes(database)
    log.info(f"  Nodes: {len(nodes)}")

    log.info("Fetching all NarrativeEdges…")
    edges = fetch_all_edges(database)
    log.info(f"  Edges: {len(edges)}")

    # ── Step 2: Narrative Chunking ──
    log.info("─" * 50)
    log.info("Building narrative chapters…")
    chapters = build_narrative_chapters(nodes, edges)
    log.info(f"  Chapters: {len(chapters)}")
    for i, ch in enumerate(chapters):
        t0 = ch[0]["start_time_ms"]
        t1 = ch[-1]["end_time_ms"]
        log.info(
            f"    Chapter {i + 1}: {len(ch)} nodes "
            f"({t0}ms – {t1}ms, {(t1 - t0) / 1000:.1f}s)"
        )

    # ── Step 3: AI Batch Evaluator ──
    log.info("─" * 50)
    log.info("Scoring chapters with Gemini ClipScoringAgent…")

    scored_clips: list[tuple[ClipScore, list[dict]]] = []
    for i, chapter in enumerate(chapters):
        log.info(f"  Scoring chapter {i + 1}/{len(chapters)}…")
        clip_score = score_chapter(chapter, i, len(chapters))
        if clip_score:
            log.info(f"    Score: {clip_score.final_score}/100")
            log.info(f"    Range: {clip_score.recommended_start_ms}ms – {clip_score.recommended_end_ms}ms")
            duration_s = (clip_score.recommended_end_ms - clip_score.recommended_start_ms) / 1000
            log.info(f"    Duration: {duration_s:.1f}s")
            scored_clips.append((clip_score, chapter))
        else:
            log.warning(f"    Chapter {i + 1} returned no score (skipped)")

    if not scored_clips:
        log.error("No chapters could be scored. Exiting.")
        return

    # ── Step 4: Global Ranking & Filtering ──
    log.info("─" * 50)
    log.info("Ranking and filtering clips…")
    finalized = rank_and_filter(scored_clips)
    log.info(f"  Finalized clips: {len(finalized)}")
    for rank, (cs, _) in enumerate(finalized, 1):
        log.info(
            f"    #{rank}: {cs.final_score}/100 "
            f"({cs.recommended_start_ms}ms – {cs.recommended_end_ms}ms)"
        )

    # ── Step 5: Remotion Output ──
    log.info("─" * 50)
    log.info("Constructing Remotion payloads…")
    payloads = build_payloads(finalized)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payloads, f, indent=2)
    log.info(f"Remotion payloads saved → {OUTPUT_PATH}")

    # ── Summary ──
    log.info("=" * 60)
    log.info("PHASE 4 AUTO-CURATOR COMPLETE")
    log.info(f"  Total nodes swept:    {len(nodes)}")
    log.info(f"  Narrative chapters:   {len(chapters)}")
    log.info(f"  Chapters scored:      {len(scored_clips)}")
    log.info(f"  Clips exported:       {len(payloads)}")
    for rank, p in enumerate(payloads, 1):
        duration_s = (p["clip_end_ms"] - p["clip_start_ms"]) / 1000
        log.info(
            f"    Clip #{rank}: {p['final_score']}/100 "
            f"({p['clip_start_ms']}ms – {p['clip_end_ms']}ms, "
            f"{duration_s:.1f}s)"
        )
    log.info(f"  Output: {OUTPUT_PATH}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
