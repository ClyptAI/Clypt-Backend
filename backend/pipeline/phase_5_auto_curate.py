#!/usr/bin/env python3
"""
Phase 5 (Auto-Curator): Full-Graph Sweep & Viral Clip Detection
================================================================
Mimics OpusClip by sweeping the entire semantic graph instead of relying
on vector search. Groups nodes into narrative chapters using edge topology,
scores each chapter via Gemini 3.1 Pro, and outputs a ranked array of
render payloads (clip boundaries for FFmpeg speaker-follow rendering).

Pipeline:
  1. Full-Graph Sweep  → pull all SemanticClipNodes chronologically
  2. Narrative Chunking → group nodes into chapters via NarrativeEdge links
  3. AI Batch Evaluator → score each chapter with Gemini ClipScoringAgent
  4. Global Ranking     → filter to 85+ (failsafe: Top 3)
  5. Clip Output        → remotion_payloads_array.json
"""

from __future__ import annotations

import asyncio
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

MIN_SCORE = 75
FALLBACK_TOP_N = 3
SPEAKER_GAP_MERGE_MS = 1500
MAX_CLIPS_OUTPUT = 10
NMS_OVERLAP_THRESHOLD = 0.5

# Sliding window config
WINDOW_MIN_NODES = 3
WINDOW_MAX_NODES = 10
WINDOW_STEP = 2

# Pre-filter: only send top N windows by mechanism score to Gemini
PREFILTER_TOP_N = 80

# Async scoring: max concurrent Gemini calls
MAX_CONCURRENT_SCORING = 5

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
You are a brutally honest Executive Producer for TikTok, YouTube Shorts, and Reels. You have seen thousands of viral clips and you know most content is mediocre.

Your task: analyze the provided chronological JSON array of Semantic Nodes and identify the single best clip within this window.

Scoring — use the FULL range, be ruthless:
- 95-100: Genuinely viral. Someone would stop scrolling, watch twice, and share it. Strong hook + clear payoff + no dead weight. Maybe 1-2 clips per video deserve this.
- 85-94: Very good. Clear hook and payoff but missing something — slightly slow, slightly too long, or slightly unclear without context.
- 70-84: Decent. Has a moment but the setup or payoff is weak. Would not stop a scroll but not skip-worthy either.
- 50-69: Below average. Something interesting is buried in filler. Not clippable without editing.
- Below 50: Not a clip. Boring, transitional, or completely context-dependent.

The vast majority of windows should score 50-80. Only score 85+ if you would personally post this clip.

Evaluation:
1. Hook: Does the first node demand attention? A controversial statement, high-energy delivery, or surprising visual. If the clip starts with filler or slow setup, penalize heavily.
2. Payoff: Does it build to a punchline, revelation, or emotional spike? Check content_mechanisms for humor/subversion/emotion intensity above 0.7.
3. Self-contained: Would someone understand and enjoy this with zero context? If not, penalize.
4. Pacing: 15-45 seconds is ideal. Drop filler nodes from start and end — tighter is better.

Strict Constraints:
recommended_start_ms MUST exactly match the start_time_ms of the first node you select.
recommended_end_ms must land on the end of the last complete sentence spoken within your selected nodes. If the last node ends mid-sentence or mid-phrase, pull recommended_end_ms back to the timestamp of the last word of the previous complete sentence. Never end the clip mid-phrase.
Your justification must state the specific content_mechanisms scores and vocal_delivery that drove your decision."""


class ClipScore(BaseModel):
    final_score: float = Field(ge=0.0, le=100.0)
    justification: str
    recommended_start_ms: int
    recommended_end_ms: int


# ──────────────────────────────────────────────
# Step 1: Full-graph sweep from Spanner
# ──────────────────────────────────────────────
def _fetch_all_nodes() -> list[dict]:
    """Pull every SemanticClipNode ordered by start_time_ms."""
    client = spanner.Client(project=PROJECT_ID)
    instance = client.instance(SPANNER_INSTANCE)
    database = instance.database(SPANNER_DATABASE)

    sql = """
        SELECT node_id, video_uri, start_time_ms, end_time_ms,
               transcript_text, visual_description, vocal_delivery,
               speakers, objects_present, visual_labels,
               content_mechanisms, spatial_tracking_uri
        FROM SemanticClipNode
        ORDER BY start_time_ms ASC
    """
    nodes = []
    with database.snapshot() as snapshot:
        results = snapshot.execute_sql(sql)
        for row in results:
            nodes.append({
                "node_id": row[0],
                "video_uri": row[1],
                "start_time_ms": row[2],
                "end_time_ms": row[3],
                "transcript_text": row[4],
                "visual_description": row[5],
                "vocal_delivery": row[6],
                "speakers": row[7],
                "objects_present": row[8],
                "visual_labels": row[9],
                "content_mechanisms": row[10],
                "spatial_tracking_uri": row[11],
            })
    return nodes


# ──────────────────────────────────────────────
# Step 2: Sliding window generation
# ──────────────────────────────────────────────
def _mechanism_pre_score(node: dict) -> float:
    """Quick heuristic from content_mechanisms — used for pre-filtering."""
    try:
        mechs = node.get("content_mechanisms")
        if isinstance(mechs, str):
            mechs = json.loads(mechs)
        if not isinstance(mechs, dict):
            return 0.0
    except Exception:
        return 0.0

    score = 0.0
    for dim in ("humor", "emotion", "social", "expertise"):
        dim_data = mechs.get(dim, {})
        if dim_data.get("present"):
            intensity = float(dim_data.get("intensity", 0.0))
            score += intensity
    return score


def _generate_windows(nodes: list[dict]) -> list[list[dict]]:
    """Generate overlapping windows of varying sizes and pre-filter."""
    windows: list[tuple[float, list[dict]]] = []

    for size in range(WINDOW_MIN_NODES, min(WINDOW_MAX_NODES + 1, len(nodes) + 1)):
        for start in range(0, len(nodes) - size + 1, WINDOW_STEP):
            window = nodes[start : start + size]
            pre_score = sum(_mechanism_pre_score(n) for n in window) / len(window)
            windows.append((pre_score, window))

    windows.sort(key=lambda x: -x[0])
    top_windows = [w for _, w in windows[:PREFILTER_TOP_N]]
    log.info(
        f"  Generated {len(windows)} total windows, pre-filtered to top {len(top_windows)}"
    )
    return top_windows


# ──────────────────────────────────────────────
# Step 3: Async batch scoring with Gemini
# ──────────────────────────────────────────────
async def _score_window(
    client, window: list[dict], semaphore: asyncio.Semaphore, idx: int
) -> dict | None:
    """Score a single window using Gemini."""
    slim_nodes = []
    for n in window:
        mechs = n.get("content_mechanisms", "{}")
        if isinstance(mechs, str):
            try:
                mechs = json.loads(mechs)
            except Exception:
                mechs = {}
        slim_nodes.append({
            "node_id": n["node_id"],
            "start_time_ms": n["start_time_ms"],
            "end_time_ms": n["end_time_ms"],
            "transcript_text": n.get("transcript_text", ""),
            "vocal_delivery": n.get("vocal_delivery", ""),
            "content_mechanisms": mechs,
        })

    user_prompt = (
        "=== CANDIDATE WINDOW ===\n"
        f"{json.dumps(slim_nodes, indent=2)}\n\n"
        "---\n\n"
        "Score this window. If it's not worth clipping, give it a low score."
    )


    async with semaphore:
        try:
            import os
            os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
            os.environ["GOOGLE_CLOUD_LOCATION"] = GEMINI_LOCATION
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

            response = await asyncio.to_thread(
                client.models.generate_content,
                model=GEMINI_MODEL,
                contents=[user_prompt],
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

            clip = ClipScore.model_validate_json(response.text)
            log.info(f"  Window {idx}: score={clip.final_score:.0f}")
            return {
                "final_score": clip.final_score,
                "justification": clip.justification,
                "clip_start_ms": clip.recommended_start_ms,
                "clip_end_ms": clip.recommended_end_ms,
                "video_uri": window[0].get("video_uri", ""),
                "spatial_tracking_uris": [
                    n.get("spatial_tracking_uri", "") for n in window
                    if n.get("spatial_tracking_uri")
                ],
                "node_ids": [n["node_id"] for n in window],
            }

        except Exception as e:
            log.warning(f"  Window {idx} failed: {e}")
            return None


async def _batch_score_windows(windows: list[list[dict]]) -> list[dict]:
    """Score all windows concurrently with bounded parallelism."""
    import os
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
    os.environ["GOOGLE_CLOUD_LOCATION"] = GEMINI_LOCATION
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

    client = genai.Client(http_options=HttpOptions(api_version="v1"))
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCORING)

    tasks = [
        _score_window(client, w, semaphore, i) for i, w in enumerate(windows)
    ]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


# ──────────────────────────────────────────────
# Step 4: NMS + global ranking
# ──────────────────────────────────────────────
def _nms(clips: list[dict]) -> list[dict]:
    """Non-maximum suppression: remove overlapping clips, keep highest score."""
    clips.sort(key=lambda c: c["final_score"], reverse=True)
    kept: list[dict] = []

    for candidate in clips:
        suppressed = False
        for k in kept:
            start = max(candidate["clip_start_ms"], k["clip_start_ms"])
            end = min(candidate["clip_end_ms"], k["clip_end_ms"])
            if end > start:
                overlap = end - start
                shorter = min(
                    candidate["clip_end_ms"] - candidate["clip_start_ms"],
                    k["clip_end_ms"] - k["clip_start_ms"],
                )
                if shorter > 0 and overlap / shorter > NMS_OVERLAP_THRESHOLD:
                    suppressed = True
                    break
        if not suppressed:
            kept.append(candidate)
        if len(kept) >= MAX_CLIPS_OUTPUT:
            break

    return kept


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("PHASE 5 — Auto-Curator (Full-Graph Sweep)")
    log.info("=" * 60)

    # Step 1: Fetch all nodes
    log.info("Step 1: Full-graph sweep from Spanner…")
    nodes = _fetch_all_nodes()
    log.info(f"  Nodes fetched: {len(nodes)}")

    if len(nodes) < WINDOW_MIN_NODES:
        log.error(
            f"Need at least {WINDOW_MIN_NODES} nodes, got {len(nodes)}. "
            "Run phases 1-4 first."
        )
        return

    # Step 2: Generate windows
    log.info("Step 2: Generating sliding windows…")
    windows = _generate_windows(nodes)
    log.info(f"  Windows to score: {len(windows)}")

    # Step 3: Batch score
    log.info("Step 3: Scoring windows with Gemini…")
    scored = asyncio.run(_batch_score_windows(windows))
    log.info(f"  Windows scored: {len(scored)}")

    if not scored:
        log.error("No windows scored successfully!")
        return

    # Step 4: NMS + ranking
    log.info("Step 4: NMS + global ranking…")
    scored.sort(key=lambda c: c["final_score"], reverse=True)

    above_min = [c for c in scored if c["final_score"] >= MIN_SCORE]
    if above_min:
        log.info(f"  {len(above_min)} clips scored ≥{MIN_SCORE}")
        final_clips = _nms(above_min)
    else:
        log.warning(
            f"  No clips scored ≥{MIN_SCORE}. "
            f"Falling back to top {FALLBACK_TOP_N}."
        )
        final_clips = _nms(scored[:FALLBACK_TOP_N])

    log.info(f"  Final clips after NMS: {len(final_clips)}")

    # Step 5: Output
    log.info("Step 5: Writing clip render payloads…")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(final_clips, f, indent=2)
    log.info(f"Output saved → {OUTPUT_PATH}")

    # Summary
    log.info("=" * 60)
    log.info("PHASE 5 COMPLETE — Auto-Curator")
    log.info(f"  Total windows evaluated: {len(scored)}")
    log.info(f"  Final clips: {len(final_clips)}")
    for i, c in enumerate(final_clips, 1):
        duration = (c["clip_end_ms"] - c["clip_start_ms"]) / 1000
        log.info(
            f"  #{i}: {c['final_score']:.0f}/100  "
            f"{c['clip_start_ms']}ms–{c['clip_end_ms']}ms  "
            f"({duration:.1f}s)"
        )
    log.info("=" * 60)


if __name__ == "__main__":
    main()