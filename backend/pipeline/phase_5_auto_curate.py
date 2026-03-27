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

MIN_SCORE = 85
FALLBACK_TOP_N = 3
SPEAKER_GAP_MERGE_MS = 1500
MAX_CLIPS_OUTPUT = 10
NMS_OVERLAP_THRESHOLD = 0.5

# Sliding window config
WINDOW_MIN_NODES = 3
WINDOW_MAX_NODES = 8
WINDOW_STEP = 2

# Pre-filter: only send top N windows by mechanism score to Gemini
PREFILTER_TOP_N = 40

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
recommended_end_ms MUST exactly match the end_time_ms of the last node you select.
Your justification must state the specific content_mechanisms scores and vocal_delivery that drove your decision."""


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

    if not results:
        raise RuntimeError("NarrativeEdge table is empty; full-graph auto-curation requires edges.")

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
# Step 2: Sliding Window Chapter Generation
# ──────────────────────────────────────────────
def build_narrative_chapters(
    nodes: list[dict],
    edges: list[dict],
) -> list[list[dict]]:
    """Generate overlapping sliding windows of consecutive nodes as clip candidates.

    Every window of WINDOW_MIN_NODES to WINDOW_MAX_NODES consecutive nodes is
    a candidate chapter. Windows advance by WINDOW_STEP so candidates overlap
    and no moment is missed at a boundary.

    Edge connectivity is used as a tiebreaker signal (stored on each window as
    metadata) but does NOT gate whether a window is created.
    """
    if not nodes:
        return []

    # Build adjacency for edge-bonus scoring downstream
    adjacency: dict[str, set[str]] = {n["node_id"]: set() for n in nodes}
    for edge in edges:
        fid, tid = edge["from_node_id"], edge["to_node_id"]
        if fid in adjacency and tid in adjacency:
            adjacency[fid].add(tid)
            adjacency[tid].add(fid)

    chapters: list[list[dict]] = []

    for window_size in range(WINDOW_MIN_NODES, WINDOW_MAX_NODES + 1):
        for start in range(0, len(nodes) - window_size + 1, WINDOW_STEP):
            window = nodes[start : start + window_size]
            chapters.append(window)

    # Deduplicate identical windows (same start + end node)
    seen: set[tuple[str, str]] = set()
    unique: list[list[dict]] = []
    for ch in chapters:
        key = (ch[0]["node_id"], ch[-1]["node_id"])
        if key not in seen:
            seen.add(key)
            unique.append(ch)

    return unique


# ──────────────────────────────────────────────
# Content type detection
# ──────────────────────────────────────────────
CONTENT_TYPE_SCORING_GUIDANCE = {
    "comedy": """\
This is comedy/sketch content. Weight these signals heavily:
- Humor mechanisms with subversion or irony type (intensity > 0.7) → strong clip signal
- Escalation chains between adjacent nodes → look for setup→punchline structure
- Vocal delivery mentioning laughter, exaggerated tone, or character voices → high value
- A clip without a clear punchline or comedic payoff should score below 70 regardless of other qualities.""",

    "interview": """\
This is interview/podcast content. Weight these signals heavily:
- Social dynamics mechanisms (status reversal, genuine disagreement, unexpected chemistry)
- Expertise mechanisms with counterintuitive truth or casual flex type
- Moments where the conversation shifts energy — a surprising answer, a long pause, a laugh
- A clip that is just two people talking with no energy shift should score below 70.""",

    "tutorial": """\
This is tutorial/educational content. Weight these signals heavily:
- Expertise mechanisms (counterintuitive truth, elegant simplification, live demo)
- Moments where a concept clicks — an "aha" explanation or a surprising demonstration
- Causal Link edges between setup (problem) and resolution (solution)
- A clip that is just narration without a clear insight or demonstration should score below 70.""",

    "vlog": """\
This is vlog/personal content. Weight these signals heavily:
- Emotion mechanisms (vulnerability, catharsis, awe, panic)
- High vocal delivery intensity — raw reactions, genuine emotion, unexpected moments
- Tension/Release structures — buildup followed by payoff or relief
- A clip that is just b-roll or filler narration should score below 70.""",

    "unknown": "",
}


def detect_content_type(nodes: list[dict]) -> str:
    """Infer content type from the dominant mechanism distribution across all nodes."""
    counts: dict[str, float] = {"humor": 0.0, "emotion": 0.0, "social": 0.0, "expertise": 0.0}

    for node in nodes:
        mechanisms = node.get("content_mechanisms", {})
        for dim in counts:
            dim_data = mechanisms.get(dim, {})
            if isinstance(dim_data, dict) and dim_data.get("present"):
                counts[dim] += float(dim_data.get("intensity", 0.0))

    total = sum(counts.values())
    if total == 0:
        return "unknown"

    dominant = max(counts, key=lambda k: counts[k])
    dominant_ratio = counts[dominant] / total

    if dominant_ratio < 0.35:
        return "unknown"

    mapping = {
        "humor": "comedy",
        "social": "interview",
        "expertise": "tutorial",
        "emotion": "vlog",
    }
    content_type = mapping[dominant]
    log.info(f"  Content type detected: {content_type} "
             f"(humor={counts['humor']:.1f}, emotion={counts['emotion']:.1f}, "
             f"social={counts['social']:.1f}, expertise={counts['expertise']:.1f})")
    return content_type


def build_scoring_instruction(content_type: str) -> str:
    """Build the full scoring system instruction with content-type-specific guidance."""
    type_guidance = CONTENT_TYPE_SCORING_GUIDANCE.get(content_type, "")
    type_section = f"\nContent-type guidance:\n{type_guidance}\n" if type_guidance else ""
    return CLIP_SCORING_SYSTEM_INSTRUCTION + type_section


# ──────────────────────────────────────────────
# Step 2.5: Cheap pre-filter using mechanism scores
# ──────────────────────────────────────────────
def _mechanism_score(window: list[dict]) -> float:
    """Score a window cheaply using content_mechanisms intensities — no API call."""
    total = 0.0
    for node in window:
        mechanisms = node.get("content_mechanisms", {})
        for dim in ("humor", "emotion", "social", "expertise"):
            dim_data = mechanisms.get(dim, {})
            if isinstance(dim_data, dict) and dim_data.get("present"):
                total += float(dim_data.get("intensity", 0.0))
    return total / max(1, len(window))


def prefilter_windows(windows: list[list[dict]]) -> list[list[dict]]:
    """Keep only the top PREFILTER_TOP_N windows by mechanism score."""
    scored = [(w, _mechanism_score(w)) for w in windows]
    scored.sort(key=lambda x: x[1], reverse=True)
    kept = [w for w, _ in scored[:PREFILTER_TOP_N]]
    log.info(f"  Pre-filter: {len(windows)} windows → {len(kept)} (top by mechanism score)")
    if scored:
        log.info(f"  Mechanism score range: {scored[-1][1]:.3f} – {scored[0][1]:.3f}")
    return kept


# ──────────────────────────────────────────────
# Step 3: Async AI Batch Evaluator
# ──────────────────────────────────────────────
async def score_chapter_async(
    chapter: list[dict],
    chapter_idx: int,
    total: int,
    sem: asyncio.Semaphore,
    scoring_instruction: str,
) -> ClipScore | None:
    """Score one window via Gemini asynchronously, respecting the concurrency semaphore."""
    async with sem:
        os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
        os.environ["GOOGLE_CLOUD_LOCATION"] = GEMINI_LOCATION
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

        client = genai.Client(http_options=HttpOptions(api_version="v1"))

        serializable_chapter = [
            {
                "node_id": n["node_id"],
                "transcript_text": n["transcript_text"],
                "vocal_delivery": n["vocal_delivery"],
                "start_time_ms": n["start_time_ms"],
                "end_time_ms": n["end_time_ms"],
                "content_mechanisms": n["content_mechanisms"],
            }
            for n in chapter
        ]

        prompt = (
            f"=== Window {chapter_idx + 1}/{total} — Semantic Nodes ===\n"
            f"{json.dumps(serializable_chapter, indent=2)}\n\n"
            "---\n\n"
            "Identify the single best viral clip within the window above. "
            "Output your scoring as a JSON object with final_score, justification, "
            "recommended_start_ms, recommended_end_ms, and included_node_ids."
        )

        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=GEMINI_MODEL,
                contents=[prompt],
                config=genai.types.GenerateContentConfig(
                    system_instruction=scoring_instruction,
                    response_mime_type="application/json",
                    response_schema=ClipScore,
                    temperature=0.2,
                    thinking_config=genai.types.ThinkingConfig(
                        thinking_level="MEDIUM",
                    ),
                ),
            )
            raw = json.loads(response.text)
            score = ClipScore.model_validate(raw)
            log.info(f"  [{chapter_idx + 1}/{total}] score={score.final_score} "
                     f"({score.recommended_start_ms}ms–{score.recommended_end_ms}ms)")
            return score
        except Exception as e:
            log.error(f"  [{chapter_idx + 1}/{total}] scoring failed: {e}")
            return None





# ──────────────────────────────────────────────
# Step 4: Global Ranking & Filtering
# ──────────────────────────────────────────────
def rank_and_filter(
    scored_clips: list[tuple[ClipScore, list[dict]]],
) -> list[tuple[ClipScore, list[dict]]]:
    """Sort by score, deduplicate overlapping clips via NMS, cap output."""
    scored_clips.sort(key=lambda x: x[0].final_score, reverse=True)

    elite = [(cs, ch) for cs, ch in scored_clips if cs.final_score >= MIN_SCORE]
    if not elite:
        log.warning(f"No clips scored {MIN_SCORE}+. Falling back to Top {FALLBACK_TOP_N}.")
        elite = scored_clips[:FALLBACK_TOP_N]

    # Non-maximum suppression: remove clips that overlap heavily with a better one
    def _overlap_ratio(a: ClipScore, b: ClipScore) -> float:
        start = max(a.recommended_start_ms, b.recommended_start_ms)
        end = min(a.recommended_end_ms, b.recommended_end_ms)
        if end <= start:
            return 0.0
        intersection = end - start
        len_a = a.recommended_end_ms - a.recommended_start_ms
        len_b = b.recommended_end_ms - b.recommended_start_ms
        return intersection / min(len_a, len_b)

    kept: list[tuple[ClipScore, list[dict]]] = []
    for candidate_cs, candidate_ch in elite:
        suppressed = False
        for kept_cs, _ in kept:
            if _overlap_ratio(candidate_cs, kept_cs) > NMS_OVERLAP_THRESHOLD:
                suppressed = True
                break
        if not suppressed:
            kept.append((candidate_cs, candidate_ch))
        if len(kept) >= MAX_CLIPS_OUTPUT:
            break

    log.info(f"  After NMS deduplication: {len(kept)} unique clips (from {len(elite)} candidates)")
    return kept


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
async def _run():
    log.info("=" * 60)
    log.info("PHASE 5 — Auto-Curator (Full-Graph Sweep)")
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

    # ── Step 2: Sliding window generation ──
    log.info("─" * 50)
    log.info("Building sliding windows…")
    chapters = build_narrative_chapters(nodes, edges)
    log.info(f"  Windows generated: {len(chapters)} "
             f"(sizes {WINDOW_MIN_NODES}–{WINDOW_MAX_NODES} nodes, step={WINDOW_STEP})")

    # ── Step 2.5: Pre-filter ──
    log.info("─" * 50)
    log.info("Pre-filtering windows by mechanism score…")
    chapters = prefilter_windows(chapters)

    # ── Content type detection ──
    log.info("─" * 50)
    log.info("Detecting content type…")
    content_type = detect_content_type(nodes)
    scoring_instruction = build_scoring_instruction(content_type)

    # ── Step 3: Async AI Scoring ──
    log.info("─" * 50)
    log.info(f"Scoring {len(chapters)} windows with Gemini (concurrency={MAX_CONCURRENT_SCORING})…")

    sem = asyncio.Semaphore(MAX_CONCURRENT_SCORING)
    tasks = [
        score_chapter_async(chapter, i, len(chapters), sem, scoring_instruction)
        for i, chapter in enumerate(chapters)
    ]
    results = await asyncio.gather(*tasks)

    scored_clips: list[tuple[ClipScore, list[dict]]] = [
        (score, chapter)
        for score, chapter in zip(results, chapters)
        if score is not None
    ]

    if not scored_clips:
        log.error("No windows could be scored. Exiting.")
        return

    log.info(f"  Scored: {len(scored_clips)}/{len(chapters)} windows")

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
    log.info("PHASE 5 AUTO-CURATOR COMPLETE")
    log.info(f"  Total nodes swept:    {len(nodes)}")
    log.info(f"  Windows generated:    {len(chapters)}")
    log.info(f"  Windows scored:       {len(scored_clips)}")
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


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
