#!/usr/bin/env python3
"""
Phase 5: Retrieval & Production Serving
========================================
Takes a text query, embeds it via Gemini Embedding 2, performs a hybrid
Spanner query (vector KNN anchor + 1-hop graph traversal), sends the sub-graph
to Gemini 3.1 Pro for clip scoring, and outputs a Remotion render payload.

Pipeline:
  1. Embed the user query → 3072-d vector (`RETRIEVAL_QUERY`)
  2. APPROX_COSINE_DISTANCE on SemanticClipNode → anchor node
  3. Spanner Graph 1-hop traversal → context nodes
  4. ClipScoringAgent (Gemini) → optimal clip boundaries
  5. Assemble Remotion payload → remotion_payload.json
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
LOCATION = "us-central1"
SPANNER_INSTANCE = "clypt-spanner-v2"
SPANNER_DATABASE = "clypt-graph-db-v2"
GEMINI_LOCATION = "global"
GEMINI_MODEL = "gemini-3.1-pro-preview"
EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_TASK_TYPE = "RETRIEVAL_QUERY"
EMBEDDING_API_VERSION = "v1beta"

EMBEDDING_DIM = 3072

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "outputs" / "remotion_payload.json"
SPEAKER_GAP_MERGE_MS = 1500

USER_QUERY = "Find a cynical or sarcastic moment about crypto."

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase_5")
# Suppress Spanner SDK internal metrics export errors (missing instance_id in Cloud Monitoring)
logging.getLogger("opentelemetry.sdk.metrics._internal.export").setLevel(logging.CRITICAL)

# ──────────────────────────────────────────────
# Pydantic schema for ClipScoringAgent response
# ──────────────────────────────────────────────
CLIP_SCORING_SYSTEM_INSTRUCTION = (
    "You are the ClipScoringAgent. Your job is to review a raw sub-graph of "
    "video nodes (an anchor moment and its surrounding context). Determine "
    "the absolute best start and end times to create a cohesive, viral "
    "standalone short-form video clip. Ensure the clip has a clear setup "
    "and payoff."
)


class ClipScore(BaseModel):
    final_score: float = Field(ge=0.0, le=100.0)
    justification: str
    recommended_start_ms: int
    recommended_end_ms: int
    included_node_ids: list[str]


# ──────────────────────────────────────────────
# Step 1: Embed the query
# ──────────────────────────────────────────────
def embed_query(query: str) -> list[float]:
    """Generate a 3072-d query embedding for retrieval."""
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
    os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

    client = genai.Client(http_options=HttpOptions(api_version=EMBEDDING_API_VERSION))
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query],
        config=genai.types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIM,
            task_type=EMBEDDING_TASK_TYPE,
        ),
    )

    embeddings = getattr(response, "embeddings", None) or []
    if not embeddings:
        raise RuntimeError("No query embedding returned")

    values = getattr(embeddings[0], "values", None)
    if not values:
        raise RuntimeError("Empty query embedding returned")
    return list(values)


# ──────────────────────────────────────────────
# Step 2: Spanner hybrid query
# ──────────────────────────────────────────────
def find_anchor(database, query_vector: list[float]) -> dict:
    """Find the single closest node by approximate cosine distance."""
    vector_literal = "[" + ",".join(str(v) for v in query_vector) + "]"

    sql = f"""
        SELECT node_id, transcript_text, start_time_ms, end_time_ms,
               spatial_tracking_uri, speakers,
               APPROX_COSINE_DISTANCE(
                   embedding,
                   ARRAY<FLOAT32>{vector_literal},
                   options => JSON '{{"num_leaves_to_search": 10}}'
               ) AS distance
        FROM SemanticClipNode
        WHERE embedding IS NOT NULL
        ORDER BY distance ASC
        LIMIT 1
    """

    with database.snapshot() as snapshot:
        results = list(snapshot.execute_sql(sql))

    if not results:
        raise RuntimeError("No nodes found in SemanticClipNode table")

    row = results[0]
    speakers = row[5]
    if isinstance(speakers, str):
        try:
            speakers = json.loads(speakers)
        except (json.JSONDecodeError, TypeError):
            speakers = []
    elif hasattr(speakers, "serialize"):
        try:
            speakers = json.loads(speakers.serialize())
        except Exception:
            speakers = []
    if not isinstance(speakers, list):
        speakers = []

    return {
        "node_id": row[0],
        "transcript_text": row[1],
        "start_time_ms": row[2],
        "end_time_ms": row[3],
        "spatial_tracking_uri": row[4],
        "speakers": [str(s) for s in speakers if s is not None and str(s) != ""],
        "distance": row[6],
    }


def find_context_nodes(database, anchor_node_id: str) -> list[dict]:
    """1-hop graph traversal from the anchor node via Spanner Graph."""
    sql = """
        SELECT
            ctx_node_id,
            ctx_transcript,
            ctx_start_ms,
            ctx_end_ms,
            ctx_tracking_uri,
            ctx_speakers,
            edge_label
        FROM GRAPH_TABLE(
            ClyptGraph
            MATCH (anchor:SemanticClipNode)-[edge:NarrativeEdge]-(context:SemanticClipNode)
            WHERE anchor.node_id = @anchor_id
            RETURN context.node_id AS ctx_node_id,
                   context.transcript_text AS ctx_transcript,
                   context.start_time_ms AS ctx_start_ms,
                   context.end_time_ms AS ctx_end_ms,
                   context.spatial_tracking_uri AS ctx_tracking_uri,
                   context.speakers AS ctx_speakers,
                   edge.label AS edge_label
        )
        ORDER BY ctx_start_ms ASC
    """

    params = {"anchor_id": anchor_node_id}
    param_types = {"anchor_id": spanner.param_types.STRING}

    with database.snapshot() as snapshot:
        results = list(snapshot.execute_sql(sql, params=params, param_types=param_types))

    seen = set()
    nodes = []
    for row in results:
        nid = row[0]
        if nid in seen or nid == anchor_node_id:
            continue
        seen.add(nid)
        speakers = row[5]
        if isinstance(speakers, str):
            try:
                speakers = json.loads(speakers)
            except (json.JSONDecodeError, TypeError):
                speakers = []
        elif hasattr(speakers, "serialize"):
            try:
                speakers = json.loads(speakers.serialize())
            except Exception:
                speakers = []
        if not isinstance(speakers, list):
            speakers = []

        nodes.append({
            "node_id": nid,
            "transcript_text": row[1],
            "start_time_ms": row[2],
            "end_time_ms": row[3],
            "spatial_tracking_uri": row[4],
            "speakers": [str(s) for s in speakers if s is not None and str(s) != ""],
            "edge_label": row[6],
        })
    return nodes


# ──────────────────────────────────────────────
# Step 3: ClipScoringAgent via Gemini
# ──────────────────────────────────────────────
def score_subgraph(subgraph: dict) -> ClipScore:
    """Send the sub-graph to Gemini 3.1 Pro for clip scoring."""
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
    os.environ["GOOGLE_CLOUD_LOCATION"] = GEMINI_LOCATION
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    prompt = (
        f"=== Sub-Graph (anchor + 1-hop context nodes) ===\n"
        f"{json.dumps(subgraph, indent=2)}\n\n"
        "---\n\n"
        f"User Query: \"{USER_QUERY}\"\n\n"
        "Evaluate the sub-graph above and determine the optimal clip "
        "boundaries. Output your scoring as a JSON object with final_score, "
        "justification, recommended_start_ms, recommended_end_ms, and "
        "included_node_ids."
    )

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


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("PHASE 4 — Retrieval & Production Serving")
    log.info("=" * 60)

    # ── Step 1: Embed the query ──
    log.info(f"Query: \"{USER_QUERY}\"")
    log.info(f"Embedding query via {EMBEDDING_MODEL} ({EMBEDDING_TASK_TYPE})…")
    query_vector = embed_query(USER_QUERY)
    log.info(f"  Query vector: {len(query_vector)}d")

    # ── Step 2: Spanner hybrid query ──
    log.info(f"Connecting to Spanner: {SPANNER_INSTANCE}/{SPANNER_DATABASE}")
    spanner_client = spanner.Client(project=PROJECT_ID)
    instance = spanner_client.instance(SPANNER_INSTANCE)
    database = instance.database(SPANNER_DATABASE)

    log.info("Finding anchor node (APPROX_COSINE_DISTANCE)…")
    anchor = find_anchor(database, query_vector)
    log.info(
        f"  Anchor: {anchor['node_id'][:12]}… "
        f"({anchor['start_time_ms']}ms–{anchor['end_time_ms']}ms) "
        f"distance={anchor['distance']:.6f}"
    )
    log.info(f"  Transcript: {anchor['transcript_text'][:120]}…")

    log.info("Traversing 1-hop graph context…")
    context_nodes = find_context_nodes(database, anchor["node_id"])
    log.info(f"  Context nodes found: {len(context_nodes)}")
    for cn in context_nodes:
        log.info(
            f"    {cn['node_id'][:12]}… "
            f"({cn['start_time_ms']}ms–{cn['end_time_ms']}ms) "
            f"via {cn['edge_label']}"
        )

    # Assemble sub-graph ordered chronologically
    all_nodes = [
        {**anchor, "role": "anchor"},
        *[{**cn, "role": "context"} for cn in context_nodes],
    ]
    all_nodes.sort(key=lambda n: n["start_time_ms"])

    subgraph = {
        "query": USER_QUERY,
        "anchor_node_id": anchor["node_id"],
        "nodes": all_nodes,
    }

    log.info(f"Sub-graph assembled: {len(all_nodes)} nodes (chronological)")

    # ── Step 3: ClipScoringAgent ──
    log.info("─" * 50)
    log.info("Sending sub-graph to ClipScoringAgent (Gemini 3.1 Pro)…")
    clip_score = score_subgraph(subgraph)
    log.info(f"  Score: {clip_score.final_score}/100")
    log.info(f"  Justification: {clip_score.justification}")
    log.info(
        f"  Recommended: {clip_score.recommended_start_ms}ms – "
        f"{clip_score.recommended_end_ms}ms"
    )
    log.info(f"  Included nodes: {len(clip_score.included_node_ids)}")

    # ── Step 4: Remotion payload ──
    log.info("─" * 50)
    log.info("Constructing Remotion render payload…")

    included_set = set(clip_score.included_node_ids)
    included_nodes = [n for n in all_nodes if n["node_id"] in included_set]
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
    # Merge short gaps for the same speaker to reduce camera jitter.
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

    payload = {
        "clip_start_ms": clip_score.recommended_start_ms,
        "clip_end_ms": clip_score.recommended_end_ms,
        "final_score": clip_score.final_score,
        "justification": clip_score.justification,
        "combined_transcript": combined_transcript,
        "tracking_uris": tracking_uris,
        "included_node_ids": clip_score.included_node_ids,
        "active_speaker_timeline": active_speaker_timeline,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    log.info(f"Remotion payload saved → {OUTPUT_PATH}")

    # ── Summary ──
    log.info("=" * 60)
    log.info("PHASE 4 COMPLETE")
    log.info(f"  Query: \"{USER_QUERY}\"")
    log.info(f"  Anchor distance: {anchor['distance']:.6f}")
    log.info(f"  Sub-graph nodes: {len(all_nodes)}")
    log.info(f"  Clip score: {clip_score.final_score}/100")
    log.info(
        f"  Clip range: {clip_score.recommended_start_ms}ms – "
        f"{clip_score.recommended_end_ms}ms "
        f"({(clip_score.recommended_end_ms - clip_score.recommended_start_ms) / 1000:.1f}s)"
    )
    log.info(f"  Transcript: {len(combined_transcript)} chars")
    log.info(f"  Tracking URIs: {len(tracking_uris)}")
    log.info(f"  Output: {OUTPUT_PATH}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
