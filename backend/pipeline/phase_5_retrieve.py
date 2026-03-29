#!/usr/bin/env python3
"""
Phase 5: Retrieval & Production Serving
========================================
Takes a text query, embeds it via Gemini Embedding 2, performs a hybrid
Spanner query (vector KNN anchor + 1-hop graph traversal), sends the sub-graph
to Gemini 3.1 Pro for clip scoring, and outputs a render payload.

Pipeline:
  1. Embed the user query → 3072-d vector (`RETRIEVAL_QUERY`)
  2. APPROX_COSINE_DISTANCE on SemanticClipNode → anchor node
  3. Spanner Graph 1-hop traversal → context nodes
  4. ClipScoringAgent (Gemini) → optimal clip boundaries
  5. Assemble render payload → remotion_payload.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from google import genai
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

EMBEDDING_DIM = 3072

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "outputs" / "remotion_payload.json"
SPEAKER_GAP_MERGE_MS = 1500

DEFAULT_QUERY = "Find a cynical or sarcastic moment about crypto."

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
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
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
    if len(values) != EMBEDDING_DIM:
        raise RuntimeError(
            f"Query embedding size mismatch: got {len(values)}, expected {EMBEDDING_DIM}"
        )
    return list(values)


# ──────────────────────────────────────────────
# Step 2 & 3: Spanner hybrid query (KNN + graph)
# ──────────────────────────────────────────────
def hybrid_query(query_vector: list[float], top_k: int = 5) -> list[dict]:
    """KNN anchor + 1-hop graph traversal."""
    spanner_client = spanner.Client(project=PROJECT_ID)
    instance = spanner_client.instance(SPANNER_INSTANCE)
    database = instance.database(SPANNER_DATABASE)

    vec_str = "[" + ",".join(f"{v:.8f}" for v in query_vector) + "]"

    knn_sql = f"""
        SELECT node_id, video_uri, start_time_ms, end_time_ms,
               transcript_text, visual_description, vocal_delivery,
               speakers, content_mechanisms, spatial_tracking_uri,
               APPROX_COSINE_DISTANCE(
                   ARRAY<FLOAT64>{vec_str}, embedding
               ) AS distance
        FROM SemanticClipNode
        ORDER BY distance ASC
        LIMIT {top_k}
    """

    nodes = []
    with database.snapshot() as snapshot:
        results = snapshot.execute_sql(knn_sql)
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
                "content_mechanisms": row[8],
                "spatial_tracking_uri": row[9],
                "distance": row[10],
            })

    if not nodes:
        return []

    anchor_id = nodes[0]["node_id"]
    log.info(f"  Anchor node: {anchor_id} (distance={nodes[0]['distance']:.4f})")

    graph_sql = f"""
        SELECT ne.from_node_id, ne.to_node_id, ne.edge_type,
               ne.narrative_classification, ne.confidence_score,
               scn.start_time_ms, scn.end_time_ms,
               scn.transcript_text, scn.vocal_delivery,
               scn.content_mechanisms, scn.spatial_tracking_uri
        FROM NarrativeEdge ne
        JOIN SemanticClipNode scn
          ON scn.node_id = CASE
               WHEN ne.from_node_id = @anchor THEN ne.to_node_id
               ELSE ne.from_node_id
             END
        WHERE ne.from_node_id = @anchor OR ne.to_node_id = @anchor
    """
    params = {"anchor": anchor_id}
    param_types = {"anchor": spanner.param_types.STRING}

    context_nodes = []
    with database.snapshot() as snapshot:
        results = snapshot.execute_sql(graph_sql, params=params, param_types=param_types)
        for row in results:
            neighbor_id = row[1] if row[0] == anchor_id else row[0]
            context_nodes.append({
                "node_id": neighbor_id,
                "edge_type": row[2],
                "narrative_classification": row[3],
                "confidence_score": row[4],
                "start_time_ms": row[5],
                "end_time_ms": row[6],
                "transcript_text": row[7],
                "vocal_delivery": row[8],
                "content_mechanisms": row[9],
                "spatial_tracking_uri": row[10],
            })

    log.info(f"  Context nodes from graph traversal: {len(context_nodes)}")

    all_nodes = nodes + context_nodes
    all_nodes.sort(key=lambda n: n.get("start_time_ms", 0))

    seen = set()
    unique = []
    for n in all_nodes:
        if n["node_id"] not in seen:
            seen.add(n["node_id"])
            unique.append(n)

    return unique


# ──────────────────────────────────────────────
# Step 4: Clip scoring via Gemini
# ──────────────────────────────────────────────
def score_clip(sub_graph: list[dict]) -> ClipScore:
    """Send sub-graph to Gemini for clip boundary scoring."""
    import os
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
    os.environ["GOOGLE_CLOUD_LOCATION"] = GEMINI_LOCATION
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

    from google.genai.types import HttpOptions
    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    slim = [
        {
            "node_id": n["node_id"],
            "start_time_ms": n["start_time_ms"],
            "end_time_ms": n["end_time_ms"],
            "transcript_text": n.get("transcript_text", ""),
            "vocal_delivery": n.get("vocal_delivery", ""),
            "content_mechanisms": n.get("content_mechanisms", "{}"),
        }
        for n in sub_graph
    ]

    user_prompt = (
        "=== SUB-GRAPH ===\n"
        f"{json.dumps(slim, indent=2)}\n\n"
        "---\n\n"
        "Determine the best clip boundaries and score."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[user_prompt],
        config=genai.types.GenerateContentConfig(
            system_instruction=CLIP_SCORING_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=ClipScore,
            temperature=0.2,
        ),
    )

    return ClipScore.model_validate_json(response.text)



# ──────────────────────────────────────────────
# Step 5: Assemble render payload
# ──────────────────────────────────────────────
def assemble_payload(
    clip: ClipScore, sub_graph: list[dict], query: str
) -> dict:
    """Build a render-ready payload from clip scoring output."""
    return {
        "video_uri": sub_graph[0].get("video_uri", ""),
        "clip_start_ms": clip.recommended_start_ms,
        "clip_end_ms": clip.recommended_end_ms,
        "final_score": clip.final_score,
        "justification": clip.justification,
        "query": query,
        "included_node_ids": clip.included_node_ids,
        "spatial_tracking_uris": [
            n.get("spatial_tracking_uri", "") for n in sub_graph
            if n.get("spatial_tracking_uri")
        ],
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("PHASE 5 — Retrieval & Production Serving")
    log.info("=" * 60)

    query = DEFAULT_QUERY
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    log.info(f"Query: {query}")

    # Step 1: Embed
    log.info("Embedding query…")
    query_vector = embed_query(query)
    log.info(f"  Vector: {len(query_vector)}d")

    # Step 2+3: Hybrid query
    log.info("Running hybrid query (KNN + graph)…")
    sub_graph = hybrid_query(query_vector)
    log.info(f"  Sub-graph: {len(sub_graph)} nodes")

    if not sub_graph:
        log.warning("No nodes found — the graph may be empty.")
        return

    for n in sub_graph:
        log.info(
            f"    {n['node_id'][:8]}… {n['start_time_ms']}ms–{n['end_time_ms']}ms "
            f"dist={n.get('distance', 'N/A')}"
        )

    # Step 4: Score
    log.info("Scoring clip boundaries with Gemini…")
    clip = score_clip(sub_graph)
    log.info(f"  Score: {clip.final_score}")
    log.info(f"  Boundaries: {clip.recommended_start_ms}ms – {clip.recommended_end_ms}ms")
    log.info(f"  Justification: {clip.justification[:200]}")

    # Step 5: Payload
    payload = assemble_payload(clip, sub_graph, query)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    log.info(f"Output saved → {OUTPUT_PATH}")

    log.info("=" * 60)
    log.info("PHASE 5 COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()