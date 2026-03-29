#!/usr/bin/env python3
"""
Phase 2B: Narrative Edge Mapping
=================================
Passes the Phase 2A Semantic Nodes into Gemini 3.1 Pro as a text-only
reasoning pass. The model acts as a graph topology engine to draw
directional narrative edges between nodes.

Inputs:
  - phase_2a_nodes.json  (local, array of Semantic Nodes)

Output:
  - phase_2b_narrative_edges.json  (array of directional edges)
"""

import json
import logging
import os
from enum import Enum
from pathlib import Path

from google import genai
from google.genai.types import HttpOptions
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
PROJECT_ID = "clypt-v2"
LOCATION = "global"
MODEL_ID = "gemini-3.1-pro-preview"

ROOT = Path(__file__).resolve().parent.parent
NODES_PATH = ROOT / "outputs" / "phase_2a_nodes.json"
OUTPUT_PATH = ROOT / "outputs" / "phase_2b_narrative_edges.json"

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase_2b")

# ──────────────────────────────────────────────
# System instruction (exact prompt from spec)
# ──────────────────────────────────────────────
SYSTEM_INSTRUCTION = """You are a graph topology engine. Your task is to analyze the provided array of sequential Semantic Nodes and map the structural relationships between them by drawing directional edges.

Edge taxonomy:
- Contradiction: Two claims that logically clash, or text contradicted by visual/audio evidence. (Example: "I prioritize privacy" <-> "We sell your data").
- Causal Link: One node establishes a problem or setup; the other resolves or extends it. This includes adjacent setup→punchline pairs in comedy. (Example: "Docker keeps crashing" -> "Added memory limits"). Look hard for these between consecutive nodes.
- Thematic: Same entity or topic organically resurfaces after a digression. (Example: Mentions React at min 3, tangent, React again at min 12).
- Tension / Release: Emotional stakes escalate, then resolve with payoff or relief. (Example: "24 hours to ship" -> "We actually did it. It's live.").
- Callback: Deliberate authorial reference back to an earlier moment, not organic recurrence. (Example: "Remember the API I showed you?" -> links to min 3).
- Escalation: Each successive moment more extreme or intense — sustained ramp, no release yet. This is common between adjacent nodes in comedy and challenge content. (Example: Spice challenge: mild -> hot -> Carolina Reaper).
- Subversion: Expectation is established, then broken — audience surprise, not logical clash. (Example: "I followed the tutorial exactly" -> total disaster).
- Analogy: Concept X is explained by mapping it onto a different, familiar domain Y. (Example: "A blockchain is like a Google Doc everyone edits").
- Revelation: New information that fundamentally recontextualizes prior nodes. (Example: "What I didn't mention is... this was all staged").

Two passes required:
1. Adjacent pass: scan every pair of consecutive nodes (node N and node N+1) for Causal Link, Escalation, Tension/Release, and Subversion edges. Comedy and sketch content in particular has dense short-range structure — setup and punchline are often just one node apart.
2. Global pass: scan all non-adjacent node pairs for Callback, Thematic, Contradiction, Analogy, and Revelation edges.

Only omit an edge if you are confident no meaningful relationship exists.

Output a strictly structured JSON array of these edges. Ensure from_node_start_time and to_node_start_time perfectly match the start_time values of the nodes you are connecting."""

# ──────────────────────────────────────────────
# Response schema (Pydantic)
# ──────────────────────────────────────────────
class EdgeType(str, Enum):
    CONTRADICTION = "Contradiction"
    CAUSAL_LINK = "Causal Link"
    THEMATIC = "Thematic"
    TENSION_RELEASE = "Tension / Release"
    CALLBACK = "Callback"
    ESCALATION = "Escalation"
    SUBVERSION = "Subversion"
    ANALOGY = "Analogy"
    REVELATION = "Revelation"


class NarrativeEdge(BaseModel):
    from_node_start_time: float
    to_node_start_time: float
    edge_type: EdgeType
    narrative_classification: str
    confidence_score: float = Field(ge=0.0, le=1.0)



def main():
    log.info("=" * 60)
    log.info("PHASE 2B — Narrative Edge Mapping")
    log.info("=" * 60)

    # ── Load Phase 2A nodes ──
    log.info(f"Loading nodes: {NODES_PATH}")
    nodes_json = NODES_PATH.read_text()
    nodes_data = json.loads(nodes_json)
    log.info(f"  Nodes loaded: {len(nodes_data)}")
    log.info(f"  Size: {len(nodes_json):,} chars")

    # ── Slim nodes to only fields needed for edge detection ──
    slim_nodes = [
        {
            "start_time": n["start_time"],
            "end_time": n["end_time"],
            "transcript_segment": n["transcript_segment"],
            "content_mechanisms": n["content_mechanisms"],
        }
        for n in nodes_data
    ]
    slim_json = json.dumps(slim_nodes)
    log.info(f"  Slim node payload: {len(slim_json):,} chars (was {len(nodes_json):,})")

    # ── Build the prompt ──
    user_prompt = (
        "=== SEMANTIC NODES ===\n"
        f"{slim_json}\n\n"
        "---\n\n"
        "Analyze the nodes above and map the structural narrative "
        "relationships between them. Output a JSON array of directional "
        "edges following your taxonomy. Ensure from_node_start_time and "
        "to_node_start_time exactly match the start_time values from "
        "the nodes above."
    )

    # ── Initialize Vertex AI client ──
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
    os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    log.info(f"Model: {MODEL_ID}")
    log.info("Submitting text-only request to Gemini…")

    # ── Call Gemini (text-only, no video) ──
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[user_prompt],
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=list[NarrativeEdge],
            temperature=0.2,
            thinking_config=genai.types.ThinkingConfig(
                thinking_level="MEDIUM",
            ),
        ),
    )

    log.info("Response received — parsing with Pydantic")

    # ── Parse via Pydantic and save ──
    raw_edges = json.loads(response.text)
    edges = [NarrativeEdge.model_validate(e) for e in raw_edges]
    log.info(f"Narrative edges extracted & validated: {len(edges)}")

    edges_dicts = [e.model_dump() for e in edges]
    with open(OUTPUT_PATH, "w") as f:
        json.dump(edges_dicts, f, indent=2)
    log.info(f"Output saved → {OUTPUT_PATH}")

    # ── Summary ──
    log.info("=" * 60)
    log.info("PHASE 2B COMPLETE")
    log.info(f"  Edges: {len(edges)}")

    if edges:
        type_counts: dict[str, int] = {}
        for e in edges:
            type_counts[e.edge_type.value] = type_counts.get(e.edge_type.value, 0) + 1
        for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            log.info(f"    {etype}: {count}")

        avg_conf = sum(e.confidence_score for e in edges) / len(edges)
        log.info(f"  Avg confidence: {avg_conf:.2f}")

        node_times = {n["start_time"] for n in nodes_data}
        edge_times = {e.from_node_start_time for e in edges} | {e.to_node_start_time for e in edges}
        orphans = edge_times - node_times
        if orphans:
            log.warning(f"  ⚠ Edge references to non-existent node times: {orphans}")
        else:
            log.info("  ✓ All edge references match valid node start_times")

    log.info("=" * 60)


if __name__ == "__main__":
    main()