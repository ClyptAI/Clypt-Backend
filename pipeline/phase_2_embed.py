#!/usr/bin/env python3
"""
Phase 2: Multimodal Embedding (Late Fusion)
============================================
Projects Semantic Nodes into a shared 1408-dimensional multimodal vector space
using the multimodalembedding@001 model. For each node, requests BOTH a text
embedding (from transcript + vocal delivery + mechanism summary) and a video
embedding (from the node's exact time segment), then fuses them via mean pooling.

Inputs:
  - phase_1b_nodes.json                         (local)
  - gs://clypt-test-bucket/phase_1a/video.mp4   (GCS)

Output:
  - phase_2_embeddings.json  (nodes + multimodal_embedding vectors)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from google import genai
from google.genai import types

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
PROJECT_ID = "clypt-v2"
LOCATION = "us-central1"
ROOT = Path(__file__).resolve().parent.parent
VIDEO_GCS_URI = "gs://clypt-test-bucket/phase_1a/video.mp4"
NODES_PATH = ROOT / "outputs" / "phase_1b_nodes.json"
OUTPUT_PATH = ROOT / "outputs" / "phase_2_embeddings.json"

# Gemini Embedding 2 model ID
MODEL_ID = "gemini-embedding-2-preview"

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase_2")


# ──────────────────────────────────────────────
# Embedding Client
# ──────────────────────────────────────────────
# Initialize the new Google Gen AI SDK client
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

def get_native_multimodal_embedding(text: str, video_uri: str, start_sec: float, end_sec: float):
    """
    Generates a single unified embedding for interleaved text and video.
    """
    # Create the video part with segment offsets via VideoMetadata
    video_part = types.Part(
        file_data=types.FileData(file_uri=video_uri, mime_type="video/mp4"),
        video_metadata=types.VideoMetadata(
            start_offset=f"{int(start_sec)}s",
            end_offset=f"{int(end_sec)}s",
        ),
    )

    # Configure output dimensionality and task type
    # Note: 3072 is the native size for Gemini Embedding 2.
    config = types.EmbedContentConfig(
        output_dimensionality=3072,
        task_type="RETRIEVAL_DOCUMENT",
    )

    response = client.models.embed_content(
        model=MODEL_ID,
        contents=[text, video_part],
        config=config
    )
    
    return response.embeddings[0].values


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    if not NODES_PATH.exists():
        log.error(f"Input file not found: {NODES_PATH}")
        return

    with open(NODES_PATH, "r") as f:
        nodes = json.load(f)

    log.info("=" * 60)
    log.info(f"PHASE 2: NATIVE MULTIMODAL EMBEDDING ({len(nodes)} nodes)")
    log.info(f"Model: {MODEL_ID} | Target Dim: 3072")
    log.info("=" * 60)

    results = []
    failed = 0

    for i, node in enumerate(nodes):
        node_id = node.get("node_id", f"node_{i}")
        transcript = node.get("transcript", "")
        start_t = float(node.get("start_time", 0))
        end_t = float(node.get("end_time", 0))

        # GEMINI 2 LIMIT: Video segments must be <= 120s (80s if audio is relevant)
        duration = end_t - start_t
        if duration > 120:
            log.warning(f"  Node {node_id} duration ({duration:.1f}s) exceeds 120s limit. Truncating.")
            end_t = start_t + 120

        log.info(f"[{i+1}/{len(nodes)}] Embedding node {node_id} ({duration:.1f}s)...")

        try:
            # Native fusion call
            fused_vector = get_native_multimodal_embedding(
                text=transcript,
                video_uri=VIDEO_GCS_URI,
                start_sec=start_t,
                end_sec=end_t
            )

            # Keep precision high for vector search quality
            fused_rounded = [round(x, 8) for x in fused_vector]
            
            results.append({
                **node, 
                "multimodal_embedding": fused_rounded,
                "embedding_dim": len(fused_rounded)
            })
            
            log.info(f"  Successfully generated {len(fused_rounded)}d vector.")

        except Exception as e:
            log.error(f"  FAILED node {node_id}: {e}")
            results.append({**node, "multimodal_embedding": None})
            failed += 1

        # Rate limit safety for preview model
        time.sleep(0.2)

    # ── Save ──
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Output saved → {OUTPUT_PATH}")

    # ── Summary ──
    log.info("=" * 60)
    log.info("PHASE 2 COMPLETE")
    log.info(f"  Total processed: {len(nodes)}")
    log.info(f"  Failed:          {failed}")
    log.info(f"  Output Vector:   3072-dimensional")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
