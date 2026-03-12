#!/usr/bin/env python3
"""
Phase 3: Multimodal Embedding (Late Fusion)
============================================
Projects Semantic Nodes into a shared 1408-dimensional multimodal vector space
using the multimodalembedding@001 model. For each node, requests BOTH a text
embedding (from transcript + vocal delivery + mechanism summary) and a video
embedding (from the node's exact time segment), then fuses them via mean pooling.

Inputs:
  - phase_2a_nodes.json                         (local)
  - gs://clypt-storage-v2/phase_1/video.mp4   (GCS)

Output:
  - phase_3_embeddings.json  (nodes + multimodal_embedding vectors)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import vertexai
from vertexai.vision_models import (
    MultiModalEmbeddingModel,
    Video,
    VideoSegmentConfig,
)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
PROJECT_ID = "clypt-v2"
LOCATION = "us-central1"
ROOT = Path(__file__).resolve().parent.parent
VIDEO_GCS_URI = "gs://clypt-storage-v2/phase_1/video.mp4"
NODES_PATH = ROOT / "outputs" / "phase_2a_nodes.json"
OUTPUT_PATH = ROOT / "outputs" / "phase_3_embeddings.json"

EMBEDDING_DIM = 1408
MIN_SEGMENT_SEC = 4  # API minimum interval

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase_3")


# ──────────────────────────────────────────────
# Text payload construction
# ──────────────────────────────────────────────
def _build_text_payload(node: dict) -> str:
    """Concatenate transcript, vocal delivery, and mechanism summary into a rich text string."""
    parts: list[str] = []

    transcript = node.get("transcript_segment", "").strip()
    if transcript:
        parts.append(transcript)

    vocal = node.get("vocal_delivery", "").strip()
    if vocal:
        parts.append(f"Vocal delivery: {vocal}")

    mechanisms = node.get("content_mechanisms", {})
    mech_parts: list[str] = []
    for dim_name in ("humor", "emotion", "social", "expertise"):
        dim = mechanisms.get(dim_name, {})
        if dim.get("present"):
            mech_type = dim.get("type", "")
            intensity = dim.get("intensity", 0.0)
            mech_parts.append(f"{dim_name.capitalize()}: {mech_type} ({intensity})")
    if mech_parts:
        parts.append("Content mechanisms: " + ", ".join(mech_parts))

    return " | ".join(parts)


def _mean_pool(vec_a: list[float], vec_b: list[float]) -> list[float]:
    """Element-wise mean of two equal-length vectors."""
    return [(a + b) / 2.0 for a, b in zip(vec_a, vec_b)]


def _average_vectors(vectors: list[list[float]]) -> list[float]:
    """Average multiple vectors into one."""
    if len(vectors) == 1:
        return vectors[0]
    n = len(vectors)
    result = [0.0] * len(vectors[0])
    for vec in vectors:
        for i, v in enumerate(vec):
            result[i] += v
    return [x / n for x in result]


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("PHASE 3 — Multimodal Embedding (Late Fusion)")
    log.info("=" * 60)

    # ── Load nodes ──
    log.info(f"Loading nodes: {NODES_PATH}")
    with open(NODES_PATH) as f:
        nodes = json.load(f)
    log.info(f"  Nodes loaded: {len(nodes)}")

    # ── Initialize Vertex AI ──
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    log.info(f"Model: multimodalembedding@001")
    log.info(f"Video: {VIDEO_GCS_URI}")
    log.info(f"Embedding dimension: {EMBEDDING_DIM}")

    video = Video.load_from_file(VIDEO_GCS_URI)

    # ── Process each node ──
    results: list[dict] = []
    failed = 0

    for i, node in enumerate(nodes):
        start_s = float(node["start_time"])
        end_s = float(node["end_time"])

        if end_s - start_s < MIN_SEGMENT_SEC:
            end_s = start_s + MIN_SEGMENT_SEC

        text_payload = _build_text_payload(node)

        log.info(
            f"[{i + 1}/{len(nodes)}] Node {start_s:.1f}s–{end_s:.1f}s "
            f"(text: {len(text_payload)} chars)"
        )

        try:
            embeddings = model.get_embeddings(
                video=video,
                video_segment_config=VideoSegmentConfig(
                    start_offset_sec=int(start_s),
                    end_offset_sec=int(end_s),
                ),
                contextual_text=text_payload,
                dimension=EMBEDDING_DIM,
            )

            text_vec = embeddings.text_embedding
            video_vecs = [ve.embedding for ve in embeddings.video_embeddings]

            if not text_vec:
                log.warning(f"  No text embedding returned for node {i + 1}")
                results.append({**node, "multimodal_embedding": None})
                failed += 1
                continue

            if not video_vecs:
                log.warning(f"  No video embeddings returned for node {i + 1}, using text only")
                fused = text_vec
            else:
                visual_vec = _average_vectors(video_vecs)
                fused = _mean_pool(text_vec, visual_vec)

            fused_rounded = [round(x, 8) for x in fused]
            results.append({**node, "multimodal_embedding": fused_rounded})
            log.info(
                f"  Text vec: {len(text_vec)}d | "
                f"Video segments: {len(video_vecs)} | "
                f"Fused: {len(fused_rounded)}d"
            )

        except Exception as e:
            log.error(f"  FAILED: {e}")
            results.append({**node, "multimodal_embedding": None})
            failed += 1

        if i < len(nodes) - 1:
            time.sleep(0.5)

    # ── Save ──
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Output saved → {OUTPUT_PATH}")

    # ── Summary ──
    log.info("=" * 60)
    log.info("PHASE 3 COMPLETE")
    log.info(f"  Nodes processed: {len(results)}")
    log.info(f"  Successful: {len(results) - failed}")
    log.info(f"  Failed: {failed}")
    embedded_count = sum(1 for r in results if r.get("multimodal_embedding"))
    log.info(f"  Nodes with embeddings: {embedded_count}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
