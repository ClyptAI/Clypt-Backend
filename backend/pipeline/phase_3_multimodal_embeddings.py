#!/usr/bin/env python3
"""
Phase 3: Multimodal Embedding (Gemini Embedding 2)
===================================================
Projects semantic nodes into a shared 3072-dimensional space using Gemini
Embedding 2. Each node is embedded as interleaved text + video with
`task_type=RETRIEVAL_DOCUMENT` for catalog indexing.

Inputs:
  - phase_2a_nodes.json                        (local)
  - gs://clypt-storage-v2/phase_1/video.mp4    (GCS)

Output:
  - phase_3_embeddings.json  (nodes + multimodal_embedding vectors)
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
VIDEO_GCS_URI = "gs://clypt-storage-v2/phase_1/video.mp4"
NODES_PATH = ROOT / "outputs" / "phase_2a_nodes.json"
OUTPUT_PATH = ROOT / "outputs" / "phase_3_embeddings.json"

MODEL_ID = "gemini-embedding-2-preview"
EMBEDDING_DIM = 3072
TASK_TYPE = "RETRIEVAL_DOCUMENT"
MAX_SEGMENT_SEC = 120.0
MIN_SEGMENT_SEC = 0.2

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
# Payload helpers
# ──────────────────────────────────────────────
def _build_text_payload(node: dict) -> str:
    """Concatenate transcript, vocal delivery, and mechanism summary into one text payload."""
    parts: list[str] = []

    transcript = (
        str(node.get("transcript_segment", "")).strip()
        or str(node.get("transcript", "")).strip()
        or str(node.get("transcript_text", "")).strip()
    )
    if transcript:
        parts.append(transcript)

    vocal = str(node.get("vocal_delivery", "")).strip()
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

    return " | ".join(parts).strip()


def _format_offset(seconds: float) -> str:
    value = max(0.0, float(seconds))
    token = f"{value:.3f}".rstrip("0").rstrip(".")
    if not token:
        token = "0"
    return f"{token}s"


def _normalize_segment(start_s: float, end_s: float) -> tuple[float, float]:
    start = max(0.0, float(start_s))
    end = max(start, float(end_s))

    if end - start < MIN_SEGMENT_SEC:
        end = start + MIN_SEGMENT_SEC

    if end - start > MAX_SEGMENT_SEC:
        end = start + MAX_SEGMENT_SEC

    return start, end


def _make_client():
    return genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


def _embed_node(
    client,
    text_payload: str,
    video_uri: str,
    start_s: float,
    end_s: float,
) -> list[float]:
    video_part = types.Part(
        file_data=types.FileData(file_uri=video_uri, mime_type="video/mp4"),
        video_metadata=types.VideoMetadata(
            start_offset=_format_offset(start_s),
            end_offset=_format_offset(end_s),
        ),
    )

    response = client.models.embed_content(
        model=MODEL_ID,
        contents=[text_payload, video_part],
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIM,
            task_type=TASK_TYPE,
        ),
    )

    embeddings = getattr(response, "embeddings", None) or []
    if not embeddings:
        raise RuntimeError("No embeddings returned")

    values = getattr(embeddings[0], "values", None)
    if not values:
        raise RuntimeError("Empty embedding vector returned")
    return list(values)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("PHASE 3 — Multimodal Embedding (Gemini Embedding 2)")
    log.info("=" * 60)

    if not NODES_PATH.exists():
        log.error(f"Input file not found: {NODES_PATH}")
        return

    # ── Load nodes ──
    log.info(f"Loading nodes: {NODES_PATH}")
    with open(NODES_PATH) as f:
        nodes = json.load(f)
    log.info(f"  Nodes loaded: {len(nodes)}")

    # ── Initialize Gemini Embedding 2 client ──
    client = _make_client()
    log.info(f"Model: {MODEL_ID}")
    log.info(f"Task type: {TASK_TYPE}")
    log.info(f"Video: {VIDEO_GCS_URI}")
    log.info(f"Embedding dimension: {EMBEDDING_DIM}")

    # ── Process each node ──
    results: list[dict] = []
    failed = 0

    for i, node in enumerate(nodes):
        start_s = float(node.get("start_time", 0.0))
        end_s = float(node.get("end_time", start_s))
        start_s, end_s = _normalize_segment(start_s, end_s)

        text_payload = _build_text_payload(node)
        if not text_payload:
            text_payload = "No transcript available for this video segment."

        log.info(
            f"[{i + 1}/{len(nodes)}] Node {start_s:.1f}s–{end_s:.1f}s "
            f"(text: {len(text_payload)} chars)"
        )

        try:
            fused = _embed_node(
                client=client,
                text_payload=text_payload,
                video_uri=VIDEO_GCS_URI,
                start_s=start_s,
                end_s=end_s,
            )
            if len(fused) != EMBEDDING_DIM:
                raise RuntimeError(
                    f"Embedding size mismatch: got {len(fused)}, expected {EMBEDDING_DIM}"
                )
            fused_rounded = [round(x, 8) for x in fused]
            results.append({**node, "multimodal_embedding": fused_rounded, "embedding_dim": len(fused_rounded)})
            log.info(f"  Embedding generated: {len(fused_rounded)}d")

        except Exception as e:
            log.error(f"  FAILED: {e}")
            results.append({**node, "multimodal_embedding": None})
            failed += 1

        if i < len(nodes) - 1:
            time.sleep(0.2)

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
