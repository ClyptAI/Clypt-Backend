#!/usr/bin/env python3
"""
Phase 3: Storage & Graph Binding (Spanner + GCS)
=================================================
Ingests Phase 1/2 JSON ledgers into Google Cloud Spanner (nodes + edges)
and uploads decoupled spatial tracking data to GCS.

Uses Spanner's native multi-model architecture: ScaNN vector search on
the embedding column, Spanner Graph for edge traversal.

Inputs:
  - phase_2_embeddings.json          (nodes with 1408-d vectors)
  - phase_1c_narrative_edges.json    (directional edges)
  - phase_1a_visual.json             (face/person/object/label data)
  - phase_1a_audio.json              (word-level timestamps + speakers)

Targets:
  - Spanner: instance=clypt-preyc-db, database=clypt-db
    - Table: SemanticClipNode
    - Table: NarrativeEdge
  - GCS: gs://clypt-test-bucket/tracking/[node_id].json
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from google.cloud import spanner, storage

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
PROJECT_ID = "clypt-preyc"
SPANNER_INSTANCE = "clypt-preyc-db"
SPANNER_DATABASE = "clypt-db"

GCS_BUCKET = "clypt-test-bucket"
VIDEO_GCS_URI = "gs://clypt-test-bucket/phase_1a/video.mp4"
TRACKING_PREFIX = "tracking"

ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = ROOT / "outputs" / "phase_2_embeddings.json"
EDGES_PATH = ROOT / "outputs" / "phase_1c_narrative_edges.json"
VISUAL_PATH = ROOT / "outputs" / "phase_1a_visual.json"
AUDIO_PATH = ROOT / "outputs" / "phase_1a_audio.json"

BATCH_SIZE = 100

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase_3")
# Suppress Spanner SDK internal metrics export errors (missing instance_id in Cloud Monitoring)
logging.getLogger("opentelemetry.sdk.metrics._internal.export").setLevel(logging.CRITICAL)


# ──────────────────────────────────────────────
# Edge type normalization
# ──────────────────────────────────────────────
EDGE_TYPE_MAP = {
    "Contradiction": "CONTRADICTION",
    "Causal Link": "CAUSAL_LINK",
    "Thematic": "THEMATIC",
    "Tension / Release": "TENSION_RELEASE",
    "Callback": "CALLBACK",
    "Escalation": "ESCALATION",
    "Subversion": "SUBVERSION",
    "Analogy": "ANALOGY",
    "Revelation": "REVELATION",
}


# ──────────────────────────────────────────────
# Cross-reference helpers
# ──────────────────────────────────────────────
def _overlaps(seg_start: int, seg_end: int, node_start: int, node_end: int) -> bool:
    return seg_start < node_end and seg_end > node_start


def _extract_speakers(node_start_ms: int, node_end_ms: int, audio: dict) -> list[str]:
    """Return speaker tags ordered by descending word count in node range."""
    speaker_counts: dict[str, int] = {}
    for word in audio.get("words", []):
        w_start = int(word.get("start_time_ms", 0))
        w_end = int(word.get("end_time_ms", w_start))
        # Include any temporal overlap with the node window.
        if not (w_start < node_end_ms and w_end > node_start_ms):
            continue
        tag = word.get("speaker_tag", "unknown")
        if not tag or str(tag) == "unknown":
            continue
        key = str(tag)
        speaker_counts[key] = speaker_counts.get(key, 0) + 1

    return [
        tag for tag, _ in sorted(
            speaker_counts.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
    ]


def _extract_objects_present(node_start_ms: int, node_end_ms: int, visual: dict) -> list[str]:
    """Find unique object descriptions tracked within the node's time range."""
    objects: set[str] = set()
    for obj in visual.get("object_tracking", []):
        if not _overlaps(obj.get("segment_start_ms", 0), obj.get("segment_end_ms", 0),
                         node_start_ms, node_end_ms):
            continue
        for frame in obj.get("frames", []):
            if node_start_ms <= frame.get("time_ms", 0) < node_end_ms:
                desc = obj.get("entity", {}).get("description", "")
                if desc:
                    objects.add(desc)
                break
    return sorted(objects)


def _extract_visual_labels(node_start_ms: int, node_end_ms: int, visual: dict) -> list[str]:
    """Find unique label descriptions active within the node's time range."""
    labels: set[str] = set()
    for label in visual.get("label_detections", []):
        desc = label.get("entity", {}).get("description", "")
        if not desc:
            continue
        for seg in label.get("segments", []):
            if _overlaps(seg.get("start_time_ms", 0), seg.get("end_time_ms", 0),
                         node_start_ms, node_end_ms):
                labels.add(desc)
                break
        for frame in label.get("frames", []):
            if node_start_ms <= frame.get("time_ms", 0) < node_end_ms:
                labels.add(desc)
                break
    return sorted(labels)


def _extract_spatial_tracking(
    node_start_ms: int, node_end_ms: int, visual: dict,
) -> dict:
    """Slice person_detections and face_detections for this node's time range.
    """
    tracking: dict = {
        "person_detections": [],
        "face_detections": [],
    }

    for person in visual.get("person_detections", []):
        if not _overlaps(person.get("segment_start_ms", 0),
                         person.get("segment_end_ms", 0),
                         node_start_ms, node_end_ms):
            continue
        filtered_ts = [
            ts for ts in person.get("timestamped_objects", [])
            if node_start_ms <= ts.get("time_ms", 0) < node_end_ms
        ]
        if filtered_ts:
            tracking["person_detections"].append({
                "confidence": person.get("confidence"),
                "segment_start_ms": person.get("segment_start_ms"),
                "segment_end_ms": person.get("segment_end_ms"),
                "timestamped_objects": filtered_ts,
            })

    all_face_detections = visual.get("face_detections", [])
    for face_idx, face in enumerate(all_face_detections):
        if not _overlaps(face.get("segment_start_ms", 0),
                         face.get("segment_end_ms", 0),
                         node_start_ms, node_end_ms):
            continue
        filtered_ts = [
            ts for ts in face.get("timestamped_objects", [])
            if node_start_ms <= ts.get("time_ms", 0) < node_end_ms
        ]
        if filtered_ts:
            face_entry: dict = {
                "confidence": face.get("confidence"),
                "segment_start_ms": face.get("segment_start_ms"),
                "segment_end_ms": face.get("segment_end_ms"),
                "face_track_index": face_idx,
                "timestamped_objects": filtered_ts,
            }
            tracking["face_detections"].append(face_entry)

    return tracking


# ──────────────────────────────────────────────
# Spanner batched mutation writer
# ──────────────────────────────────────────────
def _write_mutations_batched(database, mutations: list, label: str):
    """Write mutations in batches to avoid per-commit size limits."""
    total = len(mutations)
    for i in range(0, total, BATCH_SIZE):
        batch = mutations[i : i + BATCH_SIZE]
        database.batch().insert_or_update(
            table=batch[0]["table"],
            columns=batch[0]["columns"],
            values=[m["values"] for m in batch],
        )
        log.info(f"  [{label}] Committed batch {i // BATCH_SIZE + 1} "
                 f"({len(batch)} rows, {i + len(batch)}/{total})")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("PHASE 3 — Storage & Graph Binding")
    log.info("=" * 60)

    # ── Load all data ──
    log.info("Loading data files…")
    with open(EMBEDDINGS_PATH) as f:
        nodes = json.load(f)
    log.info(f"  Nodes: {len(nodes)} (from {EMBEDDINGS_PATH})")

    with open(EDGES_PATH) as f:
        edges = json.load(f)
    log.info(f"  Edges: {len(edges)} (from {EDGES_PATH})")

    with open(VISUAL_PATH) as f:
        visual = json.load(f)
    log.info(f"  Visual ledger loaded ({VISUAL_PATH})")

    with open(AUDIO_PATH) as f:
        audio = json.load(f)
    log.info(f"  Audio ledger loaded ({AUDIO_PATH})")

    # ── UUID generation & mapping ──
    log.info("Assigning UUIDs to nodes…")
    start_time_to_id: dict[float, str] = {}
    node_ids: list[str] = []

    for node in nodes:
        node_id = str(uuid.uuid4())
        node_ids.append(node_id)
        start_time_to_id[node["start_time"]] = node_id

    log.info(f"  Mapped {len(start_time_to_id)} unique start_times → UUIDs")

    # ── Connect to Spanner ──
    log.info(f"Connecting to Spanner: {SPANNER_INSTANCE}/{SPANNER_DATABASE}")
    spanner_client = spanner.Client(project=PROJECT_ID)
    instance = spanner_client.instance(SPANNER_INSTANCE)
    database = instance.database(SPANNER_DATABASE)

    # ── Clear stale data from previous runs ──
    # Edges reference nodes via foreign keys in the graph, so delete edges first
    log.info("Clearing previous run data from Spanner…")
    def _clear_tables(transaction):
        transaction.execute_update("DELETE FROM NarrativeEdge WHERE TRUE")
        transaction.execute_update("DELETE FROM SemanticClipNode WHERE TRUE")
    database.run_in_transaction(_clear_tables)
    log.info("  ✓ Cleared NarrativeEdge and SemanticClipNode tables")

    # ── Connect to GCS ──
    log.info(f"Connecting to GCS: gs://{GCS_BUCKET}")
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(GCS_BUCKET)

    # ══════════════════════════════════════════════
    # Step 1: Prepare & ingest SemanticClipNode rows
    # ══════════════════════════════════════════════
    log.info("─" * 50)
    log.info("Preparing SemanticClipNode mutations…")

    node_columns = [
        "node_id",
        "video_uri",
        "start_time_ms",
        "end_time_ms",
        "transcript_text",
        "vocal_delivery",
        "speakers",
        "objects_present",
        "visual_labels",
        "content_mechanisms",
        "embedding",
        "spatial_tracking_uri",
    ]

    node_mutations: list[dict] = []

    for i, (node, node_id) in enumerate(zip(nodes, node_ids)):
        start_s = float(node["start_time"])
        end_s = float(node["end_time"])
        start_ms = int(start_s * 1000)
        end_ms = int(end_s * 1000)

        speakers = _extract_speakers(start_ms, end_ms, audio)
        objects_present = _extract_objects_present(start_ms, end_ms, visual)
        visual_labels = _extract_visual_labels(start_ms, end_ms, visual)
        mechanisms = node.get("content_mechanisms", {})
        embedding = node.get("multimodal_embedding", [])
        tracking_uri = f"gs://{GCS_BUCKET}/{TRACKING_PREFIX}/{node_id}.json"

        node_mutations.append({
            "table": "SemanticClipNode",
            "columns": node_columns,
            "values": (
                node_id,
                VIDEO_GCS_URI,
                start_ms,
                end_ms,
                node.get("transcript_segment", ""),
                node.get("vocal_delivery", ""),
                json.dumps(speakers),
                json.dumps(objects_present),
                json.dumps(visual_labels),
                json.dumps(mechanisms),
                embedding,
                tracking_uri,
            ),
        })

        if (i + 1) % 5 == 0 or i == len(nodes) - 1:
            log.info(f"  Prepared node {i + 1}/{len(nodes)}: {start_s:.1f}s–{end_s:.1f}s "
                     f"speakers={speakers} labels={len(visual_labels)}")

    # Write node mutations in batches
    log.info(f"Writing {len(node_mutations)} node mutations to Spanner…")
    total = len(node_mutations)
    for batch_start in range(0, total, BATCH_SIZE):
        batch = node_mutations[batch_start : batch_start + BATCH_SIZE]
        with database.batch() as txn:
            txn.insert_or_update(
                table="SemanticClipNode",
                columns=node_columns,
                values=[m["values"] for m in batch],
            )
        log.info(f"  [Nodes] Committed batch {batch_start // BATCH_SIZE + 1} "
                 f"({len(batch)} rows, {batch_start + len(batch)}/{total})")

    # ══════════════════════════════════════════════
    # Step 2: Prepare & ingest NarrativeEdge rows
    # ══════════════════════════════════════════════
    log.info("─" * 50)
    log.info("Preparing NarrativeEdge mutations…")

    edge_columns = [
        "edge_id",
        "from_node_id",
        "to_node_id",
        "label",
        "narrative_classification",
        "confidence_score",
    ]

    edge_mutations: list[dict] = []
    skipped = 0

    for edge in edges:
        from_start = edge["from_node_start_time"]
        to_start = edge["to_node_start_time"]

        from_id = start_time_to_id.get(from_start)
        to_id = start_time_to_id.get(to_start)

        if not from_id or not to_id:
            log.warning(f"  Edge skipped — could not resolve start_times: "
                        f"from={from_start}, to={to_start}")
            skipped += 1
            continue

        edge_id = str(uuid.uuid4())
        raw_type = edge.get("edge_type", "")
        label = EDGE_TYPE_MAP.get(raw_type, raw_type.upper().replace(" ", "_").replace("/", "_"))

        edge_mutations.append({
            "table": "NarrativeEdge",
            "columns": edge_columns,
            "values": (
                edge_id,
                from_id,
                to_id,
                label,
                edge.get("narrative_classification", ""),
                edge.get("confidence_score", 0.0),
            ),
        })

    if skipped:
        log.warning(f"  {skipped} edge(s) skipped due to unresolved start_times")

    log.info(f"Writing {len(edge_mutations)} edge mutations to Spanner…")
    total = len(edge_mutations)
    for batch_start in range(0, total, BATCH_SIZE):
        batch = edge_mutations[batch_start : batch_start + BATCH_SIZE]
        with database.batch() as txn:
            txn.insert_or_update(
                table="NarrativeEdge",
                columns=edge_columns,
                values=[m["values"] for m in batch],
            )
        log.info(f"  [Edges] Committed batch {batch_start // BATCH_SIZE + 1} "
                 f"({len(batch)} rows, {batch_start + len(batch)}/{total})")

    # ══════════════════════════════════════════════
    # Step 3: Upload decoupled spatial tracking to GCS
    # ══════════════════════════════════════════════
    log.info("─" * 50)
    log.info("Uploading spatial tracking data to GCS…")

    for i, (node, node_id) in enumerate(zip(nodes, node_ids)):
        start_ms = int(float(node["start_time"]) * 1000)
        end_ms = int(float(node["end_time"]) * 1000)

        tracking = _extract_spatial_tracking(
            start_ms,
            end_ms,
            visual,
        )
        blob_name = f"{TRACKING_PREFIX}/{node_id}.json"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(
            json.dumps(tracking, indent=2),
            content_type="application/json",
        )

        face_count = sum(
            len(f.get("timestamped_objects", []))
            for f in tracking["face_detections"]
        )
        person_count = sum(
            len(p.get("timestamped_objects", []))
            for p in tracking["person_detections"]
        )

        if (i + 1) % 5 == 0 or i == len(nodes) - 1:
            log.info(
                f"  [{i + 1}/{len(nodes)}] gs://{GCS_BUCKET}/{blob_name} "
                f"(faces: {face_count} frames, persons: {person_count} frames)"
            )

    # ── Summary ──
    log.info("=" * 60)
    log.info("PHASE 3 COMPLETE")
    log.info(f"  Spanner nodes written:  {len(node_mutations)}")
    log.info(f"  Spanner edges written:  {len(edge_mutations)}")
    log.info(f"  Edges skipped:          {skipped}")
    log.info(f"  GCS tracking files:     {len(nodes)}")
    log.info(f"  Spanner: {SPANNER_INSTANCE}/{SPANNER_DATABASE}")
    log.info(f"  GCS prefix: gs://{GCS_BUCKET}/{TRACKING_PREFIX}/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
