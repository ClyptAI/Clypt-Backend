#!/usr/bin/env python3
"""
Phase 4: Storage & Graph Binding (Spanner + GCS)
=================================================
Ingests Phase 1/2 JSON ledgers into Google Cloud Spanner (nodes + edges)
and uploads decoupled spatial tracking data to GCS.

Uses Spanner's native multi-model architecture: ScaNN vector search on
the embedding column, Spanner Graph for edge traversal.

Inputs:
  - phase_3_embeddings.json          (nodes with 3072-d vectors)
  - phase_2b_narrative_edges.json    (directional edges)
  - phase_1_visual.json             (face/person/object/label data)
  - phase_1_audio.json              (word-level timestamps + speakers)

Targets:
  - Spanner: instance=clypt-spanner-v2, database=clypt-graph-db-v2
    - Table: SemanticClipNode
    - Table: NarrativeEdge
  - GCS: gs://clypt-storage-v2/tracking/[node_id].json
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
PROJECT_ID = "clypt-v2"
SPANNER_INSTANCE = "clypt-spanner-v2"
SPANNER_DATABASE = "clypt-graph-db-v2"

GCS_BUCKET = "clypt-storage-v2"
VIDEO_GCS_URI = "gs://clypt-storage-v2/phase_1/video.mp4"
TRACKING_PREFIX = "tracking"

ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = ROOT / "outputs" / "phase_3_embeddings.json"
EDGES_PATH = ROOT / "outputs" / "phase_2b_narrative_edges.json"
VISUAL_PATH = ROOT / "outputs" / "phase_1_visual.json"
AUDIO_PATH = ROOT / "outputs" / "phase_1_audio.json"

BATCH_SIZE = 100
EXPECTED_EMBEDDING_DIM = 3072

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase_4")
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


def _ensure_legacy_visual_from_tracks(visual: dict) -> dict:
    """Build legacy face/person detection blocks from canonical tracks when missing."""
    if not isinstance(visual, dict):
        return {"person_detections": [], "face_detections": [], "object_tracking": [], "label_detections": []}

    has_person = isinstance(visual.get("person_detections"), list)
    has_face = isinstance(visual.get("face_detections"), list)
    if has_person and has_face:
        visual.setdefault("object_tracking", [])
        visual.setdefault("label_detections", [])
        return visual

    tracks = visual.get("tracks", [])
    if not isinstance(tracks, list) or not tracks:
        visual["person_detections"] = visual.get("person_detections", [])
        visual["face_detections"] = visual.get("face_detections", [])
        visual.setdefault("object_tracking", [])
        visual.setdefault("label_detections", [])
        return visual

    meta = visual.get("video_metadata", {}) if isinstance(visual.get("video_metadata"), dict) else {}
    width = int(meta.get("width", 1920) or 1920)
    height = int(meta.get("height", 1080) or 1080)
    fps = float(meta.get("fps", 25.0) or 25.0)

    by_tid: dict[str, list[dict]] = {}
    for t in tracks:
        tid = str(t.get("track_id", ""))
        if not tid:
            continue
        by_tid.setdefault(tid, []).append(t)
    for tid in list(by_tid.keys()):
        by_tid[tid].sort(key=lambda x: int(x.get("frame_idx", -1)))

    person_dets = []
    face_dets = []
    for idx, (tid, dets) in enumerate(sorted(by_tid.items())):
        person_ts = []
        face_ts = []
        for d in dets:
            fi = int(d.get("frame_idx", 0))
            t_ms = int(round((fi / max(1e-6, fps)) * 1000.0))
            x1 = float(d.get("x1", 0.0))
            y1 = float(d.get("y1", 0.0))
            x2 = float(d.get("x2", x1 + 1.0))
            y2 = float(d.get("y2", y1 + 1.0))
            bbox = {
                "left": max(0.0, min(1.0, x1 / max(1, width))),
                "top": max(0.0, min(1.0, y1 / max(1, height))),
                "right": max(0.0, min(1.0, x2 / max(1, width))),
                "bottom": max(0.0, min(1.0, y2 / max(1, height))),
            }
            person_ts.append({"time_ms": t_ms, "bounding_box": bbox})

            bw = max(1e-6, bbox["right"] - bbox["left"])
            bh = max(1e-6, bbox["bottom"] - bbox["top"])
            face_bbox = {
                "left": max(0.0, min(1.0, bbox["left"] + 0.18 * bw)),
                "right": max(0.0, min(1.0, bbox["right"] - 0.18 * bw)),
                "top": max(0.0, min(1.0, bbox["top"] + 0.02 * bh)),
                "bottom": max(0.0, min(1.0, bbox["top"] + 0.48 * bh)),
            }
            face_ts.append({"time_ms": t_ms, "bounding_box": face_bbox})

        if person_ts:
            person_dets.append(
                {
                    "confidence": float(sum(float(d.get("confidence", 0.0)) for d in dets) / max(1, len(dets))),
                    "segment_start_ms": int(person_ts[0]["time_ms"]),
                    "segment_end_ms": int(person_ts[-1]["time_ms"]),
                    "person_track_index": idx,
                    "track_id": tid,
                    "timestamped_objects": person_ts,
                }
            )
        if face_ts:
            face_dets.append(
                {
                    "confidence": float(sum(float(d.get("confidence", 0.0)) for d in dets) / max(1, len(dets))),
                    "segment_start_ms": int(face_ts[0]["time_ms"]),
                    "segment_end_ms": int(face_ts[-1]["time_ms"]),
                    "face_track_index": idx,
                    "track_id": tid,
                    "timestamped_objects": face_ts,
                }
            )

    visual["person_detections"] = person_dets
    visual["face_detections"] = face_dets
    visual.setdefault("object_tracking", [])
    visual.setdefault("label_detections", [])
    return visual


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


def _table_exists(database, table_name: str) -> bool:
    sql = """
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_NAME = @table_name
    """
    params = {"table_name": table_name}
    param_types = {"table_name": spanner.param_types.STRING}
    with database.snapshot() as snapshot:
        rows = list(snapshot.execute_sql(sql, params=params, param_types=param_types))
    return len(rows) > 0


def _table_columns(database, table_name: str) -> set[str]:
    sql = """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = @table_name
    """
    params = {"table_name": table_name}
    param_types = {"table_name": spanner.param_types.STRING}
    with database.snapshot() as snapshot:
        rows = list(snapshot.execute_sql(sql, params=params, param_types=param_types))
    return {str(r[0]) for r in rows}


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("PHASE 4 — Storage & Graph Binding")
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
    visual = _ensure_legacy_visual_from_tracks(visual)
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

    semantic_exists = _table_exists(database, "SemanticClipNode")
    edge_exists = _table_exists(database, "NarrativeEdge")
    if not semantic_exists:
        raise RuntimeError("SemanticClipNode table not found in target database")
    if not edge_exists:
        raise RuntimeError("NarrativeEdge table not found in target database")

    semantic_cols = _table_columns(database, "SemanticClipNode")
    edge_cols = _table_columns(database, "NarrativeEdge")
    log.info(f"SemanticClipNode columns: {sorted(semantic_cols)}")
    log.info(f"NarrativeEdge columns: {sorted(edge_cols)}")

    # ── Clear stale data from previous runs ──
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

    embedding_col = "embedding"
    candidate_node_values = {
        "node_id": lambda ctx: ctx["node_id"],
        "video_uri": lambda ctx: VIDEO_GCS_URI,
        "start_time_ms": lambda ctx: ctx["start_ms"],
        "end_time_ms": lambda ctx: ctx["end_ms"],
        "transcript_text": lambda ctx: ctx["node"].get("transcript_segment", ""),
        "visual_description": lambda ctx: ctx["node"].get("visual_description", ""),
        "vocal_delivery": lambda ctx: ctx["node"].get("vocal_delivery", ""),
        "speakers": lambda ctx: json.dumps(ctx["speakers"]),
        "objects_present": lambda ctx: json.dumps(ctx["objects_present"]),
        "visual_labels": lambda ctx: json.dumps(ctx["visual_labels"]),
        "content_mechanisms": lambda ctx: json.dumps(ctx["mechanisms"]),
        "embedding": lambda ctx: ctx["embedding"],
        "multimodal_embedding": lambda ctx: ctx["embedding"],
        "spatial_tracking_uri": lambda ctx: ctx["tracking_uri"],
    }
    node_columns = [c for c in candidate_node_values.keys() if c in semantic_cols]
    if "node_id" not in node_columns:
        raise RuntimeError("SemanticClipNode missing required node_id column")
    if embedding_col not in node_columns:
        raise RuntimeError("SemanticClipNode missing required embedding column")

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
        if not isinstance(embedding, list) or len(embedding) != EXPECTED_EMBEDDING_DIM:
            raise RuntimeError(
                f"Node {i + 1} has invalid embedding size "
                f"({len(embedding) if isinstance(embedding, list) else 'non-list'}); "
                f"expected {EXPECTED_EMBEDDING_DIM}."
            )
        tracking_uri = f"gs://{GCS_BUCKET}/{TRACKING_PREFIX}/{node_id}.json"

        ctx = {
            "node_id": node_id,
            "node": node,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "speakers": speakers,
            "objects_present": objects_present,
            "visual_labels": visual_labels,
            "mechanisms": mechanisms,
            "embedding": embedding,
            "tracking_uri": tracking_uri,
        }
        node_values = tuple(candidate_node_values[c](ctx) for c in node_columns)
        node_mutations.append({
            "table": "SemanticClipNode",
            "columns": node_columns,
            "values": node_values,
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

    all_edge_columns = [
        "edge_id",
        "from_node_id",
        "to_node_id",
        "label",
        "narrative_classification",
        "confidence_score",
    ]
    edge_columns = [c for c in all_edge_columns if c in edge_cols]
    if set(all_edge_columns) - set(edge_columns):
        missing = sorted(set(all_edge_columns) - set(edge_columns))
        raise RuntimeError(f"NarrativeEdge missing required columns: {missing}")

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

        edge_ctx = {
            "edge_id": edge_id,
            "from_node_id": from_id,
            "to_node_id": to_id,
            "label": label,
            "narrative_classification": edge.get("narrative_classification", ""),
            "confidence_score": edge.get("confidence_score", 0.0),
        }
        edge_mutations.append({
            "table": "NarrativeEdge",
            "columns": edge_columns,
            "values": tuple(edge_ctx[c] for c in edge_columns),
        })

    if skipped:
        log.warning(f"  {skipped} edge(s) skipped due to unresolved start_times")

    if not edge_mutations:
        raise RuntimeError("No NarrativeEdge rows were generated; refusing degraded graph ingestion.")

    log.info(f"Writing {len(edge_mutations)} edge mutations to Spanner…")
    total = len(edge_mutations)
    for batch_start in range(0, total, BATCH_SIZE):
        batch = edge_mutations[batch_start : batch_start + BATCH_SIZE]
        if not batch:
            continue
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
    log.info("PHASE 4 COMPLETE")
    log.info(f"  Spanner nodes written:  {len(node_mutations)}")
    log.info(f"  Spanner edges written:  {len(edge_mutations)}")
    log.info(f"  Edges skipped:          {skipped}")
    log.info(f"  GCS tracking files:     {len(nodes)}")
    log.info(f"  Spanner: {SPANNER_INSTANCE}/{SPANNER_DATABASE}")
    log.info(f"  GCS prefix: gs://{GCS_BUCKET}/{TRACKING_PREFIX}/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
