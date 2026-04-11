#!/usr/bin/env python3
"""Ensure Spanner schema contains signal/provenance tables and candidate columns."""

from __future__ import annotations

import argparse

import google.cloud.spanner as spanner

_TABLE_DDLS: dict[str, str] = {
    "external_signals": """
CREATE TABLE external_signals (
    run_id STRING(128) NOT NULL,
    signal_id STRING(128) NOT NULL,
    signal_type STRING(32) NOT NULL,
    source_platform STRING(32) NOT NULL,
    source_id STRING(MAX) NOT NULL,
    author_id STRING(MAX),
    text STRING(MAX) NOT NULL,
    engagement_score FLOAT64 NOT NULL,
    published_at TIMESTAMP,
    metadata_json STRING(MAX)
) PRIMARY KEY (run_id, signal_id)
""".strip(),
    "external_signal_clusters": """
CREATE TABLE external_signal_clusters (
    run_id STRING(128) NOT NULL,
    cluster_id STRING(128) NOT NULL,
    cluster_type STRING(32) NOT NULL,
    summary_text STRING(MAX) NOT NULL,
    member_signal_ids ARRAY<STRING(128)>,
    cluster_weight FLOAT64 NOT NULL,
    embedding ARRAY<FLOAT32>,
    metadata_json STRING(MAX)
) PRIMARY KEY (run_id, cluster_id)
""".strip(),
    "node_signal_links": """
CREATE TABLE node_signal_links (
    run_id STRING(128) NOT NULL,
    node_id STRING(128) NOT NULL,
    cluster_id STRING(128) NOT NULL,
    link_type STRING(16) NOT NULL,
    hop_distance INT64 NOT NULL,
    time_offset_ms INT64 NOT NULL,
    similarity FLOAT64 NOT NULL,
    link_score FLOAT64 NOT NULL,
    evidence_json STRING(MAX),
    CONSTRAINT fk_node_signal_links_node FOREIGN KEY (run_id, node_id)
      REFERENCES semantic_nodes (run_id, node_id),
    CONSTRAINT fk_node_signal_links_cluster FOREIGN KEY (run_id, cluster_id)
      REFERENCES external_signal_clusters (run_id, cluster_id)
) PRIMARY KEY (run_id, node_id, cluster_id)
""".strip(),
    "candidate_signal_links": """
CREATE TABLE candidate_signal_links (
    run_id STRING(128) NOT NULL,
    clip_id STRING(128) NOT NULL,
    cluster_id STRING(128) NOT NULL,
    cluster_type STRING(32) NOT NULL,
    aggregated_link_score FLOAT64 NOT NULL,
    coverage_ms INT64 NOT NULL,
    direct_node_count INT64 NOT NULL,
    inferred_node_count INT64 NOT NULL,
    agreement_flags ARRAY<STRING(32)>,
    bonus_applied FLOAT64 NOT NULL,
    evidence_json STRING(MAX),
    CONSTRAINT fk_candidate_signal_links_clip FOREIGN KEY (run_id, clip_id)
      REFERENCES clip_candidates (run_id, clip_id),
    CONSTRAINT fk_candidate_signal_links_cluster FOREIGN KEY (run_id, cluster_id)
      REFERENCES external_signal_clusters (run_id, cluster_id)
) PRIMARY KEY (run_id, clip_id, cluster_id)
""".strip(),
    "prompt_source_links": """
CREATE TABLE prompt_source_links (
    run_id STRING(128) NOT NULL,
    prompt_id STRING(128) NOT NULL,
    prompt_source_type STRING(16) NOT NULL,
    source_cluster_id STRING(128),
    source_cluster_type STRING(32),
    metadata_json STRING(MAX),
    CONSTRAINT fk_prompt_source_links_cluster FOREIGN KEY (run_id, source_cluster_id)
      REFERENCES external_signal_clusters (run_id, cluster_id)
) PRIMARY KEY (run_id, prompt_id)
""".strip(),
    "subgraph_provenance": """
CREATE TABLE subgraph_provenance (
    run_id STRING(128) NOT NULL,
    subgraph_id STRING(128) NOT NULL,
    seed_source_set ARRAY<STRING(16)>,
    seed_prompt_ids ARRAY<STRING(128)>,
    source_cluster_ids ARRAY<STRING(128)>,
    support_summary_json STRING(MAX),
    canonical_selected BOOL NOT NULL,
    dedupe_overlap_ratio FLOAT64,
    selection_reason STRING(128),
    metadata_json STRING(MAX)
) PRIMARY KEY (run_id, subgraph_id)
""".strip(),
}

_CLIP_CANDIDATE_COLUMN_DDLS = {
    "external_signal_score": "ALTER TABLE clip_candidates ADD COLUMN external_signal_score FLOAT64",
    "agreement_bonus": "ALTER TABLE clip_candidates ADD COLUMN agreement_bonus FLOAT64",
    "external_attribution_json": "ALTER TABLE clip_candidates ADD COLUMN external_attribution_json STRING(MAX)",
}


def _existing_tables(database: spanner.Database) -> set[str]:
    with database.snapshot() as snapshot:
        rows = snapshot.execute_sql(
            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA=''"
        )
        return {str(row[0]) for row in rows}


def _existing_columns(database: spanner.Database, *, table_name: str) -> set[str]:
    with database.snapshot() as snapshot:
        rows = snapshot.execute_sql(
            f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='{table_name}'"
        )
        return {str(row[0]) for row in rows}


def _apply_statement(database: spanner.Database, *, statement: str, idx: int) -> None:
    operation = database.update_ddl([statement], operation_id=f"phase24_signal_schema_{idx:02d}")
    operation.result(timeout=900.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensure Phase 2-4 signal/provenance schema in Spanner.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--database", required=True)
    args = parser.parse_args()

    client = spanner.Client(project=args.project)
    database = client.instance(args.instance).database(args.database)

    existing_tables = _existing_tables(database)
    statements: list[str] = []

    for table_name, ddl in _TABLE_DDLS.items():
        if table_name not in existing_tables:
            statements.append(ddl)

    if "clip_candidates" in existing_tables:
        existing_clip_columns = _existing_columns(database, table_name="clip_candidates")
        for column_name, ddl in _CLIP_CANDIDATE_COLUMN_DDLS.items():
            if column_name not in existing_clip_columns:
                statements.append(ddl)

    if not statements:
        print("No schema changes needed.")
        return 0

    for index, statement in enumerate(statements, start=1):
        preview = statement.replace("\n", " ")[:120]
        print(f"[{index}/{len(statements)}] Applying: {preview}...")
        _apply_statement(database, statement=statement, idx=index)

    print("Schema migration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
