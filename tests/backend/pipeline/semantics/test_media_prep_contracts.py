from __future__ import annotations

import pytest


def test_media_prep_request_requires_items() -> None:
    from backend.pipeline.semantics.media_prep_contracts import NodeMediaPrepRequest

    with pytest.raises(ValueError, match="items must contain at least one node media request"):
        NodeMediaPrepRequest(
            run_id="run-1",
            source_video_gcs_uri="gs://bucket/video.mp4",
            object_prefix="phase14/run-1/node_media",
            items=[],
        )


def test_media_prep_item_requires_non_decreasing_window() -> None:
    from backend.pipeline.semantics.media_prep_contracts import NodeMediaPrepItem

    with pytest.raises(ValueError, match="start_ms must be <= end_ms"):
        NodeMediaPrepItem(node_id="node-1", start_ms=200, end_ms=100)


def test_media_prep_response_preserves_item_order() -> None:
    from backend.pipeline.semantics.media_prep_contracts import (
        NodeMediaDescriptor,
        NodeMediaPrepResponse,
    )

    response = NodeMediaPrepResponse(
        run_id="run-1",
        items=[
            NodeMediaDescriptor(node_id="node_a", file_uri="gs://bucket/node_a.mp4"),
            NodeMediaDescriptor(node_id="node_b", file_uri="gs://bucket/node_b.mp4"),
        ],
    )

    assert [item.node_id for item in response.items] == ["node_a", "node_b"]
