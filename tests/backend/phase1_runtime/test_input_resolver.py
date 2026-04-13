from __future__ import annotations

from pathlib import Path

import pytest


def test_phase1_input_resolver_loads_json_mapping_and_resolves_relative_path(tmp_path: Path):
    from backend.phase1_runtime.input_resolver import Phase1InputResolver

    video_path = tmp_path / "videos" / "test-video.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_text("video", encoding="utf-8")
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        """
{
  "https://youtube.com/watch?v=test": {
    "local_video_path": "videos/test-video.mp4"
  }
}
""".strip(),
        encoding="utf-8",
    )

    resolver = Phase1InputResolver.from_mapping_file(mapping_path)

    assert resolver.resolve_source_path(source_url="https://youtube.com/watch?v=test") == video_path.resolve()


def test_phase1_input_resolver_loads_yaml_mapping(tmp_path: Path):
    from backend.phase1_runtime.input_resolver import Phase1InputResolver

    video_path = tmp_path / "fixtures" / "clip.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_text("video", encoding="utf-8")
    mapping_path = tmp_path / "mapping.yml"
    mapping_path.write_text(
        """
https://youtube.com/watch?v=abc123:
  local_video_path: fixtures/clip.mp4
""".strip(),
        encoding="utf-8",
    )

    resolver = Phase1InputResolver.from_mapping_file(mapping_path)

    assert resolver.resolve_source_path(source_url="https://youtube.com/watch?v=abc123") == video_path.resolve()


def test_phase1_input_resolver_rejects_unmapped_source_url(tmp_path: Path):
    from backend.phase1_runtime.input_resolver import Phase1InputResolutionError, Phase1InputResolver

    video_path = tmp_path / "clip.mp4"
    video_path.write_text("video", encoding="utf-8")
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        """
{
  "https://youtube.com/watch?v=test": {
    "local_video_path": "clip.mp4"
  }
}
""".strip(),
        encoding="utf-8",
    )

    resolver = Phase1InputResolver.from_mapping_file(mapping_path)

    with pytest.raises(Phase1InputResolutionError, match="No test-bank mapping found"):
        resolver.resolve_source_path(source_url="https://youtube.com/watch?v=missing")


def test_phase1_input_resolver_allows_missing_local_when_gcs_uri_present(tmp_path: Path):
    from backend.phase1_runtime.input_resolver import Phase1InputResolver

    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        """
{
  "https://youtu.be/abc123?si=test": {
    "local_video_path": "cache/abc123.mp4",
    "local_audio_path": "cache/abc123.wav",
    "video_gcs_uri": "gs://bucket/test-bank/abc123.mp4",
    "audio_gcs_uri": "gs://bucket/test-bank/abc123.wav"
  }
}
""".strip(),
        encoding="utf-8",
    )

    resolver = Phase1InputResolver.from_mapping_file(mapping_path)
    asset = resolver.resolve_source_asset(source_url="https://youtu.be/abc123?si=different")

    assert asset.local_video_path == (tmp_path / "cache" / "abc123.mp4").resolve()
    assert asset.local_audio_path == (tmp_path / "cache" / "abc123.wav").resolve()
    assert asset.video_gcs_uri == "gs://bucket/test-bank/abc123.mp4"
    assert asset.audio_gcs_uri == "gs://bucket/test-bank/abc123.wav"
