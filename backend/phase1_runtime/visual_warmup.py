from __future__ import annotations

from dataclasses import asdict, dataclass
import os


@dataclass(frozen=True)
class VisualWarmupSpec:
    asset_id: str
    source_test_bank_url: str
    source_video_gcs_uri: str
    warmup_video_gcs_uri: str
    clip_start_ms: int
    clip_end_ms: int
    min_emitted_track_rows: int = 1
    min_pose_validated_tracklets: int = 1

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_VISUAL_WARMUP_SPEC = VisualWarmupSpec(
    asset_id="visual_people_warmup_v1",
    source_test_bank_url="https://youtu.be/64qBE35S0ek?si=bul2StVGVzUE8EL6",
    source_video_gcs_uri="gs://clypt-storage-v3/test-bank/canonical/videos/joeroganflagrant.mp4",
    warmup_video_gcs_uri="gs://clypt-storage-v3/test-bank/warmups/visual_people_warmup_v1.mp4",
    clip_start_ms=694000,
    clip_end_ms=706000,
)


def load_visual_warmup_spec_from_env() -> VisualWarmupSpec:
    base = DEFAULT_VISUAL_WARMUP_SPEC
    return VisualWarmupSpec(
        asset_id=os.environ.get("CLYPT_PHASE1_VISUAL_WARMUP_ASSET_ID", base.asset_id).strip(),
        source_test_bank_url=os.environ.get(
            "CLYPT_PHASE1_VISUAL_WARMUP_SOURCE_TEST_BANK_URL",
            base.source_test_bank_url,
        ).strip(),
        source_video_gcs_uri=os.environ.get(
            "CLYPT_PHASE1_VISUAL_WARMUP_SOURCE_VIDEO_GCS_URI",
            base.source_video_gcs_uri,
        ).strip(),
        warmup_video_gcs_uri=os.environ.get(
            "CLYPT_PHASE1_VISUAL_WARMUP_GCS_URI",
            base.warmup_video_gcs_uri,
        ).strip(),
        clip_start_ms=int(
            os.environ.get("CLYPT_PHASE1_VISUAL_WARMUP_CLIP_START_MS", str(base.clip_start_ms))
        ),
        clip_end_ms=int(
            os.environ.get("CLYPT_PHASE1_VISUAL_WARMUP_CLIP_END_MS", str(base.clip_end_ms))
        ),
        min_emitted_track_rows=int(
            os.environ.get(
                "CLYPT_PHASE1_VISUAL_WARMUP_MIN_EMITTED_TRACK_ROWS",
                str(base.min_emitted_track_rows),
            )
        ),
        min_pose_validated_tracklets=int(
            os.environ.get(
                "CLYPT_PHASE1_VISUAL_WARMUP_MIN_POSE_VALIDATED_TRACKLETS",
                str(base.min_pose_validated_tracklets),
            )
        ),
    )


__all__ = [
    "DEFAULT_VISUAL_WARMUP_SPEC",
    "VisualWarmupSpec",
    "load_visual_warmup_spec_from_env",
]
