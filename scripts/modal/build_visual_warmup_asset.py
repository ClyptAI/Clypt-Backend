from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.phase1_runtime.visual_warmup import load_visual_warmup_spec_from_env
from backend.providers.config import StorageSettings
from backend.providers.storage import GCSStorageClient, parse_gcs_uri


def _storage_client_for_uri(gcs_uri: str) -> GCSStorageClient:
    bucket, _ = parse_gcs_uri(gcs_uri)
    return GCSStorageClient(settings=StorageSettings(gcs_bucket=bucket))


def _build_clip(
    *,
    source_video_path: Path,
    output_path: Path,
    clip_start_ms: int,
    clip_end_ms: int,
) -> None:
    start_s = max(0.0, clip_start_ms / 1000.0)
    duration_s = max(0.1, (clip_end_ms - clip_start_ms) / 1000.0)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(source_video_path),
        "-t",
        f"{duration_s:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed while building visual warmup asset:\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create or refresh the canonical short visual warmup asset in GCS."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild and overwrite the target warmup clip even if it already exists.",
    )
    args = parser.parse_args()

    spec = load_visual_warmup_spec_from_env()
    storage_client = _storage_client_for_uri(spec.warmup_video_gcs_uri)

    if storage_client.exists(spec.warmup_video_gcs_uri) and not args.force:
        print(f"Warmup asset already exists: {spec.warmup_video_gcs_uri}")
        return 0

    with tempfile.TemporaryDirectory(prefix="clypt-visual-warmup-build-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        source_video_path = tmp_root / "source_video.mp4"
        warmup_video_path = tmp_root / f"{spec.asset_id}.mp4"
        print(f"Downloading source video: {spec.source_video_gcs_uri}")
        storage_client.download_file(
            gcs_uri=spec.source_video_gcs_uri,
            local_path=source_video_path,
        )
        print(
            "Building warmup clip "
            f"{spec.asset_id} [{spec.clip_start_ms}ms, {spec.clip_end_ms}ms]"
        )
        _build_clip(
            source_video_path=source_video_path,
            output_path=warmup_video_path,
            clip_start_ms=spec.clip_start_ms,
            clip_end_ms=spec.clip_end_ms,
        )
        _, object_name = parse_gcs_uri(spec.warmup_video_gcs_uri)
        uploaded = storage_client.upload_file(
            local_path=warmup_video_path,
            object_name=object_name,
        )
        print(f"Uploaded warmup asset: {uploaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
