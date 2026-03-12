#!/usr/bin/env python3
"""
Clypt Pipeline Orchestrator
============================
Prompts for a YouTube URL, then runs the full pipeline sequentially:

  Phase 1   →  Modal deterministic extraction
  FFmpeg    →  Re-encode video for Remotion compatibility
  Phase 2A  →  Content Mechanism Decomposition (Gemini chunked multimodal)
  Phase 2B  →  Narrative Edge Mapping (Gemini text-only)
  Phase 3   →  Multimodal Embedding (Gemini Embedding 2)
  Phase 4   →  Storage & Graph Binding (Spanner + GCS)
  Phase 5   →  Auto-Curate (full-graph sweep + Gemini scoring)
  Render    →  Remotion render (fetch tracking + render all compositions)
"""

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = ROOT / "pipeline"
RENDER_DIR = ROOT / "clypt-render-engine"
DOWNLOADS_DIR = ROOT / "downloads"
OUTPUTS_DIR = ROOT / "outputs"
FFMPEG_REENCODE_CRF = os.getenv("FFMPEG_REENCODE_CRF", "15")
FFMPEG_REENCODE_PRESET = os.getenv("FFMPEG_REENCODE_PRESET", "slow")
REMOTION_CRF = os.getenv("REMOTION_CRF", "16")
REMOTION_X264_PRESET = os.getenv("REMOTION_X264_PRESET", "slow")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("orchestrator")


def banner(text: str):
    log.info("=" * 60)
    log.info(text)
    log.info("=" * 60)


def reencode_video():
    """Re-encode the downloaded video to clean H.264 for Remotion compatibility.

    yt-dlp's merged output can have encoding quirks (misaligned keyframes,
    non-standard NAL units) that cause Chrome Headless Shell to throw
    PIPELINE_ERROR_DECODE during Remotion rendering. A quick FFmpeg pass
    produces a file that decodes cleanly.
    """
    video_path = DOWNLOADS_DIR / "video.mp4"
    clean_path = DOWNLOADS_DIR / "video_clean.mp4"

    if not video_path.exists():
        log.warning("No video.mp4 found in downloads/ — skipping re-encode")
        return

    if not shutil.which("ffmpeg"):
        log.warning("ffmpeg not found on PATH — skipping re-encode (Remotion may glitch)")
        return

    log.info("Re-encoding video.mp4 for Remotion compatibility…")
    log.info(f"  libx264 preset={FFMPEG_REENCODE_PRESET}, crf={FFMPEG_REENCODE_CRF}")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-c:v", "libx264", "-preset", FFMPEG_REENCODE_PRESET, "-crf", FFMPEG_REENCODE_CRF,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(clean_path),
        ],
        check=True,
        capture_output=True,
    )

    original = DOWNLOADS_DIR / "video_original.mp4"
    video_path.rename(original)
    clean_path.rename(video_path)
    size_mb = video_path.stat().st_size / 1e6
    log.info(f"Re-encode complete: {size_mb:.1f} MB (original saved as video_original.mp4)")


def setup_render_engine():
    """Link pipeline outputs into the Remotion project so it can find them.

    The video uses a hard link (or copy fallback) because Remotion's render
    bundler doesn't reliably resolve symlinks pointing outside the project.
    The payload JSON uses a symlink since it's small and loaded via require().
    """

    video_src = DOWNLOADS_DIR / "video.mp4"
    video_dst = RENDER_DIR / "public" / "video.mp4"
    payload_src = OUTPUTS_DIR / "remotion_payloads_array.json"
    payload_dst = RENDER_DIR / "src" / "remotion_payloads_array.json"

    if not video_src.exists():
        raise FileNotFoundError(
            f"Missing source video: {video_src}. Run Phase 1 first."
        )
    if not payload_src.exists():
        raise FileNotFoundError(
            f"Missing remotion payload: {payload_src}. "
            "Run Phase 5 (auto-curate) before rendering."
        )

    # ── Video: hard link (instant, no extra disk) with copy fallback ──
    if video_dst.is_symlink() or video_dst.exists():
        video_dst.unlink()
    try:
        os.link(video_src, video_dst)
        log.info(f"Hard-linked {video_dst.name} → {video_src}")
    except OSError:
        shutil.copy2(video_src, video_dst)
        log.info(f"Copied {video_dst.name} ← {video_src}")

    # ── Payload JSON: symlink is fine (loaded via require, not served) ──
    if payload_dst.is_symlink() or payload_dst.exists():
        payload_dst.unlink()
    rel = os.path.relpath(payload_src, payload_dst.parent)
    payload_dst.symlink_to(rel)
    log.info(f"Symlinked {payload_dst.name} → {rel}")


def run_fetch_tracking():
    """Run the Node.js script that downloads spatial tracking data from GCS."""
    script = RENDER_DIR / "scripts" / "fetch_tracking.js"
    if not script.exists():
        log.warning("fetch_tracking.js not found — skipping")
        return

    log.info("Fetching tracking data from GCS…")
    subprocess.run(
        ["node", str(script)],
        check=True,
        cwd=str(RENDER_DIR),
    )
    log.info("Tracking data fetched.")


def run_remotion_render():
    """Render each Remotion composition to an MP4."""
    import json

    payload_path = OUTPUTS_DIR / "remotion_payloads_array.json"
    if not payload_path.exists():
        log.error("No remotion_payloads_array.json found — cannot render")
        return

    payloads = json.loads(payload_path.read_text())
    if not isinstance(payloads, list):
        payloads = [payloads]

    out_dir = RENDER_DIR / "out"
    out_dir.mkdir(exist_ok=True)

    for i in range(len(payloads)):
        if len(payloads) == 1:
            comp_id = "ClyptViralShort"
            out_file = out_dir / "clip.mp4"
        else:
            comp_id = f"ClyptViralShort-{i + 1}"
            out_file = out_dir / f"clip-{i + 1}.mp4"

        log.info(f"Rendering {comp_id} → {out_file.name}")
        subprocess.run(
            [
                "npx",
                "remotion",
                "render",
                comp_id,
                str(out_file),
                "--crf",
                REMOTION_CRF,
                "--x264-preset",
                REMOTION_X264_PRESET,
            ],
            check=True,
            cwd=str(RENDER_DIR),
        )
        log.info(f"  ✓ {out_file.name} ({out_file.stat().st_size / 1e6:.1f} MB)")

    log.info(f"All {len(payloads)} clip(s) rendered to {out_dir}")


def main():
    banner("CLYPT PIPELINE ORCHESTRATOR")

    url = input("\nEnter YouTube URL: ").strip()
    if not url:
        log.error("No URL provided. Exiting.")
        sys.exit(1)

    log.info(f"Target: {url}\n")

    # ── Phase 1: Modal deterministic extraction ──
    from pipeline.phase_1_modal_pipeline import main as phase_1_main
    asyncio.run(phase_1_main(youtube_url=url))

    # ── FFmpeg Re-encode ──
    banner("RE-ENCODING VIDEO (FFmpeg)")
    reencode_video()

    # ── Phase 2A: Content Mechanism Decomposition ──
    from pipeline.phase_2a_make_nodes import main as phase_2a_main
    phase_2a_main()

    # ── Phase 2B: Narrative Edge Mapping ──
    from pipeline.phase_2b_draw_edges import main as phase_2b_main
    phase_2b_main()

    # ── Phase 3: Multimodal Embedding ──
    from pipeline.phase_3_multimodal_embeddings import main as phase_3_main
    phase_3_main()

    # ── Phase 4: Storage & Graph Binding ──
    from pipeline.phase_4_store_graph import main as phase_4_main
    phase_4_main()

    # ── Phase 5: Auto-Curate ──
    from pipeline.phase_5_auto_curate import main as phase_5_main
    phase_5_main()

    # ── Remotion Render ──
    banner("REMOTION RENDER")
    setup_render_engine()
    run_fetch_tracking()
    run_remotion_render()

    banner("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
