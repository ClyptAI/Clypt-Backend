#!/usr/bin/env python3
"""
Clypt Pipeline Orchestrator
============================
Prompts for a YouTube URL, then runs the full pipeline sequentially:

  Phase 1A  →  Deterministic Extraction (yt-dlp + Video Intelligence + STT)
  FFmpeg    →  Re-encode video for Remotion compatibility
  Phase 1B  →  Content Mechanism Decomposition (Gemini chunked multimodal)
  Phase 1C  →  Narrative Edge Mapping (Gemini text-only)
  Phase 2   →  Multimodal Embedding (multimodalembedding@001)
  Phase 3   →  Storage & Graph Binding (Spanner + GCS)
  Phase 4   →  Auto-Curate (full-graph sweep + Gemini scoring)
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
ENABLE_ASD_V2 = os.getenv("ENABLE_ASD_V2", "1") == "1"
RUN_TRUE_ASD_INFER = os.getenv("RUN_TRUE_ASD_INFER", "0") == "1"
ASD_INFER_REUSE_EXISTING = os.getenv("ASD_INFER_REUSE_EXISTING", "1") == "1"

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
            f"Missing source video: {video_src}. Run Phase 1A first."
        )
    if not payload_src.exists():
        raise FileNotFoundError(
            f"Missing remotion payload: {payload_src}. "
            "Run Phase 4 (auto-curate) before rendering."
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

    # ── Phase 1A: Deterministic Extraction ──
    from pipeline.phase_1a_extract import main as phase_1a_main
    asyncio.run(phase_1a_main(youtube_url=url))

    # ── Phase 1A-R: Speaker-to-Face Reconciliation ──
    from pipeline.phase_1a_reconcile import main as phase_1a_reconcile_main
    phase_1a_reconcile_main()

    # ── Phase 1A-ASD-INFER / Phase 1A-ASD / Phase 1A-FUSE ──
    if ENABLE_ASD_V2:
        if RUN_TRUE_ASD_INFER:
            argv_backup = sys.argv[:]
            try:
                from pipeline.phase_1a_asd_infer import main as phase_1a_asd_infer_main
                sys.argv = [sys.argv[0]]
                if ASD_INFER_REUSE_EXISTING:
                    sys.argv.append("--reuse-existing")
                phase_1a_asd_infer_main()
                sys.argv = argv_backup
            except Exception as e:
                sys.argv = argv_backup
                log.warning(
                    "Phase 1A-ASD-INFER failed (%s). Continuing with fallback ASD fusion.",
                    e,
                )
        else:
            log.info("Skipping true ASD inference (RUN_TRUE_ASD_INFER=0)")

        try:
            from pipeline.phase_1a_asd import main as phase_1a_asd_main
            phase_1a_asd_main()
        except Exception as e:
            log.warning("Phase 1A-ASD failed (%s). Continuing with STT-only fusion.", e)

        try:
            from pipeline.phase_1a_fuse import main as phase_1a_fuse_main
            phase_1a_fuse_main()
        except Exception as e:
            log.warning("Phase 1A-FUSE failed (%s). Render will use existing speaker data only.", e)
    else:
        log.info("ASD v2 disabled (ENABLE_ASD_V2=0)")

    # ── FFmpeg Re-encode ──
    banner("RE-ENCODING VIDEO (FFmpeg)")
    reencode_video()

    # ── Phase 1B: Content Mechanism Decomposition ──
    from pipeline.phase_1b_decompose import main as phase_1b_main
    phase_1b_main()

    # ── Phase 1C: Narrative Edge Mapping ──
    from pipeline.phase_1c_edges import main as phase_1c_main
    phase_1c_main()

    # ── Phase 2: Multimodal Embedding ──
    from pipeline.phase_2_embed import main as phase_2_main
    phase_2_main()

    # ── Phase 3: Storage & Graph Binding ──
    from pipeline.phase_3_store import main as phase_3_main
    phase_3_main()

    # ── Phase 4: Auto-Curate ──
    from pipeline.phase_4_auto_curate import main as phase_4_main
    phase_4_main()

    # ── Remotion Render ──
    banner("REMOTION RENDER")
    setup_render_engine()
    run_fetch_tracking()
    run_remotion_render()

    banner("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
