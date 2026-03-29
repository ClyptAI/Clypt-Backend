#!/usr/bin/env python3
"""
Clypt Pipeline Orchestrator
============================
Prompts for a YouTube URL, then runs the full pipeline sequentially:

  Phase 1   →  DigitalOcean async extraction
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
RENDER_DIR = ROOT.parent / "remotion-render"
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


def _truthy_env(name: str) -> bool:
    return str(os.getenv(name, "") or "").strip().lower() in {"1", "true", "yes", "on"}


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
    """Link pipeline outputs into the Remotion project so it can find them."""
    video_src = DOWNLOADS_DIR / "video.mp4"
    video_dst = RENDER_DIR / "public" / "video.mp4"
    payload_filename = (
        "remotion_payloads_array_audience.json"
        if _truthy_env("USE_AUDIENCE_SIGNAL_PAYLOADS")
        else "remotion_payloads_array.json"
    )
    payload_src = OUTPUTS_DIR / payload_filename
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


PHASE_ORDER = ["1", "2a", "2b", "3", "4", "5", "render"]


def _phase_enabled(phase: str, start_from: str) -> bool:
    try:
        return PHASE_ORDER.index(phase) >= PHASE_ORDER.index(start_from)
    except ValueError:
        return True


def main():
    banner("CLYPT PIPELINE ORCHESTRATOR")

    start_from = os.getenv("START_FROM", "1").lower().strip()
    if start_from not in PHASE_ORDER:
        log.error(f"Invalid START_FROM='{start_from}'. Valid values: {PHASE_ORDER}")
        sys.exit(1)
    if start_from != "1":
        log.info(f"START_FROM={start_from} — skipping earlier phases")

    url = ""
    if _phase_enabled("1", start_from):
        url = input("\nEnter YouTube URL: ").strip()
        if not url:
            log.error("No URL provided. Exiting.")
            sys.exit(1)
        log.info(f"Target: {url}\n")

    # ── Phase 1: DigitalOcean async extraction ──
    if _phase_enabled("1", start_from):
        from pipeline.phase_1_do_pipeline import main as phase_1_main
        phase_1_manifest = asyncio.run(phase_1_main(youtube_url=url))
        log.info(
            "Phase 1 manifest ready: job_id=%s video=%s transcript=%s visual=%s",
            phase_1_manifest.job_id,
            phase_1_manifest.canonical_video_gcs_uri,
            phase_1_manifest.artifacts.transcript.uri,
            phase_1_manifest.artifacts.visual_tracking.uri,
        )

    # ── FFmpeg Re-encode ──
    if _phase_enabled("1", start_from):
        banner("RE-ENCODING VIDEO (FFmpeg)")
        reencode_video()

    # ── Phase 2A: Content Mechanism Decomposition ──
    if _phase_enabled("2a", start_from):
        from pipeline.phase_2a_make_nodes import main as phase_2a_main
        phase_2a_main()

    # ── Phase 2B: Narrative Edge Mapping ──
    if _phase_enabled("2b", start_from):
        from pipeline.phase_2b_draw_edges import main as phase_2b_main
        phase_2b_main()

    # ── Phase 3: Multimodal Embedding ──
    if _phase_enabled("3", start_from):
        from pipeline.phase_3_multimodal_embeddings import main as phase_3_main
        phase_3_main()

    # ── Phase 4: Storage & Graph Binding ──
    if _phase_enabled("4", start_from):
        from pipeline.phase_4_store_graph import main as phase_4_main
        phase_4_main()

    # ── Phase 5: Auto-Curate ──
    if _phase_enabled("5", start_from):
        from pipeline.phase_5_auto_curate import main as phase_5_main
        phase_5_main()

    # —— Audience Signal Layer (optional, post-publish) ——
    if _phase_enabled("5", start_from) and _truthy_env("ENABLE_AUDIENCE_SIGNALS"):
        if not url:
            log.warning("ENABLE_AUDIENCE_SIGNALS is on, but no YouTube URL is available; skipping audience rerank")
        else:
            banner("AUDIENCE SIGNAL RERANK")
            from pipeline.audience.audience_signal_rerank import main as audience_signal_main
            audience_signal_main(youtube_url=url, refresh=True)

    # ── Remotion Render ──
    if _phase_enabled("render", start_from):
        banner("REMOTION RENDER")
        setup_render_engine()
        run_fetch_tracking()
        run_remotion_render()

    banner("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()