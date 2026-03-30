#!/usr/bin/env python3
"""
Clypt Pipeline Orchestrator
============================
Prompts for a YouTube URL, then runs the full pipeline sequentially:

  Phase 1   →  DigitalOcean async extraction
  FFmpeg    →  Re-encode video for clean H.264
  Phase 2A  →  Content Mechanism Decomposition (Gemini chunked multimodal)
  Phase 2B  →  Narrative Edge Mapping (Gemini text-only)
  Phase 3   →  Multimodal Embedding (Gemini Embedding 2)
  Phase 4   →  Storage & Graph Binding (Spanner + GCS)
  Phase 5   →  Auto-Curate (full-graph sweep + Gemini scoring)
  Render    →  FFmpeg speaker-follow render (9:16 vertical clips)
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = ROOT / "pipeline"
DOWNLOADS_DIR = ROOT / "downloads"
OUTPUTS_DIR = ROOT / "outputs"
FFMPEG_REENCODE_CRF = os.getenv("FFMPEG_REENCODE_CRF", "15")
FFMPEG_REENCODE_PRESET = os.getenv("FFMPEG_REENCODE_PRESET", "slow")
FFMPEG_RENDER_CRF = os.getenv("FFMPEG_RENDER_CRF", "20")
FFMPEG_RENDER_PRESET = os.getenv("FFMPEG_RENDER_PRESET", "fast")

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
    """Re-encode the downloaded video to clean H.264.

    yt-dlp's merged output can have encoding quirks (misaligned keyframes,
    non-standard NAL units). A quick FFmpeg pass produces a file that
    decodes cleanly for the speaker-follow crop+render step.
    """
    video_path = DOWNLOADS_DIR / "video.mp4"
    clean_path = DOWNLOADS_DIR / "video_clean.mp4"

    if not video_path.exists():
        log.warning("No video.mp4 found in downloads/ — skipping re-encode")
        return

    if not shutil.which("ffmpeg"):
        log.warning("ffmpeg not found on PATH — skipping re-encode")
        return

    log.info("Re-encoding video.mp4 for clean H.264…")
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


def run_ffmpeg_render():
    """Render 9:16 vertical clips using FFmpeg with active-speaker-follow tracking.

    Uses the clip boundaries from Phase 5 (remotion_payloads_array.json) and
    the Phase 1 tracking data (phase_1_visual.json + phase_1_audio.json) to
    produce speaker-follow vertical clips via FFmpeg crop+scale.

    If no Phase 5 payload is found, falls back to the standalone speaker-follow
    renderer which picks its own windows based on speaker engagement scoring.
    """
    # Try to import the speaker-follow renderer
    sys.path.insert(0, str(ROOT / "test-render"))
    from render_speaker_follow_clips import (
        load_json,
        build_track_index,
        build_mask_stability_index_from_visual,
        build_frame_detection_index,
        build_face_index,
        select_binding_sets,
        select_render_tracks,
        choose_windows,
        choose_window_composition,
        choose_active_speaker_segments,
        choose_adaptive_split_segments,
        build_single_track_path,
        build_camera_path,
        build_overlay_box_path,
        motion_profile_for_composition,
        render_clip as ffmpeg_render_clip,
        render_split_clip,
        concat_render_segments,
        active_speaker_state_for_interval,
        overlap_follow_decisions_for_interval,
        AdaptiveSegment,
        VIDEO_PATH as DEFAULT_VIDEO_PATH,
        VISUAL_PATH as DEFAULT_VISUAL_PATH,
        AUDIO_PATH as DEFAULT_AUDIO_PATH,
        OUTPUT_DIR as DEFAULT_OUTPUT_DIR,
        OUT_W,
        OUT_H,
        SPLIT_PANEL_HEIGHT,
    )
    import tempfile

    video_path = DOWNLOADS_DIR / "video.mp4"
    visual_path = OUTPUTS_DIR / "phase_1_visual.json"
    audio_path = OUTPUTS_DIR / "phase_1_audio.json"

    if not video_path.exists():
        raise FileNotFoundError(f"Missing source video: {video_path}")
    if not visual_path.exists():
        raise FileNotFoundError(f"Missing visual tracking: {visual_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio tracking: {audio_path}")

    # Load Phase 1 tracking data
    visual = load_json(visual_path)
    audio = load_json(audio_path)

    bindings, raw_bindings, follow_bindings, binding_source = select_binding_sets(audio)
    tracks, track_source = select_render_tracks(visual)
    if not tracks or not bindings:
        raise RuntimeError("Need non-empty tracks and speaker_bindings to render clips")

    meta = dict(visual.get("video_metadata", {}))
    fps = float(meta.get("fps", 23.976))
    src_w = int(meta.get("width", 1920))
    src_h = int(meta.get("height", 1080))
    video_duration_s = float(meta.get("duration_ms", 0)) / 1000.0

    frame_detection_index = build_frame_detection_index(tracks)
    mask_stability_index = build_mask_stability_index_from_visual(visual)
    person_track_index = build_track_index(
        tracks,
        mask_stability_index=mask_stability_index,
        src_w=src_w,
        src_h=src_h,
    )
    available_track_ids = {str(tid) for tid in person_track_index.keys() if str(tid)}
    prefer_local_track_ids = binding_source.endswith("_local")
    face_track_index = build_face_index(
        list(visual.get("face_detections", [])),
        src_w=src_w, src_h=src_h, fps=fps,
    )
    active_speakers_local = list(audio.get("active_speakers_local", []))
    overlap_follow_decisions = list(audio.get("overlap_follow_decisions", []))
    speaker_candidate_debug = list(audio.get("speaker_candidate_debug", []))

    # Determine clip windows — prefer Phase 5 curated clips, fall back to auto-pick
    payload_filename = (
        "remotion_payloads_array_audience.json"
        if _truthy_env("USE_AUDIENCE_SIGNAL_PAYLOADS")
        else "remotion_payloads_array.json"
    )
    payload_path = OUTPUTS_DIR / payload_filename
    clips_dir = OUTPUTS_DIR / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    if payload_path.exists():
        payloads = json.loads(payload_path.read_text())
        if not isinstance(payloads, list):
            payloads = [payloads]
        log.info(f"Using {len(payloads)} Phase 5 curated clip(s) from {payload_path.name}")

        windows = []
        for i, payload in enumerate(payloads):
            start_ms = int(payload.get("clip_start_ms", 0))
            end_ms = int(payload.get("clip_end_ms", start_ms))
            start_s = start_ms / 1000.0
            end_s = end_ms / 1000.0
            score = float(payload.get("final_score", 0))
            windows.append((start_s, end_s, score))
    else:
        log.warning(
            f"No Phase 5 payload found ({payload_path.name}). "
            "Falling back to auto-selected speaker-follow windows."
        )
        windows = choose_windows(bindings, video_duration_s)

    log.info(f"Rendering {len(windows)} clip(s) with FFmpeg speaker-follow tracking")
    log.info(f"  Track source: {track_source} ({len(person_track_index)} person tracks)")
    log.info(f"  Binding source: {binding_source}")

    outputs = []
    for idx, (start_s, end_s, score) in enumerate(windows, start=1):
        clip_name = f"clip_{idx}.mp4"
        out_path = clips_dir / clip_name
        duration_s = end_s - start_s

        # Choose composition mode (single person vs two-speaker split/shared)
        composition = choose_window_composition(
            clip_start_s=start_s,
            clip_end_s=end_s,
            fps=fps, src_w=src_w, src_h=src_h,
            bindings=bindings,
            person_track_index=person_track_index,
            face_track_index=face_track_index,
            speaker_candidate_debug=speaker_candidate_debug,
        )

        log.info(
            f"  Clip {idx}/{len(windows)}: {start_s:.1f}s–{end_s:.1f}s "
            f"({duration_s:.1f}s) score={score:.0f} mode={composition.mode}"
        )

        # Get interval-specific data
        clip_follow_bindings = [
            b for b in (follow_bindings if follow_bindings else bindings)
            if float(b["end_time_ms"]) / 1000.0 > start_s
            and float(b["start_time_ms"]) / 1000.0 < end_s
        ]

        # Build camera path and render based on composition mode
        profile = motion_profile_for_composition(composition.mode)

        if composition.mode in ("two_split", "two_shared") and composition.primary_track_id and composition.secondary_track_id:
            # Two-speaker modes use segment-based rendering
            if composition.mode == "two_split":
                split_profile = motion_profile_for_composition("two_split", out_h=SPLIT_PANEL_HEIGHT)
                segments = choose_adaptive_split_segments(
                    primary_track_id=composition.primary_track_id,
                    secondary_track_id=composition.secondary_track_id,
                    clip_start_s=int(start_s), clip_end_s=int(end_s),
                    fps=fps, src_w=src_w, src_h=src_h,
                    bindings=bindings,
                    person_track_index=person_track_index,
                    face_track_index=face_track_index,
                    frame_detection_index=frame_detection_index,
                )
            else:
                segments = [
                    AdaptiveSegment(
                        mode="two_shared",
                        start_s=float(start_s), end_s=float(end_s),
                        primary_track_id=composition.primary_track_id,
                        secondary_track_id=composition.secondary_track_id,
                    )
                ]

            with tempfile.TemporaryDirectory(prefix="clypt-render-") as tmpdir:
                segment_paths = []
                for seg_idx, segment in enumerate(segments, start=1):
                    seg_path = Path(tmpdir) / f"seg_{seg_idx:02d}.mp4"
                    segment_paths.append(seg_path)
                    seg_profile = motion_profile_for_composition(segment.mode, out_h=(SPLIT_PANEL_HEIGHT if segment.mode == "two_split" else OUT_H))

                    if segment.mode == "two_split" and segment.primary_track_id and segment.secondary_track_id:
                        upper_x, upper_y = build_single_track_path(
                            track_id=segment.primary_track_id,
                            clip_start_s=segment.start_s, clip_end_s=segment.end_s,
                            fps=fps, src_w=src_w, src_h=src_h,
                            person_track_index=person_track_index,
                            face_track_index=face_track_index,
                            frame_detection_index=frame_detection_index,
                            out_h=SPLIT_PANEL_HEIGHT, motion_profile=seg_profile,
                        )
                        lower_x, lower_y = build_single_track_path(
                            track_id=segment.secondary_track_id,
                            clip_start_s=segment.start_s, clip_end_s=segment.end_s,
                            fps=fps, src_w=src_w, src_h=src_h,
                            person_track_index=person_track_index,
                            face_track_index=face_track_index,
                            frame_detection_index=frame_detection_index,
                            out_h=SPLIT_PANEL_HEIGHT, motion_profile=seg_profile,
                        )
                        render_split_clip(
                            video_path=video_path, out_path=seg_path,
                            start_s=int(segment.start_s),
                            duration_s=int(segment.end_s - segment.start_s),
                            upper_x_keyframes=upper_x, upper_y_keyframes=upper_y,
                            upper_overlay_paths=[],
                            lower_x_keyframes=lower_x, lower_y_keyframes=lower_y,
                            lower_overlay_paths=[],
                            src_w=src_w, src_h=src_h,
                            camera_zoom=seg_profile.camera_zoom,
                        )
                    else:
                        # Shared or single fallback
                        seg_overlap_decisions = overlap_follow_decisions_for_interval(
                            overlap_follow_decisions,
                            float(segment.start_s), float(segment.end_s),
                        )
                        if seg_overlap_decisions:
                            x_kf, y_kf = build_camera_path(
                                clip_start_s=segment.start_s, clip_end_s=segment.end_s,
                                fps=fps, src_w=src_w, src_h=src_h,
                                bindings=clip_follow_bindings if clip_follow_bindings else bindings,
                                person_track_index=person_track_index,
                                face_track_index=face_track_index,
                                frame_detection_index=frame_detection_index,
                                motion_profile=seg_profile,
                                overlap_follow_decisions=seg_overlap_decisions,
                                prefer_local_track_ids=prefer_local_track_ids,
                                mask_stability_index=mask_stability_index,
                            )
                        else:
                            x_kf, y_kf = build_single_track_path(
                                track_id=segment.primary_track_id,
                                clip_start_s=segment.start_s, clip_end_s=segment.end_s,
                                fps=fps, src_w=src_w, src_h=src_h,
                                person_track_index=person_track_index,
                                face_track_index=face_track_index,
                                frame_detection_index=frame_detection_index,
                                out_h=OUT_H, motion_profile=seg_profile,
                            )
                        ffmpeg_render_clip(
                            video_path=video_path, out_path=seg_path,
                            start_s=int(segment.start_s),
                            duration_s=int(segment.end_s - segment.start_s),
                            x_keyframes=x_kf, y_keyframes=y_kf,
                            overlay_paths=[],
                            src_h=src_h, camera_zoom=seg_profile.camera_zoom,
                        )
                concat_render_segments(segment_paths, out_path)
        else:
            # Single-speaker mode
            single_segments = choose_active_speaker_segments(
                clip_start_s=float(start_s), clip_end_s=float(end_s),
                bindings=bindings,
                fallback_track_id=composition.primary_track_id,
            )
            with tempfile.TemporaryDirectory(prefix="clypt-render-single-") as tmpdir:
                segment_paths = []
                for seg_idx, segment in enumerate(single_segments, start=1):
                    seg_path = Path(tmpdir) / f"seg_{seg_idx:02d}.mp4"
                    segment_paths.append(seg_path)
                    seg_overlap_decisions = overlap_follow_decisions_for_interval(
                        overlap_follow_decisions,
                        float(segment.start_s), float(segment.end_s),
                    )
                    if seg_overlap_decisions:
                        x_kf, y_kf = build_camera_path(
                            clip_start_s=segment.start_s, clip_end_s=segment.end_s,
                            fps=fps, src_w=src_w, src_h=src_h,
                            bindings=clip_follow_bindings if clip_follow_bindings else bindings,
                            person_track_index=person_track_index,
                            face_track_index=face_track_index,
                            frame_detection_index=frame_detection_index,
                            motion_profile=profile,
                            overlap_follow_decisions=seg_overlap_decisions,
                            prefer_local_track_ids=prefer_local_track_ids,
                            mask_stability_index=mask_stability_index,
                        )
                    else:
                        x_kf, y_kf = build_single_track_path(
                            track_id=segment.primary_track_id,
                            clip_start_s=segment.start_s, clip_end_s=segment.end_s,
                            fps=fps, src_w=src_w, src_h=src_h,
                            person_track_index=person_track_index,
                            face_track_index=face_track_index,
                            frame_detection_index=frame_detection_index,
                            out_h=OUT_H, motion_profile=profile,
                        )
                    ffmpeg_render_clip(
                        video_path=video_path, out_path=seg_path,
                        start_s=int(segment.start_s),
                        duration_s=int(segment.end_s - segment.start_s),
                        x_keyframes=x_kf, y_keyframes=y_kf,
                        overlay_paths=[],
                        src_h=src_h, camera_zoom=profile.camera_zoom,
                    )
                concat_render_segments(segment_paths, out_path)

        size_mb = out_path.stat().st_size / 1e6
        log.info(f"    ✓ {clip_name} ({size_mb:.1f} MB)")
        outputs.append(out_path)

    log.info(f"All {len(outputs)} clip(s) rendered to {clips_dir}")


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

    # ── FFmpeg Speaker-Follow Render ──
    if _phase_enabled("render", start_from):
        banner("FFMPEG SPEAKER-FOLLOW RENDER")
        run_ffmpeg_render()

    banner("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()