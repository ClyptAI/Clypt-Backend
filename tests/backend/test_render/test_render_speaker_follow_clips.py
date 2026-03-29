import importlib.util
import sys
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[3] / 'backend' / 'test-render' / 'render_speaker_follow_clips.py'
    name = 'render_speaker_follow_clips_test'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_phase1_debug_module():
    path = Path(__file__).resolve().parents[3] / 'backend' / 'test-render' / 'render_phase1_debug_video.py'
    name = 'render_phase1_debug_video_test'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_bad_partial_body_detection(track_id="Global_Person_0"):
    return {
        "track_id": track_id,
        "bbox": [0.62, 0.55, 0.96, 0.98],
        "score": 0.91,
        "frame_idx": 100,
    }


def make_clean_body_detection(track_id="Global_Person_0"):
    return {
        "track_id": track_id,
        "bbox": [0.38, 0.12, 0.66, 0.92],
        "score": 0.88,
        "frame_idx": 100,
    }


def make_duplicate_fragment_detection(track_id="Global_Person_0"):
    return {
        "track_id": track_id,
        "bbox": [0.39, 0.13, 0.65, 0.91],
        "score": 0.89,
        "frame_idx": 100,
    }


def make_only_fragment_detection(track_id="Global_Person_0"):
    return {
        "track_id": track_id,
        "bbox": [0.70, 0.60, 0.98, 0.99],
        "score": 0.90,
        "frame_idx": 100,
    }


def make_context_fallback_detection(track_id="Global_Person_0"):
    return {
        "track_id": track_id,
        "bbox": [0.18, 0.10, 0.82, 0.96],
        "score": 0.18,
        "frame_idx": 100,
    }


def make_low_overlap_same_frame_detection(track_id="Global_Person_0"):
    return {
        "track_id": track_id,
        "bbox": [0.03, 0.06, 0.22, 0.88],
        "score": 0.87,
        "frame_idx": 100,
    }


def make_raw_render_primary_detection(track_id="Global_Person_0"):
    return {
        "track_id": track_id,
        "frame_idx": 0,
        "x_center": 2400.0,
        "y_center": 900.0,
        "width": 520.0,
        "height": 900.0,
        "score": 0.94,
        "bbox": [2140.0, 450.0, 2660.0, 1350.0],
    }


def make_raw_render_duplicate_detection(track_id="Global_Person_0"):
    return {
        "track_id": track_id,
        "frame_idx": 0,
        "x_center": 2410.0,
        "y_center": 905.0,
        "width": 500.0,
        "height": 890.0,
        "score": 0.90,
        "bbox": [2160.0, 460.0, 2660.0, 1350.0],
    }


def make_hybrid_debug_entry(
    *,
    start_time_ms=0,
    end_time_ms=800,
    active_audio_speaker_id="SPEAKER_00",
    active_audio_local_track_id="track_audio_A",
    chosen_track_id="A",
    chosen_local_track_id="track_A",
    decision_source="audio_boosted_visual",
    ambiguous=False,
    top_1_top_2_margin=0.041,
    candidates=None,
):
    return {
        "word": "hello",
        "start_time_ms": start_time_ms,
        "end_time_ms": end_time_ms,
        "active_audio_speaker_id": active_audio_speaker_id,
        "active_audio_local_track_id": active_audio_local_track_id,
        "chosen_track_id": chosen_track_id,
        "chosen_local_track_id": chosen_local_track_id,
        "decision_source": decision_source,
        "ambiguous": ambiguous,
        "top_1_top_2_margin": top_1_top_2_margin,
        "candidates": candidates
        or [
            {
                "local_track_id": "track_A",
                "track_id": "A",
                "blended_score": 0.31,
                "asd_probability": 0.19,
                "body_prior": 0.57,
                "detection_confidence": 0.94,
            },
            {
                "local_track_id": "track_B",
                "track_id": "B",
                "blended_score": 0.29,
                "asd_probability": 0.17,
                "body_prior": 0.52,
                "detection_confidence": 0.90,
            },
        ],
    }
