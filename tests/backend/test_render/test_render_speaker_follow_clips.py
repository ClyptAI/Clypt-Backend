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


def make_large_full_body_detection(track_id="Global_Person_0"):
    return {
        "track_id": track_id,
        "bbox": [0.22, 0.08, 0.58, 0.96],
        "score": 0.58,
        "frame_idx": 100,
    }


def make_small_false_positive_detection(track_id="Global_Person_0"):
    return {
        "track_id": track_id,
        "bbox": [0.66, 0.54, 0.82, 0.88],
        "score": 0.97,
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



def test_target_scoring_rejects_partial_body_candidate():
    mod = load_module()
    bad = make_bad_partial_body_detection()
    good = make_clean_body_detection()

    assert mod.score_render_target_candidate(good, frame_width=1280, frame_height=720) > mod.score_render_target_candidate(
        bad,
        frame_width=1280,
        frame_height=720,
    )


def test_choose_clean_render_target_prefers_primary_box_over_duplicate_fragment():
    mod = load_module()
    primary = make_clean_body_detection()
    duplicate = make_duplicate_fragment_detection()

    chosen = mod.choose_clean_render_target(
        target_track_id="Global_Person_0",
        frame_detections=[primary, duplicate],
        frame_width=1280,
        frame_height=720,
    )

    assert chosen == primary


def test_choose_clean_render_target_prefers_dominant_larger_box_over_smaller_false_positive():
    mod = load_module()
    larger = make_large_full_body_detection()
    smaller = make_small_false_positive_detection()

    chosen = mod.choose_clean_render_target(
        target_track_id="Global_Person_0",
        frame_detections=[smaller, larger],
        frame_width=1280,
        frame_height=720,
    )

    assert chosen == larger


def test_group_duplicate_render_targets_keeps_clean_primary_only():
    mod = load_module()
    primary = make_clean_body_detection()
    duplicate = make_duplicate_fragment_detection()

    grouped = mod._group_duplicate_render_targets(
        [primary, duplicate],
        frame_width=1280,
        frame_height=720,
    )

    assert grouped == [primary]


def test_group_duplicate_render_targets_keeps_low_overlap_same_frame_candidates():
    mod = load_module()
    primary = make_clean_body_detection()
    low_overlap = make_low_overlap_same_frame_detection()

    grouped = mod._group_duplicate_render_targets(
        [primary, low_overlap],
        frame_width=1280,
        frame_height=720,
    )

    assert len(grouped) == 2
    assert primary in grouped
    assert low_overlap in grouped


def test_build_single_track_path_uses_clean_raw_render_target():
    mod = load_module()
    person_track_index = {
        "A": [
            mod.Detection(frame_idx=0, x_center=800, y_center=900, width=500, height=900),
        ]
    }
    frame_detection_index = {
        0: [
            make_raw_render_duplicate_detection("A"),
            make_raw_render_primary_detection("A"),
        ]
    }

    x_keyframes, _ = mod.build_single_track_path(
        track_id="A",
        clip_start_s=0,
        clip_end_s=1,
        fps=24.0,
        src_w=3840,
        src_h=2160,
        person_track_index=person_track_index,
        face_track_index={},
        frame_detection_index=frame_detection_index,
    )

    assert x_keyframes
    assert x_keyframes[0][1] > 1000


def test_resolve_follow_identity_returns_active_speaker_track_id():
    mod = load_module()
    bindings = [
        {"track_id": "A", "start_time_ms": 0, "end_time_ms": 900, "word_count": 10},
        {"track_id": "B", "start_time_ms": 1000, "end_time_ms": 1900, "word_count": 10},
    ]

    assert mod.resolve_follow_identity(bindings, 500) == "A"


def test_resolve_follow_identity_prefers_new_speaker_on_boundary():
    mod = load_module()
    bindings = [
        {"track_id": "A", "start_time_ms": 0, "end_time_ms": 1000, "word_count": 10},
        {"track_id": "B", "start_time_ms": 1000, "end_time_ms": 2000, "word_count": 10},
    ]

    assert mod.resolve_follow_identity(bindings, 1000) == "B"


def test_resolve_follow_identity_falls_back_to_single_visible_frame_track():
    mod = load_module()
    bindings = []
    frame_detections = [make_clean_body_detection("track_43")]

    assert mod.resolve_follow_identity(
        bindings,
        1000,
        frame_detections=frame_detections,
    ) == "track_43"


def test_resolve_follow_identity_single_visible_frame_track_overrides_stay_wide_overlap():
    mod = load_module()
    bindings = []
    overlap_follow_decisions = [
        {
            "start_time_ms": 500,
            "end_time_ms": 1500,
            "camera_target_track_id": None,
            "stay_wide": True,
        }
    ]
    frame_detections = [make_clean_body_detection("track_43")]

    assert mod.resolve_follow_identity(
        bindings,
        1000,
        overlap_follow_decisions=overlap_follow_decisions,
        frame_detections=frame_detections,
    ) == "track_43"


def test_resolve_follow_identity_prefers_dominant_track_when_unbound_and_fragment_present():
    mod = load_module()
    bindings = []
    frame_detections = [
        {
            "track_id": "track_51",
            "bbox": [0.097, 0.092, 0.629, 0.994],
            "score": 0.948,
            "frame_idx": 3769,
        },
        {
            "track_id": "track_67",
            "bbox": [0.001, 0.420, 0.224, 0.987],
            "score": 0.681,
            "frame_idx": 3769,
        },
    ]

    assert mod.resolve_follow_identity(
        bindings,
        157200,
        frame_detections=frame_detections,
    ) == "track_51"


def test_follow_target_stability_adjustment_overlap_prefers_camera_target():
    mod = load_module()
    det = make_clean_body_detection("A")
    overlap = [{"start_time_ms": 0, "end_time_ms": 60_000, "camera_target_track_id": "A"}]
    adj_a = mod.follow_target_stability_adjustment(
        track_id="A",
        frame_idx=0,
        abs_ms=5000,
        frame_width=1280,
        frame_height=720,
        det=det,
        mask_stability_index=None,
        overlap_follow_decisions=overlap,
        prefer_local_track_ids=False,
        last_resolved_track_id=None,
    )
    adj_b = mod.follow_target_stability_adjustment(
        track_id="B",
        frame_idx=0,
        abs_ms=5000,
        frame_width=1280,
        frame_height=720,
        det=det,
        mask_stability_index=None,
        overlap_follow_decisions=overlap,
        prefer_local_track_ids=False,
        last_resolved_track_id=None,
    )
    assert adj_a - adj_b == mod.RENDER_OVERLAP_CAMERA_TARGET_BIAS


def test_follow_target_stability_adjustment_memory_reduces_track_flip():
    mod = load_module()
    det_a = make_clean_body_detection("A")
    det_b = {**make_clean_body_detection("B"), "track_id": "B"}
    with_memory = mod.follow_target_stability_adjustment(
        track_id="A",
        frame_idx=0,
        abs_ms=1000,
        frame_width=1280,
        frame_height=720,
        det=det_a,
        mask_stability_index=None,
        overlap_follow_decisions=None,
        prefer_local_track_ids=False,
        last_resolved_track_id="A",
    )
    without_memory = mod.follow_target_stability_adjustment(
        track_id="A",
        frame_idx=0,
        abs_ms=1000,
        frame_width=1280,
        frame_height=720,
        det=det_a,
        mask_stability_index=None,
        overlap_follow_decisions=None,
        prefer_local_track_ids=False,
        last_resolved_track_id="B",
    )
    assert with_memory == mod.RENDER_SHORT_TERM_TRACK_MEMORY_BIAS
    assert without_memory == 0.0
    # Same geometry/score for A vs B: memory breaks the tie toward the previous track.
    picked = mod.tiebreak_two_visible_tracks_for_follow(
        frame_detections=[det_a, det_b],
        frame_width=1280,
        frame_height=720,
        abs_ms=1000,
        video_fps=24.0,
        mask_stability_index=None,
        overlap_follow_decisions=None,
        prefer_local_track_ids=False,
        last_resolved_track_id="B",
    )
    assert picked == "B"


def test_follow_target_stability_adjustment_backward_compatible_without_signals():
    mod = load_module()
    det = make_clean_body_detection("A")
    assert (
        mod.follow_target_stability_adjustment(
            track_id="A",
            frame_idx=0,
            abs_ms=0,
            frame_width=1280,
            frame_height=720,
            det=det,
            mask_stability_index=None,
            overlap_follow_decisions=None,
            prefer_local_track_ids=False,
            last_resolved_track_id=None,
        )
        == 0.0
    )


def test_follow_target_stability_adjustment_overlap_boosts_short_term_memory():
    mod = load_module()
    det = make_clean_body_detection("A")
    overlap = [{"start_time_ms": 0, "end_time_ms": 60_000, "camera_target_track_id": None, "stay_wide": True}]
    in_overlap = mod.follow_target_stability_adjustment(
        track_id="A",
        frame_idx=0,
        abs_ms=5000,
        frame_width=1280,
        frame_height=720,
        det=det,
        mask_stability_index=None,
        overlap_follow_decisions=overlap,
        prefer_local_track_ids=False,
        last_resolved_track_id="A",
    )
    outside = mod.follow_target_stability_adjustment(
        track_id="A",
        frame_idx=0,
        abs_ms=5000,
        frame_width=1280,
        frame_height=720,
        det=det,
        mask_stability_index=None,
        overlap_follow_decisions=None,
        prefer_local_track_ids=False,
        last_resolved_track_id="A",
    )
    assert in_overlap - outside == mod.RENDER_OVERLAP_SHORT_TERM_MEMORY_BOOST


def test_tiebreak_two_visible_hysteresis_keeps_last_resolved_on_small_margin(monkeypatch):
    mod = load_module()
    det_a = {**make_clean_body_detection("A"), "track_id": "A"}
    det_b = {**make_clean_body_detection("B"), "track_id": "B"}

    def patched_score(det, fw, fh):
        tid = str(det.get("track_id") or "")
        if tid == "A":
            return 1.0
        if tid == "B":
            return 1.19
        return mod.score_render_target_candidate(det, fw, fh)

    monkeypatch.setattr(mod, "score_render_target_candidate", patched_score)
    picked = mod.tiebreak_two_visible_tracks_for_follow(
        frame_detections=[det_a, det_b],
        frame_width=1280,
        frame_height=720,
        abs_ms=1000,
        video_fps=24.0,
        mask_stability_index=None,
        overlap_follow_decisions=None,
        prefer_local_track_ids=False,
        last_resolved_track_id="A",
    )
    assert picked == "A"


def test_tiebreak_two_visible_hysteresis_yields_to_clear_leader(monkeypatch):
    mod = load_module()
    det_a = {**make_clean_body_detection("A"), "track_id": "A"}
    det_b = {**make_clean_body_detection("B"), "track_id": "B"}

    def patched_score(det, fw, fh):
        tid = str(det.get("track_id") or "")
        if tid == "A":
            return 1.0
        if tid == "B":
            return 1.30
        return mod.score_render_target_candidate(det, fw, fh)

    monkeypatch.setattr(mod, "score_render_target_candidate", patched_score)
    picked = mod.tiebreak_two_visible_tracks_for_follow(
        frame_detections=[det_a, det_b],
        frame_width=1280,
        frame_height=720,
        abs_ms=1000,
        video_fps=24.0,
        mask_stability_index=None,
        overlap_follow_decisions=None,
        prefer_local_track_ids=False,
        last_resolved_track_id="A",
    )
    assert picked == "B"


def test_resolve_follow_identity_symmetric_bodies_follow_overlap_camera_target():
    mod = load_module()
    bindings = []
    frame_detections = [
        {"track_id": "A", "bbox": [0.05, 0.10, 0.40, 0.90], "score": 0.88, "frame_idx": 0},
        {"track_id": "B", "bbox": [0.55, 0.10, 0.90, 0.90], "score": 0.88, "frame_idx": 0},
    ]
    overlap_follow_decisions = [
        {
            "start_time_ms": 0,
            "end_time_ms": 60_000,
            "camera_target_track_id": "B",
            "stay_wide": False,
        }
    ]
    assert (
        mod.resolve_follow_identity(
            bindings,
            5000,
            frame_detections=frame_detections,
            overlap_follow_decisions=overlap_follow_decisions,
            video_fps=24.0,
        )
        == "B"
    )


def test_build_camera_path_uses_clean_follow_box_for_active_speaker():
    mod = load_module()
    mod.KEYFRAME_STEP_S = 0.5
    person_track_index = {
        "A": [
            mod.Detection(frame_idx=0, x_center=3000, y_center=1200, width=520, height=900),
        ]
    }
    face_track_index = {}
    bindings = [
        {"track_id": "A", "start_time_ms": 0, "end_time_ms": 900, "word_count": 10},
    ]
    frame_detection_index = {
        0: [
            make_only_fragment_detection("A"),
            make_clean_body_detection("A"),
        ],
    }

    x_keyframes, y_keyframes = mod.build_camera_path(
        clip_start_s=0,
        clip_end_s=1,
        fps=24.0,
        src_w=3840,
        src_h=2160,
        bindings=bindings,
        person_track_index=person_track_index,
        face_track_index=face_track_index,
        frame_detection_index=frame_detection_index,
    )

    motion_profile = mod.motion_profile_for_composition("single_person")
    crop_w, crop_h = mod.crop_dimensions(3840, 2160, camera_zoom=motion_profile.camera_zoom)
    half_w = crop_w / 2.0
    half_h = crop_h / 2.0
    clean_bbox = make_clean_body_detection("A")["bbox"]
    clean_cx = ((clean_bbox[0] + clean_bbox[2]) / 2.0) * 3840.0
    clean_cy = ((clean_bbox[1] + clean_bbox[3]) / 2.0) * 2160.0
    expected_x = mod.clamp(clean_cx, half_w, 3840 - half_w) - half_w
    expected_y = mod.clamp(clean_cy, half_h, 2160 - half_h) - half_h

    assert x_keyframes
    assert y_keyframes
    assert abs(x_keyframes[0][1] - expected_x) < 1.0
    assert abs(y_keyframes[0][1] - expected_y) < 1.0


def test_resolve_follow_box_can_rescue_fragment_track_with_larger_nearby_track():
    mod = load_module()
    frame_detections = [
        {
            "track_id": "track_2",
            "bbox": [0.002, 0.427, 0.237, 0.891],
            "score": 0.590,
            "frame_idx": 334,
        },
        {
            "track_id": "track_12",
            "bbox": [0.110, 0.089, 0.636, 0.993],
            "score": 0.927,
            "frame_idx": 334,
        },
    ]

    chosen = mod.resolve_follow_box(
        "track_2",
        frame_detections,
        frame_width=1920,
        frame_height=1080,
    )

    assert chosen is not None
    assert chosen["track_id"] == "track_12"


def test_build_camera_path_prefers_new_speaker_clean_box_on_boundary():
    mod = load_module()
    mod.KEYFRAME_STEP_S = 0.5
    person_track_index = {
        "A": [
            mod.Detection(frame_idx=24, x_center=900, y_center=900, width=400, height=800),
        ],
        "B": [
            mod.Detection(frame_idx=24, x_center=2500, y_center=900, width=400, height=800),
        ],
    }
    bindings = [
        {"track_id": "A", "start_time_ms": 0, "end_time_ms": 1000, "word_count": 10},
        {"track_id": "B", "start_time_ms": 1000, "end_time_ms": 2000, "word_count": 10},
    ]
    frame_detection_index = {
        24: [
            make_clean_body_detection("A"),
            make_clean_body_detection("B"),
        ],
    }
    # Make B clearly the more relevant box by shifting it to the right.
    frame_detection_index[24][1]["bbox"] = [0.60, 0.12, 0.88, 0.92]

    x_keyframes, _ = mod.build_camera_path(
        clip_start_s=1,
        clip_end_s=2,
        fps=24.0,
        src_w=3840,
        src_h=2160,
        bindings=bindings,
        person_track_index=person_track_index,
        face_track_index={},
        frame_detection_index=frame_detection_index,
    )

    assert x_keyframes
    assert x_keyframes[0][1] > 1000


def test_choose_clean_render_target_returns_none_when_only_fragments_exist():
    mod = load_module()
    bad1 = make_bad_partial_body_detection()
    bad2 = make_only_fragment_detection()

    chosen = mod.choose_clean_render_target(
        target_track_id="Global_Person_0",
        frame_detections=[bad1, bad2],
        frame_width=1280,
        frame_height=720,
    )

    assert chosen is None


def test_fallback_anchor_for_missing_clean_target_widens_to_context():
    mod = load_module()
    fragment = make_bad_partial_body_detection()
    context = make_context_fallback_detection()

    fallback = mod.fallback_anchor_for_missing_clean_target(
        [fragment, context],
        "single_person",
        1280,
        720,
    )

    assert fallback is not None
    assert fallback.width > (fragment["bbox"][2] - fragment["bbox"][0]) * 1280
    assert fallback.height > (fragment["bbox"][3] - fragment["bbox"][1]) * 720
    assert abs(fallback.x_center - ((context["bbox"][0] + context["bbox"][2]) / 2.0) * 1280) < 180
    assert abs(fallback.y_center - ((context["bbox"][1] + context["bbox"][3]) / 2.0) * 720) < 100


def test_build_camera_path_uses_context_fallback_when_clean_target_is_missing():
    mod = load_module()
    mod.KEYFRAME_STEP_S = 0.5
    person_track_index = {
        "A": [
            mod.Detection(frame_idx=0, x_center=3000, y_center=1200, width=260, height=420),
        ]
    }
    bindings = [
        {"track_id": "A", "start_time_ms": 0, "end_time_ms": 900, "word_count": 10},
    ]
    frame_detection_index = {
        0: [
            make_bad_partial_body_detection("A"),
            make_context_fallback_detection("A"),
        ],
    }

    x_keyframes, y_keyframes = mod.build_camera_path(
        clip_start_s=0,
        clip_end_s=1,
        fps=24.0,
        src_w=3840,
        src_h=2160,
        bindings=bindings,
        person_track_index=person_track_index,
        face_track_index={},
        frame_detection_index=frame_detection_index,
    )

    motion_profile = mod.motion_profile_for_composition("single_person")
    crop_w, crop_h = mod.crop_dimensions(3840, 2160, camera_zoom=motion_profile.camera_zoom)
    half_w = crop_w / 2.0
    half_h = crop_h / 2.0
    context_bbox = make_context_fallback_detection("A")["bbox"]
    context_cx = ((context_bbox[0] + context_bbox[2]) / 2.0) * 3840.0
    context_cy = ((context_bbox[1] + context_bbox[3]) / 2.0) * 2160.0
    expected_x = mod.clamp(context_cx, half_w, 3840 - half_w) - half_w
    expected_y = mod.clamp(context_cy, half_h, 2160 - half_h) - half_h

    assert x_keyframes
    assert y_keyframes
    assert abs(x_keyframes[0][1] - expected_x) < 1.0
    assert abs(y_keyframes[0][1] - expected_y) < 1.0
    fragment_bbox = make_bad_partial_body_detection("A")["bbox"]
    fragment_cx = ((fragment_bbox[0] + fragment_bbox[2]) / 2.0) * 3840.0
    fragment_expected_x = mod.clamp(fragment_cx, half_w, 3840 - half_w) - half_w
    assert abs(x_keyframes[0][1] - fragment_expected_x) > 100.0


def test_choose_windows_are_non_overlapping():
    mod = load_module()
    mod.CLIP_DURATION_S = 20
    bindings = [
        {'track_id': 'A', 'start_time_ms': 0, 'end_time_ms': 30_000, 'word_count': 120},
        {'track_id': 'B', 'start_time_ms': 35_000, 'end_time_ms': 65_000, 'word_count': 120},
        {'track_id': 'C', 'start_time_ms': 70_000, 'end_time_ms': 100_000, 'word_count': 120},
    ]

    windows = mod.choose_windows(bindings, duration_s=120)

    assert len(windows) >= 3
    for idx, (start_a, end_a, _) in enumerate(windows):
        for start_b, end_b, _ in windows[idx + 1:]:
            assert end_a <= start_b or end_b <= start_a


def test_build_overlay_path_returns_keyframes_in_output_space():
    mod = load_module()
    mod.CLIP_DURATION_S = 20
    person_track_index = {
        'A': [
            mod.Detection(frame_idx=0, x_center=1000, y_center=900, width=500, height=900),
            mod.Detection(frame_idx=12, x_center=1100, y_center=920, width=520, height=920),
        ]
    }
    face_track_index = {
        'A': [
            mod.Detection(frame_idx=0, x_center=1020, y_center=760, width=180, height=180),
            mod.Detection(frame_idx=12, x_center=1120, y_center=780, width=190, height=190),
        ]
    }
    bindings = [
        {'track_id': 'A', 'start_time_ms': 0, 'end_time_ms': 1000, 'word_count': 10},
    ]
    x_keyframes = [(0.0, 700.0), (0.5, 720.0)]
    y_keyframes = [(0.0, 200.0), (0.5, 210.0)]

    box_x, box_y, box_w, box_h = mod.build_overlay_path(
        track_id='A',
        clip_start_s=0,
        clip_end_s=1,
        fps=24.0,
        src_w=3840,
        src_h=2160,
        x_keyframes=x_keyframes,
        y_keyframes=y_keyframes,
        person_track_index=person_track_index,
        face_track_index=face_track_index,
    )

    assert box_x
    assert len(box_x) == len(box_y) == len(box_w) == len(box_h)
    assert all(0 <= value <= mod.OUT_W for _, value in box_x)
    assert all(0 <= value <= mod.OUT_H for _, value in box_y)
    assert all(value > 0 for _, value in box_w)
    assert all(value > 0 for _, value in box_h)


def test_track_anchor_falls_back_to_body_when_face_is_implausible():
    mod = load_module()
    person_track_index = {
        'A': [
            mod.Detection(frame_idx=0, x_center=1000, y_center=900, width=500, height=900),
        ]
    }
    face_track_index = {
        'A': [
            mod.Detection(frame_idx=0, x_center=1040, y_center=1330, width=120, height=120),
        ]
    }

    det = mod._track_anchor_candidate('A', 0, 24.0, 3840, 2160, person_track_index, face_track_index)

    assert det is not None
    assert det.width == 500
    assert det.height == 900
    assert det.y_center == 900


def test_track_anchor_prefers_body_even_when_face_is_plausible():
    mod = load_module()
    person_track_index = {
        'A': [
            mod.Detection(frame_idx=0, x_center=1000, y_center=900, width=500, height=900),
        ]
    }
    face_track_index = {
        'A': [
            mod.Detection(frame_idx=0, x_center=1010, y_center=560, width=170, height=170),
        ]
    }

    det = mod._track_anchor_candidate('A', 0, 24.0, 3840, 2160, person_track_index, face_track_index)

    assert det is not None
    assert det.x_center == 1000
    assert det.y_center == 900
    assert det.width == 500
    assert det.height == 900

def test_build_camera_path_inserts_hard_cut_on_speaker_switch():
    mod = load_module()
    mod.KEYFRAME_STEP_S = 0.5
    person_track_index = {
        'A': [mod.Detection(frame_idx=0, x_center=900, y_center=900, width=400, height=800)],
        'B': [mod.Detection(frame_idx=24, x_center=2500, y_center=900, width=400, height=800)],
    }
    face_track_index = {}
    bindings = [
        {'track_id': 'A', 'start_time_ms': 0, 'end_time_ms': 900, 'word_count': 10},
        {'track_id': 'B', 'start_time_ms': 1000, 'end_time_ms': 1900, 'word_count': 10},
    ]

    x_keyframes, _ = mod.build_camera_path(
        clip_start_s=0,
        clip_end_s=2,
        fps=24.0,
        src_w=3840,
        src_h=2160,
        bindings=bindings,
        person_track_index=person_track_index,
        face_track_index=face_track_index,
    )

    cut_pairs = [
        (t0, x0, t1, x1)
        for (t0, x0), (t1, x1) in zip(x_keyframes, x_keyframes[1:])
        if (t1 - t0) < 0.01 and abs(x1 - x0) > 100
    ]
    assert cut_pairs, x_keyframes


def test_choose_active_speaker_segments_switches_tracks_within_single_person_window():
    mod = load_module()
    bindings = [
        {'track_id': 'A', 'start_time_ms': 0, 'end_time_ms': 1_200, 'word_count': 12},
        {'track_id': 'B', 'start_time_ms': 1_200, 'end_time_ms': 2_600, 'word_count': 14},
        {'track_id': 'A', 'start_time_ms': 2_600, 'end_time_ms': 4_000, 'word_count': 10},
    ]

    segments = mod.choose_active_speaker_segments(
        clip_start_s=0,
        clip_end_s=4,
        bindings=bindings,
    )

    assert [segment.primary_track_id for segment in segments] == ['A', 'B', 'A']
    assert [segment.mode for segment in segments] == ['single_person', 'single_person', 'single_person']
    assert segments[0].start_s == 0.0
    assert segments[-1].end_s == 4.0


def test_choose_active_speaker_segments_debounces_tiny_blip():
    mod = load_module()
    bindings = [
        {'track_id': 'A', 'start_time_ms': 0, 'end_time_ms': 1_200, 'word_count': 12},
        {'track_id': 'B', 'start_time_ms': 1_200, 'end_time_ms': 1_450, 'word_count': 1},
        {'track_id': 'A', 'start_time_ms': 1_450, 'end_time_ms': 3_000, 'word_count': 11},
    ]

    segments = mod.choose_active_speaker_segments(
        clip_start_s=0,
        clip_end_s=3,
        bindings=bindings,
    )

    assert len(segments) == 1
    assert segments[0].primary_track_id == 'A'
    assert segments[0].start_s == 0.0
    assert segments[0].end_s == 3.0


def test_choose_active_speaker_segments_prefers_follow_bindings_when_present():
    mod = load_module()
    audio = {
        "speaker_bindings": [
            {"track_id": "A", "start_time_ms": 0, "end_time_ms": 1200, "word_count": 12},
            {"track_id": "B", "start_time_ms": 1200, "end_time_ms": 1500, "word_count": 1},
            {"track_id": "A", "start_time_ms": 1500, "end_time_ms": 3000, "word_count": 11},
        ],
        "speaker_follow_bindings": [
            {"track_id": "A", "start_time_ms": 0, "end_time_ms": 3000, "word_count": 24},
        ],
    }

    bindings = mod.select_render_bindings(audio)
    segments = mod.choose_active_speaker_segments(
        clip_start_s=0,
        clip_end_s=3,
        bindings=bindings,
    )

    assert len(segments) == 1
    assert segments[0].primary_track_id == "A"
    assert segments[0].start_s == 0.0
    assert segments[0].end_s == 3.0


def test_select_binding_sets_retains_raw_and_follow_streams():
    mod = load_module()
    audio = {
        "speaker_bindings": [
            {"track_id": "raw_A", "start_time_ms": 0, "end_time_ms": 1000, "word_count": 5},
        ],
        "speaker_follow_bindings": [
            {"track_id": "follow_A", "start_time_ms": 0, "end_time_ms": 1000, "word_count": 5},
        ],
    }

    selected, raw_bindings, follow_bindings, source = mod.select_binding_sets(audio)

    assert selected == follow_bindings
    assert raw_bindings[0]["track_id"] == "raw_A"
    assert follow_bindings[0]["track_id"] == "follow_A"
    assert source == "speaker_follow_bindings"


def test_select_binding_sets_prefers_speaker_follow_bindings_local_without_experiment(monkeypatch):
    mod = load_module()
    monkeypatch.delenv("CLYPT_EXPERIMENT_LOCAL_CLIP_BINDINGS", raising=False)
    audio = {
        "speaker_bindings": [
            {"track_id": "raw_A", "start_time_ms": 0, "end_time_ms": 1000, "word_count": 5},
        ],
        "speaker_follow_bindings": [
            {"track_id": "follow_A", "start_time_ms": 0, "end_time_ms": 1000, "word_count": 5},
        ],
        "speaker_bindings_local": [
            {"track_id": "local_raw_A", "start_time_ms": 0, "end_time_ms": 1000, "word_count": 5},
        ],
        "speaker_follow_bindings_local": [
            {"track_id": "local_follow_A", "start_time_ms": 0, "end_time_ms": 1000, "word_count": 5},
        ],
    }

    selected, raw_bindings, follow_bindings, source = mod.select_binding_sets(audio)

    assert selected[0]["track_id"] == "local_follow_A"
    assert raw_bindings[0]["track_id"] == "local_raw_A"
    assert follow_bindings[0]["track_id"] == "local_follow_A"
    assert source == "speaker_follow_bindings_local"


def test_select_render_tracks_prefers_tracks_local_when_audio_has_follow_local(monkeypatch):
    mod = load_module()
    monkeypatch.delenv("CLYPT_EXPERIMENT_LOCAL_CLIP_BINDINGS", raising=False)
    visual = {
        "tracks": [{"track_id": "Global_Person_0", "frame_idx": 0}],
        "tracks_local": [{"track_id": "local-1", "frame_idx": 0}],
    }
    audio = {
        "speaker_follow_bindings_local": [
            {"track_id": "local-1", "start_time_ms": 0, "end_time_ms": 1000, "word_count": 1},
        ],
    }

    tracks, source = mod.select_render_tracks(visual, audio=audio)

    assert tracks == [{"track_id": "local-1", "frame_idx": 0}]
    assert source == "tracks_local"


def test_select_binding_sets_prefers_local_follow_stream_when_experiment_enabled(monkeypatch):
    mod = load_module()
    audio = {
        "speaker_bindings": [
            {"track_id": "raw_A", "start_time_ms": 0, "end_time_ms": 1000, "word_count": 5},
        ],
        "speaker_follow_bindings": [
            {"track_id": "follow_A", "start_time_ms": 0, "end_time_ms": 1000, "word_count": 5},
        ],
        "speaker_bindings_local": [
            {"track_id": "local_raw_A", "start_time_ms": 0, "end_time_ms": 1000, "word_count": 5},
        ],
        "speaker_follow_bindings_local": [
            {"track_id": "local_follow_A", "start_time_ms": 0, "end_time_ms": 1000, "word_count": 5},
        ],
    }

    monkeypatch.setenv("CLYPT_EXPERIMENT_LOCAL_CLIP_BINDINGS", "1")

    selected, raw_bindings, follow_bindings, source = mod.select_binding_sets(audio)

    assert selected[0]["track_id"] == "local_follow_A"
    assert raw_bindings[0]["track_id"] == "local_raw_A"
    assert follow_bindings[0]["track_id"] == "local_follow_A"
    assert source == "speaker_follow_bindings_local"


def test_select_render_tracks_prefers_local_tracks_when_experiment_enabled(monkeypatch):
    mod = load_module()
    visual = {
        "tracks": [{"track_id": "Global_Person_0", "frame_idx": 0}],
        "tracks_local": [{"track_id": "local-1", "frame_idx": 0}],
    }

    monkeypatch.setenv("CLYPT_EXPERIMENT_LOCAL_CLIP_BINDINGS", "1")

    tracks, source = mod.select_render_tracks(visual)

    assert tracks == [{"track_id": "local-1", "frame_idx": 0}]
    assert source == "tracks_local"


def test_build_overlay_filters_include_track_label_text():
    mod = load_module()
    overlay = mod.OverlayPath(
        x_keyframes=[(0.0, 0.0)],
        y_keyframes=[(0.0, 0.0)],
        box_x_keyframes=[(0.0, 100.0)],
        box_y_keyframes=[(0.0, 200.0)],
        box_w_keyframes=[(0.0, 300.0)],
        box_h_keyframes=[(0.0, 400.0)],
        color="0x00FF88",
        label="Global_Person_3",
    )

    filters = mod.build_overlay_filters([overlay])

    assert any("drawbox=" in fragment for fragment in filters)
    assert any("drawtext=" in fragment and "Global_Person_3" in fragment for fragment in filters)


def test_build_debug_timeline_filters_include_raw_and_follow_rows():
    mod = load_module()
    raw_bindings = [
        {"track_id": "A", "start_time_ms": 0, "end_time_ms": 1000, "word_count": 5},
    ]
    follow_bindings = [
        {"track_id": "B", "start_time_ms": 0, "end_time_ms": 1000, "word_count": 5},
    ]

    filters = mod.build_debug_timeline_filters(
        clip_start_s=0.0,
        clip_end_s=10.0,
        raw_bindings=raw_bindings,
        follow_bindings=follow_bindings,
    )
    joined = "\n".join(filters)

    assert "raw" in joined
    assert "follow" in joined
    assert "drawbox=" in joined


def test_build_render_debug_sidecar_payload_records_target_quality_rejections():
    mod = load_module()
    segment = mod.AdaptiveSegment(
        mode="single_person",
        start_s=0.0,
        end_s=1.0,
        primary_track_id="A",
        secondary_track_id=None,
    )
    frame_detection_index = {
        0: [
            make_bad_partial_body_detection("A"),
            make_clean_body_detection("A"),
        ],
    }

    payload = mod.build_render_debug_sidecar_payload(
        clip_name="speaker_follow_clip1_0s_40s.mp4",
        window={"start_s": 0, "end_s": 1, "score": 0.9},
        binding_source="speaker_bindings",
        debug_mode=True,
        debug_show_faces=False,
        composition=mod.CompositionPlan(mode="single_person", primary_track_id="A"),
        segments=[segment],
        fps=24.0,
        src_w=1280,
        src_h=720,
        frame_detection_index=frame_detection_index,
    )

    target_debug = payload["segments"][0]["target_debug"][0]

    assert target_debug["target_track_id"] == "A"
    assert target_debug["target_source"] == "clean_body_box"
    assert target_debug["fallback_used"] is False
    assert target_debug["candidate_count"] == 2
    assert target_debug["target_quality"] > 0
    assert any(rejection["reason"] == "partial_body" for rejection in target_debug["rejections"])


def test_build_render_debug_sidecar_payload_records_context_fallback():
    mod = load_module()
    segment = mod.AdaptiveSegment(
        mode="single_person",
        start_s=0.0,
        end_s=1.0,
        primary_track_id="A",
        secondary_track_id=None,
    )
    frame_detection_index = {
        0: [
            make_bad_partial_body_detection("A"),
            make_only_fragment_detection("A"),
            make_context_fallback_detection("Context"),
        ],
    }

    payload = mod.build_render_debug_sidecar_payload(
        clip_name="speaker_follow_clip1_0s_40s.mp4",
        window={"start_s": 0, "end_s": 1, "score": 0.9},
        binding_source="speaker_bindings",
        debug_mode=True,
        debug_show_faces=False,
        composition=mod.CompositionPlan(mode="single_person", primary_track_id="A"),
        segments=[segment],
        fps=24.0,
        src_w=1280,
        src_h=720,
        frame_detection_index=frame_detection_index,
    )

    target_debug = payload["segments"][0]["target_debug"][0]

    assert target_debug["target_track_id"] == "A"
    assert target_debug["target_source"] == "context_fallback"
    assert target_debug["fallback_used"] is True
    assert target_debug["candidate_count"] == 2
    assert target_debug["target_quality"] is not None
    assert all(rejection["reason"] == "partial_body" for rejection in target_debug["rejections"])


def test_build_render_debug_sidecar_payload_context_fallback_quality_ignores_unrelated_scores():
    mod = load_module()
    segment = mod.AdaptiveSegment(
        mode="single_person",
        start_s=0.0,
        end_s=1.0,
        primary_track_id="A",
        secondary_track_id=None,
    )
    low_score_frame_detection_index = {
        0: [
            {"track_id": "A", "bbox": [0.62, 0.55, 0.96, 0.98], "score": 0.10, "frame_idx": 100},
            {"track_id": "A", "bbox": [0.70, 0.60, 0.98, 0.99], "score": 0.12, "frame_idx": 100},
            make_context_fallback_detection("Context"),
        ],
    }
    high_score_frame_detection_index = {
        0: [
            {"track_id": "A", "bbox": [0.62, 0.55, 0.96, 0.98], "score": 0.99, "frame_idx": 100},
            {"track_id": "A", "bbox": [0.70, 0.60, 0.98, 0.99], "score": 0.98, "frame_idx": 100},
            make_context_fallback_detection("Context"),
        ],
    }

    low_payload = mod.build_render_debug_sidecar_payload(
        clip_name="speaker_follow_clip1_0s_40s.mp4",
        window={"start_s": 0, "end_s": 1, "score": 0.9},
        binding_source="speaker_bindings",
        debug_mode=True,
        debug_show_faces=False,
        composition=mod.CompositionPlan(mode="single_person", primary_track_id="A"),
        segments=[segment],
        fps=24.0,
        src_w=1280,
        src_h=720,
        frame_detection_index=low_score_frame_detection_index,
    )
    high_payload = mod.build_render_debug_sidecar_payload(
        clip_name="speaker_follow_clip1_0s_40s.mp4",
        window={"start_s": 0, "end_s": 1, "score": 0.9},
        binding_source="speaker_bindings",
        debug_mode=True,
        debug_show_faces=False,
        composition=mod.CompositionPlan(mode="single_person", primary_track_id="A"),
        segments=[segment],
        fps=24.0,
        src_w=1280,
        src_h=720,
        frame_detection_index=high_score_frame_detection_index,
    )

    low_debug = low_payload["segments"][0]["target_debug"][0]
    high_debug = high_payload["segments"][0]["target_debug"][0]

    assert low_debug["target_source"] == "context_fallback"
    assert high_debug["target_source"] == "context_fallback"
    assert low_debug["fallback_used"] is True
    assert high_debug["fallback_used"] is True
    assert low_debug["fallback_quality"] == high_debug["fallback_quality"]
    assert low_debug["target_quality"] == low_debug["fallback_quality"]
    assert high_debug["target_quality"] == high_debug["fallback_quality"]


def test_build_render_debug_sidecar_payload_includes_hybrid_decision_state():
    mod = load_module()
    segment = mod.AdaptiveSegment(
        mode="single_person",
        start_s=0.0,
        end_s=1.0,
        primary_track_id="A",
        secondary_track_id=None,
    )

    payload = mod.build_render_debug_sidecar_payload(
        clip_name="speaker_follow_clip1_0s_40s.mp4",
        window={"start_s": 0, "end_s": 1, "score": 0.9},
        binding_source="speaker_follow_bindings_local",
        debug_mode=True,
        debug_show_faces=False,
        composition=mod.CompositionPlan(mode="single_person", primary_track_id="A"),
        segments=[segment],
        fps=24.0,
        src_w=1280,
        src_h=720,
        frame_detection_index=None,
        speaker_candidate_debug=[
            make_hybrid_debug_entry(
                start_time_ms=50,
                end_time_ms=650,
                active_audio_speaker_id="SPEAKER_02",
                active_audio_local_track_id="track_audio_2",
                chosen_track_id="A",
                chosen_local_track_id="track_A",
                decision_source="audio_boosted_visual",
                ambiguous=True,
                top_1_top_2_margin=0.018,
            )
        ],
    )

    hybrid_debug = payload["segments"][0]["hybrid_debug"]

    assert hybrid_debug["active_audio_speaker_id"] == "SPEAKER_02"
    assert hybrid_debug["active_audio_local_track_id"] == "track_audio_2"
    assert hybrid_debug["chosen_track_id"] == "A"
    assert hybrid_debug["chosen_local_track_id"] == "track_A"
    assert hybrid_debug["decision_source"] == "audio_boosted_visual"
    assert hybrid_debug["ambiguous"] is True
    assert hybrid_debug["top_1_top_2_margin"] == 0.018
    assert hybrid_debug["candidate_track_ids"] == ["A", "B"]


def test_build_debug_hud_filters_include_hybrid_decision_state():
    mod = load_module()

    filters = mod.build_debug_hud_filters(
        mode="single_person",
        binding_source="speaker_follow_bindings_local",
        follow_track_ids=["A"],
        raw_track_ids=["A"],
        face_track_ids=["face_A"],
        hybrid_debug={
            "active_audio_speaker_id": "SPEAKER_02",
            "active_audio_local_track_id": "track_audio_2",
            "chosen_track_id": "A",
            "chosen_local_track_id": "track_A",
            "decision_source": "audio_boosted_visual",
            "ambiguous": True,
            "top_1_top_2_margin": 0.018,
        },
    )
    joined = "\n".join(filters)

    assert "audio_speaker=SPEAKER_02" in joined
    assert "decision=audio_boosted_visual" in joined
    assert "ambiguous=yes" in joined
    assert "audio_local=track_audio_2" in joined
    assert "chosen_local=track_A" in joined
    assert "margin=0.018" in joined


def test_choose_window_composition_prefers_shared_for_ambiguous_hybrid_turn():
    mod = load_module()
    fps = 24.0
    duration_s = 20
    person_track_index = {
        'A': [
            mod.Detection(frame_idx=int(second * fps), x_center=1240 + second, y_center=920, width=420, height=820)
            for second in range(duration_s + 1)
        ],
        'B': [
            mod.Detection(frame_idx=int(second * fps), x_center=1620 + second, y_center=920, width=410, height=810)
            for second in range(duration_s + 1)
        ],
    }
    face_track_index = {}
    bindings = [
        {'track_id': 'A', 'start_time_ms': 0, 'end_time_ms': 13_500, 'word_count': 120},
        {'track_id': 'B', 'start_time_ms': 13_500, 'end_time_ms': 16_500, 'word_count': 24},
    ]

    plan = mod.choose_window_composition(
        clip_start_s=0,
        clip_end_s=20,
        fps=fps,
        src_w=3840,
        src_h=2160,
        bindings=bindings,
        person_track_index=person_track_index,
        face_track_index=face_track_index,
        speaker_candidate_debug=[
            make_hybrid_debug_entry(
                start_time_ms=3_000,
                end_time_ms=4_000,
                chosen_track_id="A",
                decision_source="unknown",
                ambiguous=True,
                top_1_top_2_margin=0.011,
            ),
            make_hybrid_debug_entry(
                start_time_ms=4_000,
                end_time_ms=5_000,
                chosen_track_id="A",
                decision_source="audio_boosted_visual",
                ambiguous=True,
                top_1_top_2_margin=0.016,
            ),
        ],
    )

    assert plan.mode == 'two_shared'
    assert {plan.primary_track_id, plan.secondary_track_id} == {'A', 'B'}


def test_choose_window_composition_prefers_shared_for_ambiguous_local_track_turn():
    mod = load_module()
    fps = 24.0
    duration_s = 20
    person_track_index = {
        'track_A': [
            mod.Detection(frame_idx=int(second * fps), x_center=1240 + second, y_center=920, width=420, height=820)
            for second in range(duration_s + 1)
        ],
        'track_B': [
            mod.Detection(frame_idx=int(second * fps), x_center=1620 + second, y_center=920, width=410, height=810)
            for second in range(duration_s + 1)
        ],
    }
    face_track_index = {}
    bindings = [
        {'track_id': 'track_A', 'start_time_ms': 0, 'end_time_ms': 13_500, 'word_count': 120},
        {'track_id': 'track_B', 'start_time_ms': 13_500, 'end_time_ms': 16_500, 'word_count': 24},
    ]

    plan = mod.choose_window_composition(
        clip_start_s=0,
        clip_end_s=20,
        fps=fps,
        src_w=3840,
        src_h=2160,
        bindings=bindings,
        person_track_index=person_track_index,
        face_track_index=face_track_index,
        speaker_candidate_debug=[
            make_hybrid_debug_entry(
                start_time_ms=4_000,
                end_time_ms=5_000,
                chosen_track_id="Global_Person_0",
                chosen_local_track_id="track_A",
                decision_source="audio_boosted_visual",
                ambiguous=True,
                top_1_top_2_margin=0.016,
                candidates=[
                    {
                        "local_track_id": "track_A",
                        "track_id": "Global_Person_0",
                        "blended_score": 0.31,
                        "asd_probability": 0.19,
                        "body_prior": 0.57,
                        "detection_confidence": 0.94,
                    },
                    {
                        "local_track_id": "track_B",
                        "track_id": "Global_Person_1",
                        "blended_score": 0.29,
                        "asd_probability": 0.17,
                        "body_prior": 0.52,
                        "detection_confidence": 0.90,
                    },
                ],
            ),
        ],
    )

    assert plan.mode == 'two_shared'
    assert {plan.primary_track_id, plan.secondary_track_id} == {'track_A', 'track_B'}


def test_choose_window_composition_does_not_widen_from_unrelated_ambiguous_entries():
    mod = load_module()
    fps = 24.0
    duration_s = 20
    person_track_index = {
        'A': [
            mod.Detection(frame_idx=int(second * fps), x_center=1240 + second, y_center=920, width=420, height=820)
            for second in range(duration_s + 1)
        ],
        'B': [
            mod.Detection(frame_idx=int(second * fps), x_center=1620 + second, y_center=920, width=410, height=810)
            for second in range(duration_s + 1)
        ],
    }
    face_track_index = {}
    bindings = [
        {'track_id': 'A', 'start_time_ms': 0, 'end_time_ms': 13_500, 'word_count': 120},
        {'track_id': 'B', 'start_time_ms': 13_500, 'end_time_ms': 16_500, 'word_count': 24},
    ]

    plan = mod.choose_window_composition(
        clip_start_s=0,
        clip_end_s=20,
        fps=fps,
        src_w=3840,
        src_h=2160,
        bindings=bindings,
        person_track_index=person_track_index,
        face_track_index=face_track_index,
        speaker_candidate_debug=[
            make_hybrid_debug_entry(
                start_time_ms=3_000,
                end_time_ms=4_000,
                chosen_track_id="A",
                chosen_local_track_id="track_A",
                decision_source="unknown",
                ambiguous=True,
                top_1_top_2_margin=0.011,
                candidates=[
                    {
                        "local_track_id": "track_A",
                        "track_id": "A",
                        "blended_score": 0.31,
                        "asd_probability": 0.19,
                        "body_prior": 0.57,
                        "detection_confidence": 0.94,
                    },
                ],
                ),
                make_hybrid_debug_entry(
                    start_time_ms=4_000,
                    end_time_ms=5_000,
                    chosen_track_id="B",
                    chosen_local_track_id="track_B",
                    decision_source="audio_boosted_visual",
                    ambiguous=True,
                    top_1_top_2_margin=0.016,
                candidates=[
                    {
                        "local_track_id": "track_B",
                        "track_id": "B",
                        "blended_score": 0.29,
                        "asd_probability": 0.17,
                        "body_prior": 0.52,
                        "detection_confidence": 0.90,
                    },
                ],
            ),
        ],
    )

    assert plan.mode == 'single_person'
    assert plan.primary_track_id == 'A'


def test_choose_window_composition_prefers_single_person_when_one_track_dominates():
    mod = load_module()
    person_track_index = {
        'A': [
            mod.Detection(frame_idx=0, x_center=1000, y_center=900, width=450, height=850),
            mod.Detection(frame_idx=12, x_center=1020, y_center=910, width=450, height=850),
        ],
        'B': [
            mod.Detection(frame_idx=0, x_center=2500, y_center=900, width=420, height=820),
            mod.Detection(frame_idx=12, x_center=2520, y_center=910, width=420, height=820),
        ],
    }
    face_track_index = {}
    bindings = [
        {'track_id': 'A', 'start_time_ms': 0, 'end_time_ms': 14_000, 'word_count': 140},
        {'track_id': 'B', 'start_time_ms': 14_500, 'end_time_ms': 16_000, 'word_count': 8},
    ]

    plan = mod.choose_window_composition(
        clip_start_s=0,
        clip_end_s=20,
        fps=24.0,
        src_w=3840,
        src_h=2160,
        bindings=bindings,
        person_track_index=person_track_index,
        face_track_index=face_track_index,
    )

    assert plan.mode == 'single_person'
    assert plan.primary_track_id == 'A'
    assert plan.secondary_track_id is None


def test_choose_window_composition_prefers_shared_two_shot_when_tracks_are_close():
    mod = load_module()
    fps = 24.0
    duration_s = 20
    person_track_index = {
        'A': [
            mod.Detection(frame_idx=int(second * fps), x_center=1240 + second, y_center=920, width=420, height=820)
            for second in range(duration_s + 1)
        ],
        'B': [
            mod.Detection(frame_idx=int(second * fps), x_center=1620 + second, y_center=920, width=410, height=810)
            for second in range(duration_s + 1)
        ],
    }
    face_track_index = {}
    bindings = [
        {'track_id': 'A', 'start_time_ms': 0, 'end_time_ms': 9_500, 'word_count': 70},
        {'track_id': 'B', 'start_time_ms': 8_500, 'end_time_ms': 19_500, 'word_count': 64},
    ]

    plan = mod.choose_window_composition(
        clip_start_s=0,
        clip_end_s=20,
        fps=fps,
        src_w=3840,
        src_h=2160,
        bindings=bindings,
        person_track_index=person_track_index,
        face_track_index=face_track_index,
    )

    assert plan.mode == 'two_shared'
    assert {plan.primary_track_id, plan.secondary_track_id} == {'A', 'B'}


def test_choose_window_composition_prefers_split_when_tracks_are_far_apart():
    mod = load_module()
    fps = 24.0
    duration_s = 20
    person_track_index = {
        'A': [
            mod.Detection(frame_idx=int(second * fps), x_center=700 + (second * 2), y_center=920, width=470, height=860)
            for second in range(duration_s + 1)
        ],
        'B': [
            mod.Detection(frame_idx=int(second * fps), x_center=3050 + (second * 2), y_center=920, width=470, height=860)
            for second in range(duration_s + 1)
        ],
    }
    face_track_index = {}
    bindings = [
        {'track_id': 'A', 'start_time_ms': 0, 'end_time_ms': 10_000, 'word_count': 62},
        {'track_id': 'B', 'start_time_ms': 10_000, 'end_time_ms': 20_000, 'word_count': 58},
    ]

    plan = mod.choose_window_composition(
        clip_start_s=0,
        clip_end_s=20,
        fps=fps,
        src_w=3840,
        src_h=2160,
        bindings=bindings,
        person_track_index=person_track_index,
        face_track_index=face_track_index,
    )

    assert plan.mode == 'two_split'
    assert [plan.primary_track_id, plan.secondary_track_id] == ['A', 'B']


def test_choose_adaptive_split_segments_collapses_to_single_when_secondary_disappears():
    mod = load_module()
    fps = 24.0
    person_track_index = {
        'A': [
            mod.Detection(frame_idx=int(second * fps), x_center=700, y_center=920, width=470, height=860)
            for second in range(0, 21)
        ],
        'B': [
            mod.Detection(frame_idx=int(second * fps), x_center=3050, y_center=920, width=470, height=860)
            for second in range(0, 9)
        ],
    }
    face_track_index = {}
    bindings = [
        {'track_id': 'A', 'start_time_ms': 0, 'end_time_ms': 20_000, 'word_count': 80},
        {'track_id': 'B', 'start_time_ms': 0, 'end_time_ms': 7_000, 'word_count': 40},
    ]

    segments = mod.choose_adaptive_split_segments(
        primary_track_id='A',
        secondary_track_id='B',
        clip_start_s=0,
        clip_end_s=20,
        fps=fps,
        src_w=3840,
        src_h=2160,
        bindings=bindings,
        person_track_index=person_track_index,
        face_track_index=face_track_index,
    )

    assert any(segment.mode == 'two_split' for segment in segments)
    assert any(segment.mode == 'single_person' and segment.primary_track_id == 'A' for segment in segments)
    assert segments[-1].mode == 'single_person'


def test_choose_adaptive_split_segments_rejects_same_person_overlap_and_uses_single():
    mod = load_module()
    fps = 24.0
    person_track_index = {
        'A': [
            mod.Detection(frame_idx=int(second * fps), x_center=1400, y_center=920, width=470, height=860)
            for second in range(0, 21)
        ],
        'B': [
            mod.Detection(frame_idx=int(second * fps), x_center=1460, y_center=930, width=465, height=855)
            for second in range(0, 21)
        ],
    }
    face_track_index = {}
    bindings = [
        {'track_id': 'A', 'start_time_ms': 0, 'end_time_ms': 20_000, 'word_count': 100},
    ]

    segments = mod.choose_adaptive_split_segments(
        primary_track_id='A',
        secondary_track_id='B',
        clip_start_s=0,
        clip_end_s=20,
        fps=fps,
        src_w=3840,
        src_h=2160,
        bindings=bindings,
        person_track_index=person_track_index,
        face_track_index=face_track_index,
    )

    assert segments
    assert all(segment.mode == 'single_person' for segment in segments)
    assert all(segment.primary_track_id == 'A' for segment in segments)


def test_motion_profile_for_composition_makes_single_person_snappier_than_shared():
    mod = load_module()

    single = mod.motion_profile_for_composition('single_person')
    shared = mod.motion_profile_for_composition('two_shared')
    split = mod.motion_profile_for_composition('two_split', out_h=mod.SPLIT_PANEL_HEIGHT)

    assert single.camera_zoom > shared.camera_zoom
    assert single.ema_smoothing < shared.ema_smoothing
    assert single.keyframe_step_s < shared.keyframe_step_s
    assert split.camera_zoom >= shared.camera_zoom
    assert split.ema_smoothing <= shared.ema_smoothing


def test_build_track_index_from_raw_detections_uses_correct_keys():
    mod = load_module()
    detections = [
        {
            'track_id': 'A',
            'frame_idx': 0,
            'x_center': 500,
            'y_center': 400,
            'width': 200,
            'height': 600,
            'score': 0.9,
        },
        {
            'track_id': 'A',
            'frame_idx': 12,
            'x_center': 510,
            'y_center': 410,
            'width': 210,
            'height': 610,
            'score': 0.88,
        },
        {
            'track_id': 'B',
            'frame_idx': 0,
            'x_center': 2000,
            'y_center': 800,
            'width': 300,
            'height': 700,
            'score': 0.85,
        },
    ]

    index = mod.build_track_index(detections)

    assert set(index.keys()) == {'A', 'B'}
    assert len(index['A']) == 2
    assert len(index['B']) == 1
    det_a_0 = index['A'][0]
    assert det_a_0.frame_idx == 0
    assert det_a_0.x_center == 500


def test_build_mask_stability_index_from_visual_returns_none_without_field():
    mod = load_module()
    assert mod.build_mask_stability_index_from_visual({}) is None
    assert mod.build_mask_stability_index_from_visual({"mask_stability_signals": None}) is None


def test_parse_mask_stability_signals_payload_hybrid_dict_with_aggregate_keys():
    """Worker Phase-1 dict may bundle bbox aggregates plus ``entries``; parser uses entries only."""
    mod = load_module()
    raw = {
        "signal_version": "worker_bbox_v3",
        "mask_signal_meta": {"payload_version": "1.1.0", "segmentation_provenance": "yolo_seg_masks:yolo26m-seg"},
        "segmentation_mask_proxies": {"active": True},
        "entries": [
            {
                "frame_idx": 0,
                "track_id": "track_A",
                "stability": 0.88,
                "x_center": 0.42,
                "y_center": 0.55,
                "mask_area_in_bbox_ratio": 0.71,
            },
        ],
    }
    idx = mod.parse_mask_stability_signals_payload(raw)
    assert (0, "track_A") in idx.spatial
    assert idx.spatial[(0, "track_A")][0] == (0.42, 0.55, 0.88)


def test_build_track_index_prefers_mask_stable_duplicate_detection():
    """When one duplicate body box aligns with spatial mask stability, keep that instance (anti-jitter)."""
    mod = load_module()
    tracks = [
        {
            "track_id": "A",
            "frame_idx": 0,
            "x_center": 520.0,
            "y_center": 400.0,
            "width": 200.0,
            "height": 600.0,
            "confidence": 0.92,
        },
        {
            "track_id": "A",
            "frame_idx": 0,
            "x_center": 300.0,
            "y_center": 400.0,
            "width": 200.0,
            "height": 600.0,
            "confidence": 0.90,
        },
    ]
    raw = [
        {
            "frame_idx": 0,
            "track_id": "A",
            "x_center": 0.3,
            "y_center": 0.5,
            "stability": 1.0,
        },
    ]
    index = mod.parse_mask_stability_signals_payload(raw)
    resolved = mod.build_track_index(
        tracks,
        mask_stability_index=index,
        src_w=1000,
        src_h=800,
    )
    assert len(resolved["A"]) == 1
    assert abs(resolved["A"][0].x_center - 300.0) < 0.01


def test_tiebreak_two_visible_tracks_prefers_overlap_visible_hint():
    mod = load_module()
    left = {
        "track_id": "track_L",
        "bbox": [0.06, 0.12, 0.44, 0.92],
        "score": 0.91,
        "frame_idx": 0,
    }
    right = {
        "track_id": "track_R",
        "bbox": [0.56, 0.12, 0.94, 0.92],
        "score": 0.91,
        "frame_idx": 0,
    }
    decisions = [
        {
            "start_time_ms": 0,
            "end_time_ms": 10_000,
            "stay_wide": False,
            "camera_target_track_id": None,
            "camera_target_local_track_id": None,
            "visible_track_ids": ["track_R"],
        }
    ]
    picked = mod.tiebreak_two_visible_tracks_for_follow(
        frame_detections=[left, right],
        frame_width=1280,
        frame_height=720,
        abs_ms=500,
        video_fps=25.0,
        mask_stability_index=None,
        overlap_follow_decisions=decisions,
        prefer_local_track_ids=False,
        last_resolved_track_id=None,
    )
    assert picked == "track_R"


def test_resolve_follow_identity_tiebreak_respects_overlap_visible_hints():
    mod = load_module()
    frame_detections = [
        {
            "track_id": "track_L",
            "bbox": [0.06, 0.12, 0.44, 0.92],
            "score": 0.91,
            "frame_idx": 0,
        },
        {
            "track_id": "track_R",
            "bbox": [0.56, 0.12, 0.94, 0.92],
            "score": 0.91,
            "frame_idx": 0,
        },
    ]
    decisions = [
        {
            "start_time_ms": 0,
            "end_time_ms": 10_000,
            "stay_wide": False,
            "camera_target_track_id": None,
            "visible_track_ids": ["track_R"],
        }
    ]
    assert (
        mod.resolve_follow_identity(
            [],
            500,
            overlap_follow_decisions=decisions,
            frame_detections=frame_detections,
            video_fps=25.0,
        )
        == "track_R"
    )


def test_build_frame_detection_index_builds_per_frame_lookup():
    mod = load_module()
    detections = [
        {'track_id': 'A', 'frame_idx': 0, 'bbox': [0.1, 0.1, 0.5, 0.9], 'score': 0.9},
        {'track_id': 'B', 'frame_idx': 0, 'bbox': [0.5, 0.1, 0.9, 0.9], 'score': 0.85},
        {'track_id': 'A', 'frame_idx': 12, 'bbox': [0.12, 0.12, 0.52, 0.92], 'score': 0.88},
    ]

    index = mod.build_frame_detection_index(detections)

    assert set(index.keys()) == {0, 12}
    assert len(index[0]) == 2
    assert len(index[12]) == 1


def test_crop_dimensions_respects_nine_sixteen_aspect_ratio():
    mod = load_module()
    crop_w, crop_h = mod.crop_dimensions(3840, 2160, camera_zoom=1.0)

    actual_ratio = crop_w / crop_h
    expected_ratio = 9.0 / 16.0
    assert abs(actual_ratio - expected_ratio) < 0.01


# ─── Phase 1 debug video tests ───

def test_phase1_debug_grid_layout_non_overlapping():
    debug_mod = load_phase1_debug_module()

    faces = [
        {
            "track_id": "face_0",
            "frame_idx": 0,
            "x_center": 500,
            "y_center": 300,
            "width": 100,
            "height": 100,
        },
        {
            "track_id": "face_1",
            "frame_idx": 0,
            "x_center": 700,
            "y_center": 300,
            "width": 100,
            "height": 100,
        },
    ]

    layout = debug_mod.grid_layout(faces, src_w=1920, src_h=1080, panel_w=480)

    assert len(layout) == 2
    assert layout[0]["panel_x"] != layout[1]["panel_x"] or layout[0]["panel_y"] != layout[1]["panel_y"]


def test_phase1_debug_grid_layout_allocates_within_panel_bounds():
    debug_mod = load_phase1_debug_module()

    faces = [
        {
            "track_id": f"face_{i}",
            "frame_idx": 0,
            "x_center": 200 + i * 300,
            "y_center": 300,
            "width": 120,
            "height": 120,
        }
        for i in range(5)
    ]

    layout = debug_mod.grid_layout(faces, src_w=1920, src_h=1080, panel_w=480)

    for entry in layout:
        assert entry["panel_x"] >= 0
        assert entry["panel_y"] >= 0
        assert entry["panel_x"] + entry["panel_w"] <= 480