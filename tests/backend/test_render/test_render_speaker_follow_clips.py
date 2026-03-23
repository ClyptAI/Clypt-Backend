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

    det = mod._track_anchor_candidate('A', 0, 24.0, person_track_index, face_track_index)

    assert det is not None
    assert det.width == 500
    assert det.height == 900
    assert det.y_center == 900

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


def test_build_single_track_path_aggressive_profile_increases_motion_visibility():
    mod = load_module()
    fps = 24.0
    person_track_index = {
        'A': [
            mod.Detection(frame_idx=int(second * fps), x_center=900 + (second * 120), y_center=920, width=420, height=820)
            for second in range(5)
        ]
    }
    face_track_index = {
        'A': [
            mod.Detection(frame_idx=int(second * fps), x_center=930 + (second * 140), y_center=760, width=180, height=180)
            for second in range(5)
        ]
    }
    aggressive = mod.MotionProfile(camera_zoom=1.36, keyframe_step_s=0.2, ema_smoothing=2, y_head_bias=0.12)
    calm = mod.MotionProfile(camera_zoom=1.08, keyframe_step_s=0.5, ema_smoothing=8, y_head_bias=0.16)

    x_aggressive, _ = mod.build_single_track_path(
        track_id='A',
        clip_start_s=0,
        clip_end_s=4,
        fps=fps,
        src_w=3840,
        src_h=2160,
        person_track_index=person_track_index,
        face_track_index=face_track_index,
        motion_profile=aggressive,
    )
    x_calm, _ = mod.build_single_track_path(
        track_id='A',
        clip_start_s=0,
        clip_end_s=4,
        fps=fps,
        src_w=3840,
        src_h=2160,
        person_track_index=person_track_index,
        face_track_index=face_track_index,
        motion_profile=calm,
    )

    aggressive_span = max(v for _, v in x_aggressive) - min(v for _, v in x_aggressive)
    calm_span = max(v for _, v in x_calm) - min(v for _, v in x_calm)

    assert len(x_aggressive) > len(x_calm)
    assert aggressive_span > calm_span


def test_render_clip_draws_multiple_boxes_for_multiple_overlay_paths():
    mod = load_module()
    captured = {}

    def fake_run(cmd, capture_output, text):
        captured['cmd'] = cmd

        class Result:
            returncode = 0

        return Result()

    original_run = mod.subprocess.run
    mod.subprocess.run = fake_run
    try:
        overlay_paths = [
            mod.OverlayPath(
                x_keyframes=[(0.0, 100.0), (1.0, 120.0)],
                y_keyframes=[(0.0, 80.0), (1.0, 90.0)],
                box_x_keyframes=[(0.0, 10.0), (1.0, 12.0)],
                box_y_keyframes=[(0.0, 20.0), (1.0, 22.0)],
                box_w_keyframes=[(0.0, 150.0), (1.0, 155.0)],
                box_h_keyframes=[(0.0, 220.0), (1.0, 225.0)],
            ),
            mod.OverlayPath(
                x_keyframes=[(0.0, 200.0), (1.0, 220.0)],
                y_keyframes=[(0.0, 180.0), (1.0, 190.0)],
                box_x_keyframes=[(0.0, 30.0), (1.0, 32.0)],
                box_y_keyframes=[(0.0, 40.0), (1.0, 42.0)],
                box_w_keyframes=[(0.0, 140.0), (1.0, 142.0)],
                box_h_keyframes=[(0.0, 210.0), (1.0, 212.0)],
                color=mod.OVERLAY_SECONDARY_BOX_COLOR,
            ),
        ]

        mod.render_clip(
            video_path=Path('/tmp/fake.mp4'),
            out_path=Path('/tmp/fake_out.mp4'),
            start_s=0,
            duration_s=1,
            x_keyframes=[(0.0, 0.0)],
            y_keyframes=[(0.0, 0.0)],
            overlay_paths=overlay_paths,
            src_h=2160,
            camera_zoom=1.18,
        )
    finally:
        mod.subprocess.run = original_run

    cmd = captured['cmd']
    filter_chain = cmd[cmd.index('-vf') + 1]
    assert filter_chain.count('drawbox=') == 2


def test_build_track_index_resolves_same_frame_duplicates_by_local_continuity():
    mod = load_module()

    tracks = [
        {"track_id": "Global_Person_0", "frame_idx": 0, "x_center": 1400.0, "y_center": 800.0, "width": 420.0, "height": 820.0, "confidence": 0.93},
        {"track_id": "Global_Person_0", "frame_idx": 1, "x_center": 420.0, "y_center": 900.0, "width": 780.0, "height": 1200.0, "confidence": 0.97},
        {"track_id": "Global_Person_0", "frame_idx": 1, "x_center": 1420.0, "y_center": 805.0, "width": 430.0, "height": 830.0, "confidence": 0.91},
        {"track_id": "Global_Person_0", "frame_idx": 2, "x_center": 1445.0, "y_center": 810.0, "width": 425.0, "height": 825.0, "confidence": 0.92},
    ]

    track_index = mod.build_track_index(tracks)

    assert len(track_index["Global_Person_0"]) == 3
    frame_one = next(det for det in track_index["Global_Person_0"] if det.frame_idx == 1)
    assert frame_one.x_center == 1420.0
    assert frame_one.width == 430.0
