import importlib.util
import sys
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[3] / "backend" / "test-render" / "render_phase1_debug_video.py"
    name = "render_phase1_debug_video_overlap_test"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_active_speaker_state_uses_overlap_artifacts_without_local_mode():
    mod = load_module()
    audio = {
        "active_speakers_local": [
            {
                "start_time_ms": 0,
                "end_time_ms": 1000,
                "visible_local_track_ids": ["local_A", "local_B"],
                "visible_track_ids": ["A", "B"],
                "offscreen_audio_speaker_ids": ["SPEAKER_03"],
                "overlap": True,
                "decision_source": "phase1_overlap",
            }
        ]
    }

    state = mod.active_speaker_state_at_ms(
        audio,
        500,
        available_track_ids={"A", "B", "C"},
    )

    assert state["visible_track_ids"] == ["A", "B"]
    assert state["offscreen_audio_speaker_ids"] == ["SPEAKER_03"]
    assert state["overlap"] is True
    assert state["decision_source"] == "phase1_overlap"


def test_build_hud_lines_include_overlap_visible_and_offscreen_activity():
    mod = load_module()

    lines = mod.build_hud_lines(
        timestamp_ms=1250,
        raw_track_id="A",
        follow_track_id="A",
        current_word="hello",
        binding_source="speaker_follow_bindings_local",
        track_source="tracks_local",
        active_visible_track_ids=["A", "B"],
        offscreen_audio_speaker_ids=["SPEAKER_03", "SPEAKER_04"],
        overlap_active=True,
    )

    assert "overlap: yes" in lines
    assert "active visible: A,B" in lines
    assert "offscreen active audio: SPEAKER_03,SPEAKER_04" in lines


def test_role_style_marks_non_follow_overlap_speakers_as_active():
    mod = load_module()

    style = mod.role_style_for_track(
        "B",
        raw_track_id="A",
        follow_track_id="A",
        active_track_ids={"A", "B"},
    )

    assert style["label_suffix"] == "ACTIVE"
    assert style["thickness"] >= 3



def test_select_pose_frame_index_prefers_explicit_pose_detections():
    mod = load_module()
    visual = {
        "pose_detections": [
            {
                "track_id": "pose_A",
                "confidence": 0.91,
                "segment_start_ms": 0,
                "segment_end_ms": 1000,
                "timestamped_objects": [
                    {
                        "time_ms": 0,
                        "track_id": "pose_A",
                        "confidence": 0.91,
                        "bounding_box": {"left": 0.1, "top": 0.1, "right": 0.3, "bottom": 0.5},
                    }
                ],
            }
        ],
        "person_detections": [
            {
                "track_id": "person_A",
                "confidence": 0.5,
                "segment_start_ms": 0,
                "segment_end_ms": 1000,
                "timestamped_objects": [
                    {
                        "time_ms": 0,
                        "track_id": "person_A",
                        "confidence": 0.5,
                        "bounding_box": {"left": 0.6, "top": 0.1, "right": 0.8, "bottom": 0.5},
                    }
                ],
            }
        ],
    }

    pose_frame_index, source = mod.select_pose_frame_index(
        visual,
        fps=30.0,
        frame_width=1280,
        frame_height=720,
    )

    assert source == "pose_detections"
    assert 0 in pose_frame_index
    assert pose_frame_index[0][0]["kind"] == "pose"
    assert pose_frame_index[0][0]["track_id"] == "pose_A"


def test_build_hud_lines_include_branch_summary_and_pose_source():
    mod = load_module()

    lines = mod.build_hud_lines(
        timestamp_ms=1250,
        raw_track_id="A",
        follow_track_id="A",
        current_word="hello",
        binding_source="speaker_follow_bindings_local",
        track_source="tracks_local",
        active_visible_track_ids=["A", "B"],
        offscreen_audio_speaker_ids=["SPEAKER_03", "SPEAKER_04"],
        overlap_active=True,
        pose_track_source="person_detections",
        visual_identity_count=4,
        audio_visual_mapping_count=2,
        span_assignment_count=6,
    )

    assert "pose source: person_detections" in lines
    assert "branch outputs: visual_identities=4 mappings=2 spans=6" in lines


def test_select_render_detections_prefers_dominant_larger_box_for_active_track():
    mod = load_module()
    frame_detections = [
        {
            "track_id": "track_2",
            "frame_idx": 10,
            "x1": 0,
            "y1": 520,
            "x2": 320,
            "y2": 1080,
            "confidence": 0.55,
        },
        {
            "track_id": "track_2",
            "frame_idx": 10,
            "x1": 110,
            "y1": 85,
            "x2": 1120,
            "y2": 1030,
            "confidence": 0.51,
        },
        {
            "track_id": "track_9",
            "frame_idx": 10,
            "x1": 1200,
            "y1": 120,
            "x2": 1700,
            "y2": 980,
            "confidence": 0.88,
        },
    ]

    selected = mod.select_render_detections(
        frame_detections,
        raw_track_id="track_2",
        follow_track_id="track_2",
        active_track_ids={"track_2"},
        frame_width=1920,
        frame_height=1080,
    )

    selected_track_2 = [det for det in selected if det["track_id"] == "track_2"]
    assert len(selected_track_2) == 1
    assert selected_track_2[0]["x1"] == 110
    assert selected_track_2[0]["y1"] == 85
    assert any(det["track_id"] == "track_9" for det in selected)


def test_select_render_detections_can_rescue_active_track_with_larger_nearby_track():
    mod = load_module()
    frame_detections = [
        {
            "track_id": "track_2",
            "frame_idx": 334,
            "x1": 4,
            "y1": 460,
            "x2": 454,
            "y2": 963,
            "confidence": 0.59,
        },
        {
            "track_id": "track_12",
            "frame_idx": 334,
            "x1": 212,
            "y1": 95,
            "x2": 1220,
            "y2": 1072,
            "confidence": 0.93,
        },
    ]

    selected = mod.select_render_detections(
        frame_detections,
        raw_track_id="track_2",
        follow_track_id="track_2",
        active_track_ids={"track_2"},
        frame_width=1920,
        frame_height=1080,
    )

    assert len(selected) == 1
    assert selected[0]["track_id"] == "track_12"
    assert selected[0]["x1"] == 212
    assert selected[0]["y1"] == 95
    assert selected[0]["_render_role_track_id"] == "track_2"


def test_select_render_detections_uses_lone_visible_box_for_single_target_track():
    mod = load_module()
    frame_detections = [
        {
            "track_id": "track_12",
            "frame_idx": 334,
            "x1": 212,
            "y1": 95,
            "x2": 1220,
            "y2": 1072,
            "confidence": 0.93,
        },
    ]

    selected = mod.select_render_detections(
        frame_detections,
        raw_track_id="track_2",
        follow_track_id="track_2",
        active_track_ids={"track_2"},
        frame_width=1920,
        frame_height=1080,
    )

    assert len(selected) == 1
    assert selected[0]["track_id"] == "track_12"
    assert selected[0]["_render_role_track_id"] == "track_2"


def test_resolve_render_binding_ids_uses_lone_visible_track_when_local_binding_missing():
    mod = load_module()
    frame_detections = [
        {
            "track_id": "track_12",
            "frame_idx": 334,
            "x1": 212,
            "y1": 95,
            "x2": 1220,
            "y2": 1072,
            "confidence": 0.93,
        },
    ]

    raw_track_id, follow_track_id = mod.resolve_render_binding_ids(
        raw_track_id=None,
        follow_track_id=None,
        frame_detections=frame_detections,
    )

    assert raw_track_id == "track_12"
    assert follow_track_id == "track_12"


def test_resolve_render_binding_ids_prefers_dominant_box_over_fragment_when_unbound():
    mod = load_module()
    frame_detections = [
        {
            "track_id": "track_51",
            "frame_idx": 3769,
            "x1": 187,
            "y1": 99,
            "x2": 1207,
            "y2": 1073,
            "confidence": 0.948,
        },
        {
            "track_id": "track_67",
            "frame_idx": 3769,
            "x1": 2,
            "y1": 454,
            "x2": 430,
            "y2": 1066,
            "confidence": 0.681,
        },
    ]

    raw_track_id, follow_track_id = mod.resolve_render_binding_ids(
        raw_track_id=None,
        follow_track_id=None,
        frame_detections=frame_detections,
        frame_width=1920,
        frame_height=1080,
    )

    assert raw_track_id == "track_51"
    assert follow_track_id == "track_51"

