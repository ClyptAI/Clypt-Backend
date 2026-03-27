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
