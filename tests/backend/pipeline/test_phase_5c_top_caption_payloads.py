from __future__ import annotations

import json
from pathlib import Path
import tempfile

from backend.pipeline import phase_5c_top_caption_payloads as subject


def test_clip_context_collects_overlapping_phase2a_nodes():
    payload = {
        "clip_start_ms": 1000,
        "clip_end_ms": 5200,
        "final_score": 91.0,
        "combined_transcript": "that answer made the whole room freeze up",
        "justification": "High social tension and awkwardness.",
    }
    nodes = [
        {
            "start_time": 0.5,
            "end_time": 2.1,
            "transcript_segment": "that answer made",
            "vocal_delivery": "Dry and confrontational.",
            "content_mechanisms": {
                "humor": {"present": False, "type": "", "intensity": 0.0},
                "emotion": {"present": False, "type": "", "intensity": 0.0},
                "social": {"present": True, "type": "genuine disagreement", "intensity": 0.8},
                "expertise": {"present": False, "type": "", "intensity": 0.0},
            },
        },
        {
            "start_time": 4.4,
            "end_time": 6.8,
            "transcript_segment": "the whole room freeze up",
            "vocal_delivery": "A pause hangs in the room.",
            "content_mechanisms": {
                "humor": {"present": False, "type": "", "intensity": 0.0},
                "emotion": {"present": True, "type": "tension", "intensity": 0.6},
                "social": {"present": True, "type": "awkwardness", "intensity": 0.9},
                "expertise": {"present": False, "type": "", "intensity": 0.0},
            },
        },
    ]

    context = subject._clip_context(payload, nodes)

    assert context["dominant_mode"] == "social"
    assert context["mechanism_summary"]["social"]["type"] == "awkwardness"
    assert context["vocal_delivery_notes"] == [
        "Dry and confrontational.",
        "A pause hangs in the room.",
    ]


def test_normalize_top_caption_plan_filters_opening_transcript_repeats():
    clip_context = {
        "combined_transcript": "this got awkward fast and then it got worse",
    }
    plan = {
        "decision": "caption",
        "selected_text": "this got awkward fast",
        "selected_tone": "tense",
        "reasoning": "Testing normalization.",
        "variants": [
            {"text": "this got awkward fast", "tone": "tense", "why_it_fits": "Too close."},
            {"text": "That answer made it worse", "tone": "pointed", "why_it_fits": "Better."},
            {"text": "That answer made it worse", "tone": "pointed", "why_it_fits": "Duplicate."},
        ],
    }

    normalized = subject._normalize_top_caption_plan(plan, clip_context)

    assert normalized["top_caption"]["text"] == "That answer made it worse"
    assert [variant["text"] for variant in normalized["top_caption_variants"]] == [
        "That answer made it worse",
    ]


def test_main_writes_top_captioned_payload_using_injected_generator():
    def fake_generator(clip_context: dict, style_hint: str):
        assert clip_context["dominant_mode"] == "expertise"
        assert style_hint == ""
        return {
            "decision": "caption",
            "selected_text": "That detail changes everything",
            "selected_tone": "sharp",
            "reasoning": "The clip pivots on one clarifying detail.",
            "variants": [
                {
                    "text": "That detail changes everything",
                    "tone": "sharp",
                    "why_it_fits": "Best concise hook.",
                },
                {
                    "text": "The simplest answer wins",
                    "tone": "clean",
                    "why_it_fits": "Explainer angle.",
                },
                {
                    "text": "This is the whole point",
                    "tone": "confident",
                    "why_it_fits": "Punchy summary.",
                },
            ],
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        input_path = tmp_path / "remotion_payloads_array.json"
        nodes_path = tmp_path / "phase_2a_nodes.json"
        output_path = tmp_path / "remotion_payloads_array_top_captioned.json"

        input_payloads = [
            {
                "clip_start_ms": 0,
                "clip_end_ms": 5000,
                "final_score": 87.0,
                "combined_transcript": "here is the part that makes the explanation click",
                "captions": [{"text": "here is the part", "start_ms": 0, "end_ms": 1200}],
            }
        ]
        nodes_payload = [
            {
                "start_time": 0.0,
                "end_time": 5.0,
                "transcript_segment": "here is the part that makes the explanation click",
                "vocal_delivery": "Clear and confident.",
                "content_mechanisms": {
                    "humor": {"present": False, "type": "", "intensity": 0.0},
                    "emotion": {"present": False, "type": "", "intensity": 0.0},
                    "social": {"present": False, "type": "", "intensity": 0.0},
                    "expertise": {"present": True, "type": "elegant simplification", "intensity": 0.9},
                },
            }
        ]

        input_path.write_text(json.dumps(input_payloads), encoding="utf-8")
        nodes_path.write_text(json.dumps(nodes_payload), encoding="utf-8")

        result = subject.main(
            input_path=input_path,
            nodes_path=nodes_path,
            output_path=output_path,
            generator=fake_generator,
        )

        assert result["payload_count"] == 1
        written = json.loads(output_path.read_text(encoding="utf-8"))
        assert written[0]["captions"][0]["text"] == "here is the part"
        assert written[0]["top_caption"]["text"] == "That detail changes everything"
        assert len(written[0]["top_caption_variants"]) == 3
