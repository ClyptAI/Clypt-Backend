#!/usr/bin/env python3
"""
Phase 5C: generate editorial top-caption hooks for already-selected clips.

This stage is downstream-only:
- input: an existing Remotion payload array
- optional input: Phase 2A nodes for tone/mechanism context
- output: a new payload array with optional `top_caption` metadata

It is intentionally separate from clip selection so teams can work on
subtitles, editorial hooks, and selection logic independently.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Callable, Literal

from google import genai
from google.genai.types import HttpOptions
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUTS_DIR = ROOT / "outputs"
DEFAULT_NODES_PATH = OUTPUTS_DIR / "phase_2a_nodes.json"
DEFAULT_INPUT_CANDIDATES = (
    OUTPUTS_DIR / "remotion_payloads_array_captioned.json",
    OUTPUTS_DIR / "remotion_payloads_array.json",
    OUTPUTS_DIR / "crowd_remotion_payloads_array_captioned.json",
    OUTPUTS_DIR / "crowd_remotion_payloads_array.json",
    OUTPUTS_DIR / "remotion_payloads_array_audience.json",
)

PROJECT_ID = "clypt-v2"
GEMINI_LOCATION = "global"
GEMINI_MODEL = "gemini-3.1-pro-preview"
MAX_VARIANTS = 3
MAX_WORDS = 10
MAX_CHARS = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase_5c_top_caption")


class TopCaptionVariant(BaseModel):
    text: str = Field(default="", max_length=80)
    tone: str = Field(default="", max_length=40)
    why_it_fits: str = Field(default="", max_length=240)


class TopCaptionPlan(BaseModel):
    decision: Literal["caption", "no_caption"]
    selected_text: str = Field(default="", max_length=80)
    selected_tone: str = Field(default="", max_length=40)
    reasoning: str = Field(default="", max_length=400)
    variants: list[TopCaptionVariant] = Field(default_factory=list)


TopCaptionGenerator = Callable[[dict, str], TopCaptionPlan | dict]


TOP_CAPTION_SYSTEM_INSTRUCTION = """\
You write short editorial hook text for vertical clips.

This top caption is NOT a transcript and NOT a subtitle. Spoken subtitles will
already appear elsewhere in the frame. Your job is to add an optional
creator-style editorial line at the top of the video that makes the clip feel
native to short-form platforms.

Rules:
- Return either `caption` or `no_caption`.
- If you choose `caption`, provide exactly 3 distinct variants and mark one best pick.
- Each caption must be 4 to 10 words.
- Ground every caption in the provided clip context. Do not invent facts, names, motives, or stakes.
- Do not simply repeat the first spoken subtitle line.
- Match the clip's mode: funny, tense, contrarian, reaction, explainer, emotional, or awkward.
- Favor punchy, natural, creator-like phrasing over generic clickbait.
- If the clip already hooks itself and any top caption would weaken it, choose `no_caption`.
"""


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _resolve_input_path(input_path: str | Path | None) -> Path:
    if input_path:
        return Path(input_path)

    override = str(os.getenv("TOP_CAPTION_PAYLOAD_INPUT_PATH", "") or "").strip()
    if override:
        return Path(override)

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No clip payload file found. Checked: "
        + ", ".join(str(path) for path in DEFAULT_INPUT_CANDIDATES)
    )


def _resolve_nodes_path(nodes_path: str | Path | None) -> Path:
    if nodes_path:
        return Path(nodes_path)

    override = str(os.getenv("TOP_CAPTION_NODES_PATH", "") or "").strip()
    if override:
        return Path(override)

    return DEFAULT_NODES_PATH


def _resolve_output_path(output_path: str | Path | None, input_payload_path: Path) -> Path:
    if output_path:
        return Path(output_path)

    override = str(os.getenv("TOP_CAPTION_PAYLOAD_OUTPUT_PATH", "") or "").strip()
    if override:
        return Path(override)

    return input_payload_path.with_name(f"{input_payload_path.stem}_top_captioned.json")


def _load_payload_array(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, list) else [payload]


def _load_nodes(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def _content_hash(clip: dict) -> str:
    raw = json.dumps(clip, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def _overlapping_nodes(clip: dict, nodes: list[dict]) -> list[dict]:
    clip_start = int(clip.get("clip_start_ms", 0) or 0)
    clip_end = int(clip.get("clip_end_ms", clip_start) or clip_start)
    matched = []
    for node in nodes:
        ns = int(float(node.get("start_time", 0)) * 1000)
        ne = int(float(node.get("end_time", ns / 1000)) * 1000)
        if ns < clip_end and ne > clip_start:
            matched.append(node)
    return matched


def _build_clip_context(clip: dict, nodes: list[dict]) -> str:
    parts: list[str] = []
    transcript = str(clip.get("transcript_text", "") or "").strip()
    if not transcript:
        captions = clip.get("captions", [])
        if isinstance(captions, list) and captions:
            transcript = " ".join(
                str(c.get("text", "")) for c in captions if isinstance(c, dict)
            ).strip()
    if transcript:
        parts.append(f"Transcript: {transcript}")

    justification = str(clip.get("justification", "") or "").strip()
    if justification:
        parts.append(f"Justification: {justification}")

    overlapping = _overlapping_nodes(clip, nodes)
    for node in overlapping[:3]:
        mechs = node.get("content_mechanisms", {})
        if isinstance(mechs, str):
            try:
                mechs = json.loads(mechs)
            except Exception:
                mechs = {}
        active = []
        for dim in ("humor", "emotion", "social", "expertise"):
            d = mechs.get(dim, {})
            if d.get("present"):
                active.append(f"{dim}({d.get('type', '?')}, {d.get('intensity', 0):.1f})")
        if active:
            parts.append(f"Mechanisms: {', '.join(active)}")
        vd = str(node.get("vocal_delivery", "") or "").strip()
        if vd:
            parts.append(f"Vocal delivery: {vd}")

    score = clip.get("final_score")
    if score is not None:
        parts.append(f"Score: {score}")

    return "\n".join(parts) if parts else "No context available."


def _default_gemini_generator(clip: dict, context: str) -> TopCaptionPlan:
    import os as _os
    _os.environ.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
    _os.environ.setdefault("GOOGLE_CLOUD_LOCATION", GEMINI_LOCATION)
    _os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    user_prompt = (
        "=== CLIP CONTEXT ===\n"
        f"{context}\n\n"
        "---\n\n"
        "Based on this clip, decide whether to add an editorial top caption. "
        "If yes, propose exactly 3 variants and pick the best one."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[user_prompt],
        config=genai.types.GenerateContentConfig(
            system_instruction=TOP_CAPTION_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=TopCaptionPlan,
            temperature=0.7,
        ),
    )
    return TopCaptionPlan.model_validate_json(response.text)



def augment_payloads_with_top_captions(
    payloads: list[dict],
    nodes: list[dict],
    *,
    generator: TopCaptionGenerator | None = None,
) -> list[dict]:
    gen = generator or _default_gemini_generator
    augmented: list[dict] = []

    for i, payload in enumerate(payloads):
        context = _build_clip_context(payload, nodes)
        log.info(f"  Clip {i + 1}/{len(payloads)}: generating top caption…")

        try:
            plan = gen(payload, context)
            if isinstance(plan, dict):
                plan = TopCaptionPlan.model_validate(plan)
        except Exception as e:
            log.warning(f"  Clip {i + 1}: top caption generation failed: {e}")
            plan = TopCaptionPlan(decision="no_caption", reasoning=str(e))

        next_payload = dict(payload)
        if plan.decision == "caption" and plan.selected_text:
            word_count = len(plan.selected_text.split())
            char_count = len(plan.selected_text)
            if word_count > MAX_WORDS or char_count > MAX_CHARS:
                log.warning(
                    f"  Clip {i + 1}: caption too long ({word_count}w, {char_count}c), skipping"
                )
                next_payload["top_caption"] = None
            else:
                next_payload["top_caption"] = {
                    "text": plan.selected_text,
                    "tone": plan.selected_tone,
                    "reasoning": plan.reasoning,
                    "variants": [v.model_dump() for v in plan.variants],
                }
                log.info(f"    → \"{plan.selected_text}\" ({plan.selected_tone})")
        else:
            next_payload["top_caption"] = None
            log.info(f"    → No caption: {plan.reasoning[:80]}")

        augmented.append(next_payload)

    return augmented


def main(
    *,
    input_path: str | Path | None = None,
    nodes_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict:
    input_payload_path = _resolve_input_path(input_path)
    resolved_nodes_path = _resolve_nodes_path(nodes_path)
    resolved_output_path = _resolve_output_path(output_path, input_payload_path)

    if not input_payload_path.exists():
        raise FileNotFoundError(f"Missing clip payload input: {input_payload_path}")

    payloads = _load_payload_array(input_payload_path)
    nodes = _load_nodes(resolved_nodes_path)

    log.info("=" * 60)
    log.info("PHASE 5C - TOP CAPTION PAYLOADS")
    log.info("=" * 60)
    log.info("Input payloads: %d", len(payloads))
    log.info("Phase 2A nodes: %d", len(nodes))

    augmented = augment_payloads_with_top_captions(payloads, nodes)

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(json.dumps(augmented, indent=2), encoding="utf-8")

    captioned = sum(1 for p in augmented if p.get("top_caption"))
    log.info("Clips with top captions: %d / %d", captioned, len(augmented))
    log.info("Output saved -> %s", resolved_output_path)

    return {
        "input_payload_path": str(input_payload_path),
        "nodes_path": str(resolved_nodes_path),
        "output_path": str(resolved_output_path),
        "payload_count": len(augmented),
        "captioned_count": captioned,
    }


if __name__ == "__main__":
    main()