#!/usr/bin/env python3
"""
Phase 2A: Content Mechanism Decomposition (Chunked Processing)
===============================================================
Splits the Phase 1 audio ledger into adaptive, token-budgeted chunks
aligned on shot boundaries. Each chunk sends the audio transcript for
its time range, plus the GCS video, to Gemini 3.1 Pro. Gemini uses its
own vision capabilities to analyze the video directly — the visual
bounding-box / tracking data from Phase 1 is NOT sent here; it is
reserved exclusively for the FFmpeg speaker-follow renderer.

A merge pass deduplicates boundary nodes into the final output.

Inputs:
  - gs://clypt-storage-v2/phase_1/video.mp4  (muxed video+audio on GCS)
  - phase_1_visual.json   (shot_changes + video_metadata only, for chunk planning)
  - phase_1_audio.json    (transcript words, sliced per chunk)

Output:
  - phase_2a_nodes.json    (array of Semantic Nodes)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from google import genai
from google.genai.errors import ClientError
from google.genai.types import HttpOptions
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
PROJECT_ID = "clypt-v2"
LOCATION = "global"
MODEL_ID = "gemini-3.1-pro-preview"

ROOT = Path(__file__).resolve().parent.parent
VIDEO_GCS_URI = "gs://clypt-storage-v2/phase_1/video.mp4"
VISUAL_LEDGER_PATH = ROOT / "outputs" / "phase_1_visual.json"
AUDIO_LEDGER_PATH = ROOT / "outputs" / "phase_1_audio.json"
OUTPUT_PATH = ROOT / "outputs" / "phase_2a_nodes.json"
CHECKPOINT_PATH = ROOT / "outputs" / "phase_2a_nodes.checkpoint.json"

# Maximum text tokens to send per chunk. Kept deliberately low to avoid
# exceeding Gemini's 1M-token context window once the video and prompt
# overhead are included.
TEXT_TOKEN_BUDGET = 200_000
CHUNK_REQUEST_DELAY_S = float(os.getenv("PHASE_2A_CHUNK_REQUEST_DELAY_S", "2.0"))
MAX_RETRIES_429 = int(os.getenv("PHASE_2A_MAX_RETRIES_429", "6"))
BACKOFF_BASE_S = float(os.getenv("PHASE_2A_RETRY_BASE_S", "5.0"))
BACKOFF_MAX_S = float(os.getenv("PHASE_2A_RETRY_MAX_S", "90.0"))

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase_2a")


def resolve_video_gcs_uri() -> str:
    env_override = os.getenv("VIDEO_GCS_URI", "").strip()
    if env_override:
        return env_override
    for path in (VISUAL_LEDGER_PATH, AUDIO_LEDGER_PATH):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        uri = str(payload.get("video_gcs_uri", "") or "").strip()
        if uri:
            return uri
    return VIDEO_GCS_URI

# ──────────────────────────────────────────────
# System instruction (exact prompt from spec)
# ──────────────────────────────────────────────
SYSTEM_INSTRUCTION = """You are an expert multimodal narrative extraction engine. Your task is to perform 'Content Mechanism Decomposition' on the provided video, using the attached audio transcript ledger as your word-level ground truth and your own vision capabilities for all visual analysis.

You must deconstruct the video into a sequential array of distinct 'Semantic Nodes'. Do not summarize the video; instead, break it down into structural narrative beats.

Node granularity: Each node must represent a single atomic beat — one thought, one reaction, one point. Target 6–8 seconds per node. Never create a node shorter than 4 seconds or longer than 12 seconds. Split on any meaningful shift in thought, tone, or energy — even within a single shot. Nodes will be assembled into clips by a downstream curator, so precision matters more than length. Critical boundary rule: every node must end on the last word of a complete sentence or thought — never mid-phrase, never mid-sentence. If the natural beat boundary falls mid-sentence, pull the node end_time back to the previous sentence-ending word.

Instructions:
1. Node Segmentation & Transcript Cleaning: If the audio ledger contains words, you must assemble the transcript_segment for each node strictly using those words and timestamps. Aggressively filter out STT noise — ignore non-lexical STT hallucinations (e.g., 'hmm', 'ah', 'uh', heavy breaths, or stuttered fragments). Only stitch together the mathematically verified timestamps of actual, meaningful spoken words. The start_time and end_time of your node must match the first and last word you selected from the JSON. If the audio ledger is empty (no words), you must still produce nodes based on your own visual analysis (shot changes, scene transitions, on-screen activity). In this case, set transcript_segment to an empty string and anchor start_time/end_time to natural visual boundaries.

2. Cross-Modal Synthesis: Do not rely solely on the transcript. You must use your own vision to synthesize the visual actions, facial expressions, and vocal prosody from the video file with the text to resolve ambiguities (e.g., sarcastic tone vs. literal text). Provide a brief visual_description based on what you directly observe in the video frames.

D. Vocal Delivery: In the vocal_delivery field, describe the audio events happening in this node. Explicitly mention laughter, voice cracks, whispering, sarcastic tones, shouting, or notable background sounds. Do not put these in the transcript; describe them here.

3. Layer 2 Content Mechanism Decomposition: For each node, you must analyze why the moment works by evaluating four specific dimensions. If a dimension is present, classify its type and intensity (0.0 to 1.0). Use the following taxonomy:

Humor: Look for irony, subversion, callback, or parody. Example: A crypto bro parody evaluating the 'ironic elevation' of a situation.

Emotion: Look for vulnerability, catharsis, inspiration, awe, or panic. Example: A creator opening up about burnout represents high parasocial vulnerability.

Social Dynamics: Look for status reversal, genuine disagreement, unexpected chemistry, or riffing. Example: A podcast guest gives an answer that completely stuns the host, causing an energy shift and a status reversal.

Expertise: Look for a counterintuitive truth, casual flex, elegant simplification, or a live demo. Example: A developer elegantly explaining OAuth in a single, simple sentence that validates audience confusion.

4. Confidence Score Calibration: The confidence_score field represents how likely this node is to become a viral standalone clip. Use the full range ruthlessly — most nodes should score 0.3–0.6. Reserve 0.8–1.0 for moments with a genuine hook, clear payoff, and strong content mechanism (intense humor, emotional spike, or status reversal). A node with only mild conversation or filler should score 0.2–0.4. Aim for at most 2–4 nodes above 0.75 per video. If you score more than 30% of nodes above 0.7, you are miscalibrated.

Output: You must output a strictly structured JSON array of these nodes conforming to the requested schema. Ensure your content_mechanisms analysis accurately reflects the psychological and narrative weight of the moment."""

# ──────────────────────────────────────────────
# Response schema (Pydantic)
# ──────────────────────────────────────────────
class MechanismDimension(BaseModel):
    present: bool
    type: str
    intensity: float = Field(ge=0.0, le=1.0)


class ContentMechanisms(BaseModel):
    humor: MechanismDimension
    emotion: MechanismDimension
    social: MechanismDimension
    expertise: MechanismDimension


class SemanticNode(BaseModel):
    start_time: float
    end_time: float
    transcript_segment: str
    visual_description: str
    vocal_delivery: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    content_mechanisms: ContentMechanisms


# ──────────────────────────────────────────────
# Token estimation
# ──────────────────────────────────────────────
# JSON-heavy ledgers (lots of digits, punctuation) tend to average closer to
# ~2 characters per token rather than 4, so we use a conservative estimate.
CHARS_PER_TOKEN = 2
PROMPT_OVERHEAD_TOKENS = 10_000


# ──────────────────────────────────────────────
# Chunk planning
# ──────────────────────────────────────────────
@dataclass
class Chunk:
    index: int
    start_ms: int
    end_ms: int
    shot_indices: list[int] = field(default_factory=list)
    estimated_tokens: int = 0


def _overlaps(seg_start: int, seg_end: int, chunk_start: int, chunk_end: int) -> bool:
    return seg_start < chunk_end and seg_end > chunk_start


def _estimate_shot_tokens(
    shot_start: int,
    shot_end: int,
    visual: dict,
    audio: dict,
) -> int:
    """Estimate the token cost of one shot by measuring the serialized size of
    the data that would be sliced for this shot's time range."""
    dummy_chunk = Chunk(index=0, start_ms=shot_start, end_ms=shot_end)
    sliced_vis, sliced_aud = _slice_ledger_for_chunk(dummy_chunk, visual, audio)
    total_chars = len(json.dumps(sliced_vis)) + len(json.dumps(sliced_aud))
    return total_chars // CHARS_PER_TOKEN


def _plan_chunks(visual: dict, audio: dict) -> list[Chunk]:
    """Group consecutive shots into chunks that fit within TEXT_TOKEN_BUDGET."""
    shots = visual.get("shot_changes", [])
    if not shots:
        # Fallback: synthesize pseudo-shots from available duration.
        duration_ms = 0
        if isinstance(visual.get("video_metadata"), dict):
            duration_ms = int(visual["video_metadata"].get("duration_ms", 0) or 0)
        if duration_ms <= 0:
            words = audio.get("words", []) if isinstance(audio, dict) else []
            if words:
                duration_ms = int(words[-1].get("end_time_ms", 0) or 0)
        if duration_ms <= 0:
            tracks = visual.get("tracks", [])
            if tracks and isinstance(visual.get("video_metadata"), dict):
                fps = float(visual["video_metadata"].get("fps", 25.0) or 25.0)
                max_fi = max(int(t.get("frame_idx", 0)) for t in tracks)
                duration_ms = int(round((max_fi / max(1e-6, fps)) * 1000.0))
        if duration_ms <= 0:
            duration_ms = 10_000
        shots = []
        start = 0
        while start < duration_ms:
            end = min(duration_ms, start + 10_000)
            shots.append({"start_time_ms": start, "end_time_ms": end})
            if end >= duration_ms:
                break
            start = end

    budget = TEXT_TOKEN_BUDGET - PROMPT_OVERHEAD_TOKENS
    chunks: list[Chunk] = []
    current = Chunk(index=0, start_ms=shots[0]["start_time_ms"], end_ms=shots[0]["end_time_ms"])

    for i, shot in enumerate(shots):
        shot_tokens = _estimate_shot_tokens(
            shot["start_time_ms"], shot["end_time_ms"], visual, audio
        )

        if current.estimated_tokens + shot_tokens > budget and current.shot_indices:
            chunks.append(current)
            current = Chunk(
                index=len(chunks),
                start_ms=shot["start_time_ms"],
                end_ms=shot["end_time_ms"],
            )

        current.shot_indices.append(i)
        current.end_ms = shot["end_time_ms"]
        current.estimated_tokens += shot_tokens

    if current.shot_indices:
        chunks.append(current)

    return chunks



# ──────────────────────────────────────────────
# Ledger slicing
# ──────────────────────────────────────────────
def _slice_ledger_for_chunk(
    chunk: Chunk,
    visual: dict,
    audio: dict,
) -> tuple[dict, dict]:
    """Filter ledgers to only include data within the chunk's time range.

    Only shot_changes and video_metadata are kept from the visual ledger
    (for Gemini scene-structure context).  Bounding-box data (tracks,
    face_detections, person_detections, object_tracking, label_detections)
    is deliberately excluded — that data is reserved for the FFmpeg
    speaker-follow renderer and would only waste Gemini tokens here.
    """
    start = chunk.start_ms
    end = chunk.end_ms

    # Keep only structural visual metadata — no bounding boxes
    sliced_visual: dict = {
        "shot_changes": [
            s for s in visual.get("shot_changes", [])
            if _overlaps(s["start_time_ms"], s["end_time_ms"], start, end)
        ],
        "video_metadata": visual.get("video_metadata", {}),
    }

    sliced_audio = {
        "words": [
            w for w in audio.get("words", [])
            if start <= w["start_time_ms"] < end
        ],
    }

    return sliced_visual, sliced_audio


# ──────────────────────────────────────────────
# Node merging / deduplication
# ──────────────────────────────────────────────
DEDUP_TOLERANCE_S = 1.0


def _merge_nodes(all_chunk_nodes: list[list[SemanticNode]]) -> list[SemanticNode]:
    """Merge nodes from all chunks: sort by start_time, deduplicate boundary overlaps."""
    flat: list[SemanticNode] = []
    for chunk_nodes in all_chunk_nodes:
        flat.extend(chunk_nodes)

    flat.sort(key=lambda n: n.start_time)

    if not flat:
        return flat

    merged: list[SemanticNode] = [flat[0]]
    for node in flat[1:]:
        prev = merged[-1]
        time_overlap = prev.end_time - node.start_time
        if time_overlap > -DEDUP_TOLERANCE_S and abs(prev.start_time - node.start_time) < DEDUP_TOLERANCE_S:
            if node.confidence_score > prev.confidence_score:
                merged[-1] = node
        else:
            merged.append(node)

    return merged


MIN_NODE_DURATION_S = float(os.getenv("CLYPT_MIN_NODE_DURATION_S", "8.0"))


def _enforce_min_duration(nodes: list[SemanticNode]) -> list[SemanticNode]:
    """Merge any node shorter than MIN_NODE_DURATION_S into its neighbor."""
    if not nodes:
        return nodes
    changed = True
    while changed:
        changed = False
        out: list[SemanticNode] = []
        i = 0
        while i < len(nodes):
            node = nodes[i]
            duration = node.end_time - node.start_time
            if duration < MIN_NODE_DURATION_S and len(nodes) > 1:
                # Merge with whichever neighbor is shorter (prefer next, fallback prev)
                if i + 1 < len(nodes):
                    nxt = nodes[i + 1]
                    merged_node = SemanticNode(
                        start_time=node.start_time,
                        end_time=nxt.end_time,
                        transcript_segment=(node.transcript_segment + " " + nxt.transcript_segment).strip(),
                        visual_description=node.visual_description,
                        vocal_delivery=node.vocal_delivery,
                        confidence_score=max(node.confidence_score, nxt.confidence_score),
                        content_mechanisms=node.content_mechanisms if node.confidence_score >= nxt.confidence_score else nxt.content_mechanisms,
                    )
                    out.append(merged_node)
                    i += 2
                else:
                    # Last node is short — merge into previous
                    prev = out.pop()
                    merged_node = SemanticNode(
                        start_time=prev.start_time,
                        end_time=node.end_time,
                        transcript_segment=(prev.transcript_segment + " " + node.transcript_segment).strip(),
                        visual_description=prev.visual_description,
                        vocal_delivery=prev.vocal_delivery,
                        confidence_score=max(prev.confidence_score, node.confidence_score),
                        content_mechanisms=prev.content_mechanisms if prev.confidence_score >= node.confidence_score else node.content_mechanisms,
                    )
                    out.append(merged_node)
                    i += 1
                changed = True
            else:
                out.append(node)
                i += 1
        nodes = out
    return nodes


def _load_checkpoint() -> dict[int, list[dict]]:
    """Load completed chunk results so reruns can resume after quota failures."""
    if not CHECKPOINT_PATH.exists():
        return {}
    try:
        payload = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    chunk_results = payload.get("chunk_results", {})
    if not isinstance(chunk_results, dict):
        return {}
    out: dict[int, list[dict]] = {}
    for key, value in chunk_results.items():
        if not isinstance(value, list):
            continue
        try:
            out[int(key)] = value
        except Exception:
            continue
    return out


def _save_checkpoint(chunk_results: dict[int, list[dict]]) -> None:
    """Persist per-chunk nodes so reruns do not restart from chunk 1."""
    payload = {"chunk_results": {str(k): v for k, v in sorted(chunk_results.items())}}
    CHECKPOINT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _request_nodes_with_retry(client: genai.Client, user_prompt: str, video_gcs_uri: str = VIDEO_GCS_URI) -> list[dict]:
    """Retry Vertex generate_content on quota throttling with exponential backoff."""
    for attempt in range(MAX_RETRIES_429 + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[
                    genai.types.Part.from_uri(
                        file_uri=video_gcs_uri,
                        mime_type="video/mp4",
                    ),
                    user_prompt,
                ],
                config=genai.types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=list[SemanticNode],
                    temperature=0.2,
                    thinking_config=genai.types.ThinkingConfig(
                        thinking_level="MEDIUM",
                    ),
                ),
            )
            return json.loads(response.text)
        except ClientError as e:
            is_quota = getattr(e, "status_code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e)
            if not is_quota or attempt >= MAX_RETRIES_429:
                raise
            sleep_s = min(BACKOFF_MAX_S, BACKOFF_BASE_S * (2 ** attempt))
            log.warning(
                "  Gemini 429 quota hit on attempt %d/%d; retrying in %.1fs",
                attempt + 1,
                MAX_RETRIES_429 + 1,
                sleep_s,
            )
            time.sleep(sleep_s)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("PHASE 2A — Content Mechanism Decomposition (Chunked)")
    log.info("=" * 60)

    # ── Load Phase 1 ledgers ──
    log.info(f"Loading visual ledger (shot_changes + metadata only): {VISUAL_LEDGER_PATH}")
    with open(VISUAL_LEDGER_PATH) as f:
        visual_raw = json.load(f)
    log.info(f"  Raw size: {VISUAL_LEDGER_PATH.stat().st_size:,} bytes (bounding boxes excluded from Gemini)")

    log.info(f"Loading audio ledger: {AUDIO_LEDGER_PATH}")
    with open(AUDIO_LEDGER_PATH) as f:
        audio_raw = json.load(f)
    log.info(f"  Size: {AUDIO_LEDGER_PATH.stat().st_size:,} bytes")
    video_gcs_uri = resolve_video_gcs_uri()

    # ── Plan chunks ──
    chunks = _plan_chunks(visual_raw, audio_raw)
    log.info(f"Chunk plan: {len(chunks)} chunks from {len(visual_raw.get('shot_changes', []))} shots")
    for c in chunks:
        log.info(
            f"  Chunk {c.index}: {c.start_ms / 1000:.1f}s – {c.end_ms / 1000:.1f}s "
            f"({len(c.shot_indices)} shots, ~{c.estimated_tokens:,} tokens)"
        )

    # ── Initialize Vertex AI client ──
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
    os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

    client = genai.Client(http_options=HttpOptions(api_version="v1"))
    log.info(f"Model: {MODEL_ID}")
    log.info(f"Video: {video_gcs_uri}")

    # ── Process each chunk sequentially ──
    checkpoint_nodes = _load_checkpoint()
    if checkpoint_nodes:
        log.info(f"Loaded Phase 2A checkpoint with {len(checkpoint_nodes)} completed chunks")
    all_chunk_nodes: list[list[SemanticNode]] = []

    for chunk in chunks:
        if chunk.index in checkpoint_nodes:
            restored = [SemanticNode.model_validate(n) for n in checkpoint_nodes[chunk.index]]
            log.info("\u2500" * 50)
            log.info(
                f"Processing chunk {chunk.index + 1}/{len(chunks)}: "
                f"{chunk.start_ms / 1000:.1f}s \u2013 {chunk.end_ms / 1000:.1f}s"
            )
            log.info(f"  Using checkpointed result ({len(restored)} nodes)")
            all_chunk_nodes.append(restored)
            continue
        log.info("─" * 50)
        log.info(
            f"Processing chunk {chunk.index + 1}/{len(chunks)}: "
            f"{chunk.start_ms / 1000:.1f}s – {chunk.end_ms / 1000:.1f}s"
        )

        sliced_visual, sliced_audio = _slice_ledger_for_chunk(chunk, visual_raw, audio_raw)
        shot_changes_json = json.dumps(sliced_visual.get("shot_changes", []))
        audio_json = json.dumps(sliced_audio)

        total_chars = len(shot_changes_json) + len(audio_json)
        log.info(f"  Shot changes: {len(sliced_visual.get('shot_changes', []))} in range")
        log.info(f"  Audio slice: {len(audio_json):,} chars")
        log.info(f"  Total: {total_chars:,} chars (~{total_chars // 4:,} tokens)")

        start_s = chunk.start_ms / 1000
        end_s = chunk.end_ms / 1000

        user_prompt = (
            f"Analyze the video segment from {start_s:.1f}s to {end_s:.1f}s. "
            f"Focus ONLY on this time range. Do not produce nodes outside this window.\n\n"
            f"Below is the audio transcript ledger filtered to this time range, plus shot change "
            f"boundaries for scene structure. Use your own vision on the video for all visual analysis "
            f"(facial expressions, actions, on-screen text, scene composition).\n\n"
            f"=== SHOT CHANGES ({start_s:.1f}s – {end_s:.1f}s) ===\n"
            f"{shot_changes_json}\n\n"
            f"=== AUDIO LEDGER ({start_s:.1f}s – {end_s:.1f}s) ===\n"
            f"{audio_json}\n\n"
            f"Now decompose this segment into Semantic Nodes following your instructions."
        )

        log.info("  Submitting to Gemini…")
        raw_nodes = _request_nodes_with_retry(client, user_prompt, video_gcs_uri)
        chunk_nodes = [SemanticNode.model_validate(n) for n in raw_nodes]
        log.info(f"  Nodes from chunk: {len(chunk_nodes)}")
        all_chunk_nodes.append(chunk_nodes)
        checkpoint_nodes[chunk.index] = [n.model_dump() for n in chunk_nodes]
        _save_checkpoint(checkpoint_nodes)
        if CHUNK_REQUEST_DELAY_S > 0:
            time.sleep(CHUNK_REQUEST_DELAY_S)

    # ── Merge and deduplicate ──
    log.info("─" * 50)
    log.info("Merging nodes from all chunks…")
    total_raw = sum(len(cn) for cn in all_chunk_nodes)
    nodes = _merge_nodes(all_chunk_nodes)
    log.info(f"  Raw nodes: {total_raw} → Merged: {len(nodes)} (deduped {total_raw - len(nodes)})")

    # ── Save ──
    nodes_dicts = [n.model_dump() for n in nodes]
    with open(OUTPUT_PATH, "w") as f:
        json.dump(nodes_dicts, f, indent=2)
    log.info(f"Output saved → {OUTPUT_PATH}")
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        log.info(f"Checkpoint cleared → {CHECKPOINT_PATH}")

    # ── Summary ──
    log.info("=" * 60)
    log.info("PHASE 2A COMPLETE")
    log.info(f"  Chunks processed: {len(chunks)}")
    log.info(f"  Final nodes: {len(nodes)}")
    if nodes:
        log.info(f"  Time range: {nodes[0].start_time:.1f}s → {nodes[-1].end_time:.1f}s")
        mechanisms_present = sum(
            1 for n in nodes
            for dim in [n.content_mechanisms.humor, n.content_mechanisms.emotion,
                        n.content_mechanisms.social, n.content_mechanisms.expertise]
            if dim.present
        )
        log.info(f"  Active mechanisms: {mechanisms_present} across {len(nodes)} nodes")
    log.info("=" * 60)


if __name__ == "__main__":
    main()