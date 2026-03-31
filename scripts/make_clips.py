"""
Cut viral short clips from Phase 1 data using Claude (standalone from the main pipeline).

Usage (from repo root):
    python scripts/make_clips.py

Reads:  backend/outputs/phase_1_audio.json
        backend/downloads/video.mp4
Writes: backend/outputs/claude_clips/clip_01.mp4, ...
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIO_JSON = ROOT / "backend" / "outputs" / "phase_1_audio.json"
VIDEO_PATH = ROOT / "backend" / "downloads" / "video.mp4"
CLIPS_DIR = ROOT / "backend" / "outputs" / "claude_clips"

FFMPEG = os.getenv("FFMPEG_PATH", "").strip() or "ffmpeg"


def build_transcript(words: list[dict]) -> str:
    """Group words into lines, preserving timestamps at sentence boundaries."""
    lines = []
    sentence: list[str] = []
    sentence_start = None

    for w in words:
        word = w["word"]
        t = w["start_time_ms"]
        if sentence_start is None:
            sentence_start = t
        sentence.append(word)
        if word.rstrip(".,!?").lower() != word.rstrip(".,!?") or len(sentence) >= 20:
            ts = f"[{sentence_start // 1000}.{(sentence_start % 1000) // 100}s]"
            lines.append(f"{ts} {' '.join(sentence)}")
            sentence = []
            sentence_start = None

    if sentence:
        ts = f"[{sentence_start // 1000}.{(sentence_start % 1000) // 100}s]"
        lines.append(f"{ts} {' '.join(sentence)}")

    return "\n".join(lines)


def ask_claude_for_clips(transcript: str, total_duration_s: float) -> list[dict]:
    import anthropic

    client = anthropic.Anthropic()

    prompt = f"""You are a viral short-form video editor. Below is a word-level transcript of a video ({total_duration_s:.0f}s long). Each line starts with a timestamp like [32.5s].

Find the 3 best segments to cut as standalone viral short clips (30-60 seconds each). Pick moments that are funny, surprising, emotional, or have a clear narrative arc that works standalone.

Transcript:
{transcript}

Respond with ONLY a JSON array, no other text. Each object must have:
- "title": short clip title
- "start_s": start time in seconds (float)
- "end_s": end time in seconds (float)
- "reason": one sentence why this moment is viral

Example:
[
  {{"title": "Professional Asian CEO", "start_s": 6.8, "end_s": 55.0, "reason": "Hilarious intro that subverts expectations"}},
  ...
]"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    text = message.content[0].text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def cut_clip(video: str, start_s: float, end_s: float, out_path: str):
    dur = end_s - start_s
    subprocess.run(
        [
            FFMPEG, "-y",
            "-ss", f"{start_s:.3f}",
            "-i", video,
            "-t", f"{dur:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            out_path,
        ],
        check=True,
    )


def main():
    if not AUDIO_JSON.exists():
        print(f"ERROR: {AUDIO_JSON} not found. Run Phase 1 first.")
        sys.exit(1)
    if not VIDEO_PATH.exists():
        print(f"ERROR: {VIDEO_PATH} not found.")
        sys.exit(1)

    data = json.loads(AUDIO_JSON.read_text())
    words = data.get("words", [])
    if not words:
        print("ERROR: No words in phase_1_audio.json")
        sys.exit(1)

    total_ms = words[-1]["end_time_ms"]
    total_s = total_ms / 1000.0
    print(f"Loaded {len(words)} words, {total_s:.1f}s of audio")

    print("Building transcript...")
    transcript = build_transcript(words)

    print("Asking Claude to pick the best clips...")
    clips = ask_claude_for_clips(transcript, total_s)
    print(f"Claude picked {len(clips)} clips:")
    for i, c in enumerate(clips):
        print(f"  {i+1}. [{c['start_s']:.1f}s - {c['end_s']:.1f}s] {c['title']}")
        print(f"     {c['reason']}")

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(clips):
        out = str(CLIPS_DIR / f"clip_{i+1:02d}_{c['title'].replace(' ', '_')[:30]}.mp4")
        print(f"\nCutting clip {i+1}: {out}")
        cut_clip(str(VIDEO_PATH), float(c["start_s"]), float(c["end_s"]), out)
        size_mb = Path(out).stat().st_size / 1e6
        print(f"  Done: {size_mb:.1f} MB")

    print(f"\nAll clips saved to {CLIPS_DIR}")


if __name__ == "__main__":
    main()
