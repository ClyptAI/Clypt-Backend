from __future__ import annotations


def build_meta_prompts(*, video_duration_s: float) -> list[str]:
    """Build the default no-prompt meta-prompt set for a given video duration."""
    prompt_pool = [
        "Find the strongest standalone hook or setup-payoff moment in this video.",
        "Find the strongest emotional or reaction-driven moment in this video.",
        "Find the most surprising reveal or unexpected turn in this video.",
        "Find the strongest disagreement, challenge, or contradiction moment in this video.",
        "Find the most valuable practical insight or explanation moment in this video.",
        "Find the most audience-resonant or highly shareable moment in this video.",
        "Find the best concise anecdote or story beat in this video.",
        "Find the strongest question-and-answer exchange in this video.",
        "Find the clearest escalation or rising-intensity moment in this video.",
        "Find the best callback or return-to-topic moment in this video.",
    ]

    duration_minutes = float(video_duration_s) / 60.0
    if duration_minutes < 5:
        count = 2
    elif duration_minutes <= 15:
        count = 4
    elif duration_minutes <= 25:
        count = 6
    elif duration_minutes <= 60:
        count = 8
    else:
        count = 10
    return prompt_pool[:count]
