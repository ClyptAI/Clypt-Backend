from __future__ import annotations

from backend.pipeline.candidates.prompt_sources import build_meta_prompts


def test_build_meta_prompts_scales_by_video_duration():
    assert len(build_meta_prompts(video_duration_s=60 * 4)) == 2
    assert len(build_meta_prompts(video_duration_s=60 * 10)) == 4
    assert len(build_meta_prompts(video_duration_s=60 * 20)) == 6
    assert len(build_meta_prompts(video_duration_s=60 * 40)) == 8
    assert len(build_meta_prompts(video_duration_s=60 * 80)) == 10


def test_build_meta_prompts_returns_distinct_editorial_lenses():
    prompts = build_meta_prompts(video_duration_s=60 * 20)

    assert len(prompts) == len(set(prompts))
    assert any("hook" in prompt.lower() for prompt in prompts)
    assert any("reaction" in prompt.lower() or "emotional" in prompt.lower() for prompt in prompts)
    assert any("surprising" in prompt.lower() or "reveal" in prompt.lower() for prompt in prompts)
