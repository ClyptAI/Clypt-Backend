from __future__ import annotations

from .._embedding_utils import embed_text


def embed_prompt_texts(*, prompts: list[object]) -> list[dict]:
    """Embed retrieval prompts with the same provider used for node embeddings."""
    embedded_prompts: list[dict] = []
    for idx, prompt in enumerate(prompts, start=1):
        if isinstance(prompt, dict):
            prompt_text = str(prompt.get("text") or "").strip()
            prompt_id = str(prompt.get("prompt_id") or f"prompt_{idx:03d}")
            extra = {
                key: value
                for key, value in prompt.items()
                if key not in {"prompt_id", "text", "embedding"}
            }
        else:
            prompt_text = str(prompt).strip()
            prompt_id = f"prompt_{idx:03d}"
            extra = {}
        if not prompt_text:
            continue
        embedded_prompts.append(
            {
                "prompt_id": prompt_id,
                "text": prompt_text,
                "embedding": embed_text(text=prompt_text),
                **extra,
            }
        )
    return embedded_prompts
