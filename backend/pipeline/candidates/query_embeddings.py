from __future__ import annotations

from .._embedding_utils import embed_text


def embed_prompt_texts(*, prompts: list[str]) -> list[dict]:
    """Embed retrieval prompts with the same provider used for node embeddings."""
    embedded_prompts: list[dict] = []
    for idx, prompt in enumerate(prompts, start=1):
        embedded_prompts.append(
            {
                "prompt_id": f"prompt_{idx:03d}",
                "text": prompt,
                "embedding": embed_text(text=prompt),
            }
        )
    return embedded_prompts
