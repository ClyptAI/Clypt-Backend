from __future__ import annotations

import math
from pathlib import Path


class EcapaTdnnSpeakerVerifier:
    """Lightweight speaker verification wrapper around SpeechBrain ECAPA-TDNN.

    The model is loaded lazily so importing this module does not require
    SpeechBrain to be installed in every local test environment.
    """

    def __init__(
        self,
        *,
        model_id: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: str = "cpu",
        savedir: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.savedir = savedir
        self._classifier = None

    def similarity_paths(self, left_audio_path: str | Path, right_audio_path: str | Path) -> float:
        left_embedding = self._embedding_for_path(Path(left_audio_path))
        right_embedding = self._embedding_for_path(Path(right_audio_path))
        return _cosine_similarity(left_embedding, right_embedding)

    def _embedding_for_path(self, audio_path: Path) -> list[float]:
        classifier = self._load_classifier()
        embedding = classifier.encode_batch(str(audio_path))

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("torch is required for ECAPA speaker verification.") from exc

        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().reshape(-1).tolist()
        elif hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
            if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
                while embedding and isinstance(embedding[0], list):
                    embedding = embedding[0]
        return [float(value) for value in embedding]

    def _load_classifier(self):
        if self._classifier is not None:
            return self._classifier
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            try:
                from speechbrain.pretrained import EncoderClassifier  # pragma: no cover - compatibility
            except ImportError as exc:  # pragma: no cover - runtime dependency
                raise RuntimeError(
                    "speechbrain is required for ECAPA-TDNN speaker verification. "
                    "Install it in the Phase1 environment before enabling long-form chunking."
                ) from exc

        kwargs = {"source": self.model_id, "run_opts": {"device": self.device}}
        if self.savedir:
            kwargs["savedir"] = self.savedir
        self._classifier = EncoderClassifier.from_hparams(**kwargs)
        return self._classifier


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return float("-inf")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return float("-inf")
    return dot / (left_norm * right_norm)


__all__ = ["EcapaTdnnSpeakerVerifier"]
