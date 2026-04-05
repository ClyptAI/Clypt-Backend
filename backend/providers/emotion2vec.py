from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from typing import Any


def _default_turn_clipper(*, audio_path: Path, start_ms: int, end_ms: int) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="clypt-emotion2vec-"))
    clip_path = temp_dir / f"{start_ms}_{end_ms}.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-ss",
            f"{start_ms / 1000.0:.3f}",
            "-to",
            f"{end_ms / 1000.0:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(clip_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return clip_path


def _build_default_model():
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "funasr is required for live emotion2vec+ execution."
        ) from exc
    return AutoModel(model="iic/emotion2vec_plus_large")


def _normalize_emotion_result(result: Any) -> tuple[list[str], list[float], dict[str, float]]:
    if isinstance(result, list):
        if not result:
            raise ValueError("emotion2vec+ returned no results")
        result = result[0]
    if not isinstance(result, dict):
        raise ValueError("emotion2vec+ result must be a dict or list[dict]")

    labels = [str(label) for label in (result.get("labels") or [])]
    scores = [float(score) for score in (result.get("scores") or [])]
    per_class_scores = {
        str(label): float(score)
        for label, score in dict(result.get("per_class_scores") or {}).items()
    }

    if not labels:
        raise ValueError("emotion2vec+ result is missing labels")
    if not scores:
        raise ValueError("emotion2vec+ result is missing scores")
    if not per_class_scores:
        per_class_scores = {labels[0]: scores[0]}
    return labels, scores, per_class_scores


class Emotion2VecPlusProvider:
    def __init__(self, *, model: Any | None = None, clipper=None) -> None:
        self._model = model
        self._clipper = clipper or _default_turn_clipper

    def _ensure_model(self):
        if self._model is None:
            self._model = _build_default_model()
        return self._model

    def run(self, *, audio_path: Path, turns: list[dict]) -> dict:
        model = self._ensure_model()
        segments: list[dict] = []
        for turn in turns:
            clip_path = self._clipper(
                audio_path=audio_path,
                start_ms=int(turn["start_ms"]),
                end_ms=int(turn["end_ms"]),
            )
            raw_result = model.generate(input=str(clip_path), granularity="utterance")
            labels, scores, per_class_scores = _normalize_emotion_result(raw_result)
            segments.append(
                {
                    "turn_id": str(turn["turn_id"]),
                    "labels": labels,
                    "scores": scores,
                    "per_class_scores": per_class_scores,
                }
            )
        return {"segments": segments}


__all__ = ["Emotion2VecPlusProvider"]
