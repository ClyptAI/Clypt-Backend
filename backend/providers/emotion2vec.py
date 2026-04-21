from __future__ import annotations

import logging
import os
import time
from pathlib import Path
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_EMOTION2VEC_MODEL_ID = "iic/emotion2vec_plus_large"
_DEFAULT_FUNASR_MODEL_SOURCE = "hf"


def _resolve_funasr_hub() -> str:
    source = (
        os.getenv("FUNASR_MODEL_SOURCE", _DEFAULT_FUNASR_MODEL_SOURCE).strip().lower()
    )
    if source in {"hf", "huggingface"}:
        return "hf"
    logger.warning(
        "[emotion2vec] unsupported FUNASR_MODEL_SOURCE=%r; forcing HuggingFace",
        source,
    )
    return "hf"


def _write_turn_clip(
    *,
    audio_path: Path,
    start_ms: int,
    end_ms: int,
    output_path: Path,
) -> Path:
    import soundfile as sf

    info = sf.info(str(audio_path))
    sample_rate = int(info.samplerate)
    start_frame = max(0, int(round((max(0, start_ms) / 1000.0) * sample_rate)))
    end_frame = max(start_frame + 1, int(round((max(start_ms + 1, end_ms) / 1000.0) * sample_rate)))
    audio, _ = sf.read(
        str(audio_path),
        start=start_frame,
        stop=end_frame,
        always_2d=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sample_rate)
    return output_path


def _build_default_model():
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "funasr is required for live emotion2vec+ execution."
        ) from exc
    model_id = os.getenv("EMOTION2VEC_MODEL_ID", _DEFAULT_EMOTION2VEC_MODEL_ID)
    hub = _resolve_funasr_hub()
    logger.info("[emotion2vec] loading %s (hub=%s) ...", model_id, hub)
    t0 = time.perf_counter()
    model = AutoModel(model=model_id, hub=hub, disable_update=True)
    logger.info("[emotion2vec] model loaded in %.1f s", time.perf_counter() - t0)
    return model


def _clean_label(raw: str) -> str:
    """emotion2vec+ may return '生气/angry' style labels — keep only the English part."""
    raw = str(raw).strip()
    if "/" in raw:
        return raw.split("/", 1)[-1].strip().lower()
    return raw.lower()


def _normalize_emotion_result(result: Any) -> tuple[list[str], list[float], dict[str, float]]:
    if isinstance(result, list):
        if not result:
            raise ValueError("emotion2vec+ returned no results")
        result = result[0]
    if not isinstance(result, dict):
        raise ValueError("emotion2vec+ result must be a dict or list[dict]")

    labels = [_clean_label(label) for label in (result.get("labels") or [])]
    scores = [float(score) for score in (result.get("scores") or [])]
    per_class_scores = {
        _clean_label(label): float(score)
        for label, score in dict(result.get("per_class_scores") or {}).items()
    }

    if not labels:
        raise ValueError("emotion2vec+ result is missing labels")
    if not scores:
        raise ValueError("emotion2vec+ result is missing scores")
    if len(labels) != len(scores):
        pair_count = min(len(labels), len(scores))
        if pair_count == 0:
            raise ValueError("emotion2vec+ result labels/scores are mismatched")
        labels = labels[:pair_count]
        scores = scores[:pair_count]
    if not per_class_scores:
        per_class_scores = {labels[0]: scores[0]}
    return labels, scores, per_class_scores


def _top_label_score(
    *, labels: list[str], scores: list[float], per_class_scores: dict[str, float]
) -> tuple[str, float]:
    pair_count = min(len(labels), len(scores))
    if pair_count > 0:
        top_idx = max(range(pair_count), key=lambda idx: scores[idx])
        return labels[top_idx], float(scores[top_idx])
    if per_class_scores:
        top_label, top_score = max(per_class_scores.items(), key=lambda item: item[1])
        return str(top_label), float(top_score)
    return "?", 0.0


class Emotion2VecPlusProvider:
    def __init__(self, *, model: Any | None = None, clipper=None) -> None:
        self._model = model
        self._clipper = clipper
        self._last_run_metrics: dict[str, float | int] = {}

    @property
    def last_run_metrics(self) -> dict[str, float | int]:
        return dict(self._last_run_metrics)

    def _ensure_model(self):
        if self._model is None:
            self._model = _build_default_model()
        return self._model

    def run(self, *, audio_path: Path, turns: list[dict]) -> dict:
        model = self._ensure_model()
        segments: list[dict] = []
        total = len(turns)
        _log_every = max(1, total // 10)  # log ~every 10% of turns
        t_run = time.perf_counter()
        clip_extract_ms = 0.0
        infer_ms = 0.0
        with tempfile.TemporaryDirectory(prefix="clypt-emotion2vec-") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            for i, turn in enumerate(turns):
                turn_start_ms = int(turn["start_ms"])
                turn_end_ms = int(turn["end_ms"])
                clip_started = time.perf_counter()
                if self._clipper is not None:
                    clip_path = self._clipper(
                        audio_path=audio_path,
                        start_ms=turn_start_ms,
                        end_ms=turn_end_ms,
                    )
                else:
                    clip_path = _write_turn_clip(
                        audio_path=audio_path,
                        start_ms=turn_start_ms,
                        end_ms=turn_end_ms,
                        output_path=temp_dir / f"{turn_start_ms}_{turn_end_ms}_{i:06d}.wav",
                    )
                clip_extract_ms += (time.perf_counter() - clip_started) * 1000.0
                infer_started = time.perf_counter()
                raw_result = model.generate(input=str(clip_path), granularity="utterance")
                infer_ms += (time.perf_counter() - infer_started) * 1000.0
                labels, scores, per_class_scores = _normalize_emotion_result(raw_result)
                segments.append(
                    {
                        "turn_id": str(turn["turn_id"]),
                        "labels": labels,
                        "scores": scores,
                        "per_class_scores": per_class_scores,
                    }
                )
                done = i + 1
                if done == 1 or done % _log_every == 0 or done == total:
                    top_label, top_score = _top_label_score(
                        labels=labels,
                        scores=scores,
                        per_class_scores=per_class_scores,
                    )
                    logger.info(
                        "[emotion2vec] %d/%d turns  (top: %s %.2f)",
                        done, total,
                        top_label,
                        top_score,
                    )
        logger.info(
            "[emotion2vec] all %d turns done in %.1f s",
            total, time.perf_counter() - t_run,
        )
        self._last_run_metrics = {
            "turn_count": total,
            "clip_extract_ms": clip_extract_ms,
            "infer_ms": infer_ms,
            "total_ms": (time.perf_counter() - t_run) * 1000.0,
        }
        return {"segments": segments}


__all__ = ["Emotion2VecPlusProvider"]
