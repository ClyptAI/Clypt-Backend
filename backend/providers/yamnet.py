from __future__ import annotations

from pathlib import Path
from typing import Any


def _build_default_runner(*, device: str = "gpu"):
    try:
        import csv
        import io

        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError as exc:
        raise RuntimeError(
            "tensorflow and tensorflow-hub are required for live YAMNet execution."
        ) from exc

    normalized_device = str(device or "gpu").strip().lower()
    if normalized_device not in {"gpu", "cpu"}:
        raise ValueError("YAMNet device must be 'gpu' or 'cpu'.")

    physical_gpus = list(tf.config.list_physical_devices("GPU"))
    if normalized_device == "gpu" and not physical_gpus:
        raise RuntimeError("YAMNet was configured for GPU but no TensorFlow GPU device is available.")
    compute_device = "/GPU:0" if normalized_device == "gpu" else "/CPU:0"

    model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    with tf.io.gfile.GFile(class_map_path, "r") as handle:
        reader = csv.DictReader(io.StringIO(handle.read()))
        class_names = [row["display_name"] for row in reader]

    def _runner(*, audio_path: Path):
        with tf.device(compute_device):
            file_bytes = tf.io.read_file(str(audio_path))
            waveform, sample_rate = tf.audio.decode_wav(
                file_bytes,
                desired_channels=1,
            )
            waveform = tf.squeeze(waveform, axis=-1)
            sample_rate = int(sample_rate.numpy())
            if sample_rate != 16000:
                raise ValueError("YAMNet expects 16kHz mono audio input.")
            scores, _, _ = model(waveform)
        scores_np = scores.numpy()
        patch_hop_ms = 480
        patch_window_ms = 960
        events: list[dict[str, Any]] = []
        for idx, frame_scores in enumerate(scores_np):
            best_idx = int(frame_scores.argmax())
            confidence = float(frame_scores[best_idx])
            if confidence < 0.2:
                continue
            start_ms = idx * patch_hop_ms
            end_ms = start_ms + patch_window_ms
            events.append(
                {
                    "event_label": class_names[best_idx],
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "confidence": confidence,
                }
            )
        return events

    return _runner


class YAMNetProvider:
    def __init__(self, *, runner=None, merge_gap_ms: int = 0, device: str = "gpu") -> None:
        self._runner = runner
        self.merge_gap_ms = max(0, int(merge_gap_ms))
        self.device = str(device or "gpu").strip().lower()

    def _ensure_runner(self):
        if self._runner is None:
            self._runner = _build_default_runner(device=self.device)
        return self._runner

    def run(self, *, audio_path: Path) -> dict[str, list[dict[str, Any]]]:
        runner = self._ensure_runner()
        raw_events = list(runner(audio_path=audio_path) or [])
        if not raw_events:
            return {"events": []}

        ordered = sorted(
            raw_events,
            key=lambda item: (
                int(item["start_ms"]),
                int(item["end_ms"]),
                str(item["event_label"]),
            ),
        )
        merged: list[dict[str, Any]] = []
        for event in ordered:
            normalized = {
                "event_label": str(event["event_label"]),
                "start_ms": int(event["start_ms"]),
                "end_ms": int(event["end_ms"]),
                "confidence": float(event["confidence"]) if event.get("confidence") is not None else None,
            }
            if (
                merged
                and merged[-1]["event_label"] == normalized["event_label"]
                and normalized["start_ms"] <= (merged[-1]["end_ms"] + self.merge_gap_ms)
            ):
                merged[-1]["end_ms"] = max(merged[-1]["end_ms"], normalized["end_ms"])
                if normalized["confidence"] is not None:
                    current = merged[-1].get("confidence")
                    merged[-1]["confidence"] = (
                        normalized["confidence"]
                        if current is None
                        else max(float(current), normalized["confidence"])
                    )
            else:
                merged.append(normalized)
        return {"events": merged}


__all__ = ["YAMNetProvider"]
