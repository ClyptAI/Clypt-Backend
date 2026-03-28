from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from google.cloud import storage as gcs_storage

from backend.do_phase1_service.models import PersistedPhase1Manifest
from backend.pipeline.phase1_contract import Phase1Manifest


class StorageBackend(Protocol):
    bucket: str

    def upload_bytes(self, data: bytes, object_name: str) -> str: ...
    def upload_file(self, source_path: str | Path, object_name: str) -> str: ...


class GCSStorage:
    def __init__(self, bucket: str | None = None):
        self.bucket = bucket or os.getenv("GCS_BUCKET", "clypt-storage-v2")
        self._client = gcs_storage.Client()

    def upload_bytes(self, data: bytes, object_name: str) -> str:
        blob = self._client.bucket(self.bucket).blob(object_name)
        blob.upload_from_string(data)
        return f"gs://{self.bucket}/{object_name}"

    def upload_file(self, source_path: str | Path, object_name: str) -> str:
        blob = self._client.bucket(self.bucket).blob(object_name)
        blob.upload_from_filename(str(source_path))
        return f"gs://{self.bucket}/{object_name}"


class LocalGCSStorage:
    def __init__(self, bucket: str, root_dir: str | Path):
        self.bucket = bucket
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def upload_bytes(self, data: bytes, object_name: str) -> str:
        path = self.root_dir / object_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return f"gs://{self.bucket}/{object_name}"

    def upload_file(self, source_path: str | Path, object_name: str) -> str:
        return self.upload_bytes(Path(source_path).read_bytes(), object_name)


def persist_phase1_outputs(
    *,
    storage: StorageBackend,
    output_dir: str | Path,
    job_id: str,
    source_url: str,
    canonical_video_uri: str,
    phase_1_audio: dict,
    phase_1_visual: dict,
    timings: dict[str, int] | None = None,
) -> PersistedPhase1Manifest:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_1_audio = dict(phase_1_audio)
    phase_1_audio["speaker_candidate_debug"] = _normalize_speaker_candidate_debug(
        phase_1_audio.get("speaker_candidate_debug") or []
    )

    transcript_uri = _upload_json(storage, phase_1_audio, f"phase_1/jobs/{job_id}/phase_1_audio.json")
    visual_uri = _upload_json(storage, phase_1_visual, f"phase_1/jobs/{job_id}/phase_1_visual.json")

    manifest_payload = {
        "contract_version": "v2",
        "job_id": job_id,
        "status": "succeeded",
        "source_video": {"source_url": source_url},
        "canonical_video_gcs_uri": canonical_video_uri,
        "artifacts": {
            "transcript": {
                "uri": transcript_uri,
                "source_audio": phase_1_audio["source_audio"],
                "video_gcs_uri": canonical_video_uri,
                "words": phase_1_audio["words"],
                "speaker_bindings": phase_1_audio["speaker_bindings"],
                "audio_speaker_turns": phase_1_audio.get("audio_speaker_turns") or [],
                "speaker_bindings_local": phase_1_audio.get("speaker_bindings_local") or [],
                "speaker_follow_bindings_local": phase_1_audio.get("speaker_follow_bindings_local") or [],
                "audio_speaker_local_track_map": phase_1_audio.get("audio_speaker_local_track_map") or [],
                "speaker_candidate_debug": phase_1_audio.get("speaker_candidate_debug") or [],
                "active_speakers_local": phase_1_audio.get("active_speakers_local") or [],
                "overlap_follow_decisions": phase_1_audio.get("overlap_follow_decisions") or [],
            },
            "visual_tracking": {
                "uri": visual_uri,
                "source_video": phase_1_visual["source_video"],
                "video_gcs_uri": canonical_video_uri,
                "schema_version": phase_1_visual["schema_version"],
                "task_type": phase_1_visual["task_type"],
                "coordinate_space": phase_1_visual["coordinate_space"],
                "geometry_type": phase_1_visual["geometry_type"],
                "class_taxonomy": phase_1_visual["class_taxonomy"],
                "tracking_metrics": phase_1_visual["tracking_metrics"],
                "tracks": phase_1_visual["tracks"],
                "tracks_local": phase_1_visual.get("tracks_local") or [],
                "face_detections": phase_1_visual["face_detections"],
                "person_detections": phase_1_visual["person_detections"],
                "label_detections": phase_1_visual["label_detections"],
                "object_tracking": phase_1_visual["object_tracking"],
                "shot_changes": phase_1_visual["shot_changes"],
                "video_metadata": phase_1_visual["video_metadata"],
            },
            "events": None,
        },
        "metadata": {
            "runtime": {
                "provider": "digitalocean",
                "worker_id": os.getenv("DO_PHASE1_WORKER_ID"),
                "region": os.getenv("DO_REGION"),
            },
            "timings": {
                "ingest_ms": int((timings or {}).get("ingest_ms", 0)),
                "processing_ms": int((timings or {}).get("processing_ms", 0)),
                "upload_ms": int((timings or {}).get("upload_ms", 0)),
            },
            "quality_metrics": {
                "schema_pass_rate": float(phase_1_visual.get("tracking_metrics", {}).get("schema_pass_rate", 1.0)),
                "transcript_coverage": 1.0 if phase_1_audio.get("words") is not None else 0.0,
                "tracking_confidence": float(phase_1_visual.get("tracking_metrics", {}).get("schema_pass_rate", 1.0)),
            },
            "retry": None,
            "failure": None,
        },
    }
    validated = Phase1Manifest.model_validate(manifest_payload)
    manifest_uri = _upload_json(storage, validated.model_dump(mode="json"), f"phase_1/jobs/{job_id}/manifest.json")
    return PersistedPhase1Manifest.model_validate({**validated.model_dump(mode="json"), "manifest_uri": manifest_uri})


def _upload_json(storage: StorageBackend, payload: dict, object_name: str) -> str:
    return storage.upload_bytes(json.dumps(payload, indent=2).encode("utf-8"), object_name)


def _normalize_speaker_candidate_debug(entries: list[dict]) -> list[dict]:
    normalized_entries: list[dict] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            continue

        normalized_candidates = _normalize_speaker_candidate_debug_candidates(
            raw_entry.get("candidates") or []
        )
        chosen_local_track_id = _coerce_optional_str(raw_entry.get("chosen_local_track_id"))
        chosen_track_id = _coerce_optional_str(raw_entry.get("chosen_track_id"))
        if (not chosen_local_track_id or not chosen_track_id) and normalized_candidates:
            top_candidate = normalized_candidates[0]
            chosen_local_track_id = chosen_local_track_id or top_candidate["local_track_id"]
            chosen_track_id = chosen_track_id or top_candidate["track_id"]

        decision_source = raw_entry.get("decision_source")
        if decision_source not in {"visual", "audio_boosted_visual", "unknown"}:
            decision_source = "visual" if normalized_candidates else "unknown"

        top_margin = raw_entry.get("top_1_top_2_margin")
        if top_margin is None and len(normalized_candidates) >= 2:
            top_margin = float(
                normalized_candidates[0]["blended_score"] - normalized_candidates[1]["blended_score"]
            )

        normalized_entries.append(
            {
                "word": _coerce_optional_str(raw_entry.get("word")),
                "start_time_ms": int(raw_entry.get("start_time_ms", 0) or 0),
                "end_time_ms": int(
                    raw_entry.get("end_time_ms", raw_entry.get("start_time_ms", 0)) or 0
                ),
                "active_audio_speaker_id": _coerce_optional_str(
                    raw_entry.get("active_audio_speaker_id") or raw_entry.get("speaker_id")
                ),
                "active_audio_local_track_id": _coerce_optional_str(
                    raw_entry.get("active_audio_local_track_id")
                ),
                "chosen_track_id": chosen_track_id,
                "chosen_local_track_id": chosen_local_track_id,
                "decision_source": decision_source,
                "ambiguous": bool(
                    raw_entry.get("ambiguous", len(normalized_candidates) > 1)
                ),
                "top_1_top_2_margin": (
                    None if top_margin is None else float(top_margin)
                ),
                "candidates": normalized_candidates,
            }
        )
    return normalized_entries


def _normalize_speaker_candidate_debug_candidates(candidates: list[dict]) -> list[dict]:
    normalized_candidates: list[dict] = []
    for raw_candidate in candidates[:3]:
        if not isinstance(raw_candidate, dict):
            continue
        local_track_id = _coerce_optional_str(
            raw_candidate.get("local_track_id") or raw_candidate.get("clean_local_track_id")
        )
        track_id = _coerce_optional_str(
            raw_candidate.get("track_id") or raw_candidate.get("visual_identity_id")
        )
        if not local_track_id:
            local_track_id = track_id or "unknown_local_track"
        if not track_id:
            track_id = local_track_id

        blended_score = raw_candidate.get("blended_score")
        if blended_score is None:
            blended_score = raw_candidate.get("composite_score")
        if blended_score is None:
            blended_score = raw_candidate.get("score")
        if blended_score is None:
            blended_score = 0.0

        normalized_candidates.append(
            {
                "local_track_id": str(local_track_id),
                "track_id": str(track_id),
                "blended_score": float(blended_score),
                "asd_probability": _coerce_optional_float(raw_candidate.get("asd_probability")),
                "body_prior": _coerce_optional_float(raw_candidate.get("body_prior")),
                "detection_confidence": _coerce_optional_float(
                    raw_candidate.get("detection_confidence")
                ),
            }
        )
    return normalized_candidates


def _coerce_optional_str(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
