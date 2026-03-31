from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from google.cloud import storage as gcs_storage

from backend.do_phase1_service.models import PersistedPhase1Manifest
from backend.pipeline.phase1.metrics_scorecard import compute_phase1_scorecard
from backend.pipeline.phase1_contract import Phase1Manifest


class StorageBackend(Protocol):
    bucket: str

    def upload_bytes(self, data: bytes, object_name: str) -> str: ...
    def upload_file(self, source_path: str | Path, object_name: str) -> str: ...


class GCSStorage:
    def __init__(self, bucket: str | None = None):
        self.bucket = bucket or os.getenv("GCS_BUCKET", "clypt-storage-v3")
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

    transcript_uri = _upload_json(storage, phase_1_audio, f"phase_1/jobs/{job_id}/phase_1_audio.json")
    visual_uri = _upload_json(storage, phase_1_visual, f"phase_1/jobs/{job_id}/phase_1_visual.json")

    timings_payload = {
        "ingest_ms": int((timings or {}).get("ingest_ms", 0)),
        "processing_ms": int((timings or {}).get("processing_ms", 0)),
        "upload_ms": int((timings or {}).get("upload_ms", 0)),
    }
    benchmark_scorecard = compute_phase1_scorecard(
        phase_1_audio,
        phase_1_visual,
        job_timings_ms=timings_payload,
    )

    manifest_payload = {
        "contract_version": "v3",
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
                "audio_visual_mappings": phase_1_audio.get("audio_visual_mappings") or [],
                "span_assignments": phase_1_audio.get("span_assignments") or [],
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
                "mask_stability_signals": phase_1_visual.get("mask_stability_signals") or {},
                "visual_identities": phase_1_visual.get("visual_identities") or [],
            },
            "events": None,
        },
        "metadata": {
            "runtime": {
                "provider": "digitalocean",
                "worker_id": os.getenv("DO_PHASE1_WORKER_ID"),
                "region": os.getenv("DO_REGION"),
            },
            "timings": timings_payload,
            "benchmark_scorecard": benchmark_scorecard,
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
