from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.phase1_runtime.extract import run_phase1_sidecars
from backend.phase1_runtime.input_resolver import Phase1InputResolutionError, Phase1InputResolver
from backend.phase1_runtime.media import prepare_workspace_media
from backend.phase1_runtime.models import Phase1SidecarOutputs, Phase1Workspace
from backend.repository import Phase24JobRecord, RunRecord

logger = logging.getLogger(__name__)
UTC = timezone.utc


def _jsonable(value):
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


class Phase1JobRunner:
    def __init__(
        self,
        *,
        working_root: Path,
        audio_extractor=None,
        storage_client: Any,
        vibevoice_provider: Any,
        forced_aligner: Any,
        visual_extractor: Any,
        emotion_provider: Any,
        yamnet_provider: Any,
        phase24_task_queue_client: Any | None = None,
        phase14_repository: Any | None = None,
        phase24_worker_url: str | None = None,
        phase24_query_version: str | None = None,
        input_resolver: Phase1InputResolver | None = None,
        input_resolver_strict: bool = True,
    ) -> None:
        self.working_root = Path(working_root)
        self.audio_extractor = audio_extractor
        self.storage_client = storage_client
        self.vibevoice_provider = vibevoice_provider
        self.forced_aligner = forced_aligner
        self.visual_extractor = visual_extractor
        self.emotion_provider = emotion_provider
        self.yamnet_provider = yamnet_provider
        self.phase24_task_queue_client = phase24_task_queue_client
        self.phase14_repository = phase14_repository
        self.phase24_worker_url = phase24_worker_url
        self.phase24_query_version = phase24_query_version
        self.input_resolver = input_resolver
        self.input_resolver_strict = input_resolver_strict

    def _upsert_run_record(
        self,
        *,
        run_id: str,
        source_url: str | None,
        source_video_gcs_uri: str | None,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.phase14_repository is None:
            return
        get_run = getattr(self.phase14_repository, "get_run", None)
        existing = get_run(run_id) if callable(get_run) else None
        now = datetime.now(UTC)
        merged_metadata = dict(existing.metadata if existing is not None else {})
        if metadata:
            merged_metadata.update(metadata)
        self.phase14_repository.upsert_run(
            RunRecord(
                run_id=run_id,
                source_url=source_url,
                source_video_gcs_uri=source_video_gcs_uri,
                status=status,
                created_at=existing.created_at if existing is not None else now,
                updated_at=now,
                metadata=merged_metadata,
            )
        )

    def _upsert_phase24_job_record(
        self,
        *,
        run_id: str,
        status: str,
        task_name: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.phase14_repository is None:
            return
        get_phase24_job = getattr(self.phase14_repository, "get_phase24_job", None)
        existing = get_phase24_job(run_id) if callable(get_phase24_job) else None
        now = datetime.now(UTC)
        merged_metadata = dict(existing.metadata if existing is not None else {})
        if metadata:
            merged_metadata.update(metadata)
        self.phase14_repository.upsert_phase24_job(
            Phase24JobRecord(
                run_id=run_id,
                status=status,
                attempt_count=existing.attempt_count if existing is not None else 0,
                last_error=None,
                worker_name=existing.worker_name if existing is not None else None,
                task_name=task_name or (existing.task_name if existing is not None else None),
                locked_at=existing.locked_at if existing is not None else None,
                updated_at=now,
                completed_at=existing.completed_at if existing is not None else None,
                metadata=merged_metadata,
            )
        )

    def _enqueue_phase24(
        self,
        *,
        job_id: str,
        source_ref: str,
        video_gcs_uri: str,
        phase1_outputs_gcs_uri: str,
        runtime_controls: dict[str, Any],
    ) -> dict[str, Any]:
        if self.phase24_task_queue_client is None:
            raise ValueError("phase24_task_queue_client is required when queue mode is enabled")

        payload = {
            "run_id": job_id,
            "source_url": source_ref,
            "source_video_gcs_uri": video_gcs_uri,
            "phase1_outputs_gcs_uri": phase1_outputs_gcs_uri,
            "phase3_long_range_top_k": int(runtime_controls.get("phase3_long_range_top_k") or 3),
            "phase4_extra_prompt_texts": list(runtime_controls.get("phase4_extra_prompt_texts") or []),
            "query_version": self.phase24_query_version,
        }
        task_name = self.phase24_task_queue_client.enqueue_phase24(
            run_id=job_id,
            payload=payload,
            worker_url=self.phase24_worker_url,
        )
        metadata = {
            "query_version": self.phase24_query_version,
            "task_name": task_name,
            "phase1_outputs_gcs_uri": phase1_outputs_gcs_uri,
        }
        self._upsert_run_record(
            run_id=job_id,
            source_url=source_ref,
            source_video_gcs_uri=video_gcs_uri,
            status="PHASE24_QUEUED",
            metadata=metadata,
        )
        self._upsert_phase24_job_record(
            run_id=job_id,
            status="queued",
            task_name=task_name,
            metadata=metadata,
        )
        logger.info("[phase24] enqueued Cloud Task %s", task_name)
        return {
            "run_id": job_id,
            "status": "queued",
            "task_name": task_name,
            "artifact_paths": {},
        }

    def _persist_phase24_handoff(
        self,
        *,
        job_id: str,
        workspace: Phase1Workspace,
        phase1_outputs: Phase1SidecarOutputs,
    ) -> str:
        handoff_payload = _jsonable(phase1_outputs)
        handoff_path = workspace.metadata_dir / "phase24_handoff.json"
        handoff_path.write_text(
            json.dumps(handoff_payload, ensure_ascii=True, separators=(",", ":")),
            encoding="utf-8",
        )
        handoff_object_name = f"phase1/{job_id}/phase24_inputs/phase1_outputs.json"
        handoff_gcs_uri = self.storage_client.upload_file(
            local_path=handoff_path,
            object_name=handoff_object_name,
        )
        logger.info("[gcs]    phase24 handoff uploaded → %s", handoff_gcs_uri)
        return handoff_gcs_uri

    def run_job(
        self,
        *,
        job_id: str,
        source_url: str | None,
        source_path: str | None,
        runtime_controls: dict | None,
    ) -> dict[str, Any]:
        runtime_controls = dict(runtime_controls or {})
        if bool(source_url) == bool(source_path):
            raise ValueError("Provide exactly one of source_url or source_path")

        resolved_source_path = source_path
        if source_url is not None:
            if self.input_resolver is None:
                raise Phase1InputResolutionError(
                    "source_url requires test-bank mapping, but no Phase1InputResolver is configured. "
                    "Provide source_path or configure CLYPT_PHASE1_TEST_BANK_PATH."
                )
            try:
                resolved_source_path = str(self.input_resolver.resolve_source_path(source_url=source_url))
                logger.info("[media]  test-bank source_url resolved to %s", resolved_source_path)
            except Phase1InputResolutionError as exc:
                raise Phase1InputResolutionError(
                    f"{exc} URL download mode has been removed; add test-bank mapping or use source_path."
                ) from exc

        if resolved_source_path is None:
            raise ValueError(
                "Unable to resolve local source_path for Phase 1 media. "
                "Provide source_path directly or configure a test-bank mapping for source_url."
            )
        if self.input_resolver is not None and source_url is not None and not self.input_resolver_strict:
            logger.info(
                "[media]  input_resolver_strict=0 is ignored now that URL download is removed."
                )
        workspace = Phase1Workspace.create(root=self.working_root, run_id=job_id)
        logger.info("[media]  workspace: %s", workspace.root)

        t_media = time.perf_counter()
        prepare_workspace_media(
            source_path=resolved_source_path,
            workspace=workspace,
            audio_extractor=self.audio_extractor,
        )
        logger.info("[media]  local media + audio prep done in %.1f s", time.perf_counter() - t_media)

        t_upload = time.perf_counter()
        logger.info("[gcs]    uploading video to GCS ...")
        video_gcs_uri = self.storage_client.upload_file(
            local_path=workspace.video_path,
            object_name=f"phase1/{job_id}/source_video.mp4",
        )
        logger.info("[gcs]    uploaded → %s (%.1f s)", video_gcs_uri, time.perf_counter() - t_upload)

        source_ref = source_url or str(source_path)
        result: dict[str, Any] = {}
        run_phase14 = bool(runtime_controls.get("run_phase14"))

        if run_phase14:
            if self.phase24_task_queue_client is None:
                raise RuntimeError(
                    "run_phase14 requested with queue mode, but Cloud Tasks client is unavailable. "
                    "Set CLYPT_PHASE24_WORKER_URL and ensure google-cloud-tasks is installed."
                )
            enqueue_done = threading.Event()
            enqueue_error: list[Exception] = []
            enqueue_summary: list[dict[str, Any]] = []

            def _on_audio_done(partial: Phase1SidecarOutputs) -> None:
                try:
                    phase1_outputs_gcs_uri = self._persist_phase24_handoff(
                        job_id=job_id,
                        workspace=workspace,
                        phase1_outputs=partial,
                    )
                    summary = self._enqueue_phase24(
                        job_id=job_id,
                        source_ref=source_ref,
                        video_gcs_uri=video_gcs_uri,
                        phase1_outputs_gcs_uri=phase1_outputs_gcs_uri,
                        runtime_controls=runtime_controls,
                    )
                    enqueue_summary.append(summary)
                    logger.info(
                        "[phase24] queue-mode handoff complete — Cloud Task enqueued while RF-DETR finishes"
                    )
                except Exception as exc:  # pragma: no cover - defensive passthrough
                    enqueue_error.append(exc)
                finally:
                    enqueue_done.set()

            phase1_outputs = run_phase1_sidecars(
                source_url=source_ref,
                video_gcs_uri=video_gcs_uri,
                workspace=workspace,
                vibevoice_provider=self.vibevoice_provider,
                forced_aligner=self.forced_aligner,
                visual_extractor=self.visual_extractor,
                emotion_provider=self.emotion_provider,
                yamnet_provider=self.yamnet_provider,
                on_audio_chain_complete=_on_audio_done,
            )
            result["phase1"] = _jsonable(phase1_outputs)
            if not enqueue_done.is_set():
                raise RuntimeError(
                    "audio-chain callback did not fire; unable to enqueue phase24 work."
                )
            if enqueue_error:
                raise enqueue_error[0]
            if not enqueue_summary:
                raise RuntimeError("phase24 enqueue callback completed without a summary.")
            result["summary"] = enqueue_summary[0]
        else:
            # Original path: run sidecars, no phases 2-4
            phase1_outputs = run_phase1_sidecars(
                source_url=source_ref,
                video_gcs_uri=video_gcs_uri,
                workspace=workspace,
                vibevoice_provider=self.vibevoice_provider,
                forced_aligner=self.forced_aligner,
                visual_extractor=self.visual_extractor,
                emotion_provider=self.emotion_provider,
                yamnet_provider=self.yamnet_provider,
            )
            result["phase1"] = _jsonable(phase1_outputs)
            self._upsert_run_record(
                run_id=job_id,
                source_url=source_ref,
                source_video_gcs_uri=video_gcs_uri,
                status="PHASE1_DONE",
                metadata={"query_version": self.phase24_query_version},
            )

        return result


__all__ = ["Phase1JobRunner"]
