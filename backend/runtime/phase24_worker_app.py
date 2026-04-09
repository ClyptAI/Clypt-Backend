from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import tempfile
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.phase1_runtime.models import Phase1SidecarOutputs
from backend.providers import VertexEmbeddingClient, VertexGeminiClient, load_provider_settings
from backend.providers.storage import GCSStorageClient
from backend.repository import Phase24JobRecord, RunRecord, SpannerPhase14Repository

from .phase14_live import V31LivePhase14Runner

logger = logging.getLogger(__name__)
UTC = timezone.utc


class Phase24TaskPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    source_url: str | None = None
    source_uri: str | None = None
    source_video_gcs_uri: str | None = None
    phase1_outputs: dict[str, Any]
    phase3_long_range_top_k: int = 3
    phase4_extra_prompt_texts: list[str] = Field(default_factory=list)
    query_version: str | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> "Phase24TaskPayload":
        if not (self.source_url or self.source_uri):
            raise ValueError("Provide source_url or source_uri")
        return self

    @property
    def source_reference(self) -> str:
        return self.source_url or self.source_uri or ""


@dataclass(slots=True)
class Phase24WorkerService:
    repository: Any
    runner: Any
    service_name: str
    environment: str
    default_query_version: str
    max_attempts: int = 3
    running_lease_timeout_s: int = 1800

    def handle_task(
        self,
        payload: Phase24TaskPayload,
        *,
        job_id: str,
        attempt: int,
    ) -> dict[str, Any]:
        query_version = payload.query_version or self.default_query_version
        current_job = self.repository.get_phase24_job(payload.run_id)
        lease_state = self._acquire_phase24_lease(
            run_id=payload.run_id,
            job_id=job_id,
            attempt=attempt,
            query_version=query_version,
        )
        if lease_state["status"] == "succeeded":
            self._emit_log(
                run_id=payload.run_id,
                job_id=job_id,
                phase="phase24",
                event="run_terminal",
                attempt=attempt,
                query_version=query_version,
                status="success",
                final_status="PHASE24_DONE",
            )
            return {
                "run_id": payload.run_id,
                "status": "already_succeeded",
                "summary": {
                    "run_id": payload.run_id,
                    "artifact_paths": {},
                    "metadata": {"query_version": query_version},
                },
            }
        if lease_state["status"] == "running" and not lease_state["acquired"]:
            self._emit_log(
                run_id=payload.run_id,
                job_id=job_id,
                phase="phase24",
                event="duplicate_dispatch_skipped",
                attempt=attempt,
                query_version=query_version,
                status="already_running",
            )
            return {
                "run_id": payload.run_id,
                "status": "already_running",
                "summary": {
                    "run_id": payload.run_id,
                    "artifact_paths": {},
                    "metadata": {"query_version": query_version},
                },
            }

        current_run = self.repository.get_run(payload.run_id)
        now = datetime.now(UTC)
        if attempt > self.max_attempts:
            error_payload = {
                "code": "MaxAttemptsExceeded",
                "message": f"attempt {attempt} exceeded max_attempts={self.max_attempts}",
            }
            self.repository.upsert_phase24_job(
                Phase24JobRecord(
                    run_id=payload.run_id,
                    status="failed",
                    attempt_count=attempt,
                    last_error=error_payload,
                    worker_name=self.service_name,
                    task_name=job_id,
                    locked_at=now,
                    updated_at=now,
                    completed_at=now,
                    metadata={"query_version": query_version},
                )
            )
            self.repository.upsert_run(
                RunRecord(
                    run_id=payload.run_id,
                    source_url=payload.source_reference,
                    source_video_gcs_uri=payload.source_video_gcs_uri
                    or (payload.phase1_outputs.get("phase1_audio") or {}).get("video_gcs_uri"),
                    status="FAILED",
                    created_at=current_run.created_at if current_run is not None else now,
                    updated_at=now,
                    metadata={
                        **(current_run.metadata if current_run is not None else {}),
                        "query_version": query_version,
                        "job_id": job_id,
                    },
                )
            )
            self._emit_log(
                run_id=payload.run_id,
                job_id=job_id,
                phase="phase24",
                event="run_terminal",
                attempt=attempt,
                query_version=query_version,
                status="terminal_failure",
                error_code=error_payload["code"],
                error_message=error_payload["message"],
                final_status="FAILED",
            )
            return {
                "run_id": payload.run_id,
                "status": "max_attempts_exceeded",
                "summary": {
                    "run_id": payload.run_id,
                    "artifact_paths": {},
                    "metadata": {"query_version": query_version},
                },
            }
        self.repository.upsert_run(
            RunRecord(
                run_id=payload.run_id,
                source_url=payload.source_reference,
                source_video_gcs_uri=payload.source_video_gcs_uri
                or (payload.phase1_outputs.get("phase1_audio") or {}).get("video_gcs_uri"),
                status="PHASE24_RUNNING",
                created_at=current_run.created_at if current_run is not None else now,
                updated_at=now,
                metadata={
                    **(current_run.metadata if current_run is not None else {}),
                    "query_version": query_version,
                    "job_id": job_id,
                },
            )
        )

        try:
            runner = replace(
                self.runner,
                query_version=query_version,
                log_event=lambda **event: self._emit_log(**event),
            )
        except TypeError:
            runner = self.runner
            runner.query_version = query_version
            runner.log_event = lambda **event: self._emit_log(**event)
        phase1_outputs_payload, temp_video_dir = self._prepare_phase1_outputs(payload)
        try:
            summary = runner.run(
                run_id=payload.run_id,
                source_url=payload.source_reference,
                phase1_outputs=Phase1SidecarOutputs(**phase1_outputs_payload),
                phase3_long_range_top_k=payload.phase3_long_range_top_k,
                phase4_extra_prompt_texts=payload.phase4_extra_prompt_texts,
                job_id=job_id,
                attempt=attempt,
            )
        except Exception as exc:
            failed_at = datetime.now(UTC)
            error_payload = {
                "code": exc.__class__.__name__,
                "message": str(exc)[:2048],
            }
            is_terminal = attempt >= self.max_attempts
            self.repository.upsert_phase24_job(
                Phase24JobRecord(
                    run_id=payload.run_id,
                    status="failed" if is_terminal else "queued",
                    attempt_count=max(1, int(attempt)),
                    last_error=error_payload,
                    worker_name=self.service_name,
                    task_name=job_id,
                    locked_at=failed_at,
                    updated_at=failed_at,
                    completed_at=failed_at if is_terminal else None,
                    metadata={
                        **(current_job.metadata if current_job is not None else {}),
                        "query_version": query_version,
                    },
                )
            )
            self.repository.upsert_run(
                RunRecord(
                    run_id=payload.run_id,
                    source_url=payload.source_reference,
                    source_video_gcs_uri=payload.source_video_gcs_uri
                    or (payload.phase1_outputs.get("phase1_audio") or {}).get("video_gcs_uri"),
                    status="FAILED" if is_terminal else "PHASE24_QUEUED",
                    created_at=current_run.created_at if current_run is not None else failed_at,
                    updated_at=failed_at,
                    metadata={
                        **(current_run.metadata if current_run is not None else {}),
                        "query_version": query_version,
                        "job_id": job_id,
                    },
                )
            )
            if not is_terminal:
                self._emit_log(
                    run_id=payload.run_id,
                    job_id=job_id,
                    phase="phase24",
                    event="phase_retry_scheduled",
                    attempt=attempt,
                    query_version=query_version,
                    status="retrying",
                    error_code=error_payload["code"],
                    error_message=error_payload["message"],
                )
            raise
        finally:
            if temp_video_dir is not None:
                temp_video_dir.cleanup()

        completed_at = datetime.now(UTC)
        summary_payload = summary.model_dump(mode="json") if hasattr(summary, "model_dump") else dict(summary)
        self.repository.upsert_phase24_job(
            Phase24JobRecord(
                run_id=payload.run_id,
                status="succeeded",
                attempt_count=max(1, int(attempt)),
                last_error=None,
                worker_name=self.service_name,
                task_name=job_id,
                locked_at=completed_at,
                updated_at=completed_at,
                completed_at=completed_at,
                metadata={
                    **(current_job.metadata if current_job is not None else {}),
                    "query_version": query_version,
                },
            )
        )
        self.repository.upsert_run(
            RunRecord(
                run_id=payload.run_id,
                source_url=payload.source_reference,
                source_video_gcs_uri=payload.source_video_gcs_uri
                or (payload.phase1_outputs.get("phase1_audio") or {}).get("video_gcs_uri"),
                status="PHASE24_DONE",
                created_at=current_run.created_at if current_run is not None else completed_at,
                updated_at=completed_at,
                metadata={
                    **(current_run.metadata if current_run is not None else {}),
                    **(summary_payload.get("metadata") or {}),
                    "query_version": query_version,
                    "job_id": job_id,
                },
            )
        )
        return {
            "run_id": payload.run_id,
            "status": "succeeded",
            "summary": summary_payload,
        }

    def _acquire_phase24_lease(
        self,
        *,
        run_id: str,
        job_id: str,
        attempt: int,
        query_version: str,
    ) -> dict[str, Any]:
        acquire = getattr(self.repository, "acquire_phase24_job_lease", None)
        if not callable(acquire):
            raise RuntimeError("repository must implement acquire_phase24_job_lease")
        return acquire(
            run_id=run_id,
            job_id=job_id,
            worker_name=self.service_name,
            attempt=attempt,
            query_version=query_version,
            running_timeout_s=self.running_lease_timeout_s,
        )

    def _prepare_phase1_outputs(
        self,
        payload: Phase24TaskPayload,
    ) -> tuple[dict[str, Any], tempfile.TemporaryDirectory[str] | None]:
        phase1_outputs_payload = dict(payload.phase1_outputs)
        phase1_audio = dict(phase1_outputs_payload.get("phase1_audio") or {})
        local_video_path = phase1_audio.get("local_video_path")
        requires_local_video = (
            getattr(self.runner, "node_media_preparer", None) is None
            and getattr(self.runner, "storage_client", None) is not None
        )
        if requires_local_video and (not local_video_path or not Path(str(local_video_path)).exists()):
            video_gcs_uri = payload.source_video_gcs_uri or phase1_audio.get("video_gcs_uri")
            if not video_gcs_uri:
                raise ValueError(
                    "phase1_outputs.phase1_audio.video_gcs_uri or source_video_gcs_uri is required"
                )
            temp_dir = tempfile.TemporaryDirectory(prefix=f"phase24-{payload.run_id}-")
            local_path = Path(temp_dir.name) / "source_video.mp4"
            try:
                self.runner.storage_client.download_file(
                    gcs_uri=video_gcs_uri,
                    local_path=local_path,
                )
            except Exception:
                temp_dir.cleanup()
                raise
            phase1_audio["local_video_path"] = str(local_path)
            phase1_outputs_payload["phase1_audio"] = phase1_audio
            return phase1_outputs_payload, temp_dir
        if phase1_audio:
            phase1_outputs_payload["phase1_audio"] = phase1_audio
        return phase1_outputs_payload, None

    def _emit_log(
        self,
        *,
        run_id: str,
        job_id: str,
        phase: str,
        event: str,
        attempt: int,
        query_version: str | None,
        status: str,
        duration_ms: float | None = None,
        severity: str = "INFO",
        error_code: str | None = None,
        error_message: str | None = None,
        **extra: Any,
    ) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "severity": severity,
            "service": self.service_name,
            "environment": self.environment,
            "run_id": run_id,
            "job_id": job_id,
            "phase": phase,
            "event": event,
            "attempt": attempt,
            "query_version": query_version,
            "duration_ms": duration_ms,
            "status": status,
            "error_code": error_code,
            "error_message": error_message,
            **extra,
        }
        message = json.dumps(
            {key: value for key, value in payload.items() if value is not None},
            ensure_ascii=True,
            separators=(",", ":"),
        )
        if severity.upper() == "ERROR":
            logger.error(message)
        else:
            logger.info(message)


def build_default_phase24_worker_service() -> Phase24WorkerService:
    settings = load_provider_settings()
    repository = SpannerPhase14Repository.from_settings(settings=settings.spanner)
    runner = V31LivePhase14Runner.from_env(
        llm_client=VertexGeminiClient(settings=settings.vertex),
        embedding_client=VertexEmbeddingClient(settings=settings.vertex),
        flash_model=settings.vertex.flash_model,
        storage_client=GCSStorageClient(settings=settings.storage),
        repository=repository,
        query_version=settings.phase24_worker.query_version,
        debug_snapshots=settings.phase24_worker.debug_snapshots,
    )
    return Phase24WorkerService(
        repository=repository,
        runner=runner,
        service_name=settings.phase24_worker.service_name,
        environment=settings.phase24_worker.environment,
        default_query_version=settings.phase24_worker.query_version,
        max_attempts=settings.phase24_worker.max_attempts,
    )


def _parse_attempt(request: Request) -> int:
    execution_count = request.headers.get("X-CloudTasks-TaskExecutionCount")
    retry_count = request.headers.get("X-CloudTasks-TaskRetryCount")
    raw_value = execution_count or retry_count
    if raw_value is None:
        return 1
    try:
        parsed = int(raw_value)
    except ValueError:
        return 1
    # Cloud Tasks counters are zero-based; convert to human attempt number.
    return max(1, parsed + 1)


def create_app(*, service: Phase24WorkerService | None = None) -> FastAPI:
    app = FastAPI(title="Clypt Phase24 Worker")
    app.state.service = service

    def _service() -> Phase24WorkerService:
        if app.state.service is None:
            app.state.service = build_default_phase24_worker_service()
        return app.state.service

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    def _handle(payload: Phase24TaskPayload, request: Request) -> dict[str, Any]:
        task_name = request.headers.get("X-CloudTasks-TaskName") or payload.run_id
        try:
            return _service().handle_task(
                payload,
                job_id=task_name,
                attempt=_parse_attempt(request),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/")
    def root_task(payload: Phase24TaskPayload, request: Request) -> dict[str, Any]:
        return _handle(payload, request)

    @app.post("/tasks/phase24")
    def phase24_task(payload: Phase24TaskPayload, request: Request) -> dict[str, Any]:
        return _handle(payload, request)

    return app


app = create_app()


__all__ = [
    "Phase24TaskPayload",
    "Phase24WorkerService",
    "app",
    "build_default_phase24_worker_service",
    "create_app",
]
