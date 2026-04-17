from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import tempfile
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.phase1_runtime.models import Phase1SidecarOutputs
from backend.providers import (
    LocalOpenAIQwenClient,
    RemoteNodeMediaPrepClient,
    VertexEmbeddingClient,
    load_provider_settings,
)
from backend.providers.storage import GCSStorageClient
from backend.repository import Phase24JobRecord, RunRecord, SpannerPhase14Repository

from .phase14_live import V31LivePhase14Runner
from .phase24_error_policy import (
    Phase24FailFastError,
    Phase24FailureClass,
    classify_phase24_exception,
)

logger = logging.getLogger(__name__)
UTC = timezone.utc


class Phase24TaskPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    source_url: str | None = None
    source_uri: str | None = None
    source_video_gcs_uri: str | None = None
    phase1_outputs: dict[str, Any] | None = None
    phase1_outputs_gcs_uri: str | None = None
    phase3_long_range_top_k: int | None = None
    phase4_extra_prompt_texts: list[str] = Field(default_factory=list)
    query_version: str | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> "Phase24TaskPayload":
        if not (self.source_url or self.source_uri):
            raise ValueError("Provide source_url or source_uri")
        if self.phase1_outputs is None and not self.phase1_outputs_gcs_uri:
            raise ValueError("Provide phase1_outputs or phase1_outputs_gcs_uri")
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
    fail_fast_p95_latency_ms: float = 0.0
    fail_fast_preemption_threshold: int = 0
    admission_metrics_path: str | None = None

    @staticmethod
    def _payload_video_gcs_uri(payload: Phase24TaskPayload) -> str | None:
        if payload.source_video_gcs_uri:
            return payload.source_video_gcs_uri
        if payload.phase1_outputs:
            return (payload.phase1_outputs.get("phase1_audio") or {}).get("video_gcs_uri")
        return None

    def handle_task(
        self,
        payload: Phase24TaskPayload,
        *,
        job_id: str,
        attempt: int,
    ) -> dict[str, Any]:
        query_version = payload.query_version or self.default_query_version
        source_video_gcs_uri = self._payload_video_gcs_uri(payload)
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
                    source_video_gcs_uri=source_video_gcs_uri,
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
                source_video_gcs_uri=source_video_gcs_uri,
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
                log_event=lambda **event: self._emit_and_check_admission(**event),
            )
        except TypeError:
            runner = self.runner
            runner.query_version = query_version
            runner.log_event = lambda **event: self._emit_and_check_admission(**event)
        phase1_outputs_payload, temp_video_dir = self._prepare_phase1_outputs(payload)
        try:
            self._assert_preemption_fail_fast()
            summary = runner.run(
                run_id=payload.run_id,
                source_url=payload.source_reference,
                phase1_outputs=Phase1SidecarOutputs(**phase1_outputs_payload),
                phase3_long_range_top_k=payload.phase3_long_range_top_k,
                phase4_extra_prompt_texts=payload.phase4_extra_prompt_texts,
                job_id=job_id,
                attempt=attempt,
            )
            summary_payload = (
                summary.model_dump(mode="json")
                if hasattr(summary, "model_dump")
                else dict(summary)
            )
            p95_latency_ms = self._extract_p95_latency_ms(summary_payload)
            if (
                self.fail_fast_p95_latency_ms > 0
                and p95_latency_ms is not None
                and p95_latency_ms > self.fail_fast_p95_latency_ms
            ):
                raise Phase24FailFastError(
                    "p95 latency threshold breached: "
                    f"p95_ms={p95_latency_ms:.2f} threshold_ms={self.fail_fast_p95_latency_ms:.2f}"
                )
        except Exception as exc:
            failed_at = datetime.now(UTC)
            error_payload = {
                "code": exc.__class__.__name__,
                "message": str(exc)[:2048],
            }
            failure_class = classify_phase24_exception(exc)
            should_retry = (
                failure_class == Phase24FailureClass.TRANSIENT and attempt < self.max_attempts
            )
            is_terminal = not should_retry
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
                    source_video_gcs_uri=source_video_gcs_uri,
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
                    failure_class=failure_class.value,
                )
            else:
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
                    failure_class=failure_class.value,
                )
            raise
        finally:
            if temp_video_dir is not None:
                temp_video_dir.cleanup()

        completed_at = datetime.now(UTC)
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
                source_video_gcs_uri=source_video_gcs_uri,
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
        temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if payload.phase1_outputs is not None:
            phase1_outputs_payload = dict(payload.phase1_outputs)
        else:
            if not payload.phase1_outputs_gcs_uri:
                raise ValueError("phase1_outputs_gcs_uri is required when phase1_outputs is omitted")
            if getattr(self.runner, "storage_client", None) is None:
                raise ValueError("runner.storage_client is required to load phase1_outputs from GCS")
            temp_dir = tempfile.TemporaryDirectory(prefix=f"phase24-{payload.run_id}-")
            handoff_path = Path(temp_dir.name) / "phase1_outputs.json"
            try:
                self.runner.storage_client.download_file(
                    gcs_uri=payload.phase1_outputs_gcs_uri,
                    local_path=handoff_path,
                )
                phase1_outputs_payload = json.loads(
                    handoff_path.read_text(encoding="utf-8")
                )
            except Exception:
                temp_dir.cleanup()
                raise

        # Node-media prep runs on the remote RTX host, so the H200 does not
        # need to have the source video on local disk. We only require that
        # video_gcs_uri is present (the remote NVENC endpoint downloads from
        # GCS itself).
        phase1_audio = dict(phase1_outputs_payload.get("phase1_audio") or {})
        video_gcs_uri = (
            payload.source_video_gcs_uri or phase1_audio.get("video_gcs_uri")
        )
        if not video_gcs_uri:
            raise ValueError(
                "phase1_outputs.phase1_audio.video_gcs_uri or source_video_gcs_uri is required "
                "(remote NVENC node-media prep fetches from GCS)."
            )
        phase1_audio["video_gcs_uri"] = video_gcs_uri
        phase1_outputs_payload["phase1_audio"] = phase1_audio
        return phase1_outputs_payload, temp_dir

    @staticmethod
    def _extract_p95_latency_ms(summary_payload: dict[str, Any]) -> float | None:
        metadata = summary_payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            return None
        candidates = [
            metadata.get("subgraph_review_p95_latency_ms"),
            ((metadata.get("diagnostics") or {}).get("latency_ms") or {}).get("p95"),
            ((metadata.get("phase4") or {}).get("diagnostics") or {}).get("latency_ms", {}).get("p95"),
        ]
        for value in candidates:
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _read_preemption_count(self) -> int | None:
        if self.fail_fast_preemption_threshold <= 0 or not self.admission_metrics_path:
            return None
        metrics_path = Path(self.admission_metrics_path)
        if not metrics_path.exists():
            return None
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        try:
            return int(payload.get("preemption_count") or 0)
        except (TypeError, ValueError):
            return None

    def _assert_preemption_fail_fast(self) -> None:
        if self.fail_fast_preemption_threshold <= 0:
            return
        preemption_count = self._read_preemption_count()
        if preemption_count is None:
            return
        if preemption_count >= self.fail_fast_preemption_threshold:
            raise Phase24FailFastError(
                "preemption threshold breached: "
                f"count={preemption_count} threshold={self.fail_fast_preemption_threshold}"
            )

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

    def _emit_and_check_admission(self, **event: Any) -> None:
        self._emit_log(**event)
        self._assert_preemption_fail_fast()


def build_default_phase24_worker_service() -> Phase24WorkerService:
    settings = load_provider_settings()
    repository = SpannerPhase14Repository.from_settings(settings=settings.spanner)
    repository.bootstrap_schema()
    generation_backend = (settings.vertex.generation_backend or "").strip().lower()
    if generation_backend != "local_openai":
        raise ValueError(
            "Phase24 local runtime supports only GENAI_GENERATION_BACKEND=local_openai "
            f"(got {generation_backend!r})."
        )
    llm_client = LocalOpenAIQwenClient(settings=settings.local_generation)
    # In local_openai mode, all phase2-4 LLM calls must target the locally served model.
    local_flash_model = (
        (settings.local_generation.model or "").strip()
        or (settings.vertex.flash_model or "").strip()
    )
    if not local_flash_model:
        raise ValueError(
            "Local OpenAI generation requires CLYPT_LOCAL_LLM_MODEL (or GENAI_FLASH_MODEL as fallback)."
        )
    # Node-media prep runs exclusively on the RTX 6000 Ada NVENC host; there
    # is no in-process fallback. Always wire the remote client.
    node_media_preparer = RemoteNodeMediaPrepClient(settings=settings.node_media_prep)
    runner = V31LivePhase14Runner.from_env(
        llm_client=llm_client,
        embedding_client=VertexEmbeddingClient(settings=settings.vertex),
        flash_model=local_flash_model,
        storage_client=GCSStorageClient(settings=settings.storage),
        node_media_preparer=node_media_preparer,
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
        fail_fast_p95_latency_ms=settings.phase24_worker.fail_fast_p95_latency_ms,
        fail_fast_preemption_threshold=settings.phase24_worker.fail_fast_preemption_threshold,
        admission_metrics_path=settings.phase24_worker.admission_metrics_path,
    )


__all__ = [
    "Phase24TaskPayload",
    "Phase24WorkerService",
    "build_default_phase24_worker_service",
]
