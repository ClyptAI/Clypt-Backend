from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from backend.phase1_runtime.extract import run_phase1_sidecars
from backend.phase1_runtime.media import prepare_workspace_media
from backend.phase1_runtime.models import Phase1Workspace
from backend.providers.youtube import YouTubeDownloader


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
        downloader=None,
        audio_extractor=None,
        storage_client: Any,
        pyannote_client: Any,
        visual_extractor: Any,
        emotion_provider: Any,
        yamnet_provider: Any,
        phase14_runner: Any | None = None,
    ) -> None:
        self.working_root = Path(working_root)
        self.downloader = downloader or YouTubeDownloader()
        self.audio_extractor = audio_extractor
        self.storage_client = storage_client
        self.pyannote_client = pyannote_client
        self.visual_extractor = visual_extractor
        self.emotion_provider = emotion_provider
        self.yamnet_provider = yamnet_provider
        self.phase14_runner = phase14_runner

    def run_job(
        self,
        *,
        job_id: str,
        source_url: str | None,
        source_path: str | None,
        runtime_controls: dict | None,
    ) -> dict[str, Any]:
        runtime_controls = dict(runtime_controls or {})
        workspace = Phase1Workspace.create(root=self.working_root, run_id=job_id)
        prepare_workspace_media(
            source_url=source_url,
            source_path=source_path,
            workspace=workspace,
            downloader=self.downloader,
            audio_extractor=self.audio_extractor,
        )
        video_gcs_uri = self.storage_client.upload_file(
            local_path=workspace.video_path,
            object_name=f"phase1/{job_id}/source_video.mp4",
        )

        source_ref = source_url or str(source_path)
        phase1_outputs = run_phase1_sidecars(
            source_url=source_ref,
            video_gcs_uri=video_gcs_uri,
            workspace=workspace,
            pyannote_client=self.pyannote_client,
            visual_extractor=self.visual_extractor,
            emotion_provider=self.emotion_provider,
            yamnet_provider=self.yamnet_provider,
            identify_voiceprints=list(runtime_controls.get("identify_voiceprints") or []),
        )

        result: dict[str, Any] = {
            "phase1": _jsonable(phase1_outputs),
        }
        if runtime_controls.get("run_phase14") and self.phase14_runner is not None:
            summary = self.phase14_runner.run(
                run_id=job_id,
                source_url=source_ref,
                phase1_outputs=phase1_outputs,
            )
            result["summary"] = _jsonable(summary)
        return result


__all__ = ["Phase1JobRunner"]
