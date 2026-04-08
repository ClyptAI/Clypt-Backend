from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from backend.phase1_runtime.extract import run_phase1_sidecars
from backend.phase1_runtime.media import prepare_workspace_media
from backend.phase1_runtime.models import Phase1SidecarOutputs, Phase1Workspace
from backend.providers.youtube import YouTubeDownloader

logger = logging.getLogger(__name__)


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
        vibevoice_provider: Any,
        forced_aligner: Any,
        visual_extractor: Any,
        emotion_provider: Any,
        yamnet_provider: Any,
        phase14_runner: Any | None = None,
    ) -> None:
        self.working_root = Path(working_root)
        self.downloader = downloader or YouTubeDownloader()
        self.audio_extractor = audio_extractor
        self.storage_client = storage_client
        self.vibevoice_provider = vibevoice_provider
        self.forced_aligner = forced_aligner
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
        logger.info("[media]  workspace: %s", workspace.root)

        t_media = time.perf_counter()
        prepare_workspace_media(
            source_url=source_url,
            source_path=source_path,
            workspace=workspace,
            downloader=self.downloader,
            audio_extractor=self.audio_extractor,
        )
        logger.info("[media]  download + audio prep done in %.1f s", time.perf_counter() - t_media)

        t_upload = time.perf_counter()
        logger.info("[gcs]    uploading video to GCS ...")
        video_gcs_uri = self.storage_client.upload_file(
            local_path=workspace.video_path,
            object_name=f"phase1/{job_id}/source_video.mp4",
        )
        logger.info("[gcs]    uploaded → %s (%.1f s)", video_gcs_uri, time.perf_counter() - t_upload)

        source_ref = source_url or str(source_path)
        result: dict[str, Any] = {}

        if runtime_controls.get("run_phase14") and self.phase14_runner is not None:
            # Phase 1 sidecars and Phases 2-4 run concurrently:
            # - Thread A: runs all sidecars (visual + audio chain in parallel internally)
            # - Thread B: waits for audio chain callback, then immediately starts Phases 2-4
            # - Thread A finishes last (RF-DETR); after both done, overwrite tracklets with real visual

            _audio_done = threading.Event()
            _partial_outputs: list[Phase1SidecarOutputs] = []  # one-element holder

            def _on_audio_done(partial: Phase1SidecarOutputs) -> None:
                _partial_outputs.append(partial)
                _audio_done.set()

            def _run_phases_24() -> Any:
                _audio_done.wait()
                return self.phase14_runner.run(
                    run_id=job_id,
                    source_url=source_ref,
                    phase1_outputs=_partial_outputs[0],
                )

            t_p14 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as pool:
                sidecars_future = pool.submit(
                    run_phase1_sidecars,
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
                phases24_future = pool.submit(_run_phases_24)

                phase1_outputs = sidecars_future.result()   # blocks until RF-DETR done
                summary = phases24_future.result()           # blocks until Phase 4 done

            # Overwrite the placeholder (empty) tracklet artifacts with real visual data
            from backend.pipeline.artifacts import build_run_paths, save_json
            from backend.pipeline.timeline.tracklets import build_tracklet_artifacts
            paths = build_run_paths(
                output_root=self.phase14_runner.config.output_root,
                run_id=job_id,
            )
            shot_tracklet_index, tracklet_geometry = build_tracklet_artifacts(
                phase1_visual=phase1_outputs.phase1_visual
            )
            save_json(paths.shot_tracklet_index, shot_tracklet_index.model_dump(mode="json"))
            save_json(paths.tracklet_geometry, tracklet_geometry.model_dump(mode="json"))
            logger.info("[phase14] tracklet artifacts updated with real RF-DETR data")

            result["phase1"] = _jsonable(phase1_outputs)
            result["summary"] = _jsonable(summary)
            logger.info(
                "[phase14] Phase 2-4 branch joined in %.1f s (includes overlap with Phase 1 sidecars)",
                time.perf_counter() - t_p14,
            )
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

        return result


__all__ = ["Phase1JobRunner"]
