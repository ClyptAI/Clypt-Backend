from __future__ import annotations

from pathlib import Path

from backend.providers.config import StorageSettings
from backend.providers.storage import GCSStorageClient
from backend.runtime.phase6_render import Phase6RenderRequest, run_phase6_render


class _FakeStorageClient(GCSStorageClient):
    def __init__(self, tmp_path: Path) -> None:
        super().__init__(settings=StorageSettings(gcs_bucket="bucket"), storage_client=object())
        self.tmp_path = tmp_path
        self.downloads: list[tuple[str, str]] = []
        self.uploads: list[tuple[str, str]] = []

    def download_file(self, *, gcs_uri: str, local_path: Path) -> Path:
        self.downloads.append((gcs_uri, str(local_path)))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if gcs_uri.endswith(".json"):
            local_path.write_text(
                '{"run_id":"run-123","source_context_ref":"source_context.json","caption_plan_ref":"caption_plan.json","publish_metadata_ref":"publish_metadata.json","clips":[{"clip_id":"clip_001","clip_start_ms":100,"clip_end_ms":900,"caption_plan_ref":"caption_plan.json","publish_metadata_ref":"publish_metadata.json","caption_segment_ids":["clip_001_seg_001"],"caption_zone":"center_band","caption_preset_id":"karaoke_focus","review_needed":false,"review_reasons":[],"overlays":[],"segments":[]}]}',
                encoding="utf-8",
            )
        elif gcs_uri.endswith(".ass"):
            local_path.write_text("[Script Info]\nPlayResX: 1080\nPlayResY: 1920\n", encoding="utf-8")
        else:
            local_path.write_bytes(b"video")
        return local_path

    def upload_file(self, *, local_path: Path, object_name: str) -> str:
        self.uploads.append((str(local_path), object_name))
        return f"gs://bucket/{object_name}"


def test_run_phase6_render_downloads_inputs_runs_ffmpeg_and_uploads_output(tmp_path, monkeypatch) -> None:
    commands: list[list[str]] = []
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "montserrat_extra_bold_v1.ttf").write_bytes(b"font")
    monkeypatch.setenv("CLYPT_PHASE6_FONT_ASSET_DIR", str(font_dir))

    def fake_run(cmd, check, capture_output, text):  # noqa: ARG001
        commands.append(cmd)
        output_path = Path(cmd[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"rendered")
        return None

    monkeypatch.setattr("subprocess.run", fake_run)

    request = Phase6RenderRequest.from_payload(
        {
            "run_id": "run-123",
            "source_video_gcs_uri": "gs://bucket/source.mp4",
            "artifact_gcs_uris": {
                "render_plan": "gs://bucket/phase14/run-123/render_inputs/render_plan.json",
                "captions_clip_001.ass": "gs://bucket/phase14/run-123/render_inputs/captions_clip_001.ass",
            },
            "clips": [
                {
                    "clip_id": "clip_001",
                    "clip_start_ms": 100,
                    "clip_end_ms": 900,
                }
            ],
        }
    )

    result = run_phase6_render(
        request=request,
        storage_client=_FakeStorageClient(tmp_path),
        scratch_root=tmp_path,
    )

    assert result["outputs"][0]["clip_id"] == "clip_001"
    assert result["outputs"][0]["video_gcs_uri"] == "gs://bucket/phase14/run-123/render_outputs/clip_001.mp4"
    assert any("subtitles=" in token for token in commands[0])
    vf_arg = commands[0][commands[0].index("-vf") + 1]
    assert "force_original_aspect_ratio=increase" in vf_arg
    assert "crop=1080:1920" in vf_arg
    assert "pad=1080:1920" not in vf_arg
    assert any("fontsdir=" in token for token in commands[0])
    assert "-c:v" in commands[0]
    assert "h264_nvenc" in commands[0]


def test_run_phase6_render_uses_tracklet_crop_plan_when_present(tmp_path, monkeypatch) -> None:
    commands: list[list[str]] = []
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "montserrat_extra_bold_v1.ttf").write_bytes(b"font")
    monkeypatch.setenv("CLYPT_PHASE6_FONT_ASSET_DIR", str(font_dir))

    class CropPlanStorage(_FakeStorageClient):
        def download_file(self, *, gcs_uri: str, local_path: Path) -> Path:
            path = super().download_file(gcs_uri=gcs_uri, local_path=local_path)
            if gcs_uri.endswith(".json"):
                path.write_text(
                    '{"run_id":"run-123","source_context_ref":"source_context.json","caption_plan_ref":"caption_plan.json","publish_metadata_ref":"publish_metadata.json","clips":[{"clip_id":"clip_001","clip_start_ms":100,"clip_end_ms":900,"caption_plan_ref":"caption_plan.json","publish_metadata_ref":"publish_metadata.json","caption_segment_ids":["clip_001_seg_001"],"caption_zone":"lower_safe","caption_preset_id":"karaoke_focus","review_needed":false,"review_reasons":[],"overlays":[],"crop_plan":{"mode":"tracklet_follow_9x16","segments":[{"start_ms":0,"end_ms":800,"tracklet_id":"shot_0001:7","shot_id":"shot_0001","bbox_xyxy":[900.0,100.0,1200.0,900.0]}]},"segments":[]}]}',
                    encoding="utf-8",
                )
            return path

    def fake_run(cmd, check, capture_output, text):  # noqa: ARG001
        commands.append(cmd)
        output_path = Path(cmd[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"rendered")
        return None

    def fake_check_output(cmd, text):  # noqa: ARG001
        return "1920,1080"

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("subprocess.check_output", fake_check_output)

    request = Phase6RenderRequest.from_payload(
        {
            "run_id": "run-123",
            "source_video_gcs_uri": "gs://bucket/source.mp4",
            "artifact_gcs_uris": {
                "render_plan": "gs://bucket/phase14/run-123/render_inputs/render_plan.json",
                "captions_clip_001.ass": "gs://bucket/phase14/run-123/render_inputs/captions_clip_001.ass",
            },
            "clips": [{"clip_id": "clip_001", "clip_start_ms": 100, "clip_end_ms": 900}],
        }
    )

    run_phase6_render(
        request=request,
        storage_client=CropPlanStorage(tmp_path),
        scratch_root=tmp_path,
    )

    vf_arg = commands[0][commands[0].index("-vf") + 1]
    assert "crop=608:1080:" in vf_arg
    assert "if(between(t\\,0.000\\,0.800)" in vf_arg
    assert "scale=1080:1920" in vf_arg
    assert "pad=1080:1920" not in vf_arg


def test_run_phase6_render_interpolates_smooth_crop_keyframes(tmp_path, monkeypatch) -> None:
    commands: list[list[str]] = []
    command_files: list[str] = []
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "montserrat_extra_bold_v1.ttf").write_bytes(b"font")
    monkeypatch.setenv("CLYPT_PHASE6_FONT_ASSET_DIR", str(font_dir))

    class SmoothCropPlanStorage(_FakeStorageClient):
        def download_file(self, *, gcs_uri: str, local_path: Path) -> Path:
            path = super().download_file(gcs_uri=gcs_uri, local_path=local_path)
            if gcs_uri.endswith(".json"):
                path.write_text(
                    '{"run_id":"run-123","source_context_ref":"source_context.json","caption_plan_ref":"caption_plan.json","publish_metadata_ref":"publish_metadata.json","clips":[{"clip_id":"clip_001","clip_start_ms":100,"clip_end_ms":1100,"caption_plan_ref":"caption_plan.json","publish_metadata_ref":"publish_metadata.json","caption_segment_ids":["clip_001_seg_001"],"caption_zone":"lower_safe","caption_preset_id":"karaoke_focus","review_needed":false,"review_reasons":[],"overlays":[],"crop_plan":{"mode":"tracklet_follow_9x16_smooth_inside_person","crop_width":360,"crop_height":640,"keyframes":[{"time_ms":0,"x":100,"y":50},{"time_ms":500,"x":220,"y":90},{"time_ms":1000,"x":340,"y":120}]},"segments":[]}]}',
                    encoding="utf-8",
                )
            return path

    def fake_run(cmd, check, capture_output, text):  # noqa: ARG001
        commands.append(cmd)
        vf_arg = cmd[cmd.index("-vf") + 1]
        command_path = Path(vf_arg.split("sendcmd=f=", maxsplit=1)[1].split(",", maxsplit=1)[0])
        command_files.append(command_path.read_text(encoding="utf-8"))
        output_path = Path(cmd[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"rendered")
        return None

    def fake_check_output(cmd, text):  # noqa: ARG001
        return "1920,1080"

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("subprocess.check_output", fake_check_output)

    request = Phase6RenderRequest.from_payload(
        {
            "run_id": "run-123",
            "source_video_gcs_uri": "gs://bucket/source.mp4",
            "artifact_gcs_uris": {
                "render_plan": "gs://bucket/phase14/run-123/render_inputs/render_plan.json",
                "captions_clip_001.ass": "gs://bucket/phase14/run-123/render_inputs/captions_clip_001.ass",
            },
            "clips": [{"clip_id": "clip_001", "clip_start_ms": 100, "clip_end_ms": 1100}],
        }
    )

    run_phase6_render(
        request=request,
        storage_client=SmoothCropPlanStorage(tmp_path),
        scratch_root=tmp_path,
    )

    vf_arg = commands[0][commands[0].index("-vf") + 1]
    assert "sendcmd=f=" in vf_arg
    assert "crop@follow=360:640:100:50" in vf_arg
    assert "scale=1080:1920" in vf_arg
    command_text = command_files[0]
    assert "0.000 crop@follow x 100, crop@follow y 50;" in command_text
    assert "0.500 crop@follow x 220, crop@follow y 90;" in command_text
    assert "1.000 crop@follow x 340, crop@follow y 120;" in command_text


def test_run_phase6_render_uses_repo_pinned_fonts_when_env_unset(tmp_path, monkeypatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.delenv("CLYPT_PHASE6_FONT_ASSET_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    def fake_run(cmd, check, capture_output, text):  # noqa: ARG001
        commands.append(cmd)
        output_path = Path(cmd[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"rendered")
        return None

    monkeypatch.setattr("subprocess.run", fake_run)

    request = Phase6RenderRequest.from_payload(
        {
            "run_id": "run-123",
            "source_video_gcs_uri": "gs://bucket/source.mp4",
            "artifact_gcs_uris": {
                "render_plan": "gs://bucket/phase14/run-123/render_inputs/render_plan.json",
                "captions_clip_001.ass": "gs://bucket/phase14/run-123/render_inputs/captions_clip_001.ass",
            },
            "clips": [
                {
                    "clip_id": "clip_001",
                    "clip_start_ms": 100,
                    "clip_end_ms": 900,
                }
            ],
        }
    )

    result = run_phase6_render(
        request=request,
        storage_client=_FakeStorageClient(tmp_path),
        scratch_root=tmp_path,
    )

    assert result["outputs"][0]["clip_id"] == "clip_001"
    assert any("fontsdir=" in token for token in commands[0])
