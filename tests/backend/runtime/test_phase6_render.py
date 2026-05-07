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
    assert "scale=1080:1920" in vf_arg
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
                    '{"run_id":"run-123","source_context_ref":"source_context.json","caption_plan_ref":"caption_plan.json","publish_metadata_ref":"publish_metadata.json","clips":[{"clip_id":"clip_001","clip_start_ms":100,"clip_end_ms":900,"caption_plan_ref":"caption_plan.json","publish_metadata_ref":"publish_metadata.json","caption_segment_ids":["clip_001_seg_001"],"caption_zone":"lower_safe","caption_preset_id":"karaoke_focus","review_needed":false,"review_reasons":[],"overlays":[],"crop_plan":{"mode":"tracklet_follow_9x16_pose_x_dynamic_inside_person","runs":[{"run_id":"run_0001","shot_id":"shot_0001","tracklet_id":"shot_0001:7","start_ms":0,"end_ms":800}],"keyframes":[{"run_id":"run_0001","shot_id":"shot_0001","time_ms":0,"x":900,"y":100,"w":300,"h":534},{"run_id":"run_0001","shot_id":"shot_0001","time_ms":800,"x":900,"y":100,"w":300,"h":534}]},"segments":[]}]}',
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

    assert len(commands) == 3
    piece_vf = commands[0][commands[0].index("-vf") + 1]
    final_vf = commands[2][commands[2].index("-vf") + 1]
    assert "sendcmd=f=" in piece_vf
    assert "crop@follow=300:534:900:100" in piece_vf
    assert "scale=1080:1920" in piece_vf
    assert "subtitles=" not in piece_vf
    assert final_vf.startswith("subtitles=")


def test_run_phase6_render_interpolates_dynamic_pose_crop_keyframes(tmp_path, monkeypatch) -> None:
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
                    '{"run_id":"run-123","source_context_ref":"source_context.json","caption_plan_ref":"caption_plan.json","publish_metadata_ref":"publish_metadata.json","clips":[{"clip_id":"clip_001","clip_start_ms":100,"clip_end_ms":1100,"caption_plan_ref":"caption_plan.json","publish_metadata_ref":"publish_metadata.json","caption_segment_ids":["clip_001_seg_001"],"caption_zone":"lower_safe","caption_preset_id":"karaoke_focus","review_needed":false,"review_reasons":[],"overlays":[],"crop_plan":{"mode":"tracklet_follow_9x16_pose_x_dynamic_inside_person","runs":[{"run_id":"run_0001","shot_id":"shot_1","tracklet_id":"shot_1:track_1","start_ms":0,"end_ms":1000}],"keyframes":[{"run_id":"run_0001","shot_id":"shot_1","time_ms":0,"x":100,"y":50,"w":360,"h":640},{"run_id":"run_0001","shot_id":"shot_1","time_ms":500,"x":220,"y":90,"w":420,"h":746},{"run_id":"run_0001","shot_id":"shot_1","time_ms":1000,"x":340,"y":120,"w":480,"h":854}]},"segments":[]}]}',
                    encoding="utf-8",
                )
            return path

    def fake_run(cmd, check, capture_output, text):  # noqa: ARG001
        commands.append(cmd)
        if "-vf" in cmd:
            vf_arg = cmd[cmd.index("-vf") + 1]
            if "sendcmd=f=" in vf_arg:
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

    assert len(commands) == 3
    vf_arg = commands[0][commands[0].index("-vf") + 1]
    assert "sendcmd=f=" in vf_arg
    assert "crop@follow=360:640:100:50" in vf_arg
    assert "scale=1080:1920" in vf_arg
    command_text = command_files[0]
    assert "0.000 crop@follow x 100, crop@follow y 50;" in command_text
    assert "0.500 crop@follow x 220, crop@follow y 90;" in command_text
    assert "1.000 crop@follow x 340, crop@follow y 120;" in command_text
    assert commands[2][commands[2].index("-vf") + 1].startswith("subtitles=")


def test_run_phase6_render_keeps_crop_interpolation_run_local_at_shot_cut(tmp_path, monkeypatch) -> None:
    commands: list[list[str]] = []
    command_files: list[str] = []
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "montserrat_extra_bold_v1.ttf").write_bytes(b"font")
    monkeypatch.setenv("CLYPT_PHASE6_FONT_ASSET_DIR", str(font_dir))

    class CutCropPlanStorage(_FakeStorageClient):
        def download_file(self, *, gcs_uri: str, local_path: Path) -> Path:
            path = super().download_file(gcs_uri=gcs_uri, local_path=local_path)
            if gcs_uri.endswith(".json"):
                path.write_text(
                    '{"run_id":"run-123","source_context_ref":"source_context.json","caption_plan_ref":"caption_plan.json","publish_metadata_ref":"publish_metadata.json","clips":[{"clip_id":"clip_001","clip_start_ms":0,"clip_end_ms":1000,"caption_plan_ref":"caption_plan.json","publish_metadata_ref":"publish_metadata.json","caption_segment_ids":["clip_001_seg_001"],"caption_zone":"lower_safe","caption_preset_id":"karaoke_focus","review_needed":false,"review_reasons":[],"overlays":[],"crop_plan":{"mode":"tracklet_follow_9x16_pose_x_dynamic_inside_person","runs":[{"run_id":"run_0001","shot_id":"shot_1","tracklet_id":"shot_1:track_1","start_ms":0,"end_ms":500},{"run_id":"run_0002","shot_id":"shot_2","tracklet_id":"shot_2:track_9","start_ms":500,"end_ms":1000}],"keyframes":[{"run_id":"run_0001","shot_id":"shot_1","time_ms":0,"x":100,"y":50,"w":360,"h":640},{"run_id":"run_0001","shot_id":"shot_1","time_ms":480,"x":140,"y":50,"w":360,"h":640},{"run_id":"run_0002","shot_id":"shot_2","time_ms":500,"x":900,"y":80,"w":420,"h":746},{"run_id":"run_0002","shot_id":"shot_2","time_ms":1000,"x":940,"y":80,"w":420,"h":746}]},"segments":[]}]}',
                    encoding="utf-8",
                )
            return path

    def fake_run(cmd, check, capture_output, text):  # noqa: ARG001
        commands.append(cmd)
        if "-vf" in cmd:
            vf_arg = cmd[cmd.index("-vf") + 1]
            if "sendcmd=f=" in vf_arg:
                command_path = Path(vf_arg.split("sendcmd=f=", maxsplit=1)[1].split(",", maxsplit=1)[0])
                command_files.append(command_path.read_text(encoding="utf-8"))
        output_path = Path(cmd[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"rendered")
        return None

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("subprocess.check_output", lambda cmd, text: "1920,1080")  # noqa: ARG005

    request = Phase6RenderRequest.from_payload(
        {
            "run_id": "run-123",
            "source_video_gcs_uri": "gs://bucket/source.mp4",
            "artifact_gcs_uris": {
                "render_plan": "gs://bucket/phase14/run-123/render_inputs/render_plan.json",
                "captions_clip_001.ass": "gs://bucket/phase14/run-123/render_inputs/captions_clip_001.ass",
            },
            "clips": [{"clip_id": "clip_001", "clip_start_ms": 0, "clip_end_ms": 1000}],
        }
    )

    run_phase6_render(
        request=request,
        storage_client=CutCropPlanStorage(tmp_path),
        scratch_root=tmp_path,
    )

    assert len(commands) == 5
    first_piece_vf = commands[0][commands[0].index("-vf") + 1]
    second_piece_vf = commands[1][commands[1].index("-vf") + 1]
    first_piece_cmd = command_files[0]
    second_piece_cmd = command_files[1]
    assert "crop@follow=360:640:100:50" in first_piece_vf
    assert "crop@follow=420:746:900:80" in second_piece_vf
    assert "0.480 crop@follow" in first_piece_cmd
    assert "0.490 crop@follow" not in first_piece_cmd
    assert "0.000 crop@follow x 900, crop@follow y 80;" in second_piece_cmd
    assert "0.500 crop@follow x 940, crop@follow y 80;" in second_piece_cmd


def test_run_phase6_render_reanchors_dynamic_crop_xy_from_bbox_and_anchor(tmp_path, monkeypatch) -> None:
    commands: list[list[str]] = []
    command_files: list[str] = []
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "montserrat_extra_bold_v1.ttf").write_bytes(b"font")
    monkeypatch.setenv("CLYPT_PHASE6_FONT_ASSET_DIR", str(font_dir))

    class BboxCropPlanStorage(_FakeStorageClient):
        def download_file(self, *, gcs_uri: str, local_path: Path) -> Path:
            path = super().download_file(gcs_uri=gcs_uri, local_path=local_path)
            if gcs_uri.endswith(".json"):
                path.write_text(
                    '{"run_id":"run-123","source_context_ref":"source_context.json","caption_plan_ref":"caption_plan.json","publish_metadata_ref":"publish_metadata.json","clips":[{"clip_id":"clip_001","clip_start_ms":0,"clip_end_ms":1000,"caption_plan_ref":"caption_plan.json","publish_metadata_ref":"publish_metadata.json","caption_segment_ids":["clip_001_seg_001"],"caption_zone":"lower_safe","caption_preset_id":"karaoke_focus","review_needed":false,"review_reasons":[],"overlays":[],"crop_plan":{"mode":"tracklet_follow_9x16_pose_x_dynamic_inside_person","runs":[{"run_id":"run_0001","shot_id":"shot_1","tracklet_id":"shot_1:track_1","start_ms":0,"end_ms":1000}],"keyframes":[{"run_id":"run_0001","shot_id":"shot_1","time_ms":0,"x":400,"y":50,"w":420,"h":746,"anchor_x":520,"bbox_xyxy":[300,50,720,900]},{"run_id":"run_0001","shot_id":"shot_1","time_ms":1000,"x":700,"y":80,"w":360,"h":640,"anchor_x":860,"bbox_xyxy":[620,80,980,720]}]},"segments":[]}]}',
                    encoding="utf-8",
                )
            return path

    def fake_run(cmd, check, capture_output, text):  # noqa: ARG001
        commands.append(cmd)
        if "-vf" in cmd:
            vf_arg = cmd[cmd.index("-vf") + 1]
            if "sendcmd=f=" in vf_arg:
                command_path = Path(vf_arg.split("sendcmd=f=", maxsplit=1)[1].split(",", maxsplit=1)[0])
                command_files.append(command_path.read_text(encoding="utf-8"))
        output_path = Path(cmd[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"rendered")
        return None

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("subprocess.check_output", lambda cmd, text: "1920,1080")  # noqa: ARG005

    request = Phase6RenderRequest.from_payload(
        {
            "run_id": "run-123",
            "source_video_gcs_uri": "gs://bucket/source.mp4",
            "artifact_gcs_uris": {
                "render_plan": "gs://bucket/phase14/run-123/render_inputs/render_plan.json",
                "captions_clip_001.ass": "gs://bucket/phase14/run-123/render_inputs/captions_clip_001.ass",
            },
            "clips": [{"clip_id": "clip_001", "clip_start_ms": 0, "clip_end_ms": 1000}],
        }
    )

    run_phase6_render(
        request=request,
        storage_client=BboxCropPlanStorage(tmp_path),
        scratch_root=tmp_path,
    )

    assert len(commands) == 3
    vf_arg = commands[0][commands[0].index("-vf") + 1]
    assert "crop@follow=360:640:340:50" in vf_arg
    command_text = command_files[0]
    assert "0.000 crop@follow x 340, crop@follow y 50;" in command_text
    assert "1.000 crop@follow x 620, crop@follow y 80;" in command_text


def test_run_phase6_render_fails_fast_for_unknown_crop_mode(tmp_path, monkeypatch) -> None:
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "montserrat_extra_bold_v1.ttf").write_bytes(b"font")
    monkeypatch.setenv("CLYPT_PHASE6_FONT_ASSET_DIR", str(font_dir))

    class BadCropPlanStorage(_FakeStorageClient):
        def download_file(self, *, gcs_uri: str, local_path: Path) -> Path:
            path = super().download_file(gcs_uri=gcs_uri, local_path=local_path)
            if gcs_uri.endswith(".json"):
                path.write_text(
                    '{"run_id":"run-123","source_context_ref":"source_context.json","caption_plan_ref":"caption_plan.json","publish_metadata_ref":"publish_metadata.json","clips":[{"clip_id":"clip_001","clip_start_ms":0,"clip_end_ms":1000,"caption_plan_ref":"caption_plan.json","publish_metadata_ref":"publish_metadata.json","caption_segment_ids":["clip_001_seg_001"],"caption_zone":"lower_safe","caption_preset_id":"karaoke_focus","review_needed":false,"review_reasons":[],"overlays":[],"crop_plan":{"mode":"surprise_pan","keyframes":[{"time_ms":0,"x":100,"y":50,"w":360,"h":640}]},"segments":[]}]}',
                    encoding="utf-8",
                )
            return path

    monkeypatch.setattr("subprocess.check_output", lambda cmd, text: "1920,1080")  # noqa: ARG005
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, check, capture_output, text: None,  # noqa: ARG005
    )
    request = Phase6RenderRequest.from_payload(
        {
            "run_id": "run-123",
            "source_video_gcs_uri": "gs://bucket/source.mp4",
            "artifact_gcs_uris": {
                "render_plan": "gs://bucket/phase14/run-123/render_inputs/render_plan.json",
                "captions_clip_001.ass": "gs://bucket/phase14/run-123/render_inputs/captions_clip_001.ass",
            },
            "clips": [{"clip_id": "clip_001", "clip_start_ms": 0, "clip_end_ms": 1000}],
        }
    )

    import pytest

    with pytest.raises(ValueError, match="unknown Phase6 crop plan mode"):
        run_phase6_render(
            request=request,
            storage_client=BadCropPlanStorage(tmp_path),
            scratch_root=tmp_path,
        )


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
