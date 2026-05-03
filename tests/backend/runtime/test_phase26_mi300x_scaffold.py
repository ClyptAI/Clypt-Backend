from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _load_bench_module():
    path = REPO_ROOT / "scripts/bench_phase24_llm_concurrency.py"
    spec = importlib.util.spec_from_file_location("bench_phase24_llm_concurrency", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase26_mi300x_requirements_avoid_cuda_and_nvidia_deps() -> None:
    requirements = _read("requirements-do-phase26-mi300x.txt")

    assert "Phase26 MI300X ROCm runtime deps" in requirements
    assert "--extra-index-url https://download.pytorch.org/whl/cu" not in requirements
    assert "nvidia" not in requirements.lower()
    assert "cuda" not in requirements.lower()
    assert "fastapi" in requirements
    assert "uvicorn" in requirements


def test_rocm_container_launcher_uses_pinned_sglang_and_amd_device_flags() -> None:
    script = _read("scripts/do_phase26/run_sglang_qwen_rocm_container.sh")

    assert "clypt/sglang:v0.5.10-rocm720-mi30x-clypt1" in script
    assert "Qwen/Qwen3.6-35B-A3B" in script
    assert "--device=/dev/kfd" in script
    assert "--device=/dev/dri" in script
    assert "--group-add video" in script
    assert "--ipc=host" in script
    assert "--cap-add SYS_PTRACE" in script
    assert "--security-opt seccomp=unconfined" in script
    assert "HF_HUB_OFFLINE" in script
    assert "--kv-cache-dtype" in script
    assert "--grammar-backend" in script
    assert "--speculative-algorithm" in script
    assert "SG_LAUNCH_PROFILE_OVERRIDE" in script
    assert "no_buffer" in script
    assert "--disable-radix-cache" in script
    assert "SGLANG_USE_AITER" in script
    assert "SGLANG_ROCM_DISABLE_LINEARQUANT=1" in script
    assert "SGLANG_ENABLE_QUARK_QUANTIZATION=0" in script


def test_phase26_mi300x_deploy_prewarms_offline_validates_models_then_starts_worker() -> None:
    script = _read("scripts/do_phase26/deploy_phase26_mi300x_services.sh")

    prewarm = script.index("snapshot_download")
    offline = script.index("HF_HUB_OFFLINE=1 _restart_sglang_for_profile")
    validate = script.index("_wait_for_models")
    dispatch = script.index("systemctl restart clypt-phase26-dispatch.service")
    worker = script.index("systemctl restart clypt-phase26-worker.service")

    assert validate < offline
    assert prewarm < offline < dispatch < worker
    assert "docker pull" in script
    assert "build_sglang_rocm_mi300x_image.sh" in script
    assert "sglang.__version__" in script
    assert "QUANTIZATION_METHODS" in script
    assert "SG_LAUNCH_PROFILE_OVERRIDE" in script
    assert "_stop_sglang_service" in script
    assert "SG_ACCEPTANCE_PROFILES" in script
    assert "run_sglang_qwen_rocm_container.sh" in script
    assert "scripts/do_phase26/systemd/amd" in script
    assert "flock -n 9" in script
    assert "another deploy_phase26_mi300x_services.sh run is already active" in script
    assert "_wait_for_url http://127.0.0.1:9300/health" in script
    assert "_wait_for_url http://127.0.0.1:8080/health" in script


def test_amd_systemd_units_preserve_phase26_boundaries() -> None:
    sglang = _read("scripts/do_phase26/systemd/amd/clypt-phase26-sglang-qwen.service")
    dispatch = _read("scripts/do_phase26/systemd/amd/clypt-phase26-dispatch.service")
    worker = _read("scripts/do_phase26/systemd/amd/clypt-phase26-worker.service")

    assert "run_sglang_qwen_rocm_container.sh" in sglang
    assert "HF_HUB_OFFLINE=1" in sglang
    assert "ExecStop=-/usr/bin/docker rm -f clypt-phase26-sglang-qwen" in sglang
    assert "run_phase26_dispatch_service" in dispatch
    assert "--port 9300" in dispatch
    assert "run_phase26_worker" in worker
    assert "Requires=clypt-phase26-sglang-qwen.service" in worker


def test_bench_rocm_metadata_capture_is_optional_and_non_intrusive(monkeypatch) -> None:
    module = _load_bench_module()
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)

        class Result:
            returncode = 0
            stdout = "gpu stats"
            stderr = ""

        return Result()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setenv("SG_MODEL", "Qwen/Qwen3.6-35B-A3B")
    monkeypatch.setenv("SG_CONTEXT_LENGTH", "65536")
    monkeypatch.setenv("SG_MEM_FRACTION_STATIC", "0.78")
    monkeypatch.setenv("SG_KV_CACHE_DTYPE", "fp8_e4m3")
    monkeypatch.setenv("SG_SPECULATIVE_ALGORITHM", "NEXTN")
    monkeypatch.setenv("SG_DOCKER_IMAGE", "lmsysorg/sglang:v0.5.10-rocm720-mi30x")

    metadata = module._collect_rocm_metadata(
        enabled=True,
        label="before",
        include_amd_smi=True,
    )

    assert metadata["enabled"] is True
    assert metadata["label"] == "before"
    assert metadata["commands"]["rocm-smi"]["ok"] is True
    assert metadata["commands"]["amd-smi"]["ok"] is True
    assert metadata["sglang"]["model"] == "Qwen/Qwen3.6-35B-A3B"
    assert metadata["sglang"]["context_length"] == "65536"
    assert metadata["sglang"]["docker_image"] == "lmsysorg/sglang:v0.5.10-rocm720-mi30x"
    assert calls == [["rocm-smi"], ["amd-smi", "static"]]

    disabled = module._collect_rocm_metadata(enabled=False, label="before", include_amd_smi=True)
    assert disabled == {"enabled": False}
