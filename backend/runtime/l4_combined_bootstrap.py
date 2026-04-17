from __future__ import annotations

import logging
import os
import subprocess
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


def build_vibevoice_start_command() -> list[str]:
    # VibeVoice's vllm_plugin/scripts/start_server.py hardcodes `pip install -e /app[vllm]`,
    # so the VibeVoice source tree must live at /app. This matches both the
    # GCE L4 container layout (docker/phase24-media-prep/Dockerfile) and the DO
    # droplet layout (scripts/do_phase1/systemd/clypt-vllm-vibevoice.service mounts
    # /opt/clypt-phase1/vibevoice-repo to /app inside the container).
    #
    # Defaults are tuned for a dedicated 1x L4 (24 GB) running only VibeVoice
    # vLLM + ffmpeg GPU transcode. With bf16 audio encoder (patched in the
    # Dockerfile) the model footprint is ~10 GB, leaving ~11 GB of vLLM budget
    # for KV cache + activations when gpu_memory_utilization=0.90. KV cache
    # math: max_num_seqs * max_model_len * 57 KiB (Qwen2 28-layer bf16) =>
    # 4 * 16384 * 57 KiB ~= 3.6 GB, with ample headroom for profile_run.
    repo_dir = os.getenv("CLYPT_L4_VIBEVOICE_REPO_DIR", "/app")
    max_num_seqs = os.getenv("CLYPT_L4_VIBEVOICE_MAX_NUM_SEQS", "4")
    max_model_len = os.getenv("CLYPT_L4_VIBEVOICE_MAX_MODEL_LEN", "16384")
    gpu_memory_utilization = os.getenv(
        "CLYPT_L4_VIBEVOICE_GPU_MEMORY_UTILIZATION", "0.90"
    )
    return [
        "python3",
        f"{repo_dir}/vllm_plugin/scripts/start_server.py",
        # ffmpeg + libsndfile are installed in the Dockerfile; skipping the
        # launcher's apt-get run shaves ~30s off cold-start and removes a
        # runtime network dependency.
        "--skip-deps",
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]


def wait_for_vibevoice_health(
    *,
    base_url: str,
    healthcheck_path: str = "/health",
    timeout_s: float = 1500.0,
    poll_interval_s: float = 5.0,
) -> None:
    # Cold-start on Cloud Run L4 can legitimately take 10-15 minutes because
    # VibeVoice's start_server.py does `pip install -e /app[vllm]` (2-3 min),
    # then snapshot_download from HuggingFace (1-3 min), then vLLM model load
    # onto the GPU (2-5 min). 600s was too tight; 25 min gives headroom without
    # hiding genuine hangs.
    deadline = time.monotonic() + timeout_s
    health_url = base_url.rstrip("/") + healthcheck_path
    while True:
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status in (200, 204):
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"VibeVoice health check did not pass within {timeout_s:.1f}s: {health_url}"
            )
        time.sleep(poll_interval_s)


def launch_vibevoice_server() -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    # g2-standard-8 has 8 vCPU and the L4 exposes 2x NVENC + 4x NVDEC engines.
    # 16 concurrent ffmpeg workers keeps the encode pipeline saturated without
    # CPU thrash; H200's wider CPU + 141 GB host margin is what justified 64.
    env.setdefault("VIBEVOICE_FFMPEG_MAX_CONCURRENCY", "16")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    command = build_vibevoice_start_command()
    logger.info("launching in-container VibeVoice server: %s", " ".join(command))
    process = subprocess.Popen(command, env=env)
    try:
        wait_for_vibevoice_health(
            base_url=os.getenv("VIBEVOICE_VLLM_BASE_URL", "http://127.0.0.1:8000"),
            healthcheck_path=os.getenv("VIBEVOICE_VLLM_HEALTHCHECK_PATH", "/health"),
            timeout_s=float(os.getenv("CLYPT_L4_VIBEVOICE_STARTUP_TIMEOUT_S", "1500")),
            poll_interval_s=float(os.getenv("CLYPT_L4_VIBEVOICE_HEALTH_POLL_INTERVAL_S", "5")),
        )
    except Exception:
        stop_vibevoice_server(process)
        raise
    return process


def stop_vibevoice_server(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


__all__ = [
    "build_vibevoice_start_command",
    "launch_vibevoice_server",
    "stop_vibevoice_server",
    "wait_for_vibevoice_health",
]
