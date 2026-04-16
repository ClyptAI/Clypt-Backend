from __future__ import annotations

import logging
import os
import subprocess
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


def build_vibevoice_start_command() -> list[str]:
    repo_dir = os.getenv("CLYPT_L4_VIBEVOICE_REPO_DIR", "/app/vibevoice-repo")
    max_num_seqs = os.getenv("CLYPT_L4_VIBEVOICE_MAX_NUM_SEQS", "2")
    gpu_memory_utilization = os.getenv("CLYPT_L4_VIBEVOICE_GPU_MEMORY_UTILIZATION", "0.25")
    return [
        "python3",
        f"{repo_dir}/vllm_plugin/scripts/start_server.py",
        "--max-num-seqs",
        str(max_num_seqs),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]


def wait_for_vibevoice_health(
    *,
    base_url: str,
    healthcheck_path: str = "/health",
    timeout_s: float = 600.0,
    poll_interval_s: float = 5.0,
) -> None:
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
    env.setdefault("VIBEVOICE_FFMPEG_MAX_CONCURRENCY", "64")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    command = build_vibevoice_start_command()
    logger.info("launching in-container VibeVoice server: %s", " ".join(command))
    process = subprocess.Popen(command, env=env)
    try:
        wait_for_vibevoice_health(
            base_url=os.getenv("VIBEVOICE_VLLM_BASE_URL", "http://127.0.0.1:8000"),
            healthcheck_path=os.getenv("VIBEVOICE_VLLM_HEALTHCHECK_PATH", "/health"),
            timeout_s=float(os.getenv("CLYPT_L4_VIBEVOICE_STARTUP_TIMEOUT_S", "600")),
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
