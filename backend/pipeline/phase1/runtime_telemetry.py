from __future__ import annotations

import os
import subprocess
from typing import Any


def gpu_device_profile_name() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        return str(torch.cuda.get_device_name(0) or "").strip()
    except Exception:
        return "unknown"


def resolve_lrasd_runtime_profile(
    *,
    profile_raw: str,
    gpu_name: str,
) -> tuple[dict[str, int | str], str]:
    profile = str(profile_raw or "auto").strip().lower()
    warning = ""
    if profile not in {"auto", "h200", "h100", "a100", "default"}:
        warning = f"unknown CLYPT_LRASD_PROFILE={profile!r}; using auto profile resolution"
        profile = "auto"
    gpu_name_l = str(gpu_name or "").strip().lower()
    resolved = profile
    if profile == "auto":
        if "h200" in gpu_name_l:
            resolved = "h200"
        elif "h100" in gpu_name_l:
            resolved = "h100"
        elif "a100" in gpu_name_l:
            resolved = "a100"
        else:
            resolved = "default"
    tuned = {
        "h200": {"batch_size": 48, "max_inflight": 4, "prep_workers": 4},
        "h100": {"batch_size": 40, "max_inflight": 3, "prep_workers": 4},
        "a100": {"batch_size": 36, "max_inflight": 3, "prep_workers": 4},
        "default": {"batch_size": 32, "max_inflight": 4, "prep_workers": 4},
    }[resolved]
    return (
        {
            "profile": resolved,
            "gpu_name": gpu_name_l or "unknown",
            "batch_size": int(tuned["batch_size"]),
            "max_inflight": int(tuned["max_inflight"]),
            "prep_workers": int(tuned["prep_workers"]),
        },
        warning,
    )


def gpu_telemetry_snapshot() -> dict[str, float] | None:
    gpu_index = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_index = int(torch.cuda.current_device())
    except Exception:
        gpu_index = None
    cmd = ["nvidia-smi"]
    if gpu_index is not None and gpu_index >= 0:
        cmd.extend(["-i", str(gpu_index)])
    cmd.extend(
        [
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    line = ""
    for raw in (proc.stdout or "").splitlines():
        if raw.strip():
            line = raw.strip()
            break
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None
    try:
        util = float(parts[0])
        mem_used = float(parts[1])
        mem_total = float(parts[2])
    except Exception:
        return None
    mem_pct = (100.0 * mem_used / max(1.0, mem_total)) if mem_total > 0 else 0.0
    return {
        "gpu_utilization_pct": util,
        "gpu_memory_used_mb": mem_used,
        "gpu_memory_total_mb": mem_total,
        "gpu_memory_utilization_pct": mem_pct,
    }


def gpu_stage_window_metrics(
    *,
    stage_prefix: str,
    started: dict[str, Any] | None,
    ended: dict[str, Any] | None,
) -> dict[str, float | int]:
    out: dict[str, float | int] = {
        f"{stage_prefix}_gpu_telemetry_available": 1 if started and ended else 0
    }
    if not started or not ended:
        return out
    for key in (
        "gpu_utilization_pct",
        "gpu_memory_used_mb",
        "gpu_memory_total_mb",
        "gpu_memory_utilization_pct",
    ):
        sv = started.get(key)
        ev = ended.get(key)
        if isinstance(sv, (int, float)) and isinstance(ev, (int, float)):
            out[f"{stage_prefix}_{key}_sampled_mean"] = round((float(sv) + float(ev)) / 2.0, 3)
    return out

