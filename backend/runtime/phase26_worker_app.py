from __future__ import annotations

from backend.providers import load_phase26_host_settings

from .phase24_worker_app import build_default_phase24_worker_service


def build_default_phase26_worker_service():
    return build_default_phase24_worker_service(settings=load_phase26_host_settings())


__all__ = ["build_default_phase26_worker_service"]
