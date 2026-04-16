from __future__ import annotations

import argparse
import logging
import os

try:
    import uvicorn
except ModuleNotFoundError:  # pragma: no cover - test environments may not have uvicorn installed
    from types import SimpleNamespace

    uvicorn = SimpleNamespace(run=None)

from .l4_combined_bootstrap import launch_vibevoice_server, stop_vibevoice_server
from .phase24_media_prep_app import create_app


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Clypt combined L4 ASR + node-media-prep service."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")))
    return parser


def _apply_l4_runtime_env_defaults() -> None:
    os.environ.setdefault("VIBEVOICE_BACKEND", "vllm")
    os.environ.setdefault("VIBEVOICE_VLLM_BASE_URL", "http://127.0.0.1:8000")
    os.environ.setdefault("VIBEVOICE_VLLM_HEALTHCHECK_PATH", "/health")


def main() -> int:
    configure_logging()
    args = build_parser().parse_args()
    _apply_l4_runtime_env_defaults()
    process = launch_vibevoice_server()
    try:
        uvicorn.run(create_app(), host=args.host, port=args.port, reload=False)
    finally:
        stop_vibevoice_server(process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
