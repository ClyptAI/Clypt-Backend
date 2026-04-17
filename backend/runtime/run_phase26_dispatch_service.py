from __future__ import annotations

import argparse

import uvicorn

from backend.providers import load_phase26_host_settings
from backend.runtime.phase24_local_queue import Phase24LocalQueue
from backend.runtime.phase26_dispatch_service.app import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase26 dispatch FastAPI service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9300)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = load_phase26_host_settings()
    if settings.phase26_dispatch_service is None:
        raise SystemExit(
            "CLYPT_PHASE24_DISPATCH_AUTH_TOKEN must be set for the Phase26 dispatch service."
        )
    queue = Phase24LocalQueue(path=settings.phase24_local_queue.path)
    uvicorn.run(
        create_app(
            queue=queue,
            expected_auth_token=settings.phase26_dispatch_service.auth_token,
        ),
        host=args.host,
        port=args.port,
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
