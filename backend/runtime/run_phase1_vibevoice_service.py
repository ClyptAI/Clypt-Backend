from __future__ import annotations

import argparse

import uvicorn

from backend.runtime.phase1_vibevoice_service.app import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 1 VibeVoice FastAPI service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
