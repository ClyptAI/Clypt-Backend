"""Run the Clypt backend API server.

Usage:
    python -m backend.api          # default: 0.0.0.0:8000
    python -m backend.api --port 9000
"""
from __future__ import annotations

import argparse
import uvicorn

from backend.api.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Clypt Backend API")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
