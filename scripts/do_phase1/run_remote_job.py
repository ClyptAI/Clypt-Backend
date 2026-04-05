from __future__ import annotations

import argparse
import time

import httpx


class Phase1RemoteClient:
    def __init__(self, *, base_url: str, http_client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = http_client or httpx.Client(timeout=30.0)

    @property
    def jobs_url(self) -> str:
        return f"{self.base_url}/jobs"

    def job_url(self, *, job_id: str) -> str:
        return f"{self.base_url}/jobs/{job_id}"

    def logs_url(self, *, job_id: str, tail_lines: int = 200) -> str:
        return f"{self.job_url(job_id=job_id)}/logs?tail_lines={int(tail_lines)}"

    def submit_job(self, *, source_url: str | None, source_path: str | None = None, run_phase14: bool = False) -> dict:
        payload = {
            "source_url": source_url,
            "source_path": source_path,
            "runtime_controls": {
                "run_phase14": bool(run_phase14),
            },
        }
        response = self._client.post(self.jobs_url, json=payload)
        response.raise_for_status()
        return response.json()

    def get_job(self, *, job_id: str) -> dict:
        response = self._client.get(self.job_url(job_id=job_id))
        response.raise_for_status()
        return response.json()

    def get_logs(self, *, job_id: str, tail_lines: int = 200) -> dict:
        response = self._client.get(self.logs_url(job_id=job_id, tail_lines=tail_lines))
        response.raise_for_status()
        return response.json()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit a V3.1 Phase 1 remote job and tail logs.")
    parser.add_argument("--base-url", required=True, help="Phase 1 service base URL, e.g. http://HOST:8080")
    parser.add_argument("--source-url", required=True, help="YouTube or direct source URL")
    parser.add_argument("--run-phase14", action="store_true", help="Continue into live Phases 2-4 after Phase 1")
    parser.add_argument("--poll-interval-s", type=float, default=5.0)
    parser.add_argument("--tail-lines", type=int, default=200)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    client = Phase1RemoteClient(base_url=args.base_url)
    job = client.submit_job(
        source_url=args.source_url,
        run_phase14=bool(args.run_phase14),
    )
    job_id = job["job_id"]
    last_lines: list[str] = []
    while True:
        status = client.get_job(job_id=job_id)
        logs = client.get_logs(job_id=job_id, tail_lines=args.tail_lines)
        lines = list(logs.get("lines") or [])
        new_lines = lines[len(last_lines):] if len(lines) >= len(last_lines) else lines
        for line in new_lines:
            print(line)
        last_lines = lines
        if status.get("status") in {"succeeded", "failed"}:
            print(f"[phase1-remote] job {job_id} finished with status={status['status']}")
            return 0 if status["status"] == "succeeded" else 1
        time.sleep(max(0.5, float(args.poll_interval_s)))


if __name__ == "__main__":
    raise SystemExit(main())
