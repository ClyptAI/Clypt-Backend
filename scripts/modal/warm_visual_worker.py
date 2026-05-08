from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.phase1_runtime.visual_warmup import load_visual_warmup_spec_from_env


def _normalize_visual_base_url(service_url: str) -> str:
    base = service_url.strip().rstrip("/")
    if not base:
        raise ValueError("visual service URL is required")
    task_path = "/tasks/visual-extract"
    if base.endswith(task_path):
        return base[: -len(task_path)]
    return base


def _request_json(
    *,
    url: str,
    method: str,
    auth_token: str,
    payload: dict[str, Any] | None,
    timeout_s: float,
) -> tuple[int, dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Accept": "application/json",
    }
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{method} {url} returned non-JSON body: {exc}: {raw[:512]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{method} {url} returned non-object JSON: {type(parsed).__name__}")
    return status, parsed


def _wait_for_ready(
    *,
    base_url: str,
    auth_token: str,
    timeout_s: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    ready_url = f"{base_url}/ready"
    last_payload: dict[str, Any] | None = None
    last_status: int | None = None
    while time.monotonic() <= deadline:
        status, payload = _request_json(
            url=ready_url,
            method="GET",
            auth_token=auth_token,
            payload=None,
            timeout_s=min(30.0, timeout_s),
        )
        last_payload = payload
        last_status = status
        if status == 200:
            return payload
        time.sleep(max(0.1, poll_interval_s))
    raise RuntimeError(
        f"/ready did not return 200 before timeout; last_status={last_status} last_payload={last_payload}"
    )


def _wait_for_warmup_result(
    *,
    base_url: str,
    auth_token: str,
    result_path: str,
    timeout_s: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    result_url = result_path if result_path.startswith("http") else f"{base_url}{result_path}"
    last_payload: dict[str, Any] | None = None
    last_status: int | None = None
    while time.monotonic() <= deadline:
        status, payload = _request_json(
            url=result_url,
            method="GET",
            auth_token=auth_token,
            payload=None,
            timeout_s=min(30.0, timeout_s),
        )
        last_payload = payload
        last_status = status
        state = str(payload.get("status") or "").strip().lower()
        if status == 200 and state in {"", "success", "succeeded", "completed"}:
            return payload
        if status == 200 and state not in {"pending", "running", "submitted"}:
            raise RuntimeError(
                f"visual warmup returned unexpected terminal payload: status={status} payload={payload}"
            )
        time.sleep(max(0.1, poll_interval_s))
    raise RuntimeError(
        "visual warmup did not finish before timeout; "
        f"last_status={last_status} last_payload={last_payload}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Warm the Modal visual worker with the canonical person-containing warmup clip "
            "and wait until /ready confirms the live GPU worker is warm."
        )
    )
    parser.add_argument(
        "--service-url",
        default=os.environ.get("CLYPT_PHASE1_VISUAL_SERVICE_URL", "").strip(),
        help="Modal visual service base URL or /tasks/visual-extract URL.",
    )
    parser.add_argument(
        "--auth-token",
        default=(
            os.environ.get("CLYPT_PHASE1_VISUAL_SERVICE_AUTH_TOKEN")
            or os.environ.get("VISUAL_EXTRACT_AUTH_TOKEN")
            or ""
        ).strip(),
        help="Bearer token for the Modal visual service.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=1800.0,
        help="Total timeout for warmup submission, polling, and readiness confirmation.",
    )
    parser.add_argument(
        "--poll-interval-s",
        type=float,
        default=2.0,
        help="Polling interval for result and readiness checks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run a fresh warmup even if /ready already reports success.",
    )
    parser.add_argument(
        "--video-gcs-uri",
        default="",
        help="Optional override for the warmup clip GCS URI.",
    )
    args = parser.parse_args()

    if not args.service_url:
        raise RuntimeError("missing --service-url and CLYPT_PHASE1_VISUAL_SERVICE_URL")
    if not args.auth_token:
        raise RuntimeError(
            "missing --auth-token and CLYPT_PHASE1_VISUAL_SERVICE_AUTH_TOKEN/VISUAL_EXTRACT_AUTH_TOKEN"
        )

    base_url = _normalize_visual_base_url(args.service_url)
    warmup_spec = load_visual_warmup_spec_from_env()
    print(
        json.dumps(
            {
                "service_base_url": base_url,
                "warmup_asset_id": warmup_spec.asset_id,
                "warmup_video_gcs_uri": args.video_gcs_uri or warmup_spec.warmup_video_gcs_uri,
                "source_test_bank_url": warmup_spec.source_test_bank_url,
                "clip_start_ms": warmup_spec.clip_start_ms,
                "clip_end_ms": warmup_spec.clip_end_ms,
                "force": args.force,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )

    if not args.force:
        ready_status, ready_payload = _request_json(
            url=f"{base_url}/ready",
            method="GET",
            auth_token=args.auth_token,
            payload=None,
            timeout_s=min(30.0, args.timeout_s),
        )
        if ready_status == 200:
            print("Visual worker already ready.")
            print(json.dumps(ready_payload, ensure_ascii=True, indent=2, sort_keys=True))
            return 0

    submit_payload: dict[str, Any] = {}
    if args.video_gcs_uri.strip():
        submit_payload["video_gcs_uri"] = args.video_gcs_uri.strip()
    submit_status, submit_response = _request_json(
        url=f"{base_url}/tasks/visual-warmup",
        method="POST",
        auth_token=args.auth_token,
        payload=submit_payload,
        timeout_s=min(60.0, args.timeout_s),
    )
    if submit_status != 202:
        raise RuntimeError(
            f"visual warmup submit failed: status={submit_status} payload={submit_response}"
        )
    result_path = str(submit_response.get("result_path") or "").strip()
    if not result_path:
        raise RuntimeError(f"visual warmup submit missing result_path: {submit_response}")
    result = _wait_for_warmup_result(
        base_url=base_url,
        auth_token=args.auth_token,
        result_path=result_path,
        timeout_s=args.timeout_s,
        poll_interval_s=args.poll_interval_s,
    )
    ready_payload = _wait_for_ready(
        base_url=base_url,
        auth_token=args.auth_token,
        timeout_s=args.timeout_s,
        poll_interval_s=args.poll_interval_s,
    )
    print("Visual warmup completed.")
    print(json.dumps({"warmup_result": result, "ready": ready_payload}, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - operator CLI
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
