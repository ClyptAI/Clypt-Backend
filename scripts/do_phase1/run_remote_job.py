#!/usr/bin/env python3
from __future__ import annotations

import json
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse


DROPLET_HOST = "root@162.243.100.226"
SSH_KEY_PATH = Path.home() / ".ssh" / "clypt_do_ed25519"
REMOTE_RELAY_DIR = "/opt/clypt-phase1/relay"
REMOTE_RELAY_PORT = 8091
REMOTE_API_URL = "http://127.0.0.1:8080"
REMOTE_LOG_ROOT = "/var/lib/clypt/do_phase1_service/workdir/logs"
LOCAL_CACHE_DIR = Path(__file__).resolve().parents[2] / ".tmp" / "do-phase1-relay"
YTDLP_FORMAT = "bv*[vcodec~='^avc1']+ba[ext=m4a]/b[ext=mp4]/b"


class UserFacingError(RuntimeError):
    pass


def extract_youtube_video_id(url: str) -> str:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")
    if host in {"youtu.be", "www.youtu.be"}:
        if path:
            return path.split("/")[0]
        raise ValueError("Missing YouTube video id")
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
        if path == "watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0].strip()
            if video_id:
                return video_id
        if path.startswith("shorts/") or path.startswith("embed/"):
            parts = path.split("/")
            if len(parts) >= 2 and parts[1]:
                return parts[1]
    raise ValueError("Please enter a valid YouTube URL")


def relay_filename(video_id: str) -> str:
    return f"{video_id}.mp4"


def remote_relay_path(video_id: str) -> str:
    return f"{REMOTE_RELAY_DIR}/{relay_filename(video_id)}"


def remote_relay_source_url(video_id: str) -> str:
    return f"http://127.0.0.1:{REMOTE_RELAY_PORT}/{relay_filename(video_id)}"


def remote_job_log_path(job_id: str) -> str:
    return f"{REMOTE_LOG_ROOT}/{job_id}.log"


def local_cache_path(video_id: str) -> Path:
    return LOCAL_CACHE_DIR / relay_filename(video_id)


def _ssh_base() -> list[str]:
    return ["ssh", "-i", str(SSH_KEY_PATH), DROPLET_HOST]


def run_remote_bash(script: str, *, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    exports = ""
    if env:
        exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items()) + " "
    remote_command = f"{exports}bash -lc {shlex.quote(script)}"
    result = subprocess.run(
        _ssh_base() + [remote_command],
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise UserFacingError((result.stderr or result.stdout or "remote command failed").strip())
    return result


def remote_file_exists(video_id: str) -> bool:
    result = run_remote_bash(f"test -s {shlex.quote(remote_relay_path(video_id))}", check=False)
    return result.returncode == 0


def ensure_remote_relay(video_id: str) -> None:
    server_pattern = f"python3 -m http.server {REMOTE_RELAY_PORT} --bind 127.0.0.1"
    script = f"""
set -euo pipefail
mkdir -p {shlex.quote(REMOTE_RELAY_DIR)}
if ! pgrep -f {shlex.quote(server_pattern)} >/dev/null 2>&1; then
  nohup bash -lc 'cd {shlex.quote(REMOTE_RELAY_DIR)} && exec python3 -m http.server {REMOTE_RELAY_PORT} --bind 127.0.0.1' \\
    >/tmp/clypt_phase1_relay.log 2>&1 &
fi
for _ in $(seq 1 20); do
  if curl -sfI {shlex.quote(remote_relay_source_url(video_id))} >/dev/null 2>&1; then
    exit 0
  fi
  sleep 1
done
exit 1
"""
    run_remote_bash(script)


def download_video_locally(url: str, video_id: str) -> Path:
    LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = local_cache_path(video_id)
    if target.exists() and target.stat().st_size > 0:
        return target

    for stale in LOCAL_CACHE_DIR.glob(f"{video_id}.*"):
        stale.unlink()

    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover
        raise UserFacingError("yt-dlp is not available in the repo virtualenv") from exc

    outtmpl = str(LOCAL_CACHE_DIR / f"{video_id}.%(ext)s")
    opts = {
        "format": YTDLP_FORMAT,
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "quiet": False,
        "noprogress": False,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as exc:  # pragma: no cover - network/tool failure
        raise UserFacingError(f"yt-dlp download failed: {exc}") from exc

    if target.exists() and target.stat().st_size > 0:
        return target
    matches = sorted(LOCAL_CACHE_DIR.glob(f"{video_id}.*"))
    if not matches:
        raise UserFacingError("download completed but no relay file was created")
    matches[0].rename(target)
    return target


def upload_video_to_droplet(local_path: Path, video_id: str) -> None:
    run_remote_bash(f"mkdir -p {shlex.quote(REMOTE_RELAY_DIR)}")
    remote_target = f"{DROPLET_HOST}:{remote_relay_path(video_id)}"
    result = subprocess.run(
        ["scp", "-i", str(SSH_KEY_PATH), str(local_path), remote_target],
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise UserFacingError("failed to upload relay video to droplet")


def submit_job(source_url: str) -> str:
    script = """
python3 - <<'PY'
import json
import os
import urllib.request

req = urllib.request.Request(
    os.environ["REMOTE_API_URL"] + "/jobs",
    data=json.dumps({"source_url": os.environ["SOURCE_URL"]}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=30) as resp:
    print(resp.read().decode("utf-8"))
PY
"""
    result = run_remote_bash(script, env={"SOURCE_URL": source_url, "REMOTE_API_URL": REMOTE_API_URL})
    try:
        payload = json.loads(result.stdout.strip())
        return str(payload["job_id"])
    except Exception as exc:
        raise UserFacingError(f"could not parse job submission response: {result.stdout.strip()}") from exc


def get_job_status(job_id: str) -> str:
    script = """
python3 - <<'PY'
import json
import os
import urllib.request

with urllib.request.urlopen(os.environ["REMOTE_API_URL"] + "/jobs/" + os.environ["JOB_ID"], timeout=30) as resp:
    payload = json.loads(resp.read().decode("utf-8"))
print(payload["status"])
PY
"""
    result = run_remote_bash(script, env={"JOB_ID": job_id, "REMOTE_API_URL": REMOTE_API_URL})
    return result.stdout.strip()


def start_log_tail(job_id: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(_ssh_base() + [f"tail -n +1 -F {shlex.quote(remote_job_log_path(job_id))}"])


def prompt_for_url() -> str:
    url = input("YouTube URL: ").strip()
    if not url:
        raise UserFacingError("You must provide a YouTube URL")
    return url


def run_job_flow(url: str) -> int:
    video_id = extract_youtube_video_id(url)
    print(f"Video id: {video_id}")

    if remote_file_exists(video_id):
        print("Using cached relay video on droplet.")
    else:
        print("Droplet relay cache miss. Downloading locally...")
        local_video = download_video_locally(url, video_id)
        print(f"Uploading {local_video.name} to droplet relay...")
        upload_video_to_droplet(local_video, video_id)

    print("Ensuring droplet relay is running...")
    ensure_remote_relay(video_id)

    relay_url = remote_relay_source_url(video_id)
    print(f"Submitting job for {relay_url}")
    job_id = submit_job(relay_url)
    print(f"Started job: {job_id}")

    tail_proc = start_log_tail(job_id)

    def _cleanup(*_args: object) -> None:
        if tail_proc.poll() is None:
            tail_proc.terminate()
            try:
                tail_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tail_proc.kill()
        raise SystemExit(130)

    old_int = signal.signal(signal.SIGINT, _cleanup)
    old_term = signal.signal(signal.SIGTERM, _cleanup)
    try:
        while True:
            status = get_job_status(job_id)
            if status == "succeeded":
                if tail_proc.poll() is None:
                    tail_proc.terminate()
                    tail_proc.wait(timeout=5)
                print(f"\nJob {job_id} succeeded")
                return 0
            if status == "failed":
                if tail_proc.poll() is None:
                    tail_proc.terminate()
                    tail_proc.wait(timeout=5)
                print(f"\nJob {job_id} failed")
                return 1
            time.sleep(2)
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        if tail_proc.poll() is None:
            tail_proc.terminate()
            try:
                tail_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tail_proc.kill()


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    url = argv[0].strip() if argv else prompt_for_url()
    try:
        return run_job_flow(url)
    except (UserFacingError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
