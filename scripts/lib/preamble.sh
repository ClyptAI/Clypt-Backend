# shellcheck shell=bash
# Shared bash preamble helpers for Clypt deploy scripts.
#
# This library is sourced (not executed) by the active host deploy scripts
# under scripts/do_phase1/ and scripts/do_phase26/. It factors out idioms
# that were duplicated across multiple deploy scripts.
#
# Usage:
#     SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#     # shellcheck source=../lib/preamble.sh
#     source "$SCRIPT_DIR/../lib/preamble.sh"
#     require_root
#     require_dir "$REPO_DIR"
#     require_file "$ENV_FILE"
#     fail_if_repo_local_env_present "$REPO_DIR" "$ENV_FILE"
#     load_env_file "$ENV_FILE"
#
# All helpers exit 1 on failure (same behavior as the inline checks they
# replace). A leading "[preamble]" tag is used for error output so callers
# can tell the error came from the shared library.

# require_root: exit 1 if the effective UID is not 0.
require_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "[preamble] ERROR: must be run as root." >&2
    exit 1
  fi
}

# require_dir PATH: exit 1 if PATH is not an existing directory.
require_dir() {
  local path="${1:-}"
  if [[ -z "$path" ]]; then
    echo "[preamble] ERROR: require_dir called without a path argument." >&2
    exit 1
  fi
  if [[ ! -d "$path" ]]; then
    echo "[preamble] ERROR: directory not found: $path" >&2
    exit 1
  fi
}

# require_file PATH: exit 1 if PATH is not an existing regular file.
require_file() {
  local path="${1:-}"
  if [[ -z "$path" ]]; then
    echo "[preamble] ERROR: require_file called without a path argument." >&2
    exit 1
  fi
  if [[ ! -f "$path" ]]; then
    echo "[preamble] ERROR: file not found: $path" >&2
    exit 1
  fi
}

# fail_if_repo_local_env_present REPO_DIR ENV_FILE: exit 1 if a copied
# repo-root .env or .env.local is present. Host deploys must only consume
# the host-scoped env file under /etc/clypt-* plus the committed runtime env
# template under /etc/clypt/.
fail_if_repo_local_env_present() {
  local repo_dir="${1:-}"
  local env_file="${2:-<host-env-file>}"
  if [[ -z "$repo_dir" ]]; then
    echo "[preamble] ERROR: fail_if_repo_local_env_present called without a repo path." >&2
    exit 1
  fi
  local offenders=()
  local candidate=""
  for candidate in "$repo_dir/.env" "$repo_dir/.env.local"; do
    if [[ -f "$candidate" ]]; then
      offenders+=("$candidate")
    fi
  done
  if [[ "${#offenders[@]}" -gt 0 ]]; then
    echo "[preamble] ERROR: repo-local env override files are present on the host:" >&2
    printf '  - %s\n' "${offenders[@]}" >&2
    echo "[preamble] ERROR: remove or rename them before deploying; host deploys must use only $env_file plus /etc/clypt/* runtime env files." >&2
    exit 1
  fi
}

# load_env_file PATH: safely source an env file into the current shell
# with automatic export (set -a), then restore the prior flag state. The
# file must exist (caller is expected to have validated it) — we still
# re-check here so mis-orderings fail fast with a clear message.
load_env_file() {
  local path="${1:-}"
  if [[ -z "$path" ]]; then
    echo "[preamble] ERROR: load_env_file called without a path argument." >&2
    exit 1
  fi
  if [[ ! -f "$path" ]]; then
    echo "[preamble] ERROR: env file not found: $path" >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "$path"
  set +a
}

# require_google_service_account_key PATH: validate that PATH points to a
# signing-capable service-account JSON key. Token-only ADC files
# (`authorized_user`) can pass some auth checks but will fail later when the
# runtime needs V4 signed URLs.
require_google_service_account_key() {
  local path="${1:-}"
  if [[ -z "$path" ]]; then
    echo "[preamble] ERROR: require_google_service_account_key called without a path argument." >&2
    exit 1
  fi
  require_file "$path"
  python3 - "$path" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)

if not isinstance(data, dict):
    raise SystemExit(f"[preamble] ERROR: credential file is not a JSON object: {path}")
if data.get("type") != "service_account":
    raise SystemExit(
        "[preamble] ERROR: credential file must be a service-account key "
        f"(type=service_account), not {data.get('type')!r}: {path}"
    )
missing = [
    field for field in ("client_email", "private_key", "private_key_id")
    if not data.get(field)
]
if missing:
    raise SystemExit(
        "[preamble] ERROR: service-account key is missing required fields "
        f"{missing}: {path}"
    )
PY
}
