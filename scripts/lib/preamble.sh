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
