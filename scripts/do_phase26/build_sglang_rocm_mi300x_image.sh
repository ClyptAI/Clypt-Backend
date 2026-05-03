#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SG_DOCKER_BASE_IMAGE="${SG_DOCKER_BASE_IMAGE:-lmsysorg/sglang:v0.5.10-rocm720-mi30x}"
SG_DOCKER_IMAGE="${SG_DOCKER_IMAGE:-clypt/sglang:v0.5.10-rocm720-mi30x-clypt1}"
DOCKERFILE_DIR="$SCRIPT_DIR/docker/sglang-rocm-mi300x"

docker build \
  --build-arg "BASE_IMAGE=$SG_DOCKER_BASE_IMAGE" \
  -t "$SG_DOCKER_IMAGE" \
  "$DOCKERFILE_DIR"

declare -a docker_run_args=(--rm)
if [[ -e /dev/kfd && -d /dev/dri ]]; then
  docker_run_args+=(
    --device=/dev/kfd
    --device=/dev/dri
    --group-add video
    --ipc=host
    --cap-add SYS_PTRACE
    --security-opt seccomp=unconfined
  )
fi

docker run "${docker_run_args[@]}" "$SG_DOCKER_IMAGE" python - <<'PY'
import re
import sglang
from sglang.srt.layers.quantization import QUANTIZATION_METHODS

version = getattr(sglang, "__version__", "")
match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
if not match:
    raise SystemExit(f"cannot parse sglang.__version__={version!r}")
parsed = tuple(int(part) for part in match.groups())
if parsed < (0, 5, 10):
    raise SystemExit(f"SGLang {version} is too old; need >=0.5.10")
if "quark" in QUANTIZATION_METHODS or "quark_int4fp8_moe" in QUANTIZATION_METHODS:
    raise SystemExit("Quark quantization is unexpectedly enabled in the default Clypt ROCm image")
print(f"[build-sglang-rocm-mi300x] image accepted: {version}")
PY
