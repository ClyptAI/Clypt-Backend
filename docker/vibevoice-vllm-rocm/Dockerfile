# VibeVoice-ASR vLLM service image for the Phase1 AMD MI300X host.
#
# The ROCm vLLM base image must be pinned to the exact tag accepted during the
# MI300X compatibility spike. Deploy intentionally fails unless this build arg
# is provided.

ARG VLLM_ROCM_BASE_IMAGE
FROM ${VLLM_ROCM_BASE_IMAGE}

ARG VIBEVOICE_REPO_URL=https://github.com/microsoft/VibeVoice.git
ARG VIBEVOICE_REF=main

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ffmpeg \
      libsndfile1 \
      git \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${VIBEVOICE_REF}" \
      "${VIBEVOICE_REPO_URL}" /opt/vibevoice-baked \
 && pip install --no-cache-dir -e "/opt/vibevoice-baked[vllm]"

# Tokenizer files are generated at container start against the mounted HF model
# snapshot so weights stay host-cache managed and the operation is idempotent.

WORKDIR /opt/vibevoice-baked
