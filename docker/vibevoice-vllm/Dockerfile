# VibeVoice-ASR vLLM service image.
#
# Based on the official vLLM v0.14.1 image (the tested VibeVoice-compatible
# version). Unlike the old "lazy" layout, everything VibeVoice needs at
# runtime is BAKED INTO THIS IMAGE:
#
#   - ffmpeg + libsndfile1 (audio decode)
#   - git (needed at build time only)
#   - the vibevoice python package + [vllm] extras (librosa, scipy, soundfile,
#     aiortc, pandas, scikit-learn, gradio, av, etc.)
#   - pre-generated tokenizer files for microsoft/VibeVoice-ASR
#
# This change was made because the upstream start_server.py wrapper runs
# `apt-get install` and `pip install -e /app[vllm]` on every boot, which
# cost ~5 minutes per container restart under our systemd-managed unit
# (which docker-rm's on every restart). See docs/ERROR_LOG.md 2026-04-17.
#
# The /app mount still works at runtime for access to the vllm_plugin
# scripts (e.g. generate_tokenizer_files) if we ever need them, but we
# launch vllm directly at runtime instead of going through start_server.py.
#
# Build:
#   docker build -t clypt-vllm-vibevoice:latest docker/vibevoice-vllm/

FROM vllm/vllm-openai:v0.14.1

ARG VIBEVOICE_REPO_URL=https://github.com/microsoft/VibeVoice.git
ARG VIBEVOICE_REF=main

# System deps that the upstream wrapper's install_system_deps() would otherwise
# apt-install on every container start.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ffmpeg \
      libsndfile1 \
      git \
 && rm -rf /var/lib/apt/lists/*

# Clone vibevoice + install the [vllm] extras so all downstream python
# dependencies (librosa, scipy, scikit-learn, aiortc, gradio, av, etc.) are
# resident in the image's site-packages and container restarts don't redo
# ~5 minutes of pip work on every boot.
RUN git clone --depth 1 --branch "${VIBEVOICE_REF}" \
      "${VIBEVOICE_REPO_URL}" /opt/vibevoice-baked \
 && pip install --no-cache-dir -e "/opt/vibevoice-baked[vllm]"

# The upstream wrapper uses this CLI entrypoint to pre-generate
# tokenizer files against the downloaded HF snapshot. We call it once at
# container start against the model cache dir so it's idempotent.
# (Model weights themselves are NOT baked in — they live in the
# /root/.cache/huggingface mount managed by systemd.)

WORKDIR /opt/vibevoice-baked
