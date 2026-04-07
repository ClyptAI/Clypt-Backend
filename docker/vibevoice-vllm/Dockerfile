# VibeVoice-ASR vLLM service image.
#
# Based on the official vLLM v0.14.1 image (the tested VibeVoice-compatible
# version). The VibeVoice plugin repo is mounted at /app at runtime — no
# plugin code baked into the image so updates don't need a rebuild.
#
# Build:
#   docker build -t clypt-vllm-vibevoice:latest docker/vibevoice-vllm/
#
# Runtime: systemd mounts /opt/clypt-phase1/vibevoice-repo -> /app
# and runs: python3 /app/vllm_plugin/scripts/start_server.py

FROM vllm/vllm-openai:v0.14.1
