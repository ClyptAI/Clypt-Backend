FROM vllm/vllm-openai:v0.14.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    VIBEVOICE_BACKEND=vllm \
    VIBEVOICE_VLLM_BASE_URL=http://127.0.0.1:8000 \
    VIBEVOICE_VLLM_HEALTHCHECK_PATH=/health \
    CLYPT_L4_VIBEVOICE_REPO_DIR=/app \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

ARG VIBEVOICE_REPO_URL=https://github.com/microsoft/VibeVoice.git
ARG VIBEVOICE_REPO_REF=main
ARG VIBEVOICE_ASR_MODEL=microsoft/VibeVoice-ASR

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Install hf_transfer for 3-5x faster HuggingFace downloads. Applied both at
# build time (model bake below) and at runtime for any residual HF fetches
# (tokenizer configs, etc.) triggered by VibeVoice's start_server.py.
RUN pip install --no-cache-dir hf_transfer

# VibeVoice's vllm_plugin/scripts/start_server.py hardcodes `pip install -e /app[vllm]`,
# so the VibeVoice source tree MUST live directly at /app (not a subdirectory). We
# clone into a temp dir and move its contents into /app to preserve the expectation
# while still using /app as our WORKDIR. VibeVoice's pyproject.toml pins
# `packages.find.include = ["vibevoice*", "vllm_plugin*"]`, so our later-copied
# /app/backend is not swept into the editable install.
RUN git clone --depth 1 --branch "${VIBEVOICE_REPO_REF}" "${VIBEVOICE_REPO_URL}" /tmp/vibevoice-src \
    && cp -a /tmp/vibevoice-src/. /app/ \
    && rm -rf /tmp/vibevoice-src

COPY requirements-local.txt /app/requirements-local.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements-local.txt

# Bake the ~12 GB microsoft/VibeVoice-ASR snapshot into the image so cold-start
# on Cloud Run does not depend on HuggingFace CDN throughput. Subsequent
# snapshot_download calls issued by VibeVoice's start_server.py at runtime
# short-circuit because every file SHA already matches. This is the dominant
# cold-start win: pulls model weights from in-region Artifact Registry (~1 GB/s)
# instead of HF over a single HTTPS connection.
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='${VIBEVOICE_ASR_MODEL}')"

COPY backend /app/backend

# Force the VibeVoice audio encoder to stay in bfloat16 instead of upcasting
# to float32. Upstream model.py defaults the audio encoder dtype to fp32 "for
# numerical precision" even though the checkpoint on HF ships every sub-module
# as bfloat16. On a 24 GB L4 the fp32 upcast bloats the model from ~10 GB to
# ~18 GB and starves vLLM's profile_run (see docs/ERROR_LOG.md).
#
# Placed LATE so the expensive ~12 GB model-bake layer above stays cached
# across rebuilds. VibeVoice is installed editable at runtime via
# start_server.py (`pip install -e /app[vllm]`), so the patch still takes
# effect. The grep guard fails the build if upstream reshapes this line.
RUN sed -i 's|self._audio_encoder_dtype = torch.float32|self._audio_encoder_dtype = torch.bfloat16|' \
        /app/vllm_plugin/model.py \
    && grep -q 'self._audio_encoder_dtype = torch.bfloat16' /app/vllm_plugin/model.py

EXPOSE 8080

# The vllm/vllm-openai base image sets ENTRYPOINT=["vllm"], which would prepend
# "vllm" to our CMD and break CLI parsing. Clear it so CMD runs unmodified.
# The base image ships only /usr/bin/python3 (no bare "python" on PATH), so CMD
# must invoke python3 directly; using "python" causes "Application exec likely
# failed" with no stderr (exec fails before any Python code runs).
ENTRYPOINT []
CMD ["python3", "-m", "backend.runtime.run_l4_combined_service", "--host", "0.0.0.0", "--port", "8080"]
