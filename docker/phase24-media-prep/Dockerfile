FROM vllm/vllm-openai:v0.14.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    VIBEVOICE_BACKEND=vllm \
    VIBEVOICE_VLLM_BASE_URL=http://127.0.0.1:8000 \
    VIBEVOICE_VLLM_HEALTHCHECK_PATH=/health \
    CLYPT_L4_VIBEVOICE_REPO_DIR=/app/vibevoice-repo

WORKDIR /app

ARG VIBEVOICE_REPO_URL=https://github.com/microsoft/VibeVoice.git
ARG VIBEVOICE_REPO_REF=main

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-local.txt /app/requirements-local.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements-local.txt

RUN git clone --depth 1 --branch "${VIBEVOICE_REPO_REF}" "${VIBEVOICE_REPO_URL}" /app/vibevoice-repo

COPY backend /app/backend

EXPOSE 8080

# The vllm/vllm-openai base image sets ENTRYPOINT=["vllm"], which would prepend
# "vllm" to our CMD and break CLI parsing. Clear it so CMD runs unmodified.
# The base image ships only /usr/bin/python3 (no bare "python" on PATH), so CMD
# must invoke python3 directly; using "python" causes "Application exec likely
# failed" with no stderr (exec fails before any Python code runs).
ENTRYPOINT []
CMD ["python3", "-m", "backend.runtime.run_l4_combined_service", "--host", "0.0.0.0", "--port", "8080"]
