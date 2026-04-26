# Copyright (c) Meta Platforms, Inc. and affiliates.
# Carbon-Aware AI Workload Scheduler — OpenEnv Docker Space

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# git is needed for VCS deps; curl already in base image
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Copy source
COPY . /app/env
WORKDIR /app/env

# Ensure uv is available
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install dependencies only — no project install.
# [tool.uv] package = false in pyproject.toml ensures uv does not try
# to build this repo as a Python package (it is a runnable app).
# No uv.lock present so uv resolves fresh from pyproject.toml.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy venv and source from builder
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env      /app/env

# Activate the venv
ENV PATH="/app/.venv/bin:$PATH"

# Make sure server package is importable
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# HuggingFace Spaces: must listen on 7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 7860"]