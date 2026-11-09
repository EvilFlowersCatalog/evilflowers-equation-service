# syntax=docker/dockerfile:1.7

# -- build: the virtual environment, from uv.lock -------------------------------------------------------------------
FROM python:3.12-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Dependencies first (cached while only the code changes).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --all-extras --no-install-project

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --all-extras --no-editable

# -- runtime -----------------------------------------------------------------------------------------------------------
FROM python:3.12-slim

# OpenCV (rapid-latex-ocr's image preprocessing) needs libGL and GLib.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /app/.venv /app/.venv
# Deprecated entry point (`celery --workdir src -A worker`) kept for the dev Deployment's args; see src/worker.py.
COPY src/worker.py ./src/worker.py

ARG IMAGE_TAG=dev
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    IMAGE_TAG=${IMAGE_TAG} \
    MODELS_PATH=/models/latex-ocr

# Model weights live in a volume, not in the image (IP-018); LaTeX-OCR downloads them on first use.
VOLUME ["/models"]

# Prefork with one child: the model is loaded once in the child, and Flower can inspect the worker during a task.
CMD ["celery", "-A", "evilflowers_equation.worker", "worker", "--loglevel=INFO", "-E", "--concurrency=1", \
     "--queues=evilflowers_equation"]
