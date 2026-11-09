.PHONY: install install-all lint format test check worker image

# Core + dev tools: enough for the tests.
install:
	uv sync

# Everything the image has (adds LaTeX-OCR, the OpenTelemetry SDK and the Pyroscope SDK).
install-all:
	uv sync --all-extras

lint:
	uv run ruff check
	uv run ruff format --check

format:
	uv run ruff check --fix
	uv run ruff format

test:
	uv run pytest

check: lint test

# A local worker (needs a reachable REDIS_URL; LATEX_BACKEND=onnx needs `make install-all`).
worker:
	uv run celery -A evilflowers_equation.worker worker --loglevel=INFO -E --concurrency=1 --queues=evilflowers_equation

image:
	docker build --build-arg IMAGE_TAG=local@$$(git rev-parse --short HEAD) -t evilflowers-equation-service:local .
