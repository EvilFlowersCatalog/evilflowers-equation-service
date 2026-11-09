"""Deprecated entry point: ``celery --workdir src -A worker worker``.

The dev Deployment's ``args`` still copy the old image command (knowledge repo,
``roles/evilflowers_k8s``; EvilFlowersCatalog/knowledge#21). This shim keeps that command working until the role
starts the worker as ``celery -A evilflowers_equation.worker worker``; then delete it
(and its ``COPY`` in the Dockerfile).
"""

from evilflowers_equation.worker import app

__all__ = ["app"]
