"""EvilFlowers equation service: the Celery worker that turns formula regions of a PDF into crops and LaTeX."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("evilflowers-equation-service")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"
