"""Traces, metrics, JSON logs and CPU profiles of the equation worker (the cluster's telemetry contract, IP-019).

Off unless the pod points it at a collector; every call here is safe either way.
"""

from evilflowers_equation.observability import metrics
from evilflowers_equation.observability.context import bind
from evilflowers_equation.observability.logs import JsonFormatter
from evilflowers_equation.observability.setup import endpoint, is_enabled, setup, shutdown
from evilflowers_equation.observability.tracing import annotate, fail, span

__all__ = [
    "JsonFormatter",
    "annotate",
    "bind",
    "endpoint",
    "fail",
    "is_enabled",
    "metrics",
    "setup",
    "shutdown",
    "span",
]
