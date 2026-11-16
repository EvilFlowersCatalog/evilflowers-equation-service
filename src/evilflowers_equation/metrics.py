"""Equation's own metrics, on top of the standard set every worker has (``evilflowers_worker.observability.metrics``).

``{service}`` is ``equation``, so the names are the ones the dashboards already use
(``equation_recognition_duration_seconds_bucket`` …). Attributes stay low-cardinality: never a run id.
"""

from evilflowers_worker.observability.metrics import Instrument

# Seconds. Covers a 10 ms crop up to a 30 min book.
DURATION_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800)
COUNT_BUCKETS = (0, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)

RECOGNITION_DURATION = Instrument(
    "histogram",
    "{service}.recognition.duration",
    "s",
    "Time to transcribe one equation crop to LaTeX",
    DURATION_BUCKETS,
)
REGIONS = Instrument(
    "counter", "{service}.regions", "{region}", "Equation regions, by what became of them (kept, skipped, failed)"
)
#: ``equation_formulas_total{outcome}``: doubtful, clean or unread (no LaTeX), judges on or off (#9). Its judge calls
#: are the kit's ``equation_judge_total{question="formula_kind", outcome, band}``.
FORMULAS = Instrument(
    "counter", "{service}.formulas", "{formula}", "Kept equation regions, by the cheap doubt check (doubtful, clean)"
)
DOCUMENT_REGIONS = Instrument(
    "histogram", "{service}.document.regions", "{region}", "Equation regions kept per document", COUNT_BUCKETS
)


def record_recognition(model: str, outcome: str, seconds: float) -> None:
    """``outcome``: recognized, empty (the model answered nothing) or failed."""
    RECOGNITION_DURATION.record(seconds, {"model": model, "outcome": outcome})
