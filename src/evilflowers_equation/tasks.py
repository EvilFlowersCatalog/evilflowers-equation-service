"""The IP-018 `equations` task, dispatched by evilflowers-analyzer-service with keyword arguments.

``**_`` keeps the worker compatible with arguments a newer analyzer adds.

The job acknowledges late: a worker that dies mid-book has the message
redelivered, and the catalog answers 409 (`RunNotRunning`) to a run that
already finished, so a second delivery is harmless.
"""

from celery import shared_task

from evilflowers_equation.jobs import EquationsJob, Outcome


@shared_task(name="evilflowers_equation.equations", acks_late=True)
def equations(run_id: str, source: str, output_dir: str, regions: list[dict], **_) -> Outcome:
    """Formula regions found by the analyzer's layout pass: WebP crop, LaTeX and sympy's form."""
    return EquationsJob(run_id, source, output_dir, regions).run()
