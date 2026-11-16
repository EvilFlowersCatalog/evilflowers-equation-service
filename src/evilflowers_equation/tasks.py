"""The tasks: IP-018 `equations` (below) and the books digitalizer's `standalone.formulas` (``standalone``, #8).

`equations` is dispatched by evilflowers-analyzer-service with keyword arguments.

The message is a ``RegionsJob`` (``evilflowers_worker.contract``): ``run_id``, ``source``, ``output_dir``,
``regions``, and two optional fields (#5): ``job``, the batch's job name (`equations:0001-0008`) when the
analyzer sends one task per layout batch, and ``attempt``, the run's attempt returned by the catalog's
`/start` (catalog#140). Fields a newer analyzer adds are ignored.

``runtime.run_job`` validates the message (an invalid one is reported `failed` and dropped, never
retried), counts the delivery and runs the job. The job acknowledges late, so a stopped or killed
worker has the message delivered again:

- SIGTERM (scale-in, rollout) cancels the book after `WORKER_SOFT_SHUTDOWN_TIMEOUT` and Redis gets it
  back at once; that requeue does not count as a delivery (``evilflowers_worker.shutdown``).
- A killed pool child is requeued at once, a killed container after the visibility timeout. Past
  `MAX_DELIVERIES` the job is reported `failed` instead of decoded again (#6).
- A copy reads its run first and does nothing unless the run still waits for it, so it costs one
  request, not a second pass over the book.
"""

from typing import Any

from celery import shared_task
from evilflowers_worker import runtime
from evilflowers_worker.contract import RegionsJob

from evilflowers_equation import (
    standalone,
    worker,  # noqa: F401 - create_app must run in a spawned pool child too
)
from evilflowers_equation.jobs import EquationsJob


@shared_task(name="evilflowers_equation.equations", bind=True, **runtime.JOB_TASK_OPTIONS)
def equations(self, **kwargs: Any) -> str:
    """Formula regions found by the analyzer's layout pass: WebP crop, LaTeX and sympy's form."""
    return runtime.run_job(self, EquationsJob, RegionsJob, kwargs)


@shared_task(name="evilflowers_equation.standalone.formulas", bind=True, **runtime.JOB_TASK_OPTIONS)
def formulas(self, **kwargs: Any) -> dict:
    """The books digitalizer's formulas of leaf images (#8): `formulas.json` and an outcome file, no catalog call."""
    return runtime.run_task(self, standalone.run, standalone.FormulasJob, kwargs, on_capped=standalone.give_up)
