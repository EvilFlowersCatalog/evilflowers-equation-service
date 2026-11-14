"""The IP-018 `equations` task, dispatched by evilflowers-analyzer-service with keyword arguments.

``**_`` keeps the worker compatible with arguments a newer analyzer adds. Two
are optional (#5): ``job``, the batch's job name (`equations:0001-0008`) when the
analyzer sends one task per layout batch, and ``attempt``, the run's attempt
returned by the catalog's `/start` (catalog#140).

The job acknowledges late, so a stopped or killed worker has the message
delivered again:

- SIGTERM (scale-in, rollout) cancels the book after `WORKER_SOFT_SHUTDOWN_TIMEOUT`
  and Redis gets it back at once (`worker.requeue_on_sigterm`, #4).
- A killed pool child is requeued at once (`task_reject_on_worker_lost`), a killed
  container after the visibility timeout. After `MAX_DELIVERIES` starts the job is
  reported `failed` instead of decoded again (#6).
- A copy reads its run first and does nothing unless the run still waits for it,
  so it costs one request, not a second pass over the book.
"""

from celery import shared_task

from evilflowers_equation.jobs import EquationsJob, Outcome


@shared_task(name="evilflowers_equation.equations", acks_late=True, bind=True)
def equations(
    self,
    run_id: str,
    source: str,
    output_dir: str,
    regions: list[dict],
    job: str | None = None,
    attempt: int | None = None,
    **_,
) -> Outcome:
    """Formula regions found by the analyzer's layout pass: WebP crop, LaTeX and sympy's form."""
    return EquationsJob(run_id, source, output_dir, regions, job=job, attempt=attempt, task_id=self.request.id).run()
