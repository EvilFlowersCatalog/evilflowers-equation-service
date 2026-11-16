"""The Celery application: ``celery -A evilflowers_equation.worker worker``.

The task names and queues are a contract with their callers and must not change:

============================================  ===================================================  ===================
Task                                          Queue                                                Sent by
============================================  ===================================================  ===================
``evilflowers_equation.equations``            ``EQUATION_QUEUE`` (``evilflowers_equation``)        analyzer (kwargs)
``evilflowers_equation.standalone.formulas``  ``evilflowers_equation.standalone`` (bulk),          books digitalizer
                                              ``evilflowers_equation.standalone.interactive``
============================================  ===================================================  ===================

All three queues are declared; a Deployment picks what it consumes with ``--queues``. The catalog one consumes
``evilflowers_equation`` on db 7, the standalone ones the other two on db 8 (``REDIS_URL``).

Everything else is the kit's (``evilflowers_worker.runtime.create_app``): late acknowledgement, prefetch 1,
the db-wide visibility timeout, the scale-in protocol of ``evilflowers_worker.shutdown`` (SIGTERM stays a
warm shutdown; the job is cancelled after ``WORKER_SOFT_SHUTDOWN_TIMEOUT`` and, if that cancel is denied,
requeued by the worker itself after ``WORKER_CANCEL_TIMEOUT``), and telemetry with JSON logs.

Nobody waits for a task's result (the job reports to the catalog), so there is no result backend.
"""

from evilflowers_worker import runtime

from evilflowers_equation.settings import get_settings

settings = get_settings()
app = runtime.create_app(
    "equation",
    settings,
    include=["evilflowers_equation.tasks"],
    queues=[settings.queue, settings.standalone_queue, settings.interactive_queue],
    # First match wins: the standalone task must not fall to the catalog queue's `evilflowers_equation.*`.
    routes={"evilflowers_equation.standalone.*": settings.standalone_queue, "evilflowers_equation.*": settings.queue},
)
