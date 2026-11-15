"""The Celery application: ``celery -A evilflowers_equation.worker worker``.

The task name and queue are a contract with the analyzer and must not change:

==================================  ==========================================  ===============
Task                                Queue                                       Sent by
==================================  ==========================================  ===============
``evilflowers_equation.equations``  ``EQUATION_QUEUE`` (``evilflowers_equation``)  analyzer (kwargs)
==================================  ==========================================  ===============

Everything else is the kit's (``evilflowers_worker.runtime.create_app``): late acknowledgement, prefetch 1,
the db-wide visibility timeout, the scale-in protocol of ``evilflowers_worker.shutdown`` (SIGTERM stays a
warm shutdown; the job is cancelled after ``WORKER_SOFT_SHUTDOWN_TIMEOUT`` and, if that cancel is denied,
requeued by the worker itself after ``WORKER_CANCEL_TIMEOUT``), and telemetry with JSON logs.

Nobody waits for a task's result (the job reports to the catalog), so there is no result backend.
"""

from evilflowers_worker import runtime

from evilflowers_equation.settings import get_settings

settings = get_settings()
app = runtime.create_app("equation", settings, include=["evilflowers_equation.tasks"], queues=[settings.queue])
