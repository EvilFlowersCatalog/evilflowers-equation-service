"""The Celery application: ``celery -A evilflowers_equation.worker worker``.

The task name and queue are a contract with the analyzer and must not change:

==================================  ==========================================  ===============
Task                                Queue                                       Sent by
==================================  ==========================================  ===============
``evilflowers_equation.equations``  ``EQUATION_QUEUE`` (``evilflowers_equation``)  analyzer (kwargs)
==================================  ==========================================  ===============

Nobody waits for a task's result (the job reports to the catalog), so there is
no result backend.
"""

import logging

from celery import Celery, signals
from celery.apps.worker import install_worker_term_hard_handler, on_cold_shutdown
from celery.concurrency.solo import TaskPool as SoloPool

from evilflowers_equation.deliveries import DeliveryCounter
from evilflowers_equation.observability.signals import connect
from evilflowers_equation.settings import Settings, get_settings

TASK_PREFIX = "evilflowers_equation"

logger = logging.getLogger(__name__)


class CeleryConfig:
    """Shared with the sibling extraction services (analyzer, text, image) where it can be."""

    def __init__(self, settings: Settings):
        self.broker_url = settings.redis_url
        self.broker_connection_retry_on_startup = True
        # The job acknowledges late; Redis must not redeliver a book still being read.
        self.broker_transport_options = {"visibility_timeout": settings.broker_visibility_timeout}

        self.task_serializer = "json"
        self.result_serializer = "json"
        self.accept_content = ["json"]
        self.timezone = "Europe/Bratislava"
        self.enable_utc = True

        self.task_default_queue = settings.queue
        self.task_routes = {f"{TASK_PREFIX}.*": {"queue": settings.queue}}
        self.worker_prefetch_multiplier = 1
        # A worker killed mid-book (OOM in the model) gets the book redelivered; `MAX_DELIVERIES` caps
        # how often (`deliveries`, #6).
        self.task_reject_on_worker_lost = True
        # SIGTERM (a scale-in, a rollout) puts the running book back on the queue after this many
        # seconds, instead of after the visibility timeout. See `requeue_on_sigterm` (#4).
        self.worker_soft_shutdown_timeout = settings.worker_soft_shutdown_timeout
        self.worker_enable_soft_shutdown_on_idle = settings.worker_enable_soft_shutdown_on_idle
        # Monitoring (Flower, `celery events`).
        self.worker_send_task_events = True
        self.task_send_sent_event = True


def requeue_on_sigterm(sender, **_) -> None:
    """Make SIGTERM a cold shutdown with a soft-shutdown grace (#4).

    Celery's SIGTERM is a warm shutdown: it waits for the running book, Kubernetes kills the pod
    at its grace period, and the unacknowledged message returns only after the visibility timeout.
    A cold shutdown waits `worker_soft_shutdown_timeout`, then cancels the book and Redis gets the
    message back at once. This is what `REMAP_SIGTERM=SIGQUIT` does, chosen here per pool:

    - The solo pool cannot cancel a task. Its cancel raises inside the job, the job fails, and
      `acks_late` then acknowledges the message, which is lost. So under solo SIGTERM stays warm.
    - Prefork (the image's `--concurrency=1`) cancels the child, and the message goes back.

    Before the cold shutdown cancels the running book, its delivery is taken back (`requeue_cold_shutdown`).
    """
    if isinstance(sender.pool, SoloPool):
        logger.warning("--pool=solo: SIGTERM does not requeue a running book; it returns after the visibility timeout")
        return
    install_worker_term_hard_handler(sender.controller, sig="SIGTERM", callback=requeue_cold_shutdown)


def requeue_cold_shutdown(worker) -> None:
    """Celery's cold shutdown, after taking back the running books' deliveries (#6).

    They go back to the queue on purpose, not because the worker died, so a scale-in must not use up
    one of a book's `MAX_DELIVERIES`. Runs in the main process, which tracks the running requests; the
    cold shutdown cancels them (and empties that list) once the soft-shutdown timeout is over.
    """
    from celery.worker import state

    task_ids = [request.id for request in list(state.active_requests)]
    if task_ids:
        try:
            DeliveryCounter().forget(task_ids)
        except Exception:
            logger.warning("Could not take back the deliveries of %s", task_ids, exc_info=True)
    on_cold_shutdown(worker)


def create_app(settings: Settings | None = None) -> Celery:
    settings = settings or get_settings()
    app = Celery("evilflowers_equation", include=["evilflowers_equation.tasks"])
    app.config_from_object(CeleryConfig(settings))
    return app


# Traces, metrics, JSON logs and profiles; off unless the pod points them at a collector.
connect()
signals.worker_ready.connect(requeue_on_sigterm, weak=False)
app = create_app()
