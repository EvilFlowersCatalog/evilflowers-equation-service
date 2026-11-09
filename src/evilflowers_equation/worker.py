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

from celery import Celery

from evilflowers_equation.observability.signals import connect
from evilflowers_equation.settings import Settings, get_settings

TASK_PREFIX = "evilflowers_equation"


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
        # A worker killed mid-book (OOM in the model) gets the book redelivered.
        self.task_reject_on_worker_lost = True
        # Monitoring (Flower, `celery events`).
        self.worker_send_task_events = True
        self.task_send_sent_event = True


def create_app(settings: Settings | None = None) -> Celery:
    settings = settings or get_settings()
    app = Celery("evilflowers_equation", include=["evilflowers_equation.tasks"])
    app.config_from_object(CeleryConfig(settings))
    return app


# Traces, metrics, JSON logs and profiles; off unless the pod points them at a collector.
connect()
app = create_app()
