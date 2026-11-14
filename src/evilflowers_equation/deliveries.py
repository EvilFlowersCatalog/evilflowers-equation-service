"""How many times a task id has started, kept in Redis (#6).

A book that kills the worker every time (an OOM in the model) is put back on the
queue every time: by `task_reject_on_worker_lost` when the pool child dies, by the
visibility timeout when the whole container does. Neither leaves the task a
chance to acknowledge, and Celery's `redelivered` flag stays False on both paths,
so the count lives outside the process: `INCR equation:deliveries:<task_id>` when
the task starts. `EquationsJob` reports the job `failed` once the count passes
`MAX_DELIVERIES`.

A SIGTERM requeue (scale-in, rollout; see `worker.requeue_on_sigterm`) is not a
kill: the main process takes the running tasks' deliveries back (`forget`) before
the cold shutdown cancels them (`worker.requeue_cold_shutdown`), so stopping a pod
never uses up a book's tries.
"""

import logging
from collections.abc import Iterable

import redis

from evilflowers_equation.settings import Settings, get_settings

logger = logging.getLogger(__name__)

PREFIX = "equation:deliveries:"


class DeliveryCounter:
    def __init__(self, settings: Settings | None = None, client: redis.Redis | None = None):
        self.settings = settings or get_settings()
        self._client = client

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            # Connects on first use; short timeouts, because the counter must never hold a book up.
            self._client = redis.Redis.from_url(self.settings.redis_url, socket_timeout=5, socket_connect_timeout=5)
        return self._client

    def record(self, task_id: str) -> int:
        """Count one more start of `task_id`; the number of starts so far, this one included."""
        key = PREFIX + task_id
        with self.client.pipeline() as pipe:
            pipe.incr(key)
            pipe.expire(key, self.settings.delivery_ttl)
            count, _ = pipe.execute()
        return int(count)

    def forget(self, task_ids: Iterable[str]) -> None:
        """Take back one start of each task: it is going back to the queue on purpose, not killed."""
        for task_id in task_ids:
            key = PREFIX + task_id
            if self.client.decr(key) <= 0:
                self.client.delete(key)
