"""A worker for the shutdown tests in `test_worker`: the service's config and two slow tasks.

Each task, once running, counts its delivery the way `EquationsJob` does and records
its start and process id in Redis, so a test signals a task that is really running,
not a message still waiting to be picked up.
"""

import os
import time

import redis

from evilflowers_equation.deliveries import DeliveryCounter
from evilflowers_equation.settings import Settings
from evilflowers_equation.worker import create_app  # importing it connects the shutdown signals

BROKER = os.environ["SIGTERM_TEST_BROKER"]
QUEUE = "evilflowers_equation_test"

settings = Settings(_env_file=None, redis_url=BROKER, EQUATION_QUEUE=QUEUE, worker_soft_shutdown_timeout=1)
app = create_app(settings)


def _started(name: str, task_id: str) -> int:
    deliveries = DeliveryCounter(settings).record(task_id)
    client = redis.Redis.from_url(BROKER)
    client.incr(f"{name}:started")
    client.set(f"{name}:pid", os.getpid())
    return deliveries


@app.task(name="evilflowers_equation.sleepy", acks_late=True, bind=True)
def sleepy(self) -> None:
    _started("sleepy", self.request.id)
    time.sleep(120)


@app.task(name="evilflowers_equation.doomed", acks_late=True, bind=True)
def doomed(self) -> None:
    """Killed on every delivery; stops on the one past `MAX_DELIVERIES`, as the job does."""
    deliveries = _started("doomed", self.request.id)
    if deliveries > settings.max_deliveries:
        redis.Redis.from_url(BROKER).set("doomed:capped", deliveries)
        return
    time.sleep(120)
