"""The task contract with the analyzer: name, queue and argument shape; and shutdown."""

import inspect
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from celery import Celery
from celery.concurrency.solo import TaskPool as SoloPool

from evilflowers_equation import tasks
from evilflowers_equation import worker as worker_module
from evilflowers_equation.settings import Settings
from evilflowers_equation.worker import CeleryConfig, app, requeue_on_sigterm

TASKS = {"evilflowers_equation.equations": "evilflowers_equation"}


def test_every_task_is_registered_and_routed():
    app.loader.import_default_modules()
    for name, queue in TASKS.items():
        assert name in app.tasks, name
        assert app.amqp.router.route({}, name)["queue"].name == queue


def test_tasks_accept_arguments_a_newer_analyzer_adds():
    app.loader.import_default_modules()
    for name in TASKS:
        parameters = inspect.signature(app.tasks[name].run).parameters
        assert any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
        # #5: the batch's job name and the run's attempt, both optional.
        assert parameters["job"].default is None and parameters["attempt"].default is None
        assert app.tasks[name].acks_late


def test_the_task_passes_job_attempt_and_its_id(monkeypatch):
    seen = {}

    class Job:
        def __init__(self, *args, **kwargs):
            seen.update(args=args, kwargs=kwargs)

        def run(self):
            return "done"

    monkeypatch.setattr(tasks, "EquationsJob", Job)
    kwargs = {"run_id": "r", "source": "a.pdf", "output_dir": "o", "regions": [], "newer": 1}
    assert tasks.equations.apply(kwargs={**kwargs, "job": "equations:0001-0008", "attempt": 3}, task_id="t-1").get()
    assert seen["kwargs"] == {"job": "equations:0001-0008", "attempt": 3, "task_id": "t-1"}
    tasks.equations.apply(kwargs=kwargs, task_id="t-2").get()
    assert seen["kwargs"] == {"job": None, "attempt": None, "task_id": "t-2"}


def test_redelivery_settings():
    assert app.conf.worker_prefetch_multiplier == 1
    assert app.conf.task_reject_on_worker_lost is True
    assert app.conf.broker_transport_options["visibility_timeout"] >= 3600


def test_soft_shutdown_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("WORKER_SOFT_SHUTDOWN_TIMEOUT", "30")
    monkeypatch.setenv("WORKER_ENABLE_SOFT_SHUTDOWN_ON_IDLE", "true")
    config = CeleryConfig(Settings(_env_file=None))
    assert config.worker_soft_shutdown_timeout == 30.0 and config.worker_enable_soft_shutdown_on_idle is True


def test_sigterm_stays_warm_under_the_solo_pool(monkeypatch):
    installed = []
    monkeypatch.setattr(worker_module, "install_worker_term_hard_handler", lambda *a, **kw: installed.append(kw))
    requeue_on_sigterm(SimpleNamespace(pool=SoloPool(), controller=object()))
    assert installed == []
    requeue_on_sigterm(SimpleNamespace(pool=object(), controller=object()))
    assert installed == [{"sig": "SIGTERM", "callback": worker_module.requeue_cold_shutdown}]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for(condition, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.1)
    return False


def _ping(client) -> bool:
    try:
        return client.ping()
    except Exception:
        return False


@pytest.fixture
def prefork_worker(tmp_path):
    """A real Redis and a `--concurrency=1` prefork worker of `sigterm_app`, as the image runs it."""
    if shutil.which("redis-server") is None:
        pytest.skip("needs redis-server")
    redis = pytest.importorskip("redis")
    port = _free_port()
    server = subprocess.Popen(
        ["redis-server", "--port", str(port), "--save", "", "--appendonly", "no", "--dir", str(tmp_path)],
        stdout=subprocess.DEVNULL,
    )
    client = redis.Redis(port=port)
    worker = None
    log_path = tmp_path / "worker.log"
    try:
        assert _wait_for(lambda: _ping(client), 5), "redis-server did not start"
        broker = f"redis://127.0.0.1:{port}/0"
        tests = Path(__file__).parent
        # No telemetry: an unreachable collector would hold the shutdown up for its export timeout.
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("OTEL_", "PYROSCOPE_")) and key not in ("HOST_IP", "REMAP_SIGTERM")
        }
        # REDIS_URL: the main process's `requeue_cold_shutdown` reads the default settings.
        env |= {"SIGTERM_TEST_BROKER": broker, "REDIS_URL": broker, "PYTHONPATH": str(tests), "LOG_FORMAT": "text"}
        with log_path.open("w") as log:
            worker = subprocess.Popen(
                [sys.executable, "-m", "celery", "-A", "sigterm_app", "worker", "--pool=prefork", "--concurrency=1",
                 "--loglevel=INFO", "--without-mingle", "--without-gossip"],
                env=env, cwd=tests, stdout=log, stderr=subprocess.STDOUT,
            )  # fmt: skip
        # Sent the way the analyzer sends: a Celery app that knows the queue only by name.
        sender = Celery("analyzer", broker=broker, set_as_current=False)
        yield SimpleNamespace(
            redis=client,
            process=worker,
            send=lambda task: sender.send_task(f"evilflowers_equation.{task}", queue="evilflowers_equation_test"),
            log=log_path.read_text,
        )
    finally:
        if worker is not None and worker.poll() is None:
            worker.kill()
        server.terminate()
        server.wait(timeout=5)


def test_sigterm_requeues_the_running_book(prefork_worker):
    """#4 against a real Redis: SIGTERM puts a running `acks_late` task back on its queue within seconds,
    and does not count it as a delivery that died (#6)."""
    client = prefork_worker.redis
    task_id = prefork_worker.send("sleepy").id
    assert _wait_for(lambda: client.get("sleepy:started") == b"1", 20), prefork_worker.log()
    assert client.get(f"equation:deliveries:{task_id}") == b"1"
    assert client.llen("evilflowers_equation_test") == 0

    started = time.monotonic()
    prefork_worker.process.send_signal(signal.SIGTERM)
    assert _wait_for(lambda: client.llen("evilflowers_equation_test") == 1, 15), prefork_worker.log()
    assert time.monotonic() - started < 15
    assert client.hlen("unacked") == 0
    prefork_worker.process.wait(timeout=15)
    assert "Initiating Soft Shutdown" in prefork_worker.log()
    # Given back as the shutdown began: the next pod starts it as delivery 1 again.
    assert client.get(f"equation:deliveries:{task_id}") is None


def test_a_book_that_kills_the_worker_is_capped(prefork_worker):
    """#6: a child killed mid-book (an OOM) is requeued at once; after `MAX_DELIVERIES` starts it stops."""
    client = prefork_worker.redis
    prefork_worker.send("doomed")
    for start in (1, 2, 3):
        assert _wait_for(lambda start=start: client.get("doomed:started") == str(start).encode(), 20), (
            prefork_worker.log()
        )
        os.kill(int(client.get("doomed:pid")), signal.SIGKILL)
    assert _wait_for(lambda: client.get("doomed:capped") == b"4", 20), prefork_worker.log()
    # Acknowledged: nothing left to run, nothing unacknowledged.
    assert _wait_for(lambda: client.hlen("unacked") == 0 and client.llen("evilflowers_equation_test") == 0, 15), (
        prefork_worker.log()
    )
