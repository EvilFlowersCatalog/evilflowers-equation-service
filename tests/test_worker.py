"""The task contract with the analyzer (name, queue, options) and one real worker against Redis.

The real-worker test needs ``TEST_REDIS_URL`` (default ``redis://localhost:6379/15``). It is skipped
when no Redis answers, except in CI (``CI`` set), where a missing Redis is a failure.
"""

import os

import pytest
from evilflowers_worker.testing import StubCatalogServer, WorkerProcess, redis_client, wait_for

from conftest import region, write_pdf
from evilflowers_equation import tasks
from evilflowers_equation.worker import app

REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")
TASKS = {"evilflowers_equation.equations": "evilflowers_equation"}


def test_every_task_is_registered_and_routed():
    app.loader.import_default_modules()
    for name, queue in TASKS.items():
        assert name in app.tasks, name
        assert app.amqp.router.route({}, name)["queue"].name == queue


def test_the_task_acknowledges_late_and_requeues_a_lost_worker():
    task = app.tasks["evilflowers_equation.equations"]
    assert task.acks_late and task.reject_on_worker_lost
    assert tasks.equations.name == "evilflowers_equation.equations"


def test_the_kit_runs_the_delivery_semantics():
    assert app.conf.worker_prefetch_multiplier == 1
    # The kit's own grace (`evilflowers_worker.shutdown`) replaces Celery's soft shutdown.
    assert app.conf.worker_soft_shutdown_timeout == 0
    assert app.conf.broker_transport_options["visibility_timeout"] >= 3600


@pytest.fixture
def broker():
    if redis_client(REDIS_URL) is None:
        if os.environ.get("CI"):
            pytest.fail(f"no Redis at {REDIS_URL}: the real-worker test must run in CI")
        pytest.skip(f"no Redis at {REDIS_URL}")
    return REDIS_URL


def test_a_book_runs_end_to_end_in_a_real_worker(broker, tmp_path):
    """Sent as the analyzer sends it, run by a prefork worker of the image's app, reported to the catalog."""
    storage = tmp_path / "data"
    write_pdf(storage / "book.pdf")
    run = {"status": "running", "jobs": {"equations:0001-0002": "pending"}, "attempt": 3}
    with StubCatalogServer(run=run) as catalog:
        env = {"CATALOG_API_URL": catalog.url, "STORAGE_PATH": str(storage), "LATEX_BACKEND": "none"}
        with WorkerProcess("evilflowers_equation.worker", redis_url=broker, env=env) as worker:
            worker.wait_ready()
            before = worker.unacked()
            worker.send(
                "evilflowers_equation.equations",
                run_id="run-1",
                source="book.pdf",
                output_dir="out/v1",
                regions=[region(1, 0), region(2, 1)],
                job="equations:0001-0002",
                attempt=3,
                newer_field="ignored",
            )
            assert wait_for(lambda: any("/jobs/" in path for _m, path in catalog.paths()), 60), catalog.calls
            assert wait_for(lambda: worker.unacked() == before and worker.redis.llen(worker.queue) == 0, 30)

    assert catalog.paths() == [
        ("GET", "/extraction/v1/runs/run-1"),
        ("POST", "/extraction/v1/runs/run-1/regions?attempt=3"),
        ("POST", "/extraction/v1/runs/run-1/jobs/equations:0001-0002?attempt=3"),
    ]
    posted = catalog.calls[1][2]["regions"]
    assert [(r["kind"], r["image_path"]) for r in posted] == [
        ("equation", "out/v1/p001-eq00.webp"),
        ("equation", "out/v1/p002-eq01.webp"),
    ]
    assert (storage / "out/v1/p002-eq01.webp").exists()
    assert catalog.calls[2][2]["status"] == "done"
