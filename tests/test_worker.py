"""The task contract with the analyzer (name, queue, options) and one real worker against Redis.

The real-worker test needs ``TEST_REDIS_URL`` (default ``redis://localhost:6379/15``). It is skipped
when no Redis answers, except in CI (``CI`` set), where a missing Redis is a failure.
"""

import json
import os
import threading
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from evilflowers_worker.testing import StubCatalogServer, WorkerProcess, assert_outcome, redis_client, wait_for
from PIL import Image

from conftest import region, write_pdf
from evilflowers_equation import tasks
from evilflowers_equation.worker import app

REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")
TASKS = {
    "evilflowers_equation.equations": "evilflowers_equation",
    "evilflowers_equation.standalone.formulas": "evilflowers_equation.standalone",
}


def test_every_task_is_registered_and_routed():
    app.loader.import_default_modules()
    for name, queue in TASKS.items():
        assert name in app.tasks, name
        assert app.amqp.router.route({}, name)["queue"].name == queue


def test_the_tasks_acknowledge_late_and_requeue_a_lost_worker():
    for name in TASKS:
        task = app.tasks[name]
        assert task.acks_late and task.reject_on_worker_lost, name
    assert tasks.equations.name == "evilflowers_equation.equations"
    assert tasks.formulas.name == "evilflowers_equation.standalone.formulas"


def test_every_queue_is_declared():
    declared = {queue.name for queue in app.conf.task_queues}
    assert declared == {
        "evilflowers_equation",
        "evilflowers_equation.standalone",
        "evilflowers_equation.standalone.interactive",
    }


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


@contextmanager
def stub_vlm():
    """The vlm service's `POST /v1/formulas`, answering `x^{2}` for every crop."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            results = [{"key": item["key"], "text": "$x^{2}$", "secs": 0.0} for item in body["items"]]
            payload = json.dumps({"model": "stub", "results": results}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/formulas"
    finally:
        server.shutdown()


def test_a_standalone_job_runs_end_to_end_in_a_real_worker(broker, tmp_path):
    """Sent as the digitalizer sends it (with a `link`), run without a catalog: rows, outcome, then the callback."""
    storage = tmp_path / "data"
    (storage / "leaves").mkdir(parents=True)
    Image.new("RGB", (600, 800), "white").save(storage / "leaves/0001.png")
    callbacks = f"test-callbacks-{uuid.uuid4().hex[:8]}"
    with stub_vlm() as vlm:
        env = {"STORAGE_PATH": str(storage), "VLM_ENDPOINT": vlm, "CATALOG_API_URL": "http://127.0.0.1:9"}
        with WorkerProcess("evilflowers_equation.worker", redis_url=broker, env=env) as worker:
            worker.wait_ready()
            link = worker.producer().signature("digitalizer.job_done", kwargs={"job_id": "j-1"}, queue=callbacks)
            worker.send_task(
                "evilflowers_equation.standalone.formulas",
                {
                    "run_id": "run-1",
                    "job": "formulas-0001-a1",
                    "token": "tok",
                    "inputs": ["leaves/0001.png"],
                    "output_dir": "item/_studio/jobs/formulas-0001-a1",
                    "outcome": "item/_studio/jobs/formulas-0001-a1/outcome.json",
                    "regions": [
                        {"key": "region-1", "leaf": 1, "image": "leaves/0001.png", "bbox": [0.1, 0.1, 0.5, 0.2]}
                    ],
                },
                link=link,
            )
            try:
                assert wait_for(lambda: worker.queued(callbacks), 60), worker.output
                (callback,) = worker.queued(callbacks)
            finally:
                worker.redis.delete(callbacks)

    assert WorkerProcess.args(callback) == [{"job": "formulas-0001-a1", "answered": 1}]
    assert WorkerProcess.kwargs(callback) == {"job_id": "j-1"}
    assert_outcome(
        storage,
        "item/_studio/jobs/formulas-0001-a1/outcome.json",
        token="tok",
        status="done",
        after="item/_studio/jobs/formulas-0001-a1/formulas.json",
    )
    rows = json.loads((storage / "item/_studio/jobs/formulas-0001-a1/formulas.json").read_text())
    assert rows == [{"key": "region-1", "text": "x^{2}", "model": "paddleocr-vl:PaddleOCR-VL-1.6@132c45f"}]
