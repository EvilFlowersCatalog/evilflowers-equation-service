import requests

from conftest import FakeCatalog, FakeDeliveries, FakeRecognizer, region, write_pdf
from evilflowers_equation.clients.catalog import RunNotRunning, RunSuperseded
from evilflowers_equation.jobs import EquationsJob
from evilflowers_equation.observability import metrics
from evilflowers_equation.schemas import RunState


def job(settings, regions, catalog=None, recognizer=None, source="book.pdf", **kwargs):
    kwargs.setdefault("deliveries", FakeDeliveries())
    return EquationsJob(
        "run-1",
        source,
        "out/v1",
        regions,
        settings=settings,
        catalog=catalog or FakeCatalog(),
        recognizer=recognizer or FakeRecognizer(),
        **kwargs,
    )


def running(attempt=None, **jobs):
    return RunState(status="running", attempt=attempt, jobs=jobs or {"equations": "pending"})


def test_crops_recognizes_and_reports_done(settings):
    write_pdf(settings.storage_path / "book.pdf")
    catalog = FakeCatalog()
    outcome = job(settings, [region(1, 0), region(2, 1)], catalog, FakeRecognizer(r"\frac{2}{4}")).run()

    assert outcome == "done"
    assert [r.image_path for r in catalog.regions_sent] == ["out/v1/p001-eq00.webp", "out/v1/p002-eq01.webp"]
    assert (settings.storage_path / "out/v1/p001-eq00.webp").exists()
    first = catalog.regions_sent[0]
    assert first.kind == "equation" and first.latex == r"\frac{2}{4}"
    assert first.provenance == {"latex_model": "fake", "sympy": r"\frac{1}{2}"}
    [(name, report)] = catalog.reports
    assert name == "equations" and report.status == "done"
    assert report.provenance == {"image": "test", "latex_model": "fake", "crop_dpi": 200, "max_crop_pixels": 2_000_000}


def test_sympy_form_only_when_it_differs(settings):
    write_pdf(settings.storage_path / "book.pdf")
    catalog = FakeCatalog()
    job(settings, [region()], catalog, FakeRecognizer(r"x + 1")).run()
    assert "sympy" not in catalog.regions_sent[0].provenance


def test_failed_recognition_keeps_the_crop(settings):
    write_pdf(settings.storage_path / "book.pdf")
    catalog = FakeCatalog()
    assert job(settings, [region()], catalog, FakeRecognizer(error=RuntimeError("oom"))).run() == "done"
    assert catalog.regions_sent[0].latex is None
    assert catalog.regions_sent[0].image_path == "out/v1/p001-eq00.webp"


def test_a_bad_region_is_skipped_not_fatal(settings):
    write_pdf(settings.storage_path / "book.pdf", pages=1)
    catalog = FakeCatalog()
    tiny = region(1, 1, x1=0.1001)  # under min_crop_pixels: skipped
    missing_page = region(9, 2)  # raises: logged and skipped
    assert job(settings, [region(1, 0), tiny, missing_page], catalog).run() == "done"
    assert [r.seq for r in catalog.regions_sent] == [0]


def test_unreadable_pdf_reports_failed(settings):
    catalog = FakeCatalog()
    assert job(settings, [region()], catalog, source="missing.pdf").run() == "failed"
    [(name, report)] = catalog.reports
    assert name == "equations" and report.status == "failed"
    assert report.error and "missing.pdf" in report.error


def test_catalog_error_after_retries_reports_failed(settings):
    write_pdf(settings.storage_path / "book.pdf")
    catalog = FakeCatalog(fail_on={"regions": requests.HTTPError("400 Bad Request")})
    assert job(settings, [region()], catalog).run() == "failed"
    assert catalog.reports[0][1].status == "failed"


def test_a_run_the_catalog_refuses_stops_quietly(settings):
    write_pdf(settings.storage_path / "book.pdf")
    catalog = FakeCatalog(fail_on={"regions": RunNotRunning("finished")})
    assert job(settings, [region()], catalog).run() == "not_running"
    assert catalog.reports == []


def test_parallel_workers_keep_region_order_and_context(settings):
    import threading
    import time

    from evilflowers_equation.observability.context import current

    class SlowFirst:
        """The first crop takes longest, so a pool that does not keep order would reorder it."""

        name = "slow"

        def __init__(self):
            self.lock = threading.Lock()
            self.calls = 0
            self.seen = []

        def __call__(self, image):
            with self.lock:
                self.calls += 1
                call = self.calls
            time.sleep(0.05 if call == 1 else 0)
            self.seen.append((threading.current_thread().name, dict(current())))
            return f"x_{call}"

    write_pdf(settings.storage_path / "book.pdf", pages=4)
    catalog = FakeCatalog()
    recognizer = SlowFirst()
    regions = [region(page, page) for page in range(1, 5)]
    outcome = job(settings.model_copy(update={"latex_workers": 3}), regions, catalog, recognizer).run()

    assert outcome == "done"
    assert [r.seq for r in catalog.regions_sent] == [1, 2, 3, 4]
    assert sorted(r.latex for r in catalog.regions_sent) == ["x_1", "x_2", "x_3", "x_4"]
    # Recognition ran on the pool, with the task's run_id still bound for its log lines.
    assert all(name.startswith("latex") for name, _ in recognizer.seen)
    assert all(fields.get("run_id") == "run-1" for _, fields in recognizer.seen)


def test_a_slow_crop_does_not_stall_the_others(settings):
    import threading

    class OneSlow:
        """The first call waits until every other crop is done, or gives up after 5 s."""

        name = "slow"

        def __init__(self, others):
            self.others = others
            self.condition = threading.Condition()
            self.calls = 0
            self.done = 0
            self.waited_for_all = None

        def __call__(self, image):
            with self.condition:
                self.calls += 1
                first = self.calls == 1
            if first:
                with self.condition:
                    self.waited_for_all = self.condition.wait_for(lambda: self.done == self.others, timeout=5)
                return "slow"
            with self.condition:
                self.done += 1
                self.condition.notify_all()
            return "fast"

    write_pdf(settings.storage_path / "book.pdf", pages=10)
    catalog = FakeCatalog()
    recognizer = OneSlow(others=9)
    regions = [region(page, page) for page in range(1, 11)]
    assert job(settings.model_copy(update={"latex_workers": 2}), regions, catalog, recognizer).run() == "done"

    # 9 crops finished while one was stuck: more than the 2 x 2 that may wait rendered.
    assert recognizer.waited_for_all is True
    assert [r.seq for r in catalog.regions_sent] == list(range(1, 11))


def test_a_cut_short_crop_is_marked_in_provenance(settings):
    from evilflowers_equation.latex import Decoded

    class Looping(FakeRecognizer):
        last = Decoded(64, "repetition")

    write_pdf(settings.storage_path / "book.pdf")
    catalog = FakeCatalog()
    job(settings, [region()], catalog, Looping("x")).run()
    assert catalog.regions_sent[0].provenance == {"latex_model": "fake", "latex_stop": "repetition"}


def test_no_regions_still_reports_done(settings):
    write_pdf(settings.storage_path / "book.pdf")
    catalog = FakeCatalog()
    assert job(settings, [], catalog).run() == "done"
    assert catalog.regions_sent == [] and catalog.reports[0][1].status == "done"


# -- per-batch jobs and attempts (#5, catalog#140) ---------------------------------------------------


def test_a_batch_reports_under_its_job_name_and_counts_under_the_kind(settings, monkeypatch):
    recorded = []
    monkeypatch.setattr(metrics, "record_job", lambda job, outcome: recorded.append((job, outcome)))
    write_pdf(settings.storage_path / "book.pdf")
    catalog = FakeCatalog(states=[running(**{"equations:0001-0008": "pending", "equations": "done"})])
    assert job(settings, [region()], catalog, job="equations:0001-0008").run() == "done"
    [(name, report)] = catalog.reports
    assert name == "equations:0001-0008" and report.status == "done"
    assert recorded == [("equations", "done")]


def test_an_empty_batch_still_reports(settings):
    write_pdf(settings.storage_path / "book.pdf")
    catalog = FakeCatalog(states=[running(**{"equations:0009-0016": "pending"})])
    assert job(settings, [], catalog, job="equations:0009-0016").run() == "done"
    assert catalog.reports[0][0] == "equations:0009-0016"


def test_a_copy_of_a_job_that_reported_does_nothing(settings):
    write_pdf(settings.storage_path / "book.pdf")
    recognizer = FakeRecognizer()
    catalog = FakeCatalog(states=[running(equations="done")])
    assert job(settings, [region()], catalog, recognizer).run() == "already_done"
    assert recognizer.calls == 0 and catalog.regions_sent == [] and catalog.reports == []


def test_a_run_that_is_not_running_is_left_alone(settings):
    write_pdf(settings.storage_path / "book.pdf")
    catalog = FakeCatalog(states=[RunState(status="failed", jobs={"equations": "pending"})])
    assert job(settings, [region()], catalog).run() == "not_running"
    assert catalog.reports == []


def test_a_newer_attempt_stops_the_job_and_reports_superseded(settings):
    write_pdf(settings.storage_path / "book.pdf")
    recognizer = FakeRecognizer()
    # Started again: the job is pending once more, but for attempt 4.
    catalog = FakeCatalog(states=[running(attempt=4)])
    assert job(settings, [region()], catalog, recognizer, attempt=3).run() == "superseded"
    assert recognizer.calls == 0 and catalog.regions_sent == []
    [(name, report)] = catalog.reports
    assert name == "equations" and report.status == "superseded"


def test_a_reset_run_is_superseded_not_just_not_running(settings):
    # `backfill_extraction --force` puts the run back to `queued` on a new attempt.
    catalog = FakeCatalog(states=[RunState(status="queued", attempt=4, jobs={})])
    assert job(settings, [region()], catalog, attempt=3).run() == "superseded"
    assert [r.status for _, r in catalog.reports] == ["superseded"]


def test_the_run_is_read_again_while_decoding(settings):
    write_pdf(settings.storage_path / "book.pdf", pages=3)
    recognizer = FakeRecognizer()
    catalog = FakeCatalog(states=[running(attempt=3), running(attempt=3), running(attempt=4)])
    settings = settings.model_copy(update={"run_check_interval": 1e-9})
    outcome = job(settings, [region(1, 0), region(2, 1), region(3, 2)], catalog, recognizer, attempt=3).run()
    assert outcome == "superseded"
    # Before the work, then before each crop: the third read saw attempt 4, so the third crop never ran.
    assert catalog.reads == 3 and recognizer.calls == 1
    assert catalog.regions_sent == [] and [r.status for _, r in catalog.reports] == ["superseded"]


def test_a_catalog_down_mid_book_does_not_fail_it(settings):
    class Flaky(FakeCatalog):
        def run(self):
            self.reads += 1
            if self.reads > 1:
                raise requests.ConnectionError("catalog restarting")
            return running()

    write_pdf(settings.storage_path / "book.pdf")
    catalog = Flaky()
    settings = settings.model_copy(update={"run_check_interval": 1e-9})
    assert job(settings, [region(1, 0), region(2, 1)], catalog).run() == "done"
    assert catalog.reads == 3 and len(catalog.regions_sent) == 2


def test_a_superseded_409_is_final(settings):
    write_pdf(settings.storage_path / "book.pdf")
    catalog = FakeCatalog(fail_on={"regions": RunSuperseded('{"reason": "superseded"}')})
    assert job(settings, [region()], catalog, attempt=3).run() == "superseded"
    assert catalog.reports == []


# -- delivery cap (#6) ----------------------------------------------------------------------------------


def test_a_task_delivered_too_often_is_reported_failed_without_decoding(settings):
    write_pdf(settings.storage_path / "book.pdf")
    deliveries = FakeDeliveries()
    deliveries.counts["task-1"] = 3  # three starts, none of which finished
    recognizer = FakeRecognizer()
    catalog = FakeCatalog()
    assert job(settings, [region()], catalog, recognizer, task_id="task-1", deliveries=deliveries).run() == "failed"
    assert recognizer.calls == 0 and catalog.regions_sent == []
    [(name, report)] = catalog.reports
    assert name == "equations" and report.status == "failed"
    assert report.error.startswith("TooManyDeliveries: started 4 times")


def test_deliveries_under_the_cap_decode(settings):
    write_pdf(settings.storage_path / "book.pdf")
    deliveries = FakeDeliveries()
    deliveries.counts["task-1"] = 2
    assert job(settings, [region()], task_id="task-1", deliveries=deliveries).run() == "done"
    assert deliveries.counts["task-1"] == 3


def test_a_copy_that_skips_is_not_counted(settings):
    deliveries = FakeDeliveries()
    catalog = FakeCatalog(states=[running(equations="done")])
    assert job(settings, [region()], catalog, task_id="task-1", deliveries=deliveries).run() == "already_done"
    assert deliveries.counts == {}


def test_redis_down_does_not_fail_the_book(settings):
    write_pdf(settings.storage_path / "book.pdf")
    deliveries = FakeDeliveries(error=ConnectionError("redis"))
    assert job(settings, [region()], task_id="task-1", deliveries=deliveries).run() == "done"


# -- crop bound (#6) --------------------------------------------------------------------------------------


def test_an_oversized_crop_is_rendered_at_a_lower_dpi(settings, caplog):
    from PIL import Image

    write_pdf(settings.storage_path / "book.pdf")
    whole_page = region(x0=0, y0=0, x1=1, y1=1)  # 600 x 800 pt: 1667 x 2222 px (3.7 MP) at 200 dpi
    catalog = FakeCatalog()
    with caplog.at_level("INFO", logger="evilflowers_equation.jobs"):
        assert job(settings, [whole_page, region(1, 1)], catalog).run() == "done"
    with Image.open(settings.storage_path / catalog.regions_sent[0].image_path) as crop:
        assert crop.width * crop.height <= settings.max_crop_pixels
    # A formula-sized box keeps CROP_DPI.
    assert any(line.endswith("833x222 px at 200 dpi") for line in caplog.messages)
    # Logged before rendering, with the size it was rendered at.
    assert any("lowered from 200: MAX_CROP_PIXELS" in line for line in caplog.messages)
