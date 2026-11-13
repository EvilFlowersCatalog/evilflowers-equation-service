import requests

from conftest import FakeCatalog, FakeRecognizer, region, write_pdf
from evilflowers_equation.clients.catalog import RunNotRunning
from evilflowers_equation.jobs import EquationsJob


def job(settings, regions, catalog=None, recognizer=None, source="book.pdf"):
    return EquationsJob(
        "run-1",
        source,
        "out/v1",
        regions,
        settings=settings,
        catalog=catalog or FakeCatalog(),
        recognizer=recognizer or FakeRecognizer(),
    )


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
    assert report.provenance == {"image": "test", "latex_model": "fake", "crop_dpi": 200}


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
