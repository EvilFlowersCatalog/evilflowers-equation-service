import requests
from evilflowers_worker.catalog import RunNotRunning, RunSuperseded
from evilflowers_worker.contract import RegionsJob
from evilflowers_worker.observability import metrics
from evilflowers_worker.testing import FakeCatalog, running

from conftest import FakeRecognizer, catalog, region, write_pdf
from evilflowers_equation.jobs import EquationsJob


def job(settings, regions, fake=None, recognizer=None, source="book.pdf", delivery=1, **message):
    kwargs = {"run_id": "run-1", "source": source, "output_dir": "out/v1", "regions": regions, **message}
    return EquationsJob(
        RegionsJob.model_validate(kwargs),
        settings=settings,
        catalog=fake if fake is not None else catalog(),
        recognizer=recognizer or FakeRecognizer(),
        task_id="task-1",
        delivery=delivery,
    )


def statuses(fake: FakeCatalog) -> list[str]:
    return [report["status"] for report in fake.jobs]


def test_crops_recognizes_and_reports_done(settings):
    write_pdf(settings.storage_path / "book.pdf")
    fake = catalog()
    outcome = job(settings, [region(1, 0), region(2, 1)], fake, FakeRecognizer(r"\frac{2}{4}")).run()

    assert outcome == "done"
    assert [r["image_path"] for r in fake.regions_posted] == ["out/v1/p001-eq00.webp", "out/v1/p002-eq01.webp"]
    assert (settings.storage_path / "out/v1/p001-eq00.webp").exists()
    first = fake.regions_posted[0]
    assert first["kind"] == "equation" and first["latex"] == r"\frac{2}{4}"
    assert first["provenance"] == {"latex_model": "fake", "sympy": r"\frac{1}{2}"}
    [report] = fake.jobs
    assert report["name"] == "equations" and report["status"] == "done"
    assert report["provenance"] == {
        "image": "test",
        "latex_model": "fake",
        "crop_dpi": 200,
        "max_crop_pixels": 2_000_000,
    }


def test_sympy_form_only_when_it_differs(settings):
    write_pdf(settings.storage_path / "book.pdf")
    fake = catalog()
    job(settings, [region()], fake, FakeRecognizer(r"x + 1")).run()
    assert "sympy" not in fake.regions_posted[0]["provenance"]


def test_failed_recognition_keeps_the_crop(settings):
    write_pdf(settings.storage_path / "book.pdf")
    fake = catalog()
    assert job(settings, [region()], fake, FakeRecognizer(error=RuntimeError("oom"))).run() == "done"
    assert "latex" not in fake.regions_posted[0]
    assert fake.regions_posted[0]["image_path"] == "out/v1/p001-eq00.webp"


def test_a_bad_region_is_skipped_not_fatal(settings):
    write_pdf(settings.storage_path / "book.pdf", pages=1)
    fake = catalog()
    tiny = region(1, 1, x1=0.1001)  # under min_crop_pixels: skipped
    missing_page = region(9, 2)  # raises: logged and skipped
    assert job(settings, [region(1, 0), tiny, missing_page], fake).run() == "done"
    assert [r["seq"] for r in fake.regions_posted] == [0]


def test_unreadable_pdf_reports_failed(settings):
    fake = catalog()
    assert job(settings, [region()], fake, source="missing.pdf").run() == "failed"
    [report] = fake.jobs
    assert report["name"] == "equations" and report["status"] == "failed"
    assert report["error"] and "missing.pdf" in report["error"]


def test_an_output_dir_outside_the_storage_fails_the_job_before_any_crop(settings):
    write_pdf(settings.storage_path / "book.pdf")
    fake = catalog()
    recognizer = FakeRecognizer()
    outcome = EquationsJob(
        RegionsJob.model_validate(
            {"run_id": "run-1", "source": "book.pdf", "output_dir": "../escape", "regions": [region()]}
        ),
        settings=settings,
        catalog=fake,
        recognizer=recognizer,
    ).run()
    assert outcome == "failed"
    assert recognizer.calls == 0 and fake.regions_posted == []
    assert "escapes STORAGE_PATH" in fake.jobs[0]["error"]
    assert not (settings.storage_path.parent / "escape").exists()


def test_catalog_error_after_retries_reports_failed(settings):
    write_pdf(settings.storage_path / "book.pdf")
    fake = catalog(fail_on={"regions": requests.HTTPError("400 Bad Request")})
    assert job(settings, [region()], fake).run() == "failed"
    assert statuses(fake) == ["failed"]


def test_a_run_the_catalog_refuses_stops_quietly(settings):
    write_pdf(settings.storage_path / "book.pdf")
    fake = catalog(fail_on={"regions": RunNotRunning("finished")})
    assert job(settings, [region()], fake).run() == "not_running"
    assert fake.jobs == []


def test_parallel_workers_keep_region_order_and_context(settings):
    import threading
    import time

    from evilflowers_worker.observability.context import current

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
    fake = catalog()
    recognizer = SlowFirst()
    regions = [region(page, page) for page in range(1, 5)]
    outcome = job(settings.model_copy(update={"latex_workers": 3}), regions, fake, recognizer).run()

    assert outcome == "done"
    assert [r["seq"] for r in fake.regions_posted] == [1, 2, 3, 4]
    assert sorted(r["latex"] for r in fake.regions_posted) == ["x_1", "x_2", "x_3", "x_4"]
    # Recognition ran on the pool, with the task's run_id and task_id still bound for its log lines.
    assert all(name.startswith("latex") for name, _ in recognizer.seen)
    assert all(fields.get("run_id") == "run-1" and fields.get("task_id") == "task-1" for _, fields in recognizer.seen)


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
    fake = catalog()
    recognizer = OneSlow(others=9)
    regions = [region(page, page) for page in range(1, 11)]
    assert job(settings.model_copy(update={"latex_workers": 2}), regions, fake, recognizer).run() == "done"

    # 9 crops finished while one was stuck: more than the 2 x 2 that may wait rendered.
    assert recognizer.waited_for_all is True
    assert [r["seq"] for r in fake.regions_posted] == list(range(1, 11))


def test_a_cut_short_crop_is_marked_in_provenance(settings):
    from evilflowers_equation.latex import Decoded

    class Looping(FakeRecognizer):
        last = Decoded(64, "repetition")

    write_pdf(settings.storage_path / "book.pdf")
    fake = catalog()
    job(settings, [region()], fake, Looping("x")).run()
    assert fake.regions_posted[0]["provenance"] == {"latex_model": "fake", "latex_stop": "repetition"}


def test_no_regions_still_reports_done(settings):
    write_pdf(settings.storage_path / "book.pdf")
    fake = catalog()
    assert job(settings, [], fake).run() == "done"
    assert fake.regions_posted == [] and statuses(fake) == ["done"]


# -- per-batch jobs and attempts (#5, catalog#140) ---------------------------------------------------


def test_a_batch_reports_under_its_job_name_and_counts_under_the_kind(settings, monkeypatch):
    recorded = []
    monkeypatch.setattr(metrics, "record_job", lambda kind, outcome: recorded.append((kind, outcome)))
    write_pdf(settings.storage_path / "book.pdf")
    fake = catalog(running(jobs={"equations:0001-0008": "pending", "equations": "done"}))
    assert job(settings, [region()], fake, job="equations:0001-0008").run() == "done"
    [report] = fake.jobs
    assert report["name"] == "equations:0001-0008" and report["status"] == "done"
    assert recorded == [("equations", "done")]


def test_an_empty_batch_still_reports(settings):
    write_pdf(settings.storage_path / "book.pdf")
    fake = catalog(running(jobs={"equations:0009-0016": "pending"}))
    assert job(settings, [], fake, job="equations:0009-0016").run() == "done"
    assert fake.jobs[0]["name"] == "equations:0009-0016"


def test_a_copy_of_a_job_that_reported_does_nothing(settings):
    write_pdf(settings.storage_path / "book.pdf")
    recognizer = FakeRecognizer()
    fake = catalog(running(jobs={"equations": "done"}))
    assert job(settings, [region()], fake, recognizer).run() == "already_done"
    assert recognizer.calls == 0 and fake.regions_posted == [] and fake.jobs == []


def test_a_run_that_is_not_running_is_left_alone(settings):
    write_pdf(settings.storage_path / "book.pdf")
    fake = catalog(running(status="failed", jobs={"equations": "pending"}))
    assert job(settings, [region()], fake).run() == "not_running"
    assert fake.jobs == []


def test_a_newer_attempt_stops_the_job_and_reports_superseded(settings):
    write_pdf(settings.storage_path / "book.pdf")
    recognizer = FakeRecognizer()
    # Started again: the job is pending once more, but for attempt 4.
    fake = catalog(running(attempt=4, jobs={"equations": "pending"}))
    assert job(settings, [region()], fake, recognizer, attempt=3).run() == "superseded"
    assert recognizer.calls == 0 and fake.regions_posted == []
    [report] = fake.jobs
    assert report["name"] == "equations" and report["status"] == "superseded"


def test_a_reset_run_is_superseded_not_just_not_running(settings):
    # `backfill_extraction --force` puts the run back to `queued` on a new attempt.
    fake = catalog(running(status="queued", attempt=4))
    assert job(settings, [region()], fake, attempt=3).run() == "superseded"
    assert statuses(fake) == ["superseded"]


def test_the_run_is_read_again_while_decoding(settings):
    write_pdf(settings.storage_path / "book.pdf", pages=3)
    recognizer = FakeRecognizer()
    pending = {"equations": "pending"}
    fake = catalog(
        running(attempt=3, jobs=pending), running(attempt=3, jobs=pending), running(attempt=4, jobs=pending)
    )
    settings = settings.model_copy(update={"run_check_interval": 1e-9})
    outcome = job(settings, [region(1, 0), region(2, 1), region(3, 2)], fake, recognizer, attempt=3).run()
    assert outcome == "superseded"
    # Before the work, then before each crop: the third read saw attempt 4, so the third crop never ran.
    assert fake.reads == 3 and recognizer.calls == 1
    assert fake.regions_posted == [] and statuses(fake) == ["superseded"]


def test_a_zero_check_interval_reads_the_run_only_before_the_work(settings):
    write_pdf(settings.storage_path / "book.pdf", pages=3)
    fake = catalog()
    settings = settings.model_copy(update={"run_check_interval": 0})
    assert job(settings, [region(1, 0), region(2, 1), region(3, 2)], fake).run() == "done"
    assert fake.reads == 1


def test_a_catalog_down_mid_book_does_not_fail_it(settings):
    class Flaky(FakeCatalog):
        def run(self):
            self.reads += 1
            if self.reads > 1:
                raise requests.ConnectionError("catalog restarting")
            return running(jobs={"equations": "pending"})

    write_pdf(settings.storage_path / "book.pdf")
    fake = Flaky()
    settings = settings.model_copy(update={"run_check_interval": 1e-9})
    assert job(settings, [region(1, 0), region(2, 1)], fake).run() == "done"
    assert fake.reads == 3 and len(fake.regions_posted) == 2


def test_a_superseded_409_stops_the_job(settings):
    write_pdf(settings.storage_path / "book.pdf")
    fake = catalog(fail_on={"regions": RunSuperseded('{"reason": "superseded"}')})
    assert job(settings, [region()], fake, attempt=3).run() == "superseded"
    assert statuses(fake) == ["superseded"]


# -- delivery cap (#6) ----------------------------------------------------------------------------------


def test_a_task_delivered_too_often_is_reported_failed_without_decoding(settings):
    write_pdf(settings.storage_path / "book.pdf")
    recognizer = FakeRecognizer()
    fake = catalog()
    past_the_cap = settings.max_deliveries + 1
    assert job(settings, [region()], fake, recognizer, delivery=past_the_cap).run() == "failed"
    assert recognizer.calls == 0 and fake.regions_posted == [] and fake.reads == 0
    [report] = fake.jobs
    assert report["name"] == "equations" and report["status"] == "failed"
    assert report["error"].startswith(f"Gave up after {past_the_cap - 1} deliveries")


def test_deliveries_up_to_the_cap_decode(settings):
    write_pdf(settings.storage_path / "book.pdf")
    assert job(settings, [region()], delivery=settings.max_deliveries).run() == "done"


# -- crop bound (#6) --------------------------------------------------------------------------------------


def test_an_oversized_crop_is_rendered_at_a_lower_dpi(settings, caplog):
    from PIL import Image

    write_pdf(settings.storage_path / "book.pdf")
    whole_page = region(x0=0, y0=0, x1=1, y1=1)  # 600 x 800 pt: 1667 x 2222 px (3.7 MP) at 200 dpi
    fake = catalog()
    with caplog.at_level("INFO", logger="evilflowers_equation.jobs"):
        assert job(settings, [whole_page, region(1, 1)], fake).run() == "done"
    with Image.open(settings.storage_path / fake.regions_posted[0]["image_path"]) as crop:
        assert crop.width * crop.height <= settings.max_crop_pixels
    # A formula-sized box keeps CROP_DPI.
    assert any(line.endswith("833x222 px at 200 dpi") for line in caplog.messages)
    # Logged before rendering, with the size it was rendered at.
    assert any("lowered from 200: MAX_CROP_PIXELS" in line for line in caplog.messages)
