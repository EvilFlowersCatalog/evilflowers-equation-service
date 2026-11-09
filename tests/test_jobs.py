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


def test_no_regions_still_reports_done(settings):
    write_pdf(settings.storage_path / "book.pdf")
    catalog = FakeCatalog()
    assert job(settings, [], catalog).run() == "done"
    assert catalog.regions_sent == [] and catalog.reports[0][1].status == "done"
