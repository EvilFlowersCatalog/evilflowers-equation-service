import pymupdf
import pytest

from evilflowers_equation.schemas import JobReport, Region, RunState
from evilflowers_equation.settings import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Defaults only (no `.env`), storage in a temporary directory, no recognizer, short retries."""
    return Settings(_env_file=None, storage_path=tmp_path, catalog_attempts=2, image_tag="test", latex_backend="none")


class FakeCatalog:
    """Records what a job sends; `fail_on` makes one call raise. `states` are what `run()` answers, in turn
    (the last one repeats); by default a running run that waits for `equations`."""

    def __init__(self, fail_on: dict[str, BaseException] | None = None, states: list[RunState] | None = None):
        self.regions_sent: list[Region] = []
        self.reports: list[tuple[str, JobReport]] = []
        self.fail_on = fail_on or {}
        self.states = list(states or [RunState(status="running", jobs={"equations": "pending"})])
        self.reads = 0

    def run(self):
        self.reads += 1
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def regions(self, regions):
        if error := self.fail_on.get("regions"):
            raise error
        self.regions_sent.extend(regions)

    def job(self, name, report):
        if (error := self.fail_on.get("job")) and report.status == "done":
            raise error
        self.reports.append((name, report))


class FakeDeliveries:
    """`DeliveryCounter` in memory."""

    def __init__(self, error: Exception | None = None):
        self.counts: dict[str, int] = {}
        self.error = error

    def record(self, task_id):
        if self.error:
            raise self.error
        self.counts[task_id] = self.counts.get(task_id, 0) + 1
        return self.counts[task_id]


class FakeRecognizer:
    name = "fake"

    def __init__(self, answer: str | None = r"E = m c^{2}", error: Exception | None = None):
        self.answer = answer
        self.error = error
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        if self.error:
            raise self.error
        return self.answer


def write_pdf(path, pages: int = 2) -> None:
    """A small PDF with one formula-like line per page."""
    pdf = pymupdf.open()
    for _ in range(pages):
        page = pdf.new_page(width=600, height=800)
        page.insert_text((100, 120), "E = mc^2", fontsize=20)
    pdf.save(path)


def region(page: int = 1, seq: int = 0, **bbox) -> dict:
    return {"page": page, "seq": seq, "bbox": {"x0": 0.1, "y0": 0.1, "x1": 0.6, "y1": 0.2, **bbox}}
