import pymupdf
import pytest
from evilflowers_worker.testing import FakeCatalog, running

from evilflowers_equation.settings import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Defaults only (no `.env`), storage in a temporary directory, no recognizer, short retries."""
    return Settings(_env_file=None, storage_path=tmp_path, catalog_attempts=2, image_tag="test", latex_backend="none")


def catalog(*states, **kwargs) -> FakeCatalog:
    """The kit's `FakeCatalog`; by default a running run that waits for `equations`."""
    return FakeCatalog(states=states or [running(jobs={"equations": "pending"})], **kwargs)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = pymupdf.open()
    for _ in range(pages):
        page = pdf.new_page(width=600, height=800)
        page.insert_text((100, 120), "E = mc^2", fontsize=20)
    pdf.save(path)


def region(page: int = 1, seq: int = 0, **bbox) -> dict:
    return {"page": page, "seq": seq, "bbox": {"x0": 0.1, "y0": 0.1, "x1": 0.6, "y1": 0.2, **bbox}}
