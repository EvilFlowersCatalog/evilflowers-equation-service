import pytest
import requests
from pydantic import SecretStr

from evilflowers_equation.clients import catalog as catalog_module
from evilflowers_equation.clients.catalog import CatalogClient, RunNotRunning
from evilflowers_equation.clients.http import session
from evilflowers_equation.schemas import Box, JobReport, Region


class Response:
    def __init__(self, status: int, body: bytes = b"{}"):
        self.status_code = status
        self.content = body
        self.text = body.decode()

    def json(self):
        return {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class Http:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.posts = []

    def post(self, url, json, timeout):
        self.posts.append((url, json))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(catalog_module.time, "sleep", lambda _: None)


def client(settings, *answers):
    return CatalogClient("run-1", settings=settings, http=Http(*answers))


def test_session_sends_the_bearer_key():
    assert session(SecretStr("k")).headers["Authorization"] == "Bearer k"
    assert "Authorization" not in session(SecretStr("")).headers
    assert session().headers["User-Agent"].startswith("evilflowers-equation-service/")


def test_regions_in_batches(settings):
    settings = settings.model_copy(update={"catalog_batch_size": 2})
    box = Box(x0=0, y0=0, x1=1, y1=1)
    rows = [Region(page=1, seq=seq, bbox=box) for seq in range(5)]
    catalog = client(settings, Response(201), Response(201), Response(201))
    catalog.regions(rows)
    assert [len(body["regions"]) for _, body in catalog.http.posts] == [2, 2, 1]
    url, body = catalog.http.posts[0]
    assert url == "http://catalog:8000/extraction/v1/runs/run-1/regions"
    assert body["regions"][0] == {"kind": "equation", "page": 1, "seq": 0, "bbox": box.model_dump(), "provenance": {}}


def test_retries_5xx_then_succeeds(settings):
    catalog = client(settings, Response(503), Response(200))
    catalog.job("equations", JobReport(status="done"))
    assert len(catalog.http.posts) == 2


def test_gives_up_after_the_last_attempt(settings):
    catalog = client(settings, requests.ConnectionError(), requests.ConnectionError())
    with pytest.raises(requests.ConnectionError):
        catalog.job("equations", JobReport(status="done"))


def test_409_means_the_run_is_over(settings):
    with pytest.raises(RunNotRunning):
        client(settings, Response(409, b"finished")).job("equations", JobReport(status="done"))


def test_4xx_fails_at_once(settings):
    catalog = client(settings, Response(400))
    with pytest.raises(requests.HTTPError):
        catalog.job("equations", JobReport(status="done"))
    assert len(catalog.http.posts) == 1


def test_job_report_error_is_truncated():
    assert len(JobReport(status="failed", error="x" * 5000).error) == 2000
