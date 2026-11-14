import json as json_module

import pytest
import requests
from pydantic import SecretStr

from evilflowers_equation.clients import catalog as catalog_module
from evilflowers_equation.clients.catalog import CatalogClient, RunNotRunning, RunSuperseded
from evilflowers_equation.clients.http import session
from evilflowers_equation.schemas import Box, JobReport, Region


class Response:
    def __init__(self, status: int, body: bytes = b"{}"):
        self.status_code = status
        self.content = body
        self.text = body.decode()

    def json(self):
        return json_module.loads(self.content) if self.content else {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class Http:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.posts = []
        self.params = []
        self.gets = []

    def _answer(self):
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def post(self, url, json, timeout, params=None):
        self.posts.append((url, json))
        self.params.append(params)
        return self._answer()

    def get(self, url, timeout):
        self.gets.append(url)
        return self._answer()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(catalog_module.time, "sleep", lambda _: None)


def client(settings, *answers, attempt=None):
    return CatalogClient("run-1", attempt=attempt, settings=settings, http=Http(*answers))


def test_a_batch_job_reports_to_its_own_path(settings):
    catalog = client(settings, Response(200))
    catalog.job("equations:0001-0008", JobReport(status="done"))
    assert catalog.http.posts[0][0] == "http://catalog:8000/extraction/v1/runs/run-1/jobs/equations:0001-0008"
    assert catalog.http.params == [None]  # no attempt, no `?attempt=`: catalogs that predate #140


def test_every_ingestion_call_carries_the_attempt(settings):
    box = Box(x0=0, y0=0, x1=1, y1=1)
    catalog = client(settings, Response(201), Response(200), attempt=3)
    catalog.regions([Region(page=1, seq=0, bbox=box)])
    catalog.job("equations", JobReport(status="done"))
    assert catalog.http.params == [{"attempt": 3}, {"attempt": 3}]


def test_run_reads_status_attempt_and_jobs(settings):
    body = b'{"response": {"status": "running", "attempt": 4, "jobs": {"equations": "pending"}, "page_count": 3}}'
    state = client(settings, Response(200, body), attempt=3).run()
    assert (state.status, state.attempt, state.jobs) == ("running", 4, {"equations": "pending"})
    assert state.skip_reason("equations", 3) == "superseded"
    assert state.skip_reason("equations", 4) is None
    assert state.skip_reason("equations") is None  # an analyzer that sends no attempt


def test_a_superseded_409_is_told_apart_and_not_retried(settings):
    body = b'{"title": "The extraction run was started again", "additional_data": {"reason": "superseded"}}'
    catalog = client(settings, Response(409, body), attempt=3)
    with pytest.raises(RunSuperseded):
        catalog.regions([Region(page=1, seq=0, bbox=Box(x0=0, y0=0, x1=1, y1=1))])
    assert len(catalog.http.posts) == 1
    with pytest.raises(RunNotRunning) as refused:
        client(settings, Response(409, b'{"reason": "not_running"}')).job("equations", JobReport(status="done"))
    assert type(refused.value) is RunNotRunning


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
