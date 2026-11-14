"""Reporting to the catalog's extraction ingestion API (IP-018).

Every evilflowers-*-service posts its results to
``{CATALOG_API_URL}/extraction/v1/runs/{run_id}/...`` with the API key of the
catalog's service user. The services never write to the catalog database.

With an ``attempt`` (catalog#140, from the analyzer's kwargs) every ingestion
call carries ``?attempt=N``, and the catalog refuses the results of an earlier
attempt with 409 ``reason: superseded``. A 409 is final: never retried.
"""

import logging
import time
from collections.abc import Iterator, Sequence
from typing import Any

import requests

from evilflowers_equation.clients.http import session
from evilflowers_equation.observability import metrics
from evilflowers_equation.schemas import JobReport, Region, RunState, dump
from evilflowers_equation.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class RunNotRunning(Exception):
    """The catalog no longer accepts results for this run (409): it was reset or finished."""


class RunSuperseded(RunNotRunning):
    """409 `reason: superseded`: the run was started again, and these results belong to an earlier attempt."""


def batches[T](rows: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for index in range(0, len(rows), size):
        yield rows[index : index + size]


def _conflict(response: requests.Response) -> RunNotRunning:
    try:
        body = response.json()
    except ValueError:
        body = None
    reason = None
    if isinstance(body, dict):
        extra = body.get("additional_data")
        reason = body.get("reason") or (extra.get("reason") if isinstance(extra, dict) else None)
    return (RunSuperseded if reason == "superseded" else RunNotRunning)(response.text)


class CatalogClient:
    def __init__(
        self,
        run_id: str,
        *,
        attempt: int | None = None,
        settings: Settings | None = None,
        http: requests.Session | None = None,
    ):
        self.settings = settings or get_settings()
        self.base = f"{self.settings.catalog_api_url.rstrip('/')}/extraction/v1/runs/{run_id}"
        self.attempt = attempt
        self.http = http or session(self.settings.catalog_api_key)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """A call with retries on connection errors and 5xx (exponential backoff); 409 → `RunNotRunning`."""
        url = f"{self.base}{path}"
        params = {"attempt": self.attempt} if self.attempt is not None and method == "POST" else None
        attempts = self.settings.catalog_attempts
        for attempt in range(1, attempts + 1):
            try:
                if method == "GET":
                    response = self.http.get(url, timeout=self.settings.catalog_timeout)
                else:
                    response = self.http.post(url, params=params, json=payload, timeout=self.settings.catalog_timeout)
            except requests.RequestException:
                if attempt == attempts:
                    raise
                logger.warning("Catalog unreachable at %s, retrying", url)
                reason = "unreachable"
            else:
                if response.status_code == 409:
                    raise _conflict(response)
                if response.status_code < 500 or attempt == attempts:
                    response.raise_for_status()
                    return response.json() if response.content else {}
                logger.warning("Catalog answered %s on %s, retrying", response.status_code, path or "/")
                reason = str(response.status_code)
            # `/jobs`, not `/jobs/equations:0001-0008`: a batch name per label would grow with page count.
            route = "/" + path.strip("/").split("/")[0]
            metrics.CATALOG_RETRIES.add(1, {"path": route, "reason": reason})
            time.sleep(2**attempt)
        return {}

    def run(self) -> RunState:
        """The run's status, attempt and jobs, read before the work and between crops."""
        body = self._request("GET", "")
        return RunState.model_validate(body.get("response", body))

    def regions(self, regions: Sequence[Region]) -> None:
        for batch in batches(regions, self.settings.catalog_batch_size):
            self._request("POST", "/regions", {"regions": [dump(region) for region in batch]})

    def job(self, name: str, report: JobReport) -> None:
        self._request("POST", f"/jobs/{name}", dump(report))
