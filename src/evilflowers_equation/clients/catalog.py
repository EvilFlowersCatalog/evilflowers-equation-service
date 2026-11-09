"""Reporting to the catalog's extraction ingestion API (IP-018).

Every evilflowers-*-service posts its results to
``{CATALOG_API_URL}/extraction/v1/runs/{run_id}/...`` with the API key of the
catalog's service user. The services never write to the catalog database.
"""

import logging
import time
from collections.abc import Iterator, Sequence
from typing import Any

import requests

from evilflowers_equation.clients.http import session
from evilflowers_equation.observability import metrics
from evilflowers_equation.schemas import JobReport, Region, dump
from evilflowers_equation.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class RunNotRunning(Exception):
    """The catalog no longer accepts results for this run (409): it was reset or finished."""


def batches[T](rows: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for index in range(0, len(rows), size):
        yield rows[index : index + size]


class CatalogClient:
    def __init__(self, run_id: str, *, settings: Settings | None = None, http: requests.Session | None = None):
        self.settings = settings or get_settings()
        self.base = f"{self.settings.catalog_api_url.rstrip('/')}/extraction/v1/runs/{run_id}"
        self.http = http or session(self.settings.catalog_api_key)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with retries on connection errors and 5xx (exponential backoff); 409 → `RunNotRunning`."""
        attempts = self.settings.catalog_attempts
        for attempt in range(1, attempts + 1):
            try:
                response = self.http.post(f"{self.base}{path}", json=payload, timeout=self.settings.catalog_timeout)
            except requests.RequestException:
                if attempt == attempts:
                    raise
                logger.warning("Catalog unreachable at %s%s, retrying", self.base, path)
                reason = "unreachable"
            else:
                if response.status_code == 409:
                    raise RunNotRunning(response.text)
                if response.status_code < 500 or attempt == attempts:
                    response.raise_for_status()
                    return response.json() if response.content else {}
                logger.warning("Catalog answered %s on %s, retrying", response.status_code, path)
                reason = str(response.status_code)
            metrics.CATALOG_RETRIES.add(1, {"path": path, "reason": reason})
            time.sleep(2**attempt)
        return {}

    def regions(self, regions: Sequence[Region]) -> None:
        for batch in batches(regions, self.settings.catalog_batch_size):
            self._post("/regions", {"regions": [dump(region) for region in batch]})

    def job(self, name: str, report: JobReport) -> None:
        self._post(f"/jobs/{name}", dump(report))
