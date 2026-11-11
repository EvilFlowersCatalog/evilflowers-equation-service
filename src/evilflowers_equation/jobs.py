"""The IP-018 `equations` job: crop each formula region, transcribe it to LaTeX, post it, report the job.

The analyzer dispatches the job and declares it at the run's `/start`; the run
finishes only when every declared job has reported `done` or `failed`. So the
job never ends silently: whatever goes wrong is reported to the catalog, unless
the catalog itself refused the run (409, `RunNotRunning`). A single region that
fails is logged and skipped; the job still reports `done` with the others.
"""

import logging
from collections.abc import Sequence
from typing import Any, Literal

import pymupdf
from pydantic import TypeAdapter

from evilflowers_equation import crops, latex
from evilflowers_equation.clients.catalog import CatalogClient, RunNotRunning
from evilflowers_equation.latex import Recognizer
from evilflowers_equation.observability import bind, fail, metrics, span
from evilflowers_equation.schemas import JobReport, LayoutRegion, Region
from evilflowers_equation.settings import Settings, get_settings

logger = logging.getLogger(__name__)

Outcome = Literal["done", "failed", "not_running"]


class EquationsJob:
    #: The job's name in the catalog (`/jobs/<name>`, `run.provenance[<name>]`).
    name = "equations"
    _regions = TypeAdapter(list[LayoutRegion])

    def __init__(
        self,
        run_id: str,
        source: str,
        output_dir: str,
        regions: Sequence[dict],
        *,
        settings: Settings | None = None,
        catalog: CatalogClient | None = None,
        recognizer: Recognizer | None = None,
    ):
        self.settings = settings or get_settings()
        self.run_id = run_id
        self.source = source
        self.path = self.settings.source_path(source)
        self.output_dir = output_dir
        self.raw_regions = regions
        self.catalog = catalog or CatalogClient(run_id, settings=self.settings)
        self.recognizer = recognizer or latex.recognizer(self.settings)

    # -- protocol ---------------------------------------------------------------------------------

    def run(self) -> Outcome:
        with bind(run_id=self.run_id, job=self.name, source=self.source):
            try:
                kept = self.extract()
                self.catalog.regions(kept)
                self.catalog.job(self.name, JobReport(status="done", provenance=self.provenance()))
            except RunNotRunning:
                logger.warning("Run %s is not accepting results", self.run_id)
                return self._outcome("not_running")
            except Exception as error:
                logger.exception("Job %s failed for run %s", self.name, self.run_id)
                fail(error)
                self._report_failure(error)
                return self._outcome("failed")
            metrics.DOCUMENT_REGIONS.record(len(kept))
            logger.info("Run %s: %d of %d equations", self.run_id, len(kept), len(self.raw_regions))
            return self._outcome("done")

    def provenance(self) -> dict[str, Any]:
        """What produced the results, stored as `run.provenance["equations"]`."""
        provenance: dict[str, Any] = {
            "image": self.settings.image_tag,
            "latex_model": self.recognizer.name,
            "crop_dpi": self.settings.crop_dpi,
        }
        if isinstance(self.recognizer, latex.OnnxRecognizer):
            provenance["latex_threads"] = self.recognizer.threads
        return provenance

    def _report_failure(self, error: Exception) -> None:
        # Report it, or the run waits for this job forever.
        report = JobReport(status="failed", error=f"{type(error).__name__}: {error}", provenance=self.provenance())
        try:
            self.catalog.job(self.name, report)
        except RunNotRunning:
            logger.warning("Run %s is not accepting results", self.run_id)
        except Exception:
            logger.exception("Could not report the failed %s job for run %s", self.name, self.run_id)

    def _outcome(self, outcome: Outcome) -> Outcome:
        metrics.record_job(self.name, outcome)
        return outcome

    # -- extraction -------------------------------------------------------------------------------

    def extract(self) -> list[Region]:
        """Every region the analyzer found; one that fails is skipped. Raising fails the job."""
        regions = self._regions.validate_python(self.raw_regions)
        kept: list[Region] = []
        with (
            span("equations.extract", **{"equations.regions": len(regions)}) as current,
            pymupdf.open(self.path) as document,
        ):
            for region in regions:
                try:
                    result = self.equation(document, region)
                except Exception:
                    logger.exception("Run %s: equation p%s/%s skipped", self.run_id, region.page, region.seq)
                    metrics.REGIONS.add(1, {"outcome": "failed"})
                    continue
                metrics.REGIONS.add(1, {"outcome": "kept" if result else "skipped"})
                if result is not None:
                    kept.append(result)
            if current is not None:
                current.set_attribute("equations.kept", len(kept))
        return kept

    def equation(self, document: pymupdf.Document, region: LayoutRegion) -> Region | None:
        """One region: crop, LaTeX, sympy's form. None for a crop too small to be a formula."""
        page = document[region.page - 1]
        image = crops.render(page, crops.rect(region.bbox, page), self.settings.crop_dpi)
        if min(image.size) < self.settings.min_crop_pixels:
            return None

        name = crops.crop_name(self.output_dir, region.page, region.seq)
        crops.save(image, self.settings.storage_path, name, self.settings.webp_quality)

        source_latex = latex.recognize(self.recognizer, image)
        provenance: dict[str, Any] = {"latex_model": self.recognizer.name}
        normalized = latex.normalize(source_latex)
        if normalized and normalized != source_latex:
            # The model's LaTeX is what is shown; sympy's form is kept for comparison and search.
            provenance["sympy"] = normalized

        return Region(
            page=region.page,
            seq=region.seq,
            bbox=region.bbox,
            image_path=name,
            latex=source_latex,
            provenance=provenance,
        )
