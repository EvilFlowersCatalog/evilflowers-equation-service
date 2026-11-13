"""The IP-018 `equations` job: crop each formula region, transcribe it to LaTeX, post it, report the job.

The analyzer dispatches the job and declares it at the run's `/start`; the run
finishes only when every declared job has reported `done` or `failed`. So the
job never ends silently: whatever goes wrong is reported to the catalog, unless
the catalog itself refused the run (409, `RunNotRunning`). A single region that
fails is logged and skipped; the job still reports `done` with the others.
"""

import contextvars
import logging
import threading
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Literal

import pymupdf
from PIL import Image
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
            provenance["latex_max_tokens"] = self.recognizer.max_tokens
            provenance["latex_repetition_stop"] = self.recognizer.repetition_stop
            provenance["latex_workers"] = self.settings.latex_workers
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
        """Every region the analyzer found, in order; one that fails is skipped. Raising fails the job.

        The task thread renders and saves each crop (PyMuPDF is not thread-safe) and
        hands its recognition to `LATEX_WORKERS` threads; ONNX Runtime releases the GIL.
        At most two crops per worker are rendered and not yet recognized, so a long book
        does not pile up images in memory. The bound counts unfinished crops, not
        uncollected ones: a slow crop at the head does not stall the others.
        """
        regions = self._regions.validate_python(self.raw_regions)
        workers = self.settings.latex_workers
        kept: list[Region] = []
        pending: deque[tuple[LayoutRegion, str, Future]] = deque()
        unfinished = threading.BoundedSemaphore(2 * workers)

        def collect() -> None:
            region, name, future = pending.popleft()
            result = self._finish(region, name, future)
            metrics.REGIONS.add(1, {"outcome": "kept" if result else "failed"})
            if result is not None:
                kept.append(result)

        with (
            span("equations.extract", **{"equations.regions": len(regions), "latex.workers": workers}) as current,
            pymupdf.open(self.path) as document,
            ThreadPoolExecutor(max_workers=workers, thread_name_prefix="latex") as pool,
        ):
            for region in regions:
                unfinished.acquire()
                try:
                    name, image = self.crop(document, region)
                except Exception:
                    unfinished.release()
                    logger.exception("Run %s: equation p%s/%s skipped", self.run_id, region.page, region.seq)
                    metrics.REGIONS.add(1, {"outcome": "failed"})
                    continue
                if image is None:
                    unfinished.release()
                    metrics.REGIONS.add(1, {"outcome": "skipped"})
                    continue
                # A fresh copy each time: the span and the `bind` fields are context variables,
                # and one Context cannot be entered by two threads at once.
                future = pool.submit(contextvars.copy_context().run, self.transcribe, image)
                future.add_done_callback(lambda _: unfinished.release())
                pending.append((region, name, future))
                # In order, as far as the crops are done.
                while pending and pending[0][2].done():
                    collect()
            while pending:
                collect()
            if current is not None:
                current.set_attribute("equations.kept", len(kept))
        return kept

    def crop(self, document: pymupdf.Document, region: LayoutRegion) -> tuple[str, Image.Image | None]:
        """Render and save one region's crop; no image for a crop too small to be a formula."""
        page = document[region.page - 1]
        image = crops.render(page, crops.rect(region.bbox, page), self.settings.crop_dpi)
        name = crops.crop_name(self.output_dir, region.page, region.seq)
        if min(image.size) < self.settings.min_crop_pixels:
            return name, None
        crops.save(image, self.settings.storage_path, name, self.settings.webp_quality)
        return name, image

    def transcribe(self, image: Image.Image) -> tuple[str | None, dict[str, Any]]:
        """The crop's LaTeX and its provenance; runs on a worker thread."""
        source_latex = latex.recognize(self.recognizer, image)
        provenance: dict[str, Any] = {"latex_model": self.recognizer.name}
        # `last` is per thread; without LaTeX it may still describe this thread's previous crop.
        decoded = getattr(self.recognizer, "last", None) if source_latex else None
        if decoded is not None and decoded.stop != "eos":
            # Cut short: a loop or the token cap. Kept so such LaTeX can be told apart.
            provenance["latex_stop"] = decoded.stop
        return source_latex, provenance

    def _finish(self, region: LayoutRegion, name: str, future: Future) -> Region | None:
        """The region once its LaTeX is in, with sympy's form (on the task thread: sympy holds the GIL)."""
        try:
            source_latex, provenance = future.result()
            normalized = latex.normalize(source_latex)
        except Exception:
            logger.exception("Run %s: equation p%s/%s skipped", self.run_id, region.page, region.seq)
            return None
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
