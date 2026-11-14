"""The IP-018 `equations` job: crop each formula region, transcribe it to LaTeX, post it, report the job.

The analyzer dispatches the job and declares it at the run's `/start`; the run
finishes only when every declared job has reported `done` or `failed`. So the
job never ends silently: whatever goes wrong is reported to the catalog, unless
the catalog itself refused the run (409). A single region that fails is logged
and skipped; the job still reports `done` with the others.

The analyzer sends either one task per book, reported as `equations`, or one
per layout batch with a `job` kwarg (`equations:0001-0008`, #5), reported under
that name. Metrics keep the kind, `equations`, so their labels do not grow with
page count.

A task can arrive more than once: a SIGTERM requeue, a killed worker, an
analyzer that dispatched again. `/regions` is not idempotent, so before the work
the job reads its run and does nothing unless the run is `running`, still on
this task's `attempt` (catalog#140) and still waits for this job. It reads the
run again every `RUN_CHECK_INTERVAL` while it decodes, and stops there too. And
a task id that has already started `MAX_DELIVERIES` times is reported `failed`
without decoding (#6, `deliveries`).
"""

import contextvars
import logging
import threading
import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Literal

import pymupdf
from PIL import Image
from pydantic import TypeAdapter

from evilflowers_equation import crops, latex
from evilflowers_equation.clients.catalog import CatalogClient, RunNotRunning, RunSuperseded
from evilflowers_equation.deliveries import DeliveryCounter
from evilflowers_equation.latex import Recognizer
from evilflowers_equation.observability import bind, fail, metrics, span
from evilflowers_equation.schemas import JobReport, LayoutRegion, Region, SkipReason
from evilflowers_equation.settings import Settings, get_settings

logger = logging.getLogger(__name__)

Outcome = Literal["done", "failed", "not_running", "superseded", "already_done"]


class Stop(Exception):
    """The run no longer wants this job's results; raised by `check`."""

    def __init__(self, reason: SkipReason):
        super().__init__(reason)
        self.reason = reason


class TooManyDeliveries(Exception):
    """The task started more than `MAX_DELIVERIES` times; the earlier ones never finished."""


class EquationsJob:
    #: The kind of job: its name in the catalog when the analyzer names no batch, and its metric label.
    kind = "equations"
    _regions = TypeAdapter(list[LayoutRegion])

    def __init__(
        self,
        run_id: str,
        source: str,
        output_dir: str,
        regions: Sequence[dict],
        *,
        job: str | None = None,
        attempt: int | None = None,
        task_id: str | None = None,
        settings: Settings | None = None,
        catalog: CatalogClient | None = None,
        recognizer: Recognizer | None = None,
        deliveries: DeliveryCounter | None = None,
    ):
        self.settings = settings or get_settings()
        self.run_id = run_id
        self.source = source
        self.path = self.settings.source_path(source)
        self.output_dir = output_dir
        self.raw_regions = regions
        #: The job's name in the catalog (`/jobs/<name>`, `run.provenance[<name>]`).
        self.name = job or self.kind
        self.attempt = attempt
        self.task_id = task_id
        self.catalog = catalog or CatalogClient(run_id, attempt=attempt, settings=self.settings)
        self.recognizer = recognizer or latex.recognizer(self.settings)
        self.deliveries = deliveries or DeliveryCounter(self.settings)
        self._checked_at = 0.0

    # -- protocol ---------------------------------------------------------------------------------

    def run(self) -> Outcome:
        with bind(run_id=self.run_id, job=self.name, attempt=self.attempt, source=self.source, task_id=self.task_id):
            try:
                self.check()
                self._count_delivery()
                kept = self.extract()
                self.catalog.regions(kept)
                self.catalog.job(self.name, JobReport(status="done", provenance=self.provenance()))
            except Stop as stop:
                return self._stop(stop.reason)
            except RunSuperseded:
                logger.warning("Run %s was started again; attempt %s is superseded", self.run_id, self.attempt)
                return self._outcome("superseded")
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

    def check(self) -> None:
        """Read the run; raise `Stop` unless it is running, on this attempt and waiting for this job."""
        state = self.catalog.run()
        self._checked_at = time.monotonic()
        if reason := state.skip_reason(self.name, self.attempt):
            logger.warning(
                "Stopping job %s for run %s: %s (run %s, attempt %s of %s, job %s)",
                self.name,
                self.run_id,
                reason,
                state.status,
                self.attempt,
                state.attempt,
                state.jobs.get(self.name),
            )
            raise Stop(reason)

    def _recheck(self) -> None:
        """`check` again once `RUN_CHECK_INTERVAL` has passed; only `Stop` ends the book.

        A catalog that cannot be read mid-book is logged and the book goes on: it only has to be up to
        take the results, and failing hours of decoding over one missed read would be worse.
        """
        interval = self.settings.run_check_interval
        if interval <= 0 or time.monotonic() - self._checked_at < interval:
            return
        try:
            self.check()
        except Stop:
            raise
        except Exception:
            self._checked_at = time.monotonic()
            logger.warning("Could not read run %s mid-book; decoding on", self.run_id, exc_info=True)

    def _count_delivery(self) -> None:
        """Raise `TooManyDeliveries` once this task id has started more than `MAX_DELIVERIES` times.

        A Redis error is logged and ignored: the counter guards against a loop, it must not fail a book.
        """
        limit = self.settings.max_deliveries
        if self.task_id is None or limit <= 0:
            return
        try:
            count = self.deliveries.record(self.task_id)
        except Exception:
            logger.warning("Could not count the deliveries of task %s", self.task_id, exc_info=True)
            return
        if count > limit:
            raise TooManyDeliveries(
                f"started {count} times, and none of the {count - 1} before finished "
                f"(the worker was killed, OOM?); not decoded again"
            )
        if count > 1:
            logger.warning("Task %s: delivery %d of at most %d", self.task_id, count, limit)

    def provenance(self) -> dict[str, Any]:
        """What produced the results, stored as `run.provenance[<name>]`."""
        provenance: dict[str, Any] = {
            "image": self.settings.image_tag,
            "latex_model": self.recognizer.name,
            "crop_dpi": self.settings.crop_dpi,
        }
        if self.settings.max_crop_pixels:
            provenance["max_crop_pixels"] = self.settings.max_crop_pixels
        if isinstance(self.recognizer, latex.OnnxRecognizer):
            provenance["latex_threads"] = self.recognizer.threads
            provenance["latex_max_tokens"] = self.recognizer.max_tokens
            provenance["latex_repetition_stop"] = self.recognizer.repetition_stop
            provenance["latex_workers"] = self.settings.latex_workers
        return provenance

    def _stop(self, reason: SkipReason) -> Outcome:
        if reason == "superseded":
            # Tells the catalog the work was saved (`superseded{stage="early"}`); the run is left alone.
            try:
                self.catalog.job(self.name, JobReport(status="superseded"))
            except RunNotRunning:
                pass
            except Exception:
                logger.warning("Could not report job %s of run %s superseded", self.name, self.run_id, exc_info=True)
        return self._outcome(reason)

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
        metrics.record_job(self.kind, outcome)
        return outcome

    # -- extraction -------------------------------------------------------------------------------

    def extract(self) -> list[Region]:
        """Every region the analyzer found, in order; one that fails is skipped. Raising fails the job.

        The task thread renders and saves each crop (PyMuPDF is not thread-safe) and
        hands its recognition to `LATEX_WORKERS` threads; ONNX Runtime releases the GIL.
        At most two crops per worker are rendered and not yet recognized, so a long book
        does not pile up images in memory. The bound counts unfinished crops, not
        uncollected ones: a slow crop at the head does not stall the others.

        `Stop` from a re-read of the run ends it early; the crops in flight finish first.
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
                self._recheck()
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
        """Render and save one region's crop; no image for a crop too small to be a formula.

        The crop is logged before it is rendered: if it kills the worker, it is the last line (#6).
        """
        page = document[region.page - 1]
        clip = crops.rect(region.bbox, page)
        dpi = crops.bounded_dpi(clip, self.settings.crop_dpi, self.settings.max_crop_pixels)
        width, height = crops.size(clip, dpi)
        bbox = region.bbox
        logger.info(
            "Crop p%s/%s [%.3f %.3f %.3f %.3f]: %dx%d px at %d dpi%s",
            region.page,
            region.seq,
            bbox.x0,
            bbox.y0,
            bbox.x1,
            bbox.y1,
            width,
            height,
            dpi,
            "" if dpi == self.settings.crop_dpi else f" (lowered from {self.settings.crop_dpi}: MAX_CROP_PIXELS)",
        )
        image = crops.render(page, clip, dpi)
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
