"""The IP-018 `equations` job: crop each formula region, transcribe it to LaTeX, post it.

The lifecycle is the kit's ``LeafJob`` (``evilflowers_worker.jobs``): the delivery cap, the read of the
run before the work (``already_done`` / ``not_running`` / ``superseded``), ``extract``, ``publish`` and
the terminal report. This module is only the work: crops, the LaTeX thread pool and the provenance.

The analyzer sends either one task per book, reported as `equations`, or one per layout batch with a
`job` kwarg (`equations:0001-0008`, #5), reported under that name. Metrics keep the kind, `equations`,
so their labels do not grow with page count.

A single region that fails is logged and skipped; the job still reports `done` with the others. The
run is read again every `RUN_CHECK_INTERVAL` while the crops are decoded (``recheck``), and the job
stops there when the run moves on.
"""

import contextvars
import logging
import math
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import pymupdf
from evilflowers_worker.catalog import CatalogClient
from evilflowers_worker.contract import LayoutRegion, Region, RegionsJob
from evilflowers_worker.jobs import LeafJob
from evilflowers_worker.observability import span
from PIL import Image

from evilflowers_equation import crops, latex, metrics
from evilflowers_equation.latex import Recognizer
from evilflowers_equation.settings import Settings

logger = logging.getLogger(__name__)


class EquationsJob(LeafJob[RegionsJob, list[Region]]):
    #: The kind of job: its name in the catalog when the analyzer names no batch, and its metric label.
    kind = "equations"

    def __init__(
        self,
        message: RegionsJob,
        *,
        settings: Settings,
        catalog: CatalogClient | None = None,
        task_id: str | None = None,
        delivery: int = 1,
        recognizer: Recognizer | None = None,
    ):
        interval = settings.run_check_interval
        super().__init__(
            message,
            settings=settings,
            catalog=catalog,
            task_id=task_id,
            delivery=delivery,
            # RUN_CHECK_INTERVAL=0 means "only before the work", not "before every crop".
            recheck_interval=interval if interval > 0 else math.inf,
        )
        self.settings: Settings = settings
        self.path = settings.source_path(message.source)
        self.recognizer = recognizer or latex.recognizer(settings)

    # -- protocol ---------------------------------------------------------------------------------

    def publish(self, result: list[Region]) -> None:
        self.catalog.regions(result)
        metrics.DOCUMENT_REGIONS.record(len(result))
        logger.info("Run %s: %d of %d equations", self.message.run_id, len(result), len(self.message.regions))

    def provenance(self) -> dict[str, Any]:
        """What produced the results, stored as `run.provenance[<name>]`."""
        provenance: dict[str, Any] = {
            **super().provenance(),
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
        regions = self.message.regions
        # Refuses an `output_dir` that escapes the storage root here, as a failed job, rather than
        # crop by crop (each crop's error is only logged, and the job would report `done` with none).
        self.settings.storage_file(self.message.output_dir)
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
                self.recheck()
                unfinished.acquire()
                try:
                    name, image = self.crop(document, region)
                except Exception:
                    unfinished.release()
                    logger.exception("Run %s: equation p%s/%s skipped", self.message.run_id, region.page, region.seq)
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
        name = crops.crop_name(self.message.output_dir, region.page, region.seq)
        if min(image.size) < self.settings.min_crop_pixels:
            return name, None
        crops.save(image, self.settings.storage_file(name), self.settings.webp_quality)
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
            logger.exception("Run %s: equation p%s/%s skipped", self.message.run_id, region.page, region.seq)
            return None
        if normalized and normalized != source_latex:
            # The model's LaTeX is what is shown; sympy's form is kept for comparison and search.
            provenance["sympy"] = normalized
        return Region(
            kind="equation",
            page=region.page,
            seq=region.seq,
            bbox=region.bbox,
            image_path=name,
            latex=source_latex,
            provenance=provenance,
        )
