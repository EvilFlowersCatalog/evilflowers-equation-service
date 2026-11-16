"""`evilflowers_equation.standalone.formulas`: formulas of leaf images for the books digitalizer, no catalog (#8).

The message is a ``FormulasJob``, the kit's ``StandaloneJob`` (evilflowers-worker#5) plus ``regions``::

    {"key": "region-1734", "leaf": 3, "image": "digitized/…/0003.tif",
     "bbox": [0.08, 0.41, 0.62, 0.47], "task": "formula"}

- ``bbox``: fractions of the leaf image, origin top-left, ``x0, y0, x1, y1`` (as the analyzer reports regions).
- ``task``: ``formula`` (LaTeX) or ``ocr`` (a table, or a line of prose with inline math: text).
- ``image`` must be one of ``inputs``.

Each region is cropped from its leaf (each leaf is opened once) and transcribed by PaddleOCR-VL through the shared vlm
service. ``formulas.json`` in ``output_dir`` gets one row per region, in message order::

    {"key": "region-1734", "text": "E = mc^2", "model": "paddleocr-vl:<revision>"}

``text: null`` means "no answer"; the digitalizer then keeps the region as an image crop. A row may also carry
``"disputed": true`` and a ``"reason"`` (``checks``). The digitalizer judges every answer again with its own guards.

Then the outcome file, last and atomically: ``{run_id, token, status, error, formulas}``. Any error writes
``status: "failed"`` and raises, so the orchestrator's ``link_error`` fires and the traceback stays in the task. So
does a job past ``MAX_DELIVERIES`` (``give_up``). A message that does not validate raises without an outcome (the kit's
default). The task returns ``{"job", "answered"}``: what the orchestrator's ``link`` callback receives.
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from evilflowers_worker.contract import StandaloneJob
from evilflowers_worker.observability import span
from evilflowers_worker.outcome import OutcomeFile, atomic_write, write_outcome
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from evilflowers_equation import checks, crops, latex, metrics
from evilflowers_equation.latex import VlmItem
from evilflowers_equation.settings import Settings, get_settings

logger = logging.getLogger(__name__)

#: The file the rows go to, in ``output_dir``.
FORMULAS_FILE = "formulas.json"


class FormulaRegion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    #: The orchestrator's region key (``region-<id>``), echoed in the row.
    key: str = Field(min_length=1, max_length=256)
    #: The leaf number; informative only.
    leaf: int | None = None
    #: The leaf image, relative to ``STORAGE_PATH``; one of ``inputs``.
    image: str = Field(min_length=1)
    #: Fractions of the image, origin top-left: ``x0, y0, x1, y1``. Not range-checked: a box that overshoots the
    #: page by a rounding error is clamped to it (``crops.pixel_box``), not a reason to fail 32 leaves.
    bbox: tuple[float, float, float, float]
    task: latex.VlmTask = "formula"


class FormulasJob(StandaloneJob):
    """``evilflowers_equation.standalone.formulas``: the kit's standalone job plus the regions to transcribe."""

    regions: list[FormulaRegion] = Field(default_factory=list)

    @field_validator("regions")
    @classmethod
    def _unique_keys(cls, value: list[FormulaRegion]) -> list[FormulaRegion]:
        keys = [region.key for region in value]
        if len(keys) != len(set(keys)):
            raise ValueError("region keys must be unique")
        return value


class FormulasOutcome(OutcomeFile):
    status: Literal["done", "failed"]
    error: str | None = None
    #: ``formulas.json``, relative to ``STORAGE_PATH``; set when ``done``.
    formulas: str | None = None


def run(message: FormulasJob, delivery: int, *, settings: Settings | None = None, vlm=None, crosscheck=None) -> dict:
    """The task's work (``runtime.run_task``'s handler): rows, then the outcome. Raises after a ``failed`` outcome."""
    settings = settings or get_settings()
    try:
        message.prepare_output(settings)
        rows = transcribe(message, settings, vlm=vlm, crosscheck=crosscheck)
        path = message.output_file(settings, FORMULAS_FILE)
        atomic_write(path, json.dumps(rows, ensure_ascii=False).encode())
        relative = path.relative_to(settings.storage_path.resolve()).as_posix()
        write_outcome(
            settings,
            message.outcome,
            FormulasOutcome(run_id=message.run_id, token=message.token, status="done", formulas=relative),
        )
    except Exception as exc:
        logger.exception("Formulas job %s of run %s failed", message.job, message.run_id)
        _failed(settings, message, f"{type(exc).__name__}: {exc}")
        raise
    answered = sum(row["text"] is not None for row in rows)
    logger.info("Formulas job %s of run %s: %d of %d answered", message.job, message.run_id, answered, len(rows))
    return {"job": message.job, "answered": answered}


def give_up(message: FormulasJob, delivery: int) -> dict:
    """Past ``MAX_DELIVERIES``: the job killed its worker every time. A ``failed`` outcome, then raise."""
    error = f"Gave up after {delivery - 1} deliveries: the worker died or was stopped during each of them"
    _failed(get_settings(), message, error)
    raise RuntimeError(error)


def _failed(settings: Settings, message: FormulasJob, error: str) -> None:
    try:
        write_outcome(
            settings,
            message.outcome,
            FormulasOutcome(run_id=message.run_id, token=message.token, status="failed", error=error[:2000]),
        )
    except Exception:
        # The task still raises, so the orchestrator hears of it through link_error.
        logger.exception("Could not write the failed outcome of job %s", message.job)


def transcribe(message: FormulasJob, settings: Settings, *, vlm=None, crosscheck=None) -> list[dict[str, Any]]:
    """One row per region, in message order. Raises when a leaf cannot be read or the vlm stays unavailable."""
    vlm = vlm or latex.recognizer(settings.model_copy(update={"latex_backend": "paddleocr-vl"}))
    if crosscheck is None and settings.formulas_crosscheck:
        crosscheck = latex.recognizer(settings.model_copy(update={"latex_backend": "onnx"}))
    with span("formulas.extract", **{"equations.regions": len(message.regions)}) as current:
        crop_of = cut(message, settings)
        started = time.perf_counter()
        items = [
            VlmItem(region.key, crop_of[region.key], region.task)
            for region in message.regions
            if region.key in crop_of
        ]
        answers = vlm.recognize(items) if items else {}
        seconds = (time.perf_counter() - started) / max(len(items), 1)
        rows = []
        for region in message.regions:
            text = tidy(answers.get(region.key), region.task)
            row: dict[str, Any] = {"key": region.key, "text": text, "model": vlm.name}
            if text is not None:
                reason = judge(text, region, crop_of[region.key], settings, crosscheck)
                if reason:
                    row.update(disputed=True, reason=reason)
            if region.key in crop_of:
                outcome = "recognized" if text else "empty"
                metrics.record_recognition(vlm.name, outcome, seconds)
                metrics.REGIONS.add(1, {"outcome": "kept" if text else "failed"})
            else:
                metrics.REGIONS.add(1, {"outcome": "skipped"})
            rows.append(row)
        if current is not None:
            current.set_attribute("equations.kept", sum(row["text"] is not None for row in rows))
    return rows


def cut(message: FormulasJob, settings: Settings) -> dict[str, Image.Image]:
    """Each region's crop by key, every leaf opened once. A region whose box is empty is left out (no answer).

    Each crop is bounded at once (``crops.for_vlm``, about 1 MP): a job holds all its crops until the rows are
    written, and a page-sized region of a 600 dpi leaf would otherwise be ~100 MB (#6). The checks see this crop too.
    """
    inputs = set(message.inputs)
    by_image: dict[str, list[FormulaRegion]] = defaultdict(list)
    for region in message.regions:
        if region.image not in inputs:
            raise ValueError(f"region {region.key}: image {region.image!r} is not one of the job's inputs")
        by_image[region.image].append(region)
    crop_of: dict[str, Image.Image] = {}
    for relative, regions in by_image.items():
        path: Path = settings.storage_file(relative)
        with Image.open(path) as page:
            page.load()
            for region in regions:
                try:
                    crop_of[region.key] = crops.for_vlm(crops.crop_leaf(page, region.bbox))
                except ValueError:
                    logger.warning("Region %s skipped: empty box %s on %s", region.key, list(region.bbox), relative)
    return crop_of


def tidy(text: str | None, task: latex.VlmTask) -> str | None:
    """A formula's LaTeX through `latex.clean`; an `ocr` answer only stripped (`clean` would eat its `$…$`)."""
    if task == "formula":
        return latex.clean(text)
    return (text or "").strip() or None


def judge(text: str, region: FormulaRegion, crop: Image.Image, settings: Settings, crosscheck) -> str | None:
    """Why the answer is disputed (`checks`), or None. Never raises: a check that fails leaves the answer unchecked."""
    try:
        if reason := checks.repetition(text, settings.latex_repetition_stop):
            return reason
        if region.task != "formula":
            return None
        if settings.formulas_min_similarity > 0 and (
            reason := checks.rendered_unlike(text, crop, settings.formulas_min_similarity)
        ):
            return reason
        if crosscheck is not None:
            other = latex.recognize(crosscheck, crop)
            return checks.disagree(text, other, crosscheck.name, settings.formulas_crosscheck_min_ratio)
    except Exception:
        logger.exception("Quality check of region %s failed; left unchecked", region.key)
    return None
