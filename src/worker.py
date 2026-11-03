"""Equations of an extraction run (IP-018).

`evilflowers_equation.equations` receives the formula regions the analyzer's
layout pass found (region detection is the analyzer's, Q11), crops each one,
transcribes it to LaTeX and normalizes it with sympy where it parses. Every
page of the document is covered; a region that fails is logged and skipped.
"""

import logging
import os
from typing import List, Optional

import fitz
from celery import Celery

import crops
import latex
import settings
from catalog import Catalog, RunNotRunning

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("equation")

JOB = "equations"

app = Celery("evilflowers_equation", broker=settings.REDIS_URL)
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    timezone="Europe/Bratislava",
    enable_utc=True,
    worker_send_task_events=True,
    task_send_sent_event=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)


@app.task(name="evilflowers_equation.equations", queue=settings.QUEUE)
def equations(run_id: str, source: str, output_dir: str, regions: List[dict], **_):
    catalog = Catalog(run_id)
    try:
        document = fitz.open(os.path.join(settings.STORAGE_PATH, source))
    except Exception as error:
        logger.exception("Cannot open %s", source)
        catalog.job(JOB, failed=True, error=f"Cannot open the PDF: {error}")
        return

    kept = []
    with document:
        for region in regions:
            try:
                result = _equation(document, region, output_dir)
            except Exception:
                logger.exception("Run %s: equation on page %s skipped", run_id, region.get("page"))
                continue
            if result:
                kept.append(result)

    try:
        catalog.regions(kept)
        catalog.job(JOB, provenance={"image": settings.IMAGE_TAG, "latex_model": latex.model_name()})
    except RunNotRunning:
        logger.warning("Run %s is not accepting results; %d equations dropped", run_id, len(kept))
        return
    logger.info("Run %s: %d equations", run_id, len(kept))


def _equation(document: fitz.Document, region: dict, output_dir: str) -> Optional[dict]:
    page = document[region["page"] - 1]
    image = crops.render(page, crops.rect(region["bbox"], page), settings.CROP_DPI)
    if min(image.size) < 4:
        return None

    name = crops.crop_name(output_dir, region["page"], "eq", region["seq"])
    crops.save(image, name)

    source_latex = None
    try:
        source_latex = latex.recognize(image)
    except Exception:
        logger.exception("LaTeX recognition failed on page %s", region["page"])

    provenance = {"latex_model": latex.model_name()}
    normalized = latex.normalize(source_latex)
    if normalized and normalized != source_latex:
        # The model's LaTeX is what is shown; sympy's form is kept for comparison and search.
        provenance["sympy"] = normalized

    return {
        "kind": "equation",
        "page": region["page"],
        "seq": region["seq"],
        "bbox": region["bbox"],
        "image_path": name,
        "latex": source_latex,
        "provenance": provenance,
    }
