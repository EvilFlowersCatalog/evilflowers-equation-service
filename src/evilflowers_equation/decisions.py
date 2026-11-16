"""Is the crop a formula at all? Only for doubtful ones: decider-vision is asked (#9, knowledge#76).

Layout marks text lines, list markers, chemical labels and noise as formulas, and the recognizer transcribes them
anyway. Two steps, so the scarce decider-vision (~17 s a call, one replica) sees only what a cheap filter flags:

1. **Doubt** (no model, every kept formula): the LaTeX does not parse in sympy, is shorter than ``MIN_TOKENS``
   tokens, or a second reading of the crop disagrees with it (only when two backends ran). A crop with no LaTeX at
   all is ``unread``, not doubtful: a recognizer that is down, or ``LATEX_BACKEND=none``, must not send every crop to
   the judge. ``equation_formulas_total{outcome}`` counts doubtful, clean and unread, judges on or off, so the share
   is known before any call is made.
2. **Judge** (doubtful only): one ``Choice`` (``formula_kind``) to decider-vision through the kit's ``Panel``. Its
   answer goes on the region as ``decision``, ``p`` and ``decider_model`` (EvilFlowersCatalog#189); ``band`` only once
   ``JUDGE_THRESHOLDS`` has ``formula_kind`` (knowledge#65). No answer: the region is posted as before.
"""

from __future__ import annotations

import io

from evilflowers_worker.judges import Choice, Judges, JudgeSettings
from PIL import Image

from evilflowers_equation import checks, latex
from evilflowers_equation.settings import get_settings

#: The question asked of decider-vision; ``formula_kind`` keys its thresholds and its metrics.
FORMULA_KIND = Choice(
    id="formula_kind",
    instructions="What is this image?",
    criteria={
        "formula": "mathematical expression",
        "text": "ordinary words",
        "figure": "drawing or chart",
        "table": "table",
        "noise": "dots, lines, smudges",
    },
    # What the catalog hides above the threshold, once there is one.
    reject={"text", "noise"},
)
STATE = "A region a layout model marked as a formula, cropped from a page of a publication."

#: LaTeX shorter than this many tokens is doubtful (a list marker, a stray letter).
MIN_TOKENS = 3


def doubts(
    source_latex: str | None,
    normalized: str | None,
    *,
    other: str | None = None,
    other_name: str = "",
    min_ratio: float = 0.8,
) -> list[str] | None:
    """Why the formula is doubtful (empty: clean), or None when there is no LaTeX to doubt.

    ``normalized`` is sympy's form (``latex.normalize``), None when it does not parse. ``other`` is a second backend's
    reading of the same crop, when one ran.
    """
    if not source_latex:
        return None
    reasons = []
    if normalized is None:
        reasons.append("sympy cannot parse it")
    if (tokens := len(checks.TOKEN.findall(source_latex))) < MIN_TOKENS:
        reasons.append(f"{tokens} token{'s' if tokens != 1 else ''} (< {MIN_TOKENS})")
    if reason := checks.disagree(source_latex, latex.clean(other), other_name, min_ratio):
        reasons.append(reason)
    return reasons


def png(image: Image.Image, max_side: int) -> bytes:
    """The image as PNG, its longer side at most ``max_side`` (sent to the judge; decider-vision scales to 448)."""
    image = image.copy()
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


_process_judges: Judges | None = None


def judges_for(settings: JudgeSettings) -> Judges:
    """The process's judges for its own settings (their circuit breakers live as long as the process); fresh ones for
    any other settings object (tests)."""
    global _process_judges
    if settings is not get_settings():
        return Judges.from_settings(settings)
    if _process_judges is None:
        _process_judges = Judges.from_settings(settings)
    return _process_judges
