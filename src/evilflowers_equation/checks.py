"""Quality checks of a transcribed formula (#8, digitalizer design §6): they mark an answer `disputed`, never edit it.

- **Repetition** (always on): the answer ends in a loop (`repeats`, the stop the ONNX decoder already has), the way
  a VLM runs away on a crop it cannot read.
- **Render** (`FORMULAS_MIN_SIMILARITY` > 0): the LaTeX is rendered with matplotlib's mathtext and compared with the
  crop by structural similarity. mathtext reads a subset of LaTeX, so a formula it cannot render is *unchecked*, not
  disputed. The threshold is not calibrated yet, hence off by default: the digitalizer's review screen is meant to
  measure it first.
- **Cross-check** (`FORMULAS_CROSSCHECK`): LaTeX-OCR reads the crop too; two readings that differ are disputed.

Each check answers a reason, or None when it has nothing against the answer.
"""

from __future__ import annotations

import difflib
import io
import logging
import re

import numpy as np
from PIL import Image, ImageOps

from evilflowers_equation import latex

logger = logging.getLogger(__name__)

#: LaTeX tokens, roughly: a command, an escaped character, or any other single non-space character.
TOKEN = re.compile(r"\\[A-Za-z]+|\\.|\S")
#: Side of the square window the similarity is computed over.
SSIM_WINDOW = 7
#: Spacing and sizing commands that change how a formula looks, not what it says.
LAYOUT = re.compile(r"\\(left|right|big|Big|bigg|Bigg|displaystyle|textstyle|[,;:! ])|\s+")


def repetition(text: str, span: int) -> str | None:
    """A reason when the last `span` tokens of `text` repeat a unit of 1 to 8 tokens."""
    period = latex.repeats(TOKEN.findall(text), span)
    return f"ends in a loop (a {period}-token unit repeated over {span} tokens)" if period else None


# -- Render ---------------------------------------------------------------------------------------------------------


def render(text: str, dpi: int = 200) -> Image.Image | None:
    """The formula as matplotlib's mathtext draws it, black on white; None when mathtext cannot."""
    try:
        from matplotlib import mathtext
    except ImportError:
        logger.warning("The render check needs matplotlib (the `check` extra); formulas stay unchecked")
        return None
    buffer = io.BytesIO()
    try:
        mathtext.math_to_image(f"${text}$", buffer, dpi=dpi, format="png")
    except Exception:
        logger.debug("mathtext cannot render %r", text, exc_info=True)
        return None
    buffer.seek(0)
    with Image.open(buffer) as image:
        image.load()
        background = Image.new("RGBA", image.size, "white")
        return Image.alpha_composite(background, image.convert("RGBA")).convert("L")


def ink(image: Image.Image) -> Image.Image | None:
    """The image in grey, cut to its ink (pixels darker than the midpoint of its range); None when it has none."""
    grey = ImageOps.grayscale(image)
    lo, hi = grey.getextrema()
    if hi - lo < 32:
        return None
    mask = grey.point(lambda value: 255 if value < (lo + hi) / 2 else 0)
    box = mask.getbbox()
    return grey.crop(box) if box else None


def ssim(a: np.ndarray, b: np.ndarray, window: int = SSIM_WINDOW) -> float:
    """Mean structural similarity of two grey images of one shape (0..255), over square windows (Wang et al. 2004)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def mean(x: np.ndarray) -> np.ndarray:
        # Box filter over every full window, from a summed-area table.
        table = np.pad(x, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        w = window
        return (table[w:, w:] - table[:-w, w:] - table[w:, :-w] + table[:-w, :-w]) / (w * w)

    if min(a.shape) < window:
        raise ValueError(f"images of {a.shape} are smaller than the {window}px window")
    mu_a, mu_b = mean(a), mean(b)
    var_a = mean(a * a) - mu_a**2
    var_b = mean(b * b) - mu_b**2
    cov = mean(a * b) - mu_a * mu_b
    index = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return float(index.mean())


def similarity(text: str, crop: Image.Image) -> float | None:
    """How alike the rendered LaTeX and the crop look (SSIM of their ink, at the crop's size); None: not comparable."""
    drawn = render(text)
    if drawn is None:
        return None
    printed, rendered = ink(crop), ink(drawn)
    if printed is None or rendered is None or min(printed.size) < SSIM_WINDOW:
        return None
    rendered = rendered.resize(printed.size, Image.Resampling.LANCZOS)
    return ssim(np.asarray(printed), np.asarray(rendered))


def rendered_unlike(text: str, crop: Image.Image, threshold: float) -> str | None:
    score = similarity(text, crop)
    if score is None or score >= threshold:
        return None
    return f"rendered LaTeX looks unlike the crop (similarity {score:.2f} < {threshold:.2f})"


# -- Cross-check ----------------------------------------------------------------------------------------------------


def canonical(text: str) -> str:
    """The LaTeX with spacing and delimiter sizing removed: what two readings of one formula should share."""
    return LAYOUT.sub("", text)


def agreement(a: str, b: str) -> float:
    """How alike two readings are, 0..1: sympy's forms where both parse, else their canonical LaTeX."""
    sympy_a, sympy_b = latex.normalize(a), latex.normalize(b)
    if sympy_a is not None and sympy_b is not None:
        a, b = sympy_a, sympy_b
    return difflib.SequenceMatcher(None, canonical(a), canonical(b)).ratio()


def disagree(text: str, other: str | None, other_name: str, min_ratio: float) -> str | None:
    if not other:
        return None
    ratio = agreement(text, other)
    if ratio >= min_ratio:
        return None
    return f"{other_name} reads it differently (agreement {ratio:.2f} < {min_ratio:.2f}): {other[:200]}"
