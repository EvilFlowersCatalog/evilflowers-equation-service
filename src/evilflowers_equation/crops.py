"""Page geometry and crops (IP-018).

Boxes arrive as page fractions with a top-left origin on the original file's
1-based page, the same convention PyMuPDF uses for page coordinates. Crops are
written as WebP into the run's folder under the shared storage mount, where the
catalog serves and deletes them.
"""

import io
import math
from pathlib import Path

import pymupdf
from PIL import Image

from evilflowers_equation.schemas import Box


def rect(box: Box, page: pymupdf.Page) -> pymupdf.Rect:
    """The box in the page's points."""
    width, height = page.rect.width, page.rect.height
    return pymupdf.Rect(box.x0 * width, box.y0 * height, box.x1 * width, box.y1 * height)


def size(clip: pymupdf.Rect, dpi: int) -> tuple[int, int]:
    """Width and height in pixels of `clip` rendered at `dpi`."""
    scale = dpi / 72
    return round(clip.width * scale), round(clip.height * scale)


def bounded_dpi(clip: pymupdf.Rect, dpi: int, max_pixels: int) -> int:
    """`dpi`, lowered so that `clip` renders to at most `max_pixels` pixels (0: no bound).

    Worked out before rendering: the pixmap, its PNG and the PIL image of an oversized
    box are what spikes the worker's memory past its limit (#6).
    """
    width, height = size(clip, dpi)
    if max_pixels <= 0 or width * height <= max_pixels:
        return dpi
    return max(1, math.floor(dpi * math.sqrt(max_pixels / (width * height))))


def render(page: pymupdf.Page, clip: pymupdf.Rect, dpi: int) -> Image.Image:
    pixmap = page.get_pixmap(clip=clip, dpi=dpi, alpha=False)
    return Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")


def crop_name(output_dir: str, page: int, seq: int) -> str:
    """Storage name of a crop, relative to the storage root: `<output_dir>/p012-eq03.webp`."""
    return f"{output_dir.rstrip('/')}/p{page:03d}-eq{seq:02d}.webp"


def save(image: Image.Image, storage_path: Path, name: str, quality: int) -> None:
    """Write the crop under the storage root."""
    path = storage_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "WEBP", quality=quality, method=4)
