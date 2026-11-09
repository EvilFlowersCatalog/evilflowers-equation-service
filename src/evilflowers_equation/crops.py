"""Page geometry and crops (IP-018).

Boxes arrive as page fractions with a top-left origin on the original file's
1-based page, the same convention PyMuPDF uses for page coordinates. Crops are
written as WebP into the run's folder under the shared storage mount, where the
catalog serves and deletes them.
"""

import io
from pathlib import Path

import pymupdf
from PIL import Image

from evilflowers_equation.schemas import Box


def rect(box: Box, page: pymupdf.Page) -> pymupdf.Rect:
    """The box in the page's points."""
    width, height = page.rect.width, page.rect.height
    return pymupdf.Rect(box.x0 * width, box.y0 * height, box.x1 * width, box.y1 * height)


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
