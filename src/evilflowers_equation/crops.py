"""Page geometry and crops (IP-018).

Boxes arrive as page fractions with a top-left origin on the original file's
1-based page, the same convention PyMuPDF uses for page coordinates. Crops are
written as WebP into the run's folder under the shared storage mount, where the
catalog serves and deletes them.

Two page sources: a PDF page (PyMuPDF, the catalog task) and a leaf image (Pillow,
the standalone task, #8). The fractions mean the same on both.
"""

import io
import math
from collections.abc import Sequence
from pathlib import Path

import pymupdf
from evilflowers_worker.contract import Box
from PIL import Image

#: Pixels of context around a leaf crop: the vlm service was validated on crops cut like this (`recognition.PAD`).
VLM_PAD = 10
#: The vlm service's own bounds (`recognition.MAX_PIXELS`, `MAX_ASPECT`); it applies them again on arrival.
VLM_MAX_PIXELS = 1280 * 28 * 28
VLM_MAX_ASPECT = 10.0


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


def pixel_box(bbox: Sequence[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Fractions `[x0, y0, x1, y1]` (top-left origin) of an image of `size` → its pixel box, clamped to the image."""
    width, height = size
    x0, y0, x1, y1 = bbox
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))

    def clamp(value: float, limit: int) -> int:
        return min(max(round(value * limit), 0), limit)

    return clamp(left, width), clamp(top, height), clamp(right, width), clamp(bottom, height)


def crop_leaf(page: Image.Image, bbox: Sequence[float], pad: int = VLM_PAD) -> Image.Image:
    """The region of a leaf image with `pad` pixels of context, cut as the vlm service cuts it (`crop_from`)."""
    x0, y0, x1, y1 = pixel_box(bbox, page.size)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty box {list(bbox)} on a {page.width}x{page.height} page")
    box = (max(0, x0 - pad), max(0, y0 - pad), min(page.width, x1 + pad), min(page.height, y1 + pad))
    return page.crop(box).convert("RGB")


def for_vlm(image: Image.Image) -> Image.Image:
    """White-padded to a sane aspect ratio and size, downscaled to `VLM_MAX_PIXELS` (`recognition.normalize`).

    The service does this again, but a whole-page crop of a 600 dpi scan would not fit its 8 MB per image.
    """
    image = image.convert("RGB")
    width, height = image.size
    target = (max(width, 64, int(height / VLM_MAX_ASPECT)), max(height, 64, int(width / VLM_MAX_ASPECT)))
    if target != (width, height):
        canvas = Image.new("RGB", target, "white")
        canvas.paste(image, ((target[0] - width) // 2, (target[1] - height) // 2))
        image = canvas
    if image.width * image.height > VLM_MAX_PIXELS:
        scale = math.sqrt(VLM_MAX_PIXELS / (image.width * image.height))
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    return image


def save(image: Image.Image, path: Path, quality: int) -> None:
    """Write the crop to `path` (resolved under the storage root by `RuntimeSettings.storage_file`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "WEBP", quality=quality, method=4)
