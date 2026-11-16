"""Quality checks of a transcribed formula (#8): repetition, rendered-vs-crop similarity, two-model agreement."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from evilflowers_equation import checks, crops


def test_repetition_finds_a_loop():
    assert checks.repetition(r"x = " + r"\qquad" * 80, 64)
    assert checks.repetition("e_{A}," * 40, 64)
    assert checks.repetition(r"\frac{a}{b} + c", 64) is None
    assert checks.repetition(r"x" * 200, 0) is None


def test_ssim_of_an_image_with_itself_is_one():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (40, 60)).astype(np.uint8)
    assert checks.ssim(image, image) == pytest.approx(1.0)
    assert checks.ssim(image, 255 - image) < 0


def test_ink_cuts_to_the_dark_pixels():
    image = Image.new("L", (100, 50), 255)
    ImageDraw.Draw(image).rectangle((20, 10, 39, 29), fill=0)
    assert checks.ink(image).size == (20, 20)
    assert checks.ink(Image.new("L", (10, 10), 255)) is None


def test_canonical_ignores_spacing_and_sizing():
    assert checks.canonical(r"\left( a \, + b \right)") == checks.canonical("(a+b)")


def test_agreement():
    assert checks.agreement(r"E = mc^2", r"E=m c^{2}") == 1.0
    assert checks.agreement(r"E = mc^2", r"\int_0^1 f(t)\,dt") < 0.5
    assert checks.disagree(r"E = mc^2", None, "onnx", 0.8) is None


def test_pixel_box_orders_and_clamps():
    assert crops.pixel_box((0.6, 0.5, 0.1, 0.4), (1000, 1000)) == (100, 400, 600, 500)
    assert crops.pixel_box((0.0, 0.0, 1.0, 1.0), (10, 20)) == (0, 0, 10, 20)


def test_for_vlm_bounds_a_huge_crop():
    image = crops.for_vlm(Image.new("RGB", (5000, 7000), "white"))
    assert image.width * image.height <= crops.VLM_MAX_PIXELS
    sliver = crops.for_vlm(Image.new("RGB", (3, 140), "white"))
    assert sliver.size == (64, 140)


matplotlib = pytest.importorskip("matplotlib", reason="the render check needs the `check` extra")


def test_a_rendered_formula_resembles_its_own_render():
    printed = checks.render(r"\frac{a+b}{c}")
    assert printed is not None
    same = checks.similarity(r"\frac{a+b}{c}", printed)
    other = checks.similarity(r"\sum_{i=1}^{n} x_i^2 \cdot y", printed)
    assert same == pytest.approx(1.0, abs=0.05)
    assert other < same


def test_what_mathtext_cannot_render_is_unchecked():
    crop = Image.new("RGB", (100, 40), "white")
    ImageDraw.Draw(crop).text((5, 5), "x", fill="black")
    assert checks.render(r"\begin{pmatrix} a \end{pmatrix}") is None
    assert checks.rendered_unlike(r"\begin{pmatrix} a \end{pmatrix}", crop, 0.9) is None


def test_a_low_score_is_disputed():
    printed = checks.render(r"\frac{a+b}{c}")
    assert checks.rendered_unlike(r"\frac{a+b}{c}", printed, 0.5) is None
    assert "looks unlike" in checks.rendered_unlike(r"\sum_{i=1}^{n} x_i^2 \cdot y", printed, 0.99)
