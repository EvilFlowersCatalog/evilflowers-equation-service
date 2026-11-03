import os

import fitz
import pytest

import latex
import settings
import worker


def test_clean_unwraps_what_models_add():
    assert latex.clean("$$ a^2 + b^2 $$") == "a^2 + b^2"
    assert latex.clean(r"\[x\]") == "x"
    assert latex.clean("```latex\n" + r"\frac{1}{2}" + "\n```") == r"\frac{1}{2}"
    assert latex.clean("   ") is None


def test_normalize_where_it_parses():
    pytest.importorskip("sympy")
    pytest.importorskip("antlr4")
    assert latex.normalize(r"\frac{2}{4}") == r"\frac{1}{2}"
    assert latex.normalize(r"\frac{") is None


def test_crop_without_recognizer(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_PATH", str(tmp_path))
    monkeypatch.setattr(settings, "LATEX_BACKEND", "none")
    pdf = fitz.open()
    page = pdf.new_page(width=600, height=800)
    page.insert_text((100, 120), "E = mc^2", fontsize=20)
    region = {"page": 1, "seq": 0, "bbox": {"x0": 0.1, "y0": 0.1, "x1": 0.6, "y1": 0.2}}
    result = worker._equation(pdf, region, "out/v1")
    assert result["image_path"] == "out/v1/p001-eq00.webp"
    assert result["latex"] is None
    assert os.path.exists(tmp_path / "out/v1/p001-eq00.webp")
