"""LaTeX for one equation crop, and sympy normalization where it parses (IP-018).

The recognizer is chosen by `LATEX_BACKEND` and loaded once per worker process.
"""

import base64
import io
import logging
import os
from typing import Optional

import requests
from PIL import Image

import settings

logger = logging.getLogger(__name__)

_onnx = None

OLLAMA_PROMPT = (
    "Transcribe the mathematical expression in this image to LaTeX. Answer only the LaTeX, without "
    "dollar signs, code fences or explanation."
)


def _onnx_model():
    global _onnx
    if _onnx is None:
        from rapid_latex_ocr import LaTeXOCR

        os.makedirs(settings.MODELS_PATH, exist_ok=True)
        try:
            _onnx = LaTeXOCR(model_dir=settings.MODELS_PATH)
        except TypeError:  # versions without `model_dir` keep the weights next to the package
            _onnx = LaTeXOCR()
    return _onnx


def clean(latex: Optional[str]) -> Optional[str]:
    """Strip what models wrap LaTeX in: code fences, `$`, `\\[ \\]`."""
    if not latex:
        return None
    text = latex.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("\n") + 1 :] if "\n" in text else text
        text = text.removeprefix("latex").strip()
    for left, right in (("$$", "$$"), ("\\[", "\\]"), ("$", "$"), ("\\(", "\\)")):
        if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            text = text[len(left) : -len(right)].strip()
            break
    return text or None


def recognize(image: Image.Image) -> Optional[str]:
    if settings.LATEX_BACKEND == "none":
        return None
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    if settings.LATEX_BACKEND == "ollama":
        return clean(_ollama(buffer.getvalue()))

    result = _onnx_model()(buffer.getvalue())
    # (latex, elapsed) in rapid_latex_ocr 0.x; a bare string in later versions.
    return clean(result[0] if isinstance(result, (tuple, list)) else str(result))


def _ollama(png: bytes) -> str:
    headers = {"Authorization": f"Bearer {settings.OLLAMA_API_KEY}"} if settings.OLLAMA_API_KEY else {}
    response = requests.post(
        settings.OLLAMA_ENDPOINT,
        json={
            "model": settings.LATEX_MODEL,
            "stream": False,
            "messages": [
                {"role": "user", "content": OLLAMA_PROMPT, "images": [base64.b64encode(png).decode("ascii")]}
            ],
        },
        headers=headers,
        timeout=settings.LATEX_TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "")


def normalize(latex: Optional[str]) -> Optional[str]:
    """sympy's canonical LaTeX of the expression, or None when it does not parse."""
    if not latex:
        return None
    try:
        import sympy
        from sympy.parsing.latex import parse_latex

        expr = parse_latex(latex)
        # parse_latex keeps the expression unevaluated (2/4 stays 2/4) since sympy 1.14.
        # Re-evaluating through sympify canonicalizes arithmetic without computing
        # integrals or sums, which doit() would.
        try:
            expr = sympy.sympify(str(expr))
        except Exception:
            pass
        return sympy.latex(expr)
    except Exception:
        return None


def model_name() -> str:
    if settings.LATEX_BACKEND == "ollama":
        return f"ollama:{settings.LATEX_MODEL}"
    if settings.LATEX_BACKEND == "onnx":
        return "rapid-latex-ocr"
    return "none"
