"""LaTeX for one equation crop, and sympy's canonical form where it parses (IP-018).

``recognizer(settings)`` picks the backend from ``LATEX_BACKEND``; the ONNX model
is loaded once per worker process, on the first crop.
"""

import base64
import functools
import io
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar, Protocol
from unittest.mock import patch

import requests
from PIL import Image

from evilflowers_equation.clients.http import session
from evilflowers_equation.observability import annotate, metrics, span
from evilflowers_equation.settings import Settings

logger = logging.getLogger(__name__)

OLLAMA_PROMPT = (
    "Transcribe the mathematical expression in this image to LaTeX. Answer only the LaTeX, without "
    "dollar signs, code fences or explanation."
)


class Recognizer(Protocol):
    #: Recorded in each region's and the job's provenance (`latex_model`).
    name: str

    def __call__(self, image: Image.Image) -> str | None: ...


def png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def clean(latex: str | None) -> str | None:
    """Strip what models wrap LaTeX in: code fences, `$`, `\\[ \\]`, `\\( \\)`."""
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


def normalize(latex: str | None) -> str | None:
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
            logger.debug("sympify could not re-evaluate %r", latex, exc_info=True)
        return sympy.latex(expr)
    except Exception:
        return None


class NoRecognizer:
    """`none`: crops and positions only."""

    name = "none"

    def __call__(self, image: Image.Image) -> str | None:
        return None


class OnnxRecognizer:
    """`onnx`: LaTeX-OCR (pix2tex) on ONNX Runtime, CPU; weights downloaded to `MODELS_PATH` on first use.

    rapid-latex-ocr would download its weights next to the package, which is
    read-only in the pod and lost on every restart. So the weights are fetched
    here into `MODELS_PATH` (the `models` PVC) and handed to it as paths.
    """

    name = "rapid-latex-ocr"
    #: The constructor argument of `rapid_latex_ocr.LaTeXOCR` (0.0.9) → the file in `MODELS_PATH`.
    FILES: ClassVar[dict[str, str]] = {
        "image_resizer_path": "image_resizer.onnx",
        "encoder_path": "encoder.onnx",
        "decoder_path": "decoder.onnx",
        "tokenizer_json": "tokenizer.json",
    }
    WEIGHTS_URL = "https://github.com/RapidAI/RapidLaTeXOCR/releases/download/v0.0.0"

    def __init__(self, settings: Settings, http: requests.Session | None = None):
        self.models_path = settings.models_path
        self.threads = settings.latex_threads
        self.allow_spinning = settings.latex_allow_spinning
        self.http = http or session()
        self._model = None
        self._tokens: _TokenCounter | None = None
        self._lock = threading.Lock()

    def weights(self) -> dict[str, Path]:
        """The weight files in `MODELS_PATH`, downloaded first where missing."""
        self.models_path.mkdir(parents=True, exist_ok=True)
        paths = {}
        for argument, file_name in self.FILES.items():
            path = self.models_path / file_name
            if not path.exists():
                self._download(file_name, path)
            paths[argument] = path
        return paths

    def _download(self, file_name: str, path: Path) -> None:
        url = f"{self.WEIGHTS_URL}/{file_name}"
        logger.info("Downloading %s to %s", url, path)
        with span("latex.model.download", **{"url.full": url}):
            response = self.http.get(url, stream=True, timeout=300)
            response.raise_for_status()
            # A partial file must never look like a model: write aside, then rename.
            partial = path.with_suffix(path.suffix + ".partial")
            with partial.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    file.write(chunk)
            partial.replace(path)

    def _load(self):
        with self._lock:
            if self._model is None:
                with span(
                    "latex.model.load",
                    **{"latex.model": self.name, "latex.threads": self.threads, "latex.spinning": self.allow_spinning},
                ):
                    from rapid_latex_ocr import LaTeXOCR

                    weights = {argument: str(path) for argument, path in self.weights().items()}
                    with self._session_options():
                        model = LaTeXOCR(**weights)
                    self._tokens = model.encoder_decoder = _TokenCounter(model.encoder_decoder)
                    self._model = model
            return self._model

    @contextmanager
    def _session_options(self) -> Iterator[None]:
        """Apply `LATEX_THREADS` and `LATEX_ALLOW_SPINNING` to the sessions `LaTeXOCR` creates.

        rapid_latex_ocr 0.0.9 builds its three sessions (`image_resizer` in `main`,
        `encoder` and `decoder` in `models`) as `OrtInferSession(path)` and exposes no
        thread count, so ORT sizes each pool from the node's cores, not the pod's limit.
        The two modules' `OrtInferSession` is swapped for the duration of the constructor
        only. That is why the package is pinned; `tests/test_latex.py` guards it.
        """
        if self.threads <= 0 and self.allow_spinning:
            yield
            return

        from rapid_latex_ocr import main, models
        from rapid_latex_ocr.utils_load import OrtInferSession

        allow_spinning = self.allow_spinning

        class Session(OrtInferSession):
            def _init_sess_opt(self):
                super()._init_sess_opt()
                if not allow_spinning:
                    self.sess_opt.add_session_config_entry("session.intra_op.allow_spinning", "0")

        factory = functools.partial(Session, num_threads=self.threads if self.threads > 0 else -1)
        with patch.object(main, "OrtInferSession", factory), patch.object(models, "OrtInferSession", factory):
            yield

    def __call__(self, image: Image.Image) -> str | None:
        model = self._load()
        started = time.perf_counter()
        result = model(png(image))
        tokens = self._tokens.last if self._tokens else None
        if tokens is not None:
            annotate(**{"latex.tokens": tokens})
            logger.info("LaTeX-OCR: %d tokens in %.1f s", tokens, time.perf_counter() - started)
        # (latex, elapsed) in rapid_latex_ocr 0.x; a bare string in later versions.
        return clean(result[0] if isinstance(result, tuple | list) else str(result))


class _TokenCounter:
    """Wraps `LaTeXOCR.encoder_decoder` to keep the number of tokens the decoder produced for the last crop.

    The decoder stops at the end token or after `max_seq_len` (512) steps, so the count
    tells a crop that is slow per step from one that ran every step.
    """

    def __init__(self, inner):
        self.inner = inner
        self._local = threading.local()

    @property
    def last(self) -> int | None:
        return getattr(self._local, "tokens", None)

    def __call__(self, x, **kwargs):
        self._local.tokens = None
        output = self.inner(x, **kwargs)
        self._local.tokens = int(output.shape[-1])
        return output


class OllamaRecognizer:
    """`ollama`: a multimodal model behind an Ollama `/api/chat` endpoint transcribes the crop."""

    def __init__(self, settings: Settings, http: requests.Session | None = None):
        self.endpoint = settings.ollama_endpoint
        self.model = settings.latex_model
        self.timeout = settings.latex_timeout
        self.name = f"ollama:{self.model}"
        self.http = http or session(settings.ollama_api_key)

    def __call__(self, image: Image.Image) -> str | None:
        response = self.http.post(
            self.endpoint,
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": OLLAMA_PROMPT,
                        "images": [base64.b64encode(png(image)).decode("ascii")],
                    }
                ],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return clean(response.json().get("message", {}).get("content", ""))


_recognizers: dict[tuple, Recognizer] = {}
_recognizers_lock = threading.Lock()


def recognizer(settings: Settings) -> Recognizer:
    """The backend of `LATEX_BACKEND`, one per process (the ONNX model is loaded once)."""
    key = (
        settings.latex_backend,
        str(settings.models_path),
        settings.latex_threads,
        settings.latex_allow_spinning,
        settings.ollama_endpoint,
        settings.latex_model,
    )
    with _recognizers_lock:
        if key not in _recognizers:
            match settings.latex_backend:
                case "onnx":
                    _recognizers[key] = OnnxRecognizer(settings)
                case "ollama":
                    _recognizers[key] = OllamaRecognizer(settings)
                case _:
                    _recognizers[key] = NoRecognizer()
        return _recognizers[key]


def recognize(model: Recognizer, image: Image.Image) -> str | None:
    """Run the recognizer on one crop, timed and traced; a failure is logged and gives None."""
    started = time.perf_counter()
    outcome = "failed"
    try:
        with span("latex.recognize", **{"latex.model": model.name}):
            latex = model(image)
        outcome = "recognized" if latex else "empty"
        return latex
    except Exception:
        logger.exception("LaTeX recognition failed")
        return None
    finally:
        metrics.record_recognition(model.name, outcome, time.perf_counter() - started)
