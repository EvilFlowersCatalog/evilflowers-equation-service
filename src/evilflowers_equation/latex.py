"""LaTeX for one equation crop, and sympy's canonical form where it parses (IP-018).

``recognizer(settings)`` picks the backend from ``LATEX_BACKEND``; the ONNX model
is loaded once per worker process, on the first crop.
"""

from __future__ import annotations

import base64
import functools
import io
import logging
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, Protocol
from unittest.mock import patch

import numpy as np
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
        self.max_tokens = settings.latex_max_tokens
        self.repetition_stop = settings.latex_repetition_stop
        self.http = http or session()
        self._model = None
        self._decoder: GreedyDecoder | None = None
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
                    self._decoder = model.encoder_decoder = GreedyDecoder.replacing(
                        model.encoder_decoder, self.max_tokens, self.repetition_stop
                    )
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

    @property
    def last(self) -> Decoded | None:
        """How the decoder ended the last crop of the calling thread."""
        return self._decoder.last if self._decoder else None

    def __call__(self, image: Image.Image) -> str | None:
        model = self._load()
        started = time.perf_counter()
        result = model(png(image))
        decoded = self.last
        if decoded is not None:
            annotate(**{"latex.tokens": decoded.tokens, "latex.stop": decoded.stop})
            # Keep the "N tokens in S s" prefix: the Loki queries of #3 parse it.
            suffix = "" if decoded.stop == "eos" else f", stopped by {decoded.stop}"
            logger.info("LaTeX-OCR: %d tokens in %.1f s%s", decoded.tokens, time.perf_counter() - started, suffix)
        # (latex, elapsed) in rapid_latex_ocr 0.x; a bare string in later versions.
        return clean(result[0] if isinstance(result, tuple | list) else str(result))


Stop = Literal["eos", "max_tokens", "repetition"]


@dataclass(frozen=True)
class Decoded:
    #: Tokens produced, the end token included.
    tokens: int
    #: `eos`: the model finished. `max_tokens`: cut at `LATEX_MAX_TOKENS`. `repetition`: cut in a loop.
    stop: Stop


def repeats(tokens: Sequence[int], span: int, max_period: int = 8) -> int:
    """The period (1 to `max_period` tokens) with which the last `span` tokens repeat, or 0.

    A non-zero period is the decoder stuck in a loop (`\\qquad\\qquad…`, `e_{A},e_{A},…`).
    Left alone it runs to the token cap; on dev those crops took 41 % of decode time (#3).
    Replaying dev's stored LaTeX (re-tokenized, so an estimate), a span of 64 cuts ~30 %
    of decode time, and ~55 of ~10,400 regions lose anything after the loop, nearly all
    of it more garbage. Real content can repeat too (a long row of zeros is `0,` again
    and again), which is why the span is long and nothing decoded is thrown away.
    """
    if span <= 0:
        return 0
    last = tokens[-span:]
    for period in range(1, max_period + 1):
        if len(tokens) >= span + period and last == tokens[-span - period : -period]:
            return period
    return 0


class GreedyDecoder:
    """Stands in for `LaTeXOCR.encoder_decoder`: the same encoder and decoder sessions, a cheaper loop.

    rapid_latex_ocr's loop samples (top-k, softmax, `np.random.choice`) at temperature
    1e-5, which is argmax with extra work and randomness. This one takes the argmax,
    stops at `max_tokens` and when `repeats` sees a loop, and keeps how it ended per
    thread. What it returns is always a prefix of what the full loop would decode. The
    decoder graph has no KV cache, so each step still re-runs the whole prefix.
    """

    #: Positions the decoder graph takes (rapid_latex_ocr's `Decoder.max_seq_len`).
    WINDOW = 512

    def __init__(self, encoder, decoder, bos: int, eos: int, max_tokens: int, repetition_stop: int):
        self.encoder = encoder
        self.decoder = decoder
        self.bos = bos
        self.eos = eos
        self.max_tokens = max_tokens
        self.repetition_stop = repetition_stop
        self._local = threading.local()

    @classmethod
    def replacing(cls, inner, max_tokens: int, repetition_stop: int) -> GreedyDecoder:
        """From rapid_latex_ocr 0.0.9's `EncoderDecoder`."""
        return cls(inner.encoder, inner.decoder.session, inner.bos_token, inner.eos_token, max_tokens, repetition_stop)

    @property
    def last(self) -> Decoded | None:
        return getattr(self._local, "decoded", None)

    def __call__(self, x: np.ndarray, temperature: float | None = None) -> np.ndarray:
        """Token ids of one image, shape (1, n); `temperature` is accepted and ignored."""
        self._local.decoded = None
        context = self.encoder([x])[0]
        out = [self.bos]
        stop: Stop = "max_tokens"
        for _ in range(self.max_tokens):
            window = np.array([out[-self.WINDOW :]], dtype=np.int64)
            mask = np.ones_like(window, dtype=bool)
            logits = self.decoder([window, mask, context])[0]
            token = int(np.argmax(logits[0, -1]))
            out.append(token)
            if token == self.eos:
                stop = "eos"
                break
            if repeats(out[1:], self.repetition_stop):
                stop = "repetition"
                break
        self._local.decoded = Decoded(len(out) - 1, stop)
        return np.array([out[1:]], dtype=np.int64)


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
        settings.latex_max_tokens,
        settings.latex_repetition_stop,
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
