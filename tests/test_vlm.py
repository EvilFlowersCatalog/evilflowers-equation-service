"""The `paddleocr-vl` backend: an HTTP client of the shared vlm service (`POST /v1/formulas`)."""

import base64
import io

import pytest
import requests
from PIL import Image

from evilflowers_equation import latex
from evilflowers_equation.latex import PaddleOcrVlRecognizer, VlmItem, VlmUnavailable


class Response:
    def __init__(self, status: int = 200, body: dict | None = None, headers: dict | None = None):
        self.status_code = status
        self.body = body or {}
        self.headers = headers or {}
        self.text = str(body)

    def json(self):
        return self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class Http:
    """Answers each request with the next of `responses`; a callable gets the request body."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def post(self, url, json, timeout):
        self.requests.append((url, json, timeout))
        answer = self.responses.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer(json) if callable(answer) else answer


def echo(body):
    """The vlm's answer: each item's task and image size as its text."""
    results = []
    for item in body["items"]:
        image = Image.open(io.BytesIO(base64.b64decode(item["image"])))
        results.append({"key": item["key"], "text": f"{item['task']} {image.width}x{image.height}", "secs": 0.1})
    return Response(body={"model": "PaddleOCR-VL-1.6", "results": results})


def recognizer(settings, http, **overrides):
    return PaddleOcrVlRecognizer(settings.model_copy(update=overrides), http=http, sleep=lambda _s: None)


def crop(width=200, height=50):
    return Image.new("RGB", (width, height), "white")


def test_batches_with_a_timeout_per_crop(settings):
    http = Http(echo, echo)
    vlm = recognizer(settings, http, vlm_batch_size=2, vlm_timeout=30.0)
    items = [VlmItem("a", crop()), VlmItem("b", crop(), "ocr"), VlmItem("c", crop(80, 90))]

    assert vlm.recognize(items) == {"a": "formula 200x50", "b": "ocr 200x50", "c": "formula 80x90"}
    assert [len(body["items"]) for _url, body, _t in http.requests] == [2, 1]
    assert [timeout for _url, _body, timeout in http.requests] == [60.0, 30.0]
    assert http.requests[0][0] == settings.vlm_endpoint


def test_name_carries_the_revision(settings):
    assert recognizer(settings, Http(), vlm_model_revision="1.6@abc").name == "paddleocr-vl:1.6@abc"


def test_a_crop_it_cannot_read_has_no_answer(settings):
    body = {
        "model": "m",
        "results": [{"key": "a", "text": None, "error": "unreadable image"}, {"key": "b", "text": ""}],
    }
    vlm = recognizer(settings, Http(Response(body=body)))
    assert vlm.recognize([VlmItem("a", crop()), VlmItem("b", crop())]) == {"a": None, "b": None}


def test_503_waits_for_retry_after_then_retries(settings):
    waits = []
    http = Http(
        Response(503, {"detail": "model is loading"}, {"Retry-After": "7"}),
        requests.ConnectionError("refused"),
        echo,
    )
    vlm = PaddleOcrVlRecognizer(settings, http=http, sleep=waits.append)
    assert vlm.recognize([VlmItem("a", crop())]) == {"a": "formula 200x50"}
    # Retry-After when the service sends one, otherwise exponential backoff (1, 2, 4 … s).
    assert waits == [7.0, 2.0]


def test_the_wait_is_capped(settings):
    waits = []
    http = Http(Response(503, headers={"Retry-After": "9000"}), echo)
    vlm = PaddleOcrVlRecognizer(settings.model_copy(update={"vlm_max_wait": 30.0}), http=http, sleep=waits.append)
    vlm.recognize([VlmItem("a", crop())])
    assert waits == [30.0]


def test_unavailable_after_the_last_attempt(settings):
    http = Http(*[Response(503, headers={"Retry-After": "1"})] * 3)
    vlm = recognizer(settings, http, vlm_attempts=3)
    with pytest.raises(VlmUnavailable, match="after 3 attempts"):
        vlm.recognize([VlmItem("a", crop())])
    assert len(http.requests) == 3


def test_a_client_error_is_not_retried(settings):
    http = Http(Response(422, {"detail": "too many items"}))
    with pytest.raises(requests.HTTPError):
        recognizer(settings, http).recognize([VlmItem("a", crop())])
    assert len(http.requests) == 1


def test_as_the_catalog_backend_it_cleans_the_latex(settings):
    http = Http(Response(body={"model": "m", "results": [{"key": "crop", "text": "$$E = mc^2$$"}]}))
    vlm = recognizer(settings, http)
    assert vlm(crop()) == "E = mc^2"
    assert http.requests[0][1]["items"][0]["task"] == "formula"


def test_latex_backend_picks_it(settings):
    chosen = latex.recognizer(
        settings.model_copy(update={"latex_backend": "paddleocr-vl", "vlm_endpoint": "http://x"})
    )
    assert isinstance(chosen, PaddleOcrVlRecognizer)
    assert chosen.endpoint == "http://x"
