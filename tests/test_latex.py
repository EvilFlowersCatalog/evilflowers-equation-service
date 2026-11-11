from pathlib import Path

import pytest
from PIL import Image

from evilflowers_equation import latex


def test_clean_unwraps_what_models_add():
    assert latex.clean("$$ a^2 + b^2 $$") == "a^2 + b^2"
    assert latex.clean(r"\[x\]") == "x"
    assert latex.clean(r"\(y\)") == "y"
    assert latex.clean("```latex\n" + r"\frac{1}{2}" + "\n```") == r"\frac{1}{2}"
    assert latex.clean("   ") is None
    assert latex.clean(None) is None


def test_normalize_where_it_parses():
    assert latex.normalize(r"\frac{2}{4}") == r"\frac{1}{2}"
    assert latex.normalize(r"\frac{") is None
    assert latex.normalize(None) is None


def test_recognizer_follows_the_backend(settings):
    assert latex.recognizer(settings).name == "none"
    ollama = latex.recognizer(settings.model_copy(update={"latex_backend": "ollama"}))
    assert ollama.name == "ollama:qwen2.5vl:7b"
    onnx = latex.recognizer(settings.model_copy(update={"latex_backend": "onnx"}))
    assert onnx.name == "rapid-latex-ocr"
    # One per process: the ONNX model is loaded once.
    assert latex.recognizer(settings.model_copy(update={"latex_backend": "onnx"})) is onnx


def test_ollama_request_and_answer(settings):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "$x^2$"}}

    class Http:
        def post(self, url, json, timeout):
            self.sent = (url, json, timeout)
            return Response()

    http = Http()
    model = latex.OllamaRecognizer(settings.model_copy(update={"latex_backend": "ollama"}), http=http)
    assert model(Image.new("RGB", (20, 10), "white")) == "x^2"
    url, body, timeout = http.sent
    assert url == settings.ollama_endpoint and timeout == settings.latex_timeout
    assert body["model"] == "qwen2.5vl:7b" and body["messages"][0]["images"]


def test_onnx_weights_go_to_models_path(settings, tmp_path):
    class Response:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"weights"

    class Http:
        def __init__(self):
            self.urls = []

        def get(self, url, stream, timeout):
            self.urls.append(url)
            return Response()

    models = tmp_path / "models"
    (models).mkdir()
    (models / "encoder.onnx").write_bytes(b"already here")
    http = Http()
    model = latex.OnnxRecognizer(settings.model_copy(update={"models_path": models}), http=http)

    paths = model.weights()
    assert set(paths) == {"image_resizer_path", "encoder_path", "decoder_path", "tokenizer_json"}
    assert all(path.parent == models and path.exists() for path in paths.values())
    assert (models / "encoder.onnx").read_bytes() == b"already here"
    downloaded = sorted(url.rsplit("/", 1)[1] for url in http.urls)
    assert downloaded == ["decoder.onnx", "image_resizer.onnx", "tokenizer.json"]
    assert not list(models.glob("*.partial"))


@pytest.fixture
def onnx_sessions(monkeypatch, tmp_path):
    """Every ONNX Runtime session `LaTeXOCR` creates, without the real weights (`InferenceSession` is faked)."""
    rlo_main = pytest.importorskip("rapid_latex_ocr.main")
    from rapid_latex_ocr import utils_load

    created = []

    class FakeInferenceSession:
        def __init__(self, path, sess_options=None, providers=None):
            self.path = Path(path).name
            self.sess_options = sess_options
            created.append(self)

        def get_session_options(self):
            return self.sess_options

    monkeypatch.setattr(utils_load, "InferenceSession", FakeInferenceSession)
    monkeypatch.setattr(rlo_main, "TokenizerCls", lambda path: None)
    models = tmp_path / "models"
    models.mkdir()
    for file_name in latex.OnnxRecognizer.FILES.values():
        (models / file_name).write_bytes(b"weights")
    return models, created


@pytest.mark.parametrize("threads", [1, 3])
def test_onnx_sessions_use_latex_threads(settings, onnx_sessions, threads):
    models, created = onnx_sessions
    config = {"models_path": models, "latex_threads": threads, "latex_allow_spinning": False}
    latex.OnnxRecognizer(settings.model_copy(update=config))._load()

    # rapid-latex-ocr 0.0.9 has three sessions; a new version that builds them elsewhere fails here.
    assert sorted(session.path for session in created) == ["decoder.onnx", "encoder.onnx", "image_resizer.onnx"]
    for session in created:
        options = session.get_session_options()
        assert options.intra_op_num_threads == threads
        assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"

    # The patch is undone after the constructor.
    from rapid_latex_ocr import main
    from rapid_latex_ocr import models as rlo_models
    from rapid_latex_ocr.utils_load import OrtInferSession

    assert main.OrtInferSession is OrtInferSession and rlo_models.OrtInferSession is OrtInferSession


def test_onnx_sessions_keep_ort_defaults_without_latex_threads(settings, onnx_sessions):
    models, created = onnx_sessions
    latex.OnnxRecognizer(settings.model_copy(update={"models_path": models}))._load()

    assert len(created) == 3
    for session in created:
        assert session.get_session_options().intra_op_num_threads == 0


def test_onnx_counts_the_decoded_tokens(settings, onnx_sessions, caplog):
    import numpy as np

    models, _ = onnx_sessions
    model = latex.OnnxRecognizer(settings.model_copy(update={"models_path": models, "latex_threads": 2}))
    loaded = model._load()
    loaded.encoder_decoder.inner = lambda x, temperature: np.zeros((1, 7), dtype=np.int64)
    loaded.loop_image_resizer = lambda image: image
    loaded.tokenizer = type("Tokenizer", (), {"token2str": staticmethod(lambda tokens: ["x ^ 2"])})()

    with caplog.at_level("INFO", logger="evilflowers_equation.latex"):
        assert model(Image.new("RGB", (20, 10), "white")) == "x^2"
    assert model._tokens.last == 7
    assert "7 tokens" in caplog.text


def test_job_provenance_records_latex_threads(settings):
    from evilflowers_equation.jobs import EquationsJob

    onnx = latex.OnnxRecognizer(settings.model_copy(update={"latex_threads": 4}))
    job = EquationsJob("run-1", "book.pdf", "out", [], settings=settings, catalog=None, recognizer=onnx)
    assert job.provenance()["latex_threads"] == 4


def test_a_failing_recognizer_gives_none():
    class Broken:
        name = "broken"

        def __call__(self, image):
            raise RuntimeError("model crashed")

    assert latex.recognize(Broken(), Image.new("RGB", (10, 10))) is None


@pytest.mark.parametrize("answer", [None, "x"])
def test_recognize_passes_the_answer_through(answer):
    class Model:
        name = "m"

        def __call__(self, image):
            return answer

    assert latex.recognize(Model(), Image.new("RGB", (10, 10))) == answer
