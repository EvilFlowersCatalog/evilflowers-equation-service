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
