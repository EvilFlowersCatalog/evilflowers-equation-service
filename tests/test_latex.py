from pathlib import Path

import numpy as np
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


def test_onnx_replaces_the_sampling_loop(settings, onnx_sessions, caplog):
    models, _ = onnx_sessions
    model = latex.OnnxRecognizer(settings.model_copy(update={"models_path": models, "latex_max_tokens": 64}))
    loaded = model._load()

    decoder = loaded.encoder_decoder
    assert isinstance(decoder, latex.GreedyDecoder)
    assert (decoder.bos, decoder.eos, decoder.max_tokens, decoder.repetition_stop) == (1, 2, 64, 64)

    decoder.encoder = lambda inputs: [np.zeros((1, 4, 8), dtype=np.float32)]
    decoder.decoder = ScriptedDecoder([5, 6, 7, 8, 9, 10, 2])
    loaded.loop_image_resizer = lambda image: image
    loaded.tokenizer = type("Tokenizer", (), {"token2str": staticmethod(lambda tokens: ["x ^ 2"])})()

    with caplog.at_level("INFO", logger="evilflowers_equation.latex"):
        assert model(Image.new("RGB", (20, 10), "white")) == "x^2"
    assert model.last == latex.Decoded(7, "eos")
    assert "LaTeX-OCR: 7 tokens in" in caplog.text and "stopped" not in caplog.text


class ScriptedDecoder:
    """A decoder session whose logits pick `script[step]`; repeats the last token past the end."""

    VOCAB = 16

    def __init__(self, script):
        self.script = script
        self.windows = []

    def __call__(self, inputs):
        tokens, mask, _context = inputs
        assert tokens.dtype == np.int64 and mask.dtype == bool and mask.shape == tokens.shape
        self.windows.append(tokens.shape[-1])
        step = len(self.windows) - 1
        logits = np.zeros((1, tokens.shape[-1], self.VOCAB), dtype=np.float32)
        logits[0, -1, self.script[min(step, len(self.script) - 1)]] = 1.0
        return [logits]


def greedy(script, max_tokens=512, repetition_stop=0):
    return latex.GreedyDecoder(
        lambda inputs: [np.zeros((1, 4, 8), dtype=np.float32)],
        ScriptedDecoder(script),
        bos=1,
        eos=2,
        max_tokens=max_tokens,
        repetition_stop=repetition_stop,
    )


def test_greedy_decoder_stops_at_the_end_token():
    decoder = greedy([5, 6, 2, 7])
    assert decoder(np.zeros((1, 1, 8, 8))).tolist() == [[5, 6, 2]]
    assert decoder.last == latex.Decoded(3, "eos")
    # Each step feeds the whole prefix, the start token included.
    assert decoder.decoder.windows == [1, 2, 3]


def test_greedy_decoder_stops_at_max_tokens():
    decoder = greedy([5, 6, 7, 8, 9], max_tokens=4)
    assert decoder(np.zeros((1, 1, 8, 8))).tolist() == [[5, 6, 7, 8]]
    assert decoder.last == latex.Decoded(4, "max_tokens")


def test_greedy_decoder_keeps_the_last_512_positions():
    decoder = greedy([5, 6], max_tokens=520)
    decoder(np.zeros((1, 1, 8, 8)))
    assert max(decoder.decoder.windows) == 512 and len(decoder.decoder.windows) == 520


def test_greedy_decoder_stops_in_a_loop_and_keeps_what_it_decoded():
    # 3, 4 then the unit (5, 6) forever.
    decoder = greedy([3, 4] + [5, 6] * 40, repetition_stop=8)
    # Stopped once 8 tokens repeated the unit (2 + 2 + 8 decoded); nothing is thrown away.
    assert decoder(np.zeros((1, 1, 8, 8))).tolist() == [[3, 4] + [5, 6] * 5]
    assert decoder.last == latex.Decoded(12, "repetition")


def test_repeats_finds_the_shortest_period():
    assert latex.repeats([9] + [7] * 6, span=5) == 1
    assert latex.repeats([9, 1, 2, 3, 1, 2, 3, 1, 2, 3], span=6) == 3
    assert latex.repeats([1, 2, 3, 4, 5, 6, 7, 8], span=4) == 0
    assert latex.repeats([7] * 5, span=5) == 0  # needs span + period tokens
    assert latex.repeats([7] * 50, span=0) == 0  # off
    assert latex.repeats(list(range(9)) * 3, span=8, max_period=8) == 0  # period 9 is too long


def test_job_provenance_records_latex_threads(settings):
    from evilflowers_equation.jobs import EquationsJob

    config = {"latex_threads": 4, "latex_workers": 3, "latex_max_tokens": 256}
    onnx = latex.OnnxRecognizer(settings.model_copy(update=config))
    job = EquationsJob("run-1", "book.pdf", "out", [], settings=settings.model_copy(update=config), recognizer=onnx)
    provenance = job.provenance()
    assert provenance["latex_threads"] == 4 and provenance["latex_workers"] == 3
    assert provenance["latex_max_tokens"] == 256 and provenance["latex_repetition_stop"] == 64


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
