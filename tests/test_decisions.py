"""#9: the cheap doubt check, and decider-vision's `formula_kind` on doubtful formulas only."""

import io

import pytest
from evilflowers_worker.judges import CircuitBreaker, Judges, JudgeUnavailable, Thresholds, Verdict
from PIL import Image

from conftest import FakeRecognizer, catalog, region, write_pdf
from evilflowers_equation import decisions, latex, metrics
from test_jobs import job


class FakeJudge:
    """Stands in for the kit's decider-vision client: what `Panel.ask` uses of it."""

    name = "decider-vision"
    timeout = 60.0

    def __init__(self, verdict="text", p=0.91, error: Exception | None = None):
        self.verdict, self.p, self.error = verdict, p, error
        self.breaker = CircuitBreaker()
        self.calls = []

    def ask(self, question, *, state, images, timeout):
        self.calls.append((question.id, state, images))
        if self.error:
            raise self.error
        return Verdict(
            question=question.id, verdict=self.verdict, p=self.p, model="strands-decider-2b", revision="abc"
        )


def judges(settings, judge=None) -> Judges:
    return Judges(decider=None, decider_vision=judge, ollama=None, rerank=None, settings=settings)


def run(settings, answer, judge=None, *, enabled=True, regions=None):
    write_pdf(settings.storage_path / "book.pdf")
    fake = catalog()
    the_job = job(settings, regions or [region()], fake, FakeRecognizer(answer))
    the_job.judges = judges(settings, judge) if enabled else judges(settings)
    the_job.panel = the_job.judges.panel()
    assert the_job.run() == "done"
    return fake.regions_posted


@pytest.fixture
def counted(monkeypatch):
    seen = []
    monkeypatch.setattr(metrics.FORMULAS, "add", lambda value, attributes: seen.append(attributes["outcome"]))
    return seen


# -- the doubt check ------------------------------------------------------------------------------------------


def doubts(text, **kwargs):
    return decisions.doubts(text, latex.normalize(text), **kwargs)


def test_a_formula_that_parses_and_is_long_enough_is_clean():
    assert doubts(r"E = m c^{2}") == []


def test_latex_sympy_cannot_parse_is_doubtful():
    assert doubts(r"\frac{a}{b} = ") == ["sympy cannot parse it"]


def test_short_latex_is_doubtful():
    assert doubts("x") == ["1 token (< 3)"]
    assert doubts("1.") == ["sympy cannot parse it", "2 tokens (< 3)"]


def test_two_readings_that_disagree_are_doubtful():
    [reason] = doubts(r"\frac{a}{b} + c", other=r"$\sum_{i=1}^{n} q_i$", other_name="latex-ocr", min_ratio=0.8)
    assert reason.startswith("latex-ocr reads it differently")
    assert doubts(r"\frac{a}{b} + c", other=r"\frac{a}{b}+c", other_name="latex-ocr") == []


def test_no_latex_is_unread_not_doubtful():
    assert decisions.doubts(None, None) is None
    assert decisions.doubts("", None) is None


def test_the_judge_gets_a_png_no_larger_than_the_max_side():
    data = decisions.png(Image.new("RGB", (1200, 300), "white"), 448)
    with Image.open(io.BytesIO(data)) as image:
        assert image.format == "PNG" and max(image.size) == 448


# -- the job ----------------------------------------------------------------------------------------------------


def test_a_clean_formula_is_not_judged_and_posted_as_before(settings, counted):
    judge = FakeJudge()
    [posted] = run(settings, r"E = m c^{2}", judge)
    assert judge.calls == [] and counted == ["clean"]
    assert posted["provenance"].keys() <= {"latex_model", "sympy"}
    assert not {"decision", "p", "decider_model", "band", "signals"} & posted.keys()


def test_a_doubtful_formula_gets_the_judges_answer_without_a_band(settings, counted):
    judge = FakeJudge(verdict="text", p=0.91)
    [posted] = run(settings, "x", judge)
    assert counted == ["doubtful"]
    [(question, state, [image])] = judge.calls
    assert question == "formula_kind" and state == decisions.STATE
    with Image.open(io.BytesIO(image)) as crop:
        assert crop.format == "PNG" and max(crop.size) <= settings.judge_max_side
    assert posted["decision"] == "text" and posted["p"] == 0.91
    assert posted["decider_model"] == "strands-decider-2b@abc"
    # No JUDGE_THRESHOLDS for formula_kind until knowledge#65: the catalog gets no band.
    assert "band" not in posted
    assert posted["signals"] == {"doubts": ["1 token (< 3)"]}
    assert posted["provenance"]["formula_kind"]["verdict"] == "text"
    assert posted["latex"] == "x"


def test_thresholds_band_the_answer(settings):
    thresholds = {"formula_kind": Thresholds(keep=0.8, hide=0.9)}
    settings = settings.model_copy(update={"judge_thresholds": thresholds})
    [posted] = run(settings, "x", FakeJudge(verdict="noise", p=0.95))
    assert posted["band"] == "hide"


@pytest.mark.parametrize(
    "judge, enabled",
    [
        (None, False),  # judges off: counted, never called
        (FakeJudge(error=JudgeUnavailable("timeout")), True),  # decider-vision down
        (FakeJudge(error=RuntimeError("a bug")), True),  # anything else: still never fails the job
    ],
)
def test_without_an_answer_a_doubtful_formula_is_posted_as_before(settings, counted, judge, enabled):
    [posted] = run(settings, "x", judge, enabled=enabled)
    assert counted == ["doubtful"]
    assert posted["latex"] == "x" and posted["provenance"] == {"latex_model": "fake"}
    assert not {"decision", "p", "decider_model", "band", "signals"} & posted.keys()


def test_a_crop_without_latex_is_counted_unread_and_not_judged(settings, counted):
    judge = FakeJudge()
    [posted] = run(settings, None, judge)
    assert judge.calls == [] and counted == ["unread"]
    assert "latex" not in posted
