"""`evilflowers_equation.standalone.formulas` (#8): the contract with the books digitalizer, without Celery.

The message and the rows follow the contract comment on #8 and the digitalizer's executable spec
(`studio/testing/standalone.py::equation_formulas`): `regions[{key, leaf, image, bbox, task}]` in, one row
`{key, text, model}` per region in `formulas.json`, then the outcome `{run_id, token, status, error, formulas}`.
"""

import json

import pytest
from evilflowers_worker.testing import assert_outcome
from PIL import Image, ImageDraw
from pydantic import ValidationError

from evilflowers_equation import standalone
from evilflowers_equation.latex import VlmUnavailable

OUT = "item/_studio/jobs/formulas-0002-a1"
OUTCOME = f"{OUT}/outcome.json"


class FakeVlm:
    name = "paddleocr-vl:test"

    def __init__(self, answers=None, error: Exception | None = None):
        self.answers = answers or {}
        self.error = error
        self.items = []

    def recognize(self, items):
        self.items.extend(items)
        if self.error:
            raise self.error
        return {item.key: self.answers.get(item.key, "E = mc^2") for item in items}


def leaf(storage, name: str = "digitized/book/0003.png", size=(1000, 1400)) -> str:
    path = storage / name
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, "white")
    ImageDraw.Draw(image).text((100, 600), "E = mc^2", fill="black")
    image.save(path)
    return name


def message(images, regions, **overrides) -> dict:
    return {
        "run_id": "run-1",
        "job": "formulas-0002-a1",
        "token": "tok-123",
        "inputs": images,
        "output_dir": OUT,
        "outcome": OUTCOME,
        "regions": regions,
        **overrides,
    }


def region(key: str, image: str, task: str = "formula", bbox=(0.08, 0.41, 0.62, 0.47)) -> dict:
    return {"key": key, "leaf": 3, "image": image, "bbox": list(bbox), "task": task}


def test_rows_then_outcome(settings):
    image = leaf(settings.storage_path)
    vlm = FakeVlm({"region-2": "$a$ and $b$"})
    job = standalone.FormulasJob.model_validate(
        message([image], [region("region-1", image), region("region-2", image, task="ocr")], newer_field="ignored")
    )

    result = standalone.run(job, 1, settings=settings, vlm=vlm)

    assert result == {"job": "formulas-0002-a1", "answered": 2}
    outcome = assert_outcome(
        settings.storage_path,
        OUTCOME,
        run_id="run-1",
        token="tok-123",
        status="done",
        after=f"{OUT}/formulas.json",
    )
    assert outcome == {
        "run_id": "run-1",
        "token": "tok-123",
        "status": "done",
        "error": None,
        "formulas": f"{OUT}/formulas.json",
    }
    rows = json.loads((settings.storage_path / OUT / "formulas.json").read_text())
    # A formula goes through `clean`; an `ocr` answer keeps its inline `$…$`.
    assert rows == [
        {"key": "region-1", "text": "E = mc^2", "model": "paddleocr-vl:test"},
        {"key": "region-2", "text": "$a$ and $b$", "model": "paddleocr-vl:test"},
    ]
    assert [(item.key, item.task) for item in vlm.items] == [("region-1", "formula"), ("region-2", "ocr")]


def test_the_crop_is_the_box_plus_context(settings):
    image = leaf(settings.storage_path)
    vlm = FakeVlm()
    job = standalone.FormulasJob.model_validate(message([image], [region("r", image, bbox=(0.1, 0.4, 0.6, 0.5))]))
    standalone.run(job, 1, settings=settings, vlm=vlm)
    # 500 x 140 px box + 10 px on each side, then padded to a sane aspect ratio (no side under width / 10).
    assert vlm.items[0].image.size == (520, 160)


def test_no_answer_is_a_null_row(settings):
    image = leaf(settings.storage_path)
    vlm = FakeVlm({"region-1": None})
    job = standalone.FormulasJob.model_validate(message([image], [region("region-1", image)]))
    assert standalone.run(job, 1, settings=settings, vlm=vlm) == {"job": "formulas-0002-a1", "answered": 0}
    rows = json.loads((settings.storage_path / OUT / "formulas.json").read_text())
    assert rows == [{"key": "region-1", "text": None, "model": "paddleocr-vl:test"}]


def test_an_empty_box_is_a_null_row_and_never_sent(settings):
    image = leaf(settings.storage_path)
    vlm = FakeVlm()
    regions = [region("flat", image, bbox=(0.2, 0.5, 0.4, 0.5)), region("ok", image)]
    standalone.run(standalone.FormulasJob.model_validate(message([image], regions)), 1, settings=settings, vlm=vlm)
    rows = json.loads((settings.storage_path / OUT / "formulas.json").read_text())
    assert [(row["key"], row["text"]) for row in rows] == [("flat", None), ("ok", "E = mc^2")]
    assert [item.key for item in vlm.items] == ["ok"]


def test_a_box_past_the_edge_is_clamped(settings):
    image = leaf(settings.storage_path)
    vlm = FakeVlm()
    job = standalone.FormulasJob.model_validate(message([image], [region("r", image, bbox=(0.9, 0.95, 1.0003, 1.01))]))
    standalone.run(job, 1, settings=settings, vlm=vlm)
    # 900..1000 x 1330..1400 px, plus 10 px of context where the leaf has it.
    assert vlm.items[0].image.size == (110, 80)


def test_no_regions_is_done_with_no_rows(settings):
    image = leaf(settings.storage_path)
    vlm = FakeVlm()
    job = standalone.FormulasJob.model_validate(message([image], []))
    assert standalone.run(job, 1, settings=settings, vlm=vlm) == {"job": "formulas-0002-a1", "answered": 0}
    assert json.loads((settings.storage_path / OUT / "formulas.json").read_text()) == []
    assert_outcome(settings.storage_path, OUTCOME, status="done")


def test_a_loop_is_disputed(settings):
    image = leaf(settings.storage_path)
    vlm = FakeVlm({"r": r"x = " + r"\qquad" * 80})
    standalone.run(
        standalone.FormulasJob.model_validate(message([image], [region("r", image)])), 1, settings=settings, vlm=vlm
    )
    (row,) = json.loads((settings.storage_path / OUT / "formulas.json").read_text())
    assert row["disputed"] is True
    assert "loop" in row["reason"]
    assert row["text"].startswith("x = ")


def test_a_crosscheck_that_disagrees_disputes(settings):
    image = leaf(settings.storage_path)

    class Onnx:
        name = "rapid-latex-ocr"

        def __call__(self, crop):
            return r"\int_0^1 f(t) \, dt"

    vlm = FakeVlm({"same": r"E = mc^2", "other": r"E = mc^2"})
    job = standalone.FormulasJob.model_validate(message([image], [region("other", image)]))
    standalone.run(job, 1, settings=settings, vlm=vlm, crosscheck=Onnx())
    (row,) = json.loads((settings.storage_path / OUT / "formulas.json").read_text())
    assert row["disputed"] is True
    assert row["reason"].startswith("rapid-latex-ocr reads it differently")


def test_a_crosscheck_that_agrees_does_not(settings):
    image = leaf(settings.storage_path)

    class Onnx:
        name = "rapid-latex-ocr"

        def __call__(self, crop):
            return r"E=m c^{2}"

    job = standalone.FormulasJob.model_validate(message([image], [region("r", image)]))
    standalone.run(job, 1, settings=settings, vlm=FakeVlm(), crosscheck=Onnx())
    (row,) = json.loads((settings.storage_path / OUT / "formulas.json").read_text())
    assert "disputed" not in row


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"image": "digitized/book/elsewhere.png"}, "not one of the job's inputs"),
        ({"image": "digitized/book/missing.png"}, "FileNotFoundError"),
    ],
)
def test_a_bad_leaf_fails_the_job_with_an_outcome(settings, change, error):
    image = leaf(settings.storage_path)
    inputs = [image, "digitized/book/missing.png"]
    job = standalone.FormulasJob.model_validate(message(inputs, [{**region("r", image), **change}]))
    with pytest.raises(Exception):  # noqa: B017 - the task must raise, whatever the error, so link_error fires
        standalone.run(job, 1, settings=settings, vlm=FakeVlm())
    outcome = assert_outcome(settings.storage_path, OUTCOME, run_id="run-1", token="tok-123", status="failed")
    assert error in outcome["error"]
    assert outcome["formulas"] is None
    assert not (settings.storage_path / OUT / "formulas.json").exists()


def test_an_unavailable_vlm_fails_the_job(settings):
    image = leaf(settings.storage_path)
    job = standalone.FormulasJob.model_validate(message([image], [region("r", image)]))
    with pytest.raises(VlmUnavailable):
        standalone.run(job, 1, settings=settings, vlm=FakeVlm(error=VlmUnavailable("down")))
    outcome = assert_outcome(settings.storage_path, OUTCOME, status="failed")
    assert outcome["error"] == "VlmUnavailable: down"


def test_past_the_delivery_cap_the_job_fails_with_an_outcome(settings, monkeypatch):
    monkeypatch.setattr(standalone, "get_settings", lambda: settings)
    image = leaf(settings.storage_path)
    job = standalone.FormulasJob.model_validate(message([image], [region("r", image)]))
    with pytest.raises(RuntimeError, match="Gave up after 3 deliveries"):
        standalone.give_up(job, 4)
    assert_outcome(settings.storage_path, OUTCOME, token="tok-123", status="failed")


@pytest.mark.parametrize(
    "regions",
    [
        [region("r", "a.png"), region("r", "a.png")],  # duplicate keys
        [{**region("r", "a.png"), "task": "table"}],  # not a task the contract has
        [{**region("r", "a.png"), "bbox": [0.1, 0.2, 0.3]}],
    ],
)
def test_an_invalid_message_is_refused(regions):
    with pytest.raises(ValidationError):
        standalone.FormulasJob.model_validate(message(["a.png"], regions))


def test_an_output_dir_that_escapes_storage_fails(settings):
    image = leaf(settings.storage_path)
    job = standalone.FormulasJob.model_validate(message([image], [region("r", image)], output_dir="../elsewhere"))
    with pytest.raises(ValueError, match="escapes"):
        standalone.run(job, 1, settings=settings, vlm=FakeVlm())
    assert_outcome(settings.storage_path, OUTCOME, status="failed")
