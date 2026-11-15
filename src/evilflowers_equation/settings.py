"""Configuration of the worker: the kit's ``WorkerSettings`` plus the crop and LaTeX fields.

The broker, shutdown, delivery cap, storage, image tag, logs and catalog ingestion API are
``evilflowers_worker.settings.WorkerSettings`` (``REDIS_URL``, ``WORKER_SOFT_SHUTDOWN_TIMEOUT``,
``WORKER_CANCEL_TIMEOUT``, ``MAX_DELIVERIES``, ``STORAGE_PATH``, ``CATALOG_API_KEY`` …). Every field
keeps the environment variable name the deployments already set.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from evilflowers_worker.settings import WorkerSettings
from pydantic import AliasChoices, Field, SecretStr


class Settings(WorkerSettings):
    queue: str = Field("evilflowers_equation", validation_alias=AliasChoices("EQUATION_QUEUE", "QUEUE"))
    #: Deliveries before the job is reported `failed` instead of decoded again (#6). Lower than the kit's 5:
    #: each delivery of a book that kills the worker costs up to hours of CPU. 0 turns the cap off.
    max_deliveries: int = Field(3, ge=0)
    #: Seconds between reads of the run while a book is decoded (#5, catalog#140). The job stops when
    #: the run is no longer `running`, was started again (a new `attempt`) or no longer waits for it. 0 = only once,
    #: before the work.
    run_check_interval: float = Field(300.0, ge=0)

    # -- Crops ----------------------------------------------------------------------------
    crop_dpi: int = 200
    webp_quality: int = 90
    #: A crop with a side shorter than this many pixels is dropped (a stray line, not a formula).
    min_crop_pixels: int = 4
    #: Pixels a crop may have at most (#6). A larger box is rendered at a lower dpi so that it fits: a
    #: whole A4 page at 200 dpi is 3.9 MP, a poster-sized page far more, while a wide display formula is
    #: well under 1 MP and LaTeX-OCR shrinks its input below that anyway. 0 = no bound.
    max_crop_pixels: int = Field(2_000_000, ge=0)

    # -- LaTeX recognition ------------------------------------------------------------------
    #: `onnx`: LaTeX-OCR (pix2tex) weights on ONNX Runtime, CPU, downloaded to `MODELS_PATH` on first use.
    #: `ollama`: a multimodal model behind an Ollama endpoint. `none`: crops and positions only.
    latex_backend: Literal["onnx", "ollama", "none"] = "onnx"
    #: ONNX Runtime intra-op threads per session; match the pod's CPU limit. 0 = ORT default (all node cores).
    #: The decoder makes up to 512 tiny `run()` calls per crop, and more threads than the CPU quota
    #: mostly burn that quota on synchronisation (CFS throttling).
    latex_threads: int = Field(0, ge=0)
    #: ONNX Runtime's `session.intra_op.allow_spinning`. False makes idle pool threads sleep instead of
    #: busy-waiting between the decoder's calls, which can help under a CPU quota. True is ORT's default.
    latex_allow_spinning: bool = True
    #: Tokens decoded per crop at most; 512 is the decoder's window. A crop that reaches it is cut there.
    latex_max_tokens: int = Field(512, ge=1, le=512)
    #: Stop a crop once its last this-many tokens repeat a unit of 1 to 8 tokens (the decoder in a loop).
    #: 0 turns it off. See `latex.repeats` for how 64 was chosen.
    latex_repetition_stop: int = Field(64, ge=0)
    #: Crops of one book decoded at the same time, each on its own thread. `Session.run` releases the
    #: GIL, so with `LATEX_THREADS=1` this should match the pod's CPU limit (not the node's cores).
    latex_workers: int = Field(1, ge=1)
    models_path: Path = Path("/models/latex-ocr")
    ollama_endpoint: str = "http://ollama:11434/api/chat"
    ollama_api_key: SecretStr | None = None
    latex_model: str = "qwen2.5vl:7b"
    #: Seconds per Ollama request (one crop).
    latex_timeout: float = 120.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
