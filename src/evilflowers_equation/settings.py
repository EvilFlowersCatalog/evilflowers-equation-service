"""Configuration of the worker: one typed object read from the environment (and `.env`).

Every field keeps the environment variable name the deployments already set
(``REDIS_URL``, ``EQUATION_QUEUE``, ``CATALOG_API_KEY``, ``LATEX_BACKEND`` …),
so the move from the old module-level settings needs no change in the k8s manifests.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    # -- Celery -------------------------------------------------------------------
    redis_url: str = "redis://redis:6379/7"
    queue: str = Field("evilflowers_equation", validation_alias=AliasChoices("EQUATION_QUEUE", "QUEUE"))
    #: Seconds before Redis redelivers an unacknowledged message. The job acknowledges
    #: late, so this must outlast the longest book (LaTeX-OCR runs on CPU, region by region).
    broker_visibility_timeout: int = 7200

    # -- Storage and provenance -----------------------------------------------------
    #: The catalog's storage root: originals are read here, crops written.
    storage_path: Path = Path("/mnt/data")
    #: Built into the image by CI; recorded in every run's provenance and as `service.version`.
    image_tag: str = "dev"

    # -- Catalog ingestion API (IP-018) ---------------------------------------------
    catalog_api_url: str = "http://catalog:8000"
    #: API key (bearer JWT) of the catalog user holding `extraction.ingest_extraction`.
    catalog_api_key: SecretStr | None = None
    #: Rows per request; keeps each body well under Django's 2.5 MB upload limit.
    catalog_batch_size: int = 100
    catalog_attempts: int = 4
    catalog_timeout: float = 120.0

    # -- Logs ---------------------------------------------------------------------------
    #: `json` (one object per line, what Loki parses) or `text` (Celery's own format, for a terminal).
    log_format: Literal["json", "text"] = "json"

    # -- Crops ----------------------------------------------------------------------------
    crop_dpi: int = 200
    webp_quality: int = 90
    #: A crop with a side shorter than this many pixels is dropped (a stray line, not a formula).
    min_crop_pixels: int = 4

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
    models_path: Path = Path("/models/latex-ocr")
    ollama_endpoint: str = "http://ollama:11434/api/chat"
    ollama_api_key: SecretStr | None = None
    latex_model: str = "qwen2.5vl:7b"
    #: Seconds per Ollama request (one crop).
    latex_timeout: float = 120.0

    def source_path(self, source: str) -> Path:
        """Absolute path of a file the catalog names relative to its storage root."""
        return self.storage_path / source


@lru_cache
def get_settings() -> Settings:
    return Settings()
