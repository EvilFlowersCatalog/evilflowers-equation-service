"""Messages the worker receives and the catalog ingests, as typed models.

The outgoing models mirror the catalog's ingestion forms
(EvilFlowersCatalog ``apps/extraction/forms.py``): a payload that validates
here is one the catalog accepts. Boxes are page fractions with a top-left
origin on the original file's 1-based page.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: The catalog keeps this much of a failed job's error.
MAX_ERROR_LENGTH = 2000


class Box(BaseModel):
    """``{x0, y0, x1, y1}`` as page fractions, origin top-left."""

    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float


class LayoutRegion(BaseModel):
    """A formula region the analyzer's layout pass found (`equations` job input)."""

    model_config = ConfigDict(extra="ignore")

    page: int = Field(ge=1)
    seq: int = Field(ge=0)
    bbox: Box


class Region(BaseModel):
    kind: Literal["equation"] = "equation"
    page: int = Field(ge=1)
    seq: int = Field(ge=0)
    bbox: Box
    image_path: str | None = Field(None, max_length=255)
    latex: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class JobReport(BaseModel):
    status: Literal["done", "failed"]
    error: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("error")
    @classmethod
    def _truncate(cls, value: str | None) -> str | None:
        return value[:MAX_ERROR_LENGTH] if value else None


def dump(model: BaseModel) -> dict[str, Any]:
    """JSON-ready, without the absent optional fields (the catalog treats them as unset)."""
    return model.model_dump(mode="json", exclude_none=True)
