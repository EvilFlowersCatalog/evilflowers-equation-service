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
    #: `superseded`: the job saw a newer attempt of its run and stopped (catalog#140); counted, the run is left alone.
    status: Literal["done", "failed", "superseded"]
    error: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("error")
    @classmethod
    def _truncate(cls, value: str | None) -> str | None:
        return value[:MAX_ERROR_LENGTH] if value else None


#: Why a job does not do (or stops doing) its work.
#: `not_running`: the run no longer accepts results. `superseded`: the run was started again (a new attempt).
#: `already_done`: the run no longer waits for this job (a copy of it reported, or it was never declared).
SkipReason = Literal["not_running", "superseded", "already_done"]


class RunState(BaseModel):
    """What a job reads of its run (`GET /extraction/v1/runs/{id}`), before the work and between crops."""

    model_config = ConfigDict(extra="ignore")

    status: str
    #: `{job: pending | done | failed}`, as declared at the run's `/start`.
    jobs: dict[str, str] = Field(default_factory=dict)
    #: The current attempt (catalog#140); absent from catalogs that predate it.
    attempt: int | None = None

    def skip_reason(self, job: str, attempt: int | None = None) -> SkipReason | None:
        """Why a job working for `attempt` must stop, or None while the run still waits for it.

        The attempt is compared first, as the catalog does (`_require_running`): a reset run is
        `queued` on a new attempt, and after a restart the job is `pending` again, but for the new
        attempt, not for this task.
        """
        if attempt is not None and self.attempt is not None and self.attempt != attempt:
            return "superseded"
        if self.status != "running":
            return "not_running"
        if self.jobs.get(job) != "pending":
            return "already_done"
        return None


def dump(model: BaseModel) -> dict[str, Any]:
    """JSON-ready, without the absent optional fields (the catalog treats them as unset)."""
    return model.model_dump(mode="json", exclude_none=True)
