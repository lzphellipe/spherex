"""DTOs da API (Pydantic)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from spherex.domain.models import Label


class CandidateSummary(BaseModel):
    id: str
    field_id: str
    score: float = Field(ge=0, le=1)
    n: int
    rate: float
    rms: float
    matched_known: str | None = None


class Provenance(BaseModel):
    exposure_ids: list[str]
    config_hash: str
    model_version: str | None
    code_version: str
    created_at: str


class CandidateDetail(CandidateSummary):
    detection_ids: list[str]
    exposure_ids: list[str]
    stamps: list[list[list[list[float]]]]
    provenance: Provenance
    labels: list[dict]
    consensus: str | None


class LabelIn(BaseModel):
    reviewer: str = Field(min_length=1, max_length=64)
    label: Label


class JobIn(BaseModel):
    kind: str = Field(pattern="^(detect|train|eval)$")


class JobOut(BaseModel):
    id: int
    kind: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result: str | None = None
