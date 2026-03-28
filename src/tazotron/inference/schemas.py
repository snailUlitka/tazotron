"""Pydantic schemas for the inference HTTP contract."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelResponse(BaseModel):
    """Model metadata returned to backend clients."""

    model_config = ConfigDict(extra="forbid")

    externalId: str
    name: str
    version: str


class ModelMetricsResponse(BaseModel):
    """Metrics for a single model."""

    model_config = ConfigDict(extra="forbid")

    externalId: str
    accuracy: float
    loss: float


class ClassificationRequest(BaseModel):
    """Request body for X-ray classification."""

    model_config = ConfigDict(extra="forbid")

    modelExternalId: str = Field(min_length=1)
    imageBase64: str = Field(min_length=1)


class ClassificationResponse(BaseModel):
    """Classification result returned to the backend."""

    model_config = ConfigDict(extra="forbid")

    status: str
    diagnosis: str | None = None
    probability: float | None = None
    errorMessage: str | None = None


class CtToXraySegmentationRequest(BaseModel):
    """One CT segmentation payload for CT to X-ray rendering."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(min_length=1)
    fileBase64: str = Field(min_length=1)


class CtToXrayRequest(BaseModel):
    """CT study payload for CT to X-ray rendering."""

    model_config = ConfigDict(extra="forbid")

    ctFileBase64: str = Field(min_length=1)
    segmentations: list[CtToXraySegmentationRequest] = Field(min_length=1)
