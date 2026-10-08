"""Pydantic response models for the MSST API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    cuda_available: bool
    cuda_device_count: int
    default_device: str
    model_dir: str
    download_backend: str
    loaded_models: list[str] = Field(default_factory=list)


class ModelInfo(BaseModel):
    id: str
    name: str
    category: str
    category_label: str
    model_type: str
    instruments: list[str]
    target_instrument: str | None = None
    size: int | None = None
    sha256: str | None = None
    downloaded: bool = False


class ModelListResponse(BaseModel):
    total: int
    models: list[ModelInfo]


class InferenceMetadata(BaseModel):
    model_id: str | None = None
    model_type: str
    sample_rate: int
    device: str
    stems: list[str]
    output_format: str
    elapsed_seconds: float
