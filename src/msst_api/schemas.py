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
    rvc_model_dir: str | None = None
    rvc_loaded_models: list[str] = Field(default_factory=list)
    asr_model_dir: str | None = None
    v2m_model_dir: str | None = None
    asr_loaded_models: list[str] = Field(default_factory=list)
    v2m_loaded_models: list[str] = Field(default_factory=list)
    asr_streaming_backend: str | None = None


class ASRModelInfo(BaseModel):
    id: str
    name: str
    capability: str = "asr"
    kind: str
    description: str = ""
    tasks: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    downloaded: bool = False
    extra: dict | None = None


class ASRModelListResponse(BaseModel):
    total: int
    models: list[ASRModelInfo]


class ASRTranscriptionResponse(BaseModel):
    model: str
    text: str
    language: str | None = None
    phonemes: list[str] | None = None
    elapsed_seconds: float


class V2MModelInfo(BaseModel):
    id: str
    name: str
    capability: str = "v2m"
    kind: str
    description: str = ""
    tasks: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    downloaded: bool = False
    extra: dict | None = None


class V2MModelListResponse(BaseModel):
    total: int
    models: list[V2MModelInfo]


class V2MInferenceMetadata(BaseModel):
    language: str
    lyric_output_mode: str
    output_formats: list[str]
    alignment_engine: str
    chinese_asr_engine: str
    japanese_asr_engine: str
    files: list[str]
    elapsed_seconds: float



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


class RVCVoiceInfo(BaseModel):
    id: str
    name: str
    has_index: bool = False
    size: int | None = None
    downloaded: bool = True


class RVCVoiceListResponse(BaseModel):
    total: int
    voices: list[RVCVoiceInfo]
