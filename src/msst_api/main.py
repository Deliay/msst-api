"""FastAPI application exposing MSST source separation over HTTP."""

from __future__ import annotations

import base64
import json
import logging
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from .asr.manager import ASRError, ASRManager, ASRModelNotFoundError
from .audio import (
    RVC_OUTPUT_FORMATS,
    SUPPORTED_OUTPUT_FORMATS,
    build_zip,
    decode_audio,
    decode_audio_mono,
    encode_audio,
    encode_audio_dynamic,
    save_upload,
)
from .config import Settings, get_settings
from .download import DownloadError, download_model
from .registry import ModelNotFoundError, get_registry
from .resources import SharedModelCache
from .rvc_engine import (
    RVCError,
    RVCManager,
    RVCParams,
    RVCVoiceNotFoundError,
    RVCVoiceSpec,
)
from .schemas import (
    ASRModelInfo,
    ASRModelListResponse,
    ASRTranscriptionResponse,
    HealthResponse,
    InferenceMetadata,
    ModelInfo,
    ModelListResponse,
    RVCVoiceInfo,
    RVCVoiceListResponse,
    SVSVoiceInfo,
    SVSVoiceListResponse,
    V2MInferenceMetadata,
    V2MModelInfo,
    V2MModelListResponse,
)
from .separator import InferenceError, ModelManager, ModelSpec, resolve_device
from .svs import SVSError, SVSVoiceNotFoundError
from .svs.engine import SVSManager, SVSSpec
from .svs.pipeline import SVSRequest, run_svs_pipeline
from .v2m.manager import V2MError, V2MManager, V2MModelNotFoundError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("msst_api")

TAG = "MSST"
RVC_TAG = "RVC"
ASR_TAG = "ASR"
V2M_TAG = "Vocal2Midi"
SVS_TAG = "DiffSinger"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.rvc_model_dir.mkdir(parents=True, exist_ok=True)
    settings.rvc_asset_dir.mkdir(parents=True, exist_ok=True)
    settings.asr_model_dir.mkdir(parents=True, exist_ok=True)
    settings.v2m_model_dir.mkdir(parents=True, exist_ok=True)
    settings.svs_model_dir.mkdir(parents=True, exist_ok=True)
    app.state.settings = settings
    # One governor shared by MSST, RVC, ASR and Vocal2Midi so
    # MSST_MAX_LOADED_MODELS and MSST_MAX_CONCURRENCY are global budgets across
    # every subsystem.
    shared_cache = SharedModelCache(
        settings.max_loaded_models, settings.max_concurrency
    )
    app.state.manager = ModelManager(settings, cache=shared_cache)
    app.state.rvc_manager = RVCManager(settings, cache=shared_cache)
    app.state.asr_manager = ASRManager(settings, cache=shared_cache)
    app.state.v2m_manager = V2MManager(settings, cache=shared_cache)
    app.state.svs_manager = SVSManager(settings, cache=shared_cache)
    # Let the vendored v2m runtimes route their loaders through the same cache.
    from .v2m import runtime as v2m_runtime

    v2m_runtime.configure(
        settings=settings,
        cache=shared_cache,
        asr_manager=app.state.asr_manager,
        v2m_manager=app.state.v2m_manager,
    )
    logger.info(
        "MSST API ready. model_dir=%s rvc_model_dir=%s asr_model_dir=%s "
        "v2m_model_dir=%s device=%s max_loaded_models=%s max_concurrency=%s",
        settings.model_dir,
        settings.rvc_model_dir,
        settings.asr_model_dir,
        settings.v2m_model_dir,
        settings.device,
        settings.max_loaded_models,
        settings.max_concurrency,
    )
    try:
        yield
    finally:
        app.state.manager.unload()
        app.state.rvc_manager.unload()
        app.state.asr_manager.unload()
        app.state.v2m_manager.unload()
        app.state.svs_manager.unload()


app = FastAPI(
    title="MSST API",
    description=(
        "GPU inference API for Music-Source-Separation-Training (MSST). "
        "Missing models are downloaded automatically from ModelScope "
        "(default) or Hugging Face."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _manager() -> ModelManager:
    return app.state.manager


def _rvc_manager() -> RVCManager:
    return app.state.rvc_manager


def _asr_manager() -> ASRManager:
    return app.state.asr_manager


def _v2m_manager() -> V2MManager:
    return app.state.v2m_manager


def _svs_manager() -> SVSManager:
    return app.state.svs_manager


def _settings() -> Settings:
    return app.state.settings


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(ModelNotFoundError)
async def _model_not_found(request, exc: ModelNotFoundError):  # noqa: ANN001
    return JSONResponse(status_code=404, content={"detail": f"Unknown model: {exc}"})


@app.exception_handler(RVCVoiceNotFoundError)
async def _rvc_voice_not_found(request, exc: RVCVoiceNotFoundError):  # noqa: ANN001
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(RVCError)
async def _rvc_error(request, exc: RVCError):  # noqa: ANN001
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(DownloadError)
async def _download_error(request, exc: DownloadError):  # noqa: ANN001
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(ASRModelNotFoundError)
async def _asr_model_not_found(request, exc: ASRModelNotFoundError):  # noqa: ANN001
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ASRError)
async def _asr_error(request, exc: ASRError):  # noqa: ANN001
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(V2MError)
async def _v2m_error(request, exc: V2MError):  # noqa: ANN001
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(V2MModelNotFoundError)
async def _v2m_model_not_found(request, exc: V2MModelNotFoundError):  # noqa: ANN001
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(SVSVoiceNotFoundError)
async def _svs_voice_not_found(request, exc: SVSVoiceNotFoundError):  # noqa: ANN001
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(SVSError)
async def _svs_error(request, exc: SVSError):  # noqa: ANN001
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Health / catalog
# ---------------------------------------------------------------------------
@app.get("/", tags=[TAG])
def root() -> dict:
    return {
        "service": "msst-api",
        "docs": "/docs",
        "inference": "POST /api/msst/inference",
        "models": "GET /api/msst/models",
        "rvc_inference": "POST /api/rvc/inference",
        "rvc_models": "GET /api/rvc/models",
        "asr_inference": "POST /api/asr/transcribe",
        "asr_streaming": "WS /api/asr/streaming",
        "asr_models": "GET /api/asr/models",
        "v2m_inference": "POST /api/v2m/inference",
        "v2m_models": "GET /api/v2m/models",
        "diffsinger_inference": "POST /api/diffsinger/inference",
        "diffsinger_models": "GET /api/diffsinger/models",
    }


@app.get("/health", response_model=HealthResponse, tags=[TAG])
def health() -> HealthResponse:
    settings = _settings()
    manager = _manager()
    return HealthResponse(
        cuda_available=torch.cuda.is_available(),
        cuda_device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        default_device=settings.device,
        model_dir=str(settings.model_dir),
        download_backend=settings.download_backend,
        loaded_models=manager.loaded,
        rvc_model_dir=str(settings.rvc_model_dir),
        rvc_loaded_models=_rvc_manager().loaded,
        asr_model_dir=str(settings.asr_model_dir),
        v2m_model_dir=str(settings.v2m_model_dir),
        asr_loaded_models=_asr_manager().loaded,
        v2m_loaded_models=_v2m_manager().loaded,
        svs_model_dir=str(settings.svs_model_dir),
        svs_loaded_models=_svs_manager().loaded,
        asr_streaming_backend=settings.asr_streaming_backend,
    )


@app.get("/api/msst/models", response_model=ModelListResponse, tags=[TAG])
def list_models(category: str | None = None, downloaded: bool | None = None) -> ModelListResponse:
    registry = get_registry()
    items: list[ModelInfo] = []
    for entry in registry.all():
        if category and entry.category != category:
            continue
        is_downloaded = entry.default_weight_path.is_file()
        if downloaded is not None and is_downloaded != downloaded:
            continue
        items.append(
            ModelInfo(**entry.to_public_dict(), downloaded=is_downloaded)
        )
    return ModelListResponse(total=len(items), models=items)


@app.get("/api/msst/models/{model_id}", response_model=ModelInfo, tags=[TAG])
def get_model(model_id: str) -> ModelInfo:
    entry = get_registry().get(model_id)
    return ModelInfo(**entry.to_public_dict(), downloaded=entry.default_weight_path.is_file())


@app.post("/api/msst/models/{model_id}/download", response_model=ModelInfo, tags=[TAG])
def download_catalog_model(model_id: str, source: str | None = None) -> ModelInfo:
    entry = get_registry().get(model_id)
    settings = _apply_source_override(source)
    download_model(
        entry.sources,
        entry.default_weight_path,
        expected_size=entry.size,
        expected_sha=entry.sha256,
        settings=settings,
    )
    return ModelInfo(**entry.to_public_dict(), downloaded=True)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _apply_source_override(source: str | None) -> Settings:
    settings = _settings()
    if not source:
        return settings
    source = source.strip().lower()
    if source not in {"modelscope", "huggingface", "auto"}:
        raise HTTPException(status_code=400, detail=f"Invalid source: {source}")
    return replace(settings, download_backend=source)


def _resolve_catalog_model(model_id: str, source: str | None) -> tuple[str, Path, Path, Path | None]:
    registry = get_registry()
    entry = registry.get(model_id)
    settings = _apply_source_override(source)
    config_path = registry.resolve_config(entry)
    checkpoint_path = download_model(
        entry.sources,
        entry.default_weight_path,
        expected_size=entry.size,
        expected_sha=entry.sha256,
        settings=settings,
    )
    return entry.model_type, config_path, checkpoint_path, entry.default_weight_path


def _resolve_custom_model(
    *,
    model_type: str,
    config: UploadFile | None,
    checkpoint: UploadFile | None,
    config_url: str | None,
    checkpoint_url: str | None,
    temp_dir: Path,
) -> tuple[str, Path, Path, None]:
    if not model_type:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'model' (catalog id) or 'model_type' for a custom model.",
        )

    if config is not None:
        config_path = save_upload(
            config.file.read(), Path(config.filename or "config.yaml").suffix or ".yaml", temp_dir
        )
    elif config_url:
        from .download import download_url

        config_path = download_url(config_url, temp_dir / "config.yaml", _settings())
    else:
        raise HTTPException(
            status_code=400,
            detail="Custom model requires 'config' (file) or 'config_url'.",
        )

    if checkpoint is not None:
        suffix = Path(checkpoint.filename or "model.ckpt").suffix or ".ckpt"
        checkpoint_path = save_upload(checkpoint.file.read(), suffix, temp_dir)
    elif checkpoint_url:
        from .download import download_url

        name = Path(checkpoint_url.split("?", 1)[0]).name or "model.ckpt"
        checkpoint_path = download_url(checkpoint_url, _settings().model_dir / "custom" / name, _settings())
    else:
        raise HTTPException(
            status_code=400,
            detail="Custom model requires 'checkpoint' (file) or 'checkpoint_url'.",
        )
    return model_type, config_path, checkpoint_path, None


@app.post("/api/msst/inference", tags=[TAG])
def inference(
    audio: UploadFile = File(..., description="Input audio file (wav/flac/mp3/m4a/...)."),
    model: str | None = Form(None, description="Catalog model id from GET /api/msst/models."),
    model_type: str | None = Form(None, description="Architecture key for a custom model."),
    config: UploadFile | None = File(None, description="Custom model YAML config."),
    checkpoint: UploadFile | None = File(None, description="Custom model checkpoint."),
    config_url: str | None = Form(None, description="URL to download a custom config."),
    checkpoint_url: str | None = Form(None, description="URL to download a custom checkpoint."),
    device: str | None = Form(None, description="Torch device, e.g. cuda:0 or cpu."),
    batch_size: int | None = Form(None),
    num_overlap: int | None = Form(None),
    chunk_size: int | None = Form(None),
    normalize: bool | None = Form(None),
    use_tta: bool = Form(False),
    bigshifts: int = Form(1),
    extract_instrumental: bool = Form(False),
    stems: str | None = Form(None, description="Comma-separated subset of output stems."),
    output_format: str = Form("wav", description="wav | flac | mp3"),
    bit_depth: str | None = Form(None, description="e.g. PCM_16, PCM_24, 320k."),
    response_format: str = Form("zip", description="zip | json"),
    source: str | None = Form(None, description="Download backend: modelscope | huggingface | auto."),
):
    settings = _settings()
    started = time.time()

    output_format = (output_format or "wav").strip().lower()
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise HTTPException(status_code=400, detail=f"Invalid output_format: {output_format}")
    response_format = (response_format or "zip").strip().lower()
    if response_format not in {"zip", "json"}:
        raise HTTPException(status_code=400, detail="response_format must be 'zip' or 'json'")
    if bigshifts < 1:
        raise HTTPException(status_code=400, detail="bigshifts must be >= 1")

    temp_dir = Path(tempfile.mkdtemp(prefix="req-", dir=str(settings.temp_dir)))
    try:
        if model:
            resolved_type, config_path, checkpoint_path, _ = _resolve_catalog_model(model, source)
        else:
            resolved_type, config_path, checkpoint_path, _ = _resolve_custom_model(
                model_type=model_type or "",
                config=config,
                checkpoint=checkpoint,
                config_url=config_url,
                checkpoint_url=checkpoint_url,
                temp_dir=temp_dir,
            )

        spec = ModelSpec(
            model_type=resolved_type,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=(device or settings.device),
            use_tta=use_tta,
            bigshifts=bigshifts,
            extract_instrumental=extract_instrumental,
            batch_size=batch_size,
            num_overlap=num_overlap,
            chunk_size=chunk_size,
            normalize=normalize,
        )

        try:
            separator = _manager().get(spec)
        except InferenceError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        # Decode the upload at the model's expected sample rate.
        audio_bytes = audio.file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio upload")
        max_bytes = int(settings.max_upload_mb * 1024 * 1024)
        if len(audio_bytes) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds MSST_MAX_UPLOAD_MB ({settings.max_upload_mb} MB)",
            )
        upload_path = save_upload(
            audio_bytes, Path(audio.filename or "input.wav").suffix or ".wav", temp_dir
        )
        try:
            mix = decode_audio(upload_path, separator.sample_rate)
        except Exception as error:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Cannot decode audio: {error}") from error

        try:
            results = _manager().separate(spec, mix)
        except InferenceError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:  # noqa: BLE001
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail=f"Inference failed: {error}") from error

        requested = None
        if stems:
            requested = [part.strip() for part in stems.split(",") if part.strip()]
            missing = [name for name in requested if name not in results]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Requested stems not produced: {missing}. Available: {list(results)}",
                )
            results = {name: results[name] for name in requested}

        encoded: dict[str, bytes] = {}
        for name, array in results.items():
            encoded[name] = encode_audio(
                array, separator.sample_rate, fmt=output_format, subtype=bit_depth
            )

        metadata = InferenceMetadata(
            model_id=model,
            model_type=spec.model_type,
            sample_rate=separator.sample_rate,
            device=str(separator.device),
            stems=list(encoded),
            output_format=output_format,
            elapsed_seconds=round(time.time() - started, 3),
        )

        if response_format == "json":
            payload = metadata.model_dump()
            payload["stems"] = {
                name: base64.b64encode(content).decode("ascii")
                for name, content in encoded.items()
            }
            return JSONResponse(content=payload)

        files = {f"{name}.{output_format}": content for name, content in encoded.items()}
        files["manifest.json"] = json.dumps(metadata.model_dump(), indent=2).encode("utf-8")
        archive = build_zip(files)
        headers = {"Content-Disposition": 'attachment; filename="stems.zip"'}
        return Response(content=archive, media_type="application/zip", headers=headers)
    finally:
        if not settings.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# RVC (voice conversion)
# ---------------------------------------------------------------------------
_RVC_TRUE = {"1", "true", "yes", "on", "y", "t"}

#: Pedalboard effect toggles + parameter defaults (used when ``post_process``).
_RVC_EFFECT_FLAGS = (
    "reverb",
    "pitch_shift",
    "limiter",
    "gain",
    "distortion",
    "chorus",
    "bitcrush",
    "clipping",
    "compressor",
    "delay",
)
_RVC_EFFECT_FLOATS = {
    "reverb_room_size": 0.5,
    "reverb_damping": 0.5,
    "reverb_wet_level": 0.33,
    "reverb_dry_level": 0.4,
    "reverb_width": 1.0,
    "reverb_freeze_mode": 0.0,
    "pitch_shift_semitones": 0.0,
    "limiter_threshold": -6.0,
    "limiter_release": 0.05,
    "gain_db": 0.0,
    "distortion_gain": 25.0,
    "chorus_rate": 1.0,
    "chorus_depth": 0.25,
    "chorus_delay": 7.0,
    "chorus_feedback": 0.0,
    "chorus_mix": 0.5,
    "clipping_threshold": 0.0,
    "compressor_threshold": 0.0,
    "compressor_ratio": 1.0,
    "compressor_attack": 1.0,
    "compressor_release": 100.0,
    "delay_seconds": 0.5,
    "delay_feedback": 0.0,
    "delay_mix": 0.5,
}

_RVC_F0_METHODS = {"rmvpe", "crepe", "crepe-tiny", "fcpe"}


def _build_rvc_params(form) -> RVCParams:  # noqa: ANN001 - Starlette FormData
    def _str(key: str, default: str = "") -> str:
        value = form.get(key)
        return default if value is None else str(value).strip()

    def _int(key: str, default: int) -> int:
        value = form.get(key)
        try:
            return int(float(value)) if value not in (None, "") else default
        except (TypeError, ValueError):
            return default

    def _float(key: str, default: float) -> float:
        value = form.get(key)
        try:
            return float(value) if value not in (None, "") else default
        except (TypeError, ValueError):
            return default

    def _bool(key: str, default: bool = False) -> bool:
        value = form.get(key)
        if value in (None, ""):
            return default
        return str(value).strip().lower() in _RVC_TRUE

    effects: dict = {}
    for flag in _RVC_EFFECT_FLAGS:
        effects[flag] = _bool(flag)
    for name, default in _RVC_EFFECT_FLOATS.items():
        effects[name] = _float(name, default)
    effects["bitcrush_bit_depth"] = _int("bitcrush_bit_depth", 8)

    return RVCParams(
        pitch=_int("pitch", 0),
        f0_method=_str("f0_method", "rmvpe").lower() or "rmvpe",
        index_rate=_float("index_rate", 0.75),
        volume_envelope=_float("volume_envelope", 1.0),
        protect=_float("protect", 0.5),
        split_audio=_bool("split_audio"),
        f0_autotune=_bool("f0_autotune"),
        f0_autotune_strength=_float("f0_autotune_strength", 1.0),
        clean_audio=_bool("clean_audio"),
        clean_strength=_float("clean_strength", 0.5),
        resample_sr=_int("resample_sr", 0),
        sid=_int("sid", 0),
        proposed_pitch=_bool("proposed_pitch"),
        proposed_pitch_threshold=_float("proposed_pitch_threshold", 155.0),
        post_process=_bool("post_process"),
        effects=effects,
    )


@app.get("/api/rvc/models", response_model=RVCVoiceListResponse, tags=[RVC_TAG])
def list_rvc_models() -> RVCVoiceListResponse:
    voices = [
        RVCVoiceInfo(**voice.to_public_dict())
        for voice in _rvc_manager().library.all()
    ]
    return RVCVoiceListResponse(total=len(voices), voices=voices)


@app.get("/api/rvc/models/{voice_id}", response_model=RVCVoiceInfo, tags=[RVC_TAG])
def get_rvc_model(voice_id: str) -> RVCVoiceInfo:
    voice = _rvc_manager().library.get(voice_id)
    return RVCVoiceInfo(**voice.to_public_dict())


@app.post("/api/rvc/inference", tags=[RVC_TAG])
async def rvc_inference(request: Request):
    """Convert ``audio`` to the timbre of ``model`` and stream the result back."""

    settings = _settings()
    started = time.time()

    form = await request.form()
    upload = form.get("audio") or form.get("file") or form.get("input")
    if upload is None or isinstance(upload, str):
        raise HTTPException(
            status_code=400, detail="multipart field 'audio' (file) is required."
        )

    model_value = form.get("model") or form.get("model_name") or form.get("pth")
    if not model_value:
        raise HTTPException(status_code=400, detail="Field 'model' is required.")

    output_format = (str(form.get("output_format", "wav")).strip().lower().lstrip(".")) or "wav"
    if output_format not in RVC_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid output_format: {output_format}. "
            f"Expected one of {sorted(RVC_OUTPUT_FORMATS)}",
        )

    library = _rvc_manager().library
    voice = library.get(str(model_value))

    index_value = form.get("index") or form.get("index_path")
    if index_value:
        index_path = library.resolve_index(str(index_value))
    else:
        index_path = voice.index_path

    params = _build_rvc_params(form)
    if params.f0_method not in _RVC_F0_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid f0_method: {params.f0_method}. "
            f"Expected one of {sorted(_RVC_F0_METHODS)}",
        )

    embedder = (str(form.get("embedder_model") or settings.rvc_embedder).strip().lower()
                or settings.rvc_embedder)
    embedder_custom = str(form.get("embedder_model_custom") or "").strip() or None
    try:
        device = resolve_device(
            str(form.get("device") or settings.device).strip() or settings.device, settings
        )
    except InferenceError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    spec = RVCVoiceSpec(
        model_path=voice.path,
        index_path=index_path,
        device=device,
        embedder=embedder,
        embedder_custom=embedder_custom,
    )

    audio_bytes = await upload.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds MSST_MAX_UPLOAD_MB ({settings.max_upload_mb} MB)",
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="rvc-", dir=str(settings.temp_dir)))
    try:
        upload_path = save_upload(
            audio_bytes, Path(upload.filename or "input.wav").suffix or ".wav", temp_dir
        )
        try:
            mix = decode_audio_mono(upload_path, 16000)
        except Exception as error:  # noqa: BLE001
            raise HTTPException(
                status_code=400, detail=f"Cannot decode audio: {error}"
            ) from error

        formant_shifting = str(form.get("formant_shifting", "")).strip().lower() in _RVC_TRUE
        if formant_shifting:
            from .rvc.lib.utils import load_audio_infer

            def _float(key: str, default: float) -> float:
                value = form.get(key)
                try:
                    return float(value) if value not in (None, "") else default
                except (TypeError, ValueError):
                    return default

            mix = load_audio_infer(
                mix,
                16000,
                formant_shifting=True,
                formant_qfrency=_float("formant_qfrency", 1.0),
                formant_timbre=_float("formant_timbre", 1.0),
            )

        try:
            converted, sample_rate = await run_in_threadpool(
                _rvc_manager().convert, spec, mix, params
            )
        except RVCVoiceNotFoundError:
            raise
        except RVCError:
            raise
        except Exception as error:  # noqa: BLE001
            logger.exception("RVC inference failed")
            raise HTTPException(
                status_code=500, detail=f"RVC inference failed: {error}"
            ) from error

        content, media_type = encode_audio_dynamic(converted, sample_rate, output_format)
        headers = {
            "Content-Disposition": f'inline; filename="output.{output_format}"',
            "X-RVC-Model": voice.name,
            "X-RVC-Sample-Rate": str(sample_rate),
            "X-RVC-F0-Method": params.f0_method,
            "X-RVC-Elapsed": f"{time.time() - started:.3f}",
        }
        return Response(content=content, media_type=media_type, headers=headers)
    finally:
        if not settings.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# ASR (single-sentence ``asr`` + streaming ``streaming_asr``)
# ---------------------------------------------------------------------------
def _asr_model_info(manager: ASRManager, entry) -> ASRModelInfo:  # noqa: ANN001
    public = entry.to_public_dict(downloaded=manager.is_downloaded(entry))
    return ASRModelInfo(
        id=public["id"],
        name=public["name"],
        capability=public["capability"],
        kind=public["kind"],
        description=public["description"],
        tasks=public["tasks"],
        languages=public["languages"],
        downloaded=public["downloaded"],
        extra=public.get("extra"),
    )


def _v2m_model_info(manager: V2MManager, entry) -> V2MModelInfo:  # noqa: ANN001
    public = entry.to_public_dict(downloaded=manager.is_downloaded(entry))
    return V2MModelInfo(
        id=public["id"],
        name=public["name"],
        capability=public["capability"],
        kind=public["kind"],
        description=public["description"],
        tasks=public["tasks"],
        languages=public["languages"],
        downloaded=public["downloaded"],
        extra=public.get("extra"),
    )


@app.get("/api/asr/models", response_model=ASRModelListResponse, tags=[ASR_TAG])
def list_asr_models(
    task: str | None = None, downloaded: bool | None = None
) -> ASRModelListResponse:
    manager = _asr_manager()
    items: list[ASRModelInfo] = []
    for entry in manager.models():
        if task and task not in entry.tasks:
            continue
        is_downloaded = manager.is_downloaded(entry)
        if downloaded is not None and is_downloaded != downloaded:
            continue
        items.append(_asr_model_info(manager, entry))
    return ASRModelListResponse(total=len(items), models=items)


@app.get("/api/asr/models/{model_id}", response_model=ASRModelInfo, tags=[ASR_TAG])
def get_asr_model(model_id: str) -> ASRModelInfo:
    manager = _asr_manager()
    return _asr_model_info(manager, manager.get_entry(model_id))


@app.post("/api/asr/models/{model_id}/download", response_model=ASRModelInfo, tags=[ASR_TAG])
def download_asr_model(model_id: str, source: str | None = None) -> ASRModelInfo:
    manager = _asr_manager()
    entry, _path = manager.ensure_path(model_id, source=source)
    return _asr_model_info(manager, entry)


def _read_upload_limited(upload: UploadFile, settings: Settings) -> bytes:
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds MSST_MAX_UPLOAD_MB ({settings.max_upload_mb} MB)",
        )
    return data


@app.post("/api/asr/transcribe", response_model=ASRTranscriptionResponse, tags=[ASR_TAG])
@app.post(
    "/api/asr/inference",
    response_model=ASRTranscriptionResponse,
    include_in_schema=False,
    tags=[ASR_TAG],
)
def asr_transcribe(
    audio: UploadFile = File(..., description="Input audio (wav/flac/mp3/m4a/...)."),
    model: str | None = Form(None, description="ASR model id from GET /api/asr/models."),
    language: str | None = Form(None, description="Language hint, e.g. zh/ja/en/yue."),
    context: str | None = Form(None, description="Optional ASR context / prompt."),
    device: str | None = Form(None),
    source: str | None = Form(None, description="Download backend: modelscope | huggingface."),
):
    settings = _settings()
    started = time.time()
    model_id = (model or settings.asr_default_model).strip()
    temp_dir = Path(tempfile.mkdtemp(prefix="asr-", dir=str(settings.temp_dir)))
    try:
        data = _read_upload_limited(audio, settings)
        upload_path = save_upload(
            data, Path(audio.filename or "input.wav").suffix or ".wav", temp_dir
        )
        try:
            wav = decode_audio_mono(upload_path, 16000)
        except Exception as error:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Cannot decode audio: {error}") from error
        result = _asr_manager().transcribe(
            model_id,
            wav,
            language=language,
            context=context,
            device=device,
            source=source,
        )
    finally:
        if not settings.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
    return ASRTranscriptionResponse(
        model=model_id,
        text=result.text,
        language=result.language,
        phonemes=result.phonemes,
        elapsed_seconds=round(time.time() - started, 3),
    )


def _pcm_to_16k(data: bytes, fmt: str, sample_rate: int):  # noqa: ANN201
    if fmt in {"pcm_s16le", "s16le", "int16"}:
        audio = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    else:
        audio = np.frombuffer(data, dtype="<f4").astype(np.float32)
    if sample_rate != 16000 and audio.size:
        import soxr

        audio = soxr.resample(audio, sample_rate, 16000).astype(np.float32)
    return audio


@app.websocket("/api/asr/streaming")
@app.websocket("/api/asr/streaming_asr")
async def asr_streaming(websocket: WebSocket) -> None:
    """Streaming ASR (``streaming_asr``). Qwen3-ASR only.

    Protocol: the client opens the socket with ``?model=&language=&format=&sample_rate=``
    then sends binary audio frames (float32 little-endian PCM at ``sample_rate``,
    or int16 when ``format=pcm_s16le``).  The server replies with JSON messages:
    ``{"type": "ready"|"partial"|"final"|"error", ...}``.  Send ``"finish"``
    (text) or ``{"type": "finish"}` (JSON) to flush the final transcript.
    """

    await websocket.accept()
    settings = _settings()
    params = websocket.query_params
    model_id = (params.get("model") or settings.asr_default_model).strip()
    language = params.get("language")
    context = params.get("context")
    fmt = (params.get("format") or "pcm_f32").lower()
    try:
        sample_rate = int(params.get("sample_rate") or 16000)
    except ValueError:
        sample_rate = 16000

    session = None
    try:
        try:
            session = await run_in_threadpool(
                _asr_manager().open_stream,
                model_id,
                language=language,
                context=context,
                device=params.get("device"),
                source=params.get("source"),
            )
        except (ASRError, ASRModelNotFoundError) as error:
            await websocket.send_json({"type": "error", "detail": str(error)})
            await websocket.close()
            return

        await websocket.send_json({"type": "ready", "model": model_id})
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                break
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                wav = _pcm_to_16k(message["bytes"], fmt, sample_rate)
                try:
                    result = await run_in_threadpool(session.feed, wav)
                except Exception as error:  # noqa: BLE001
                    await websocket.send_json({"type": "error", "detail": str(error)})
                    break
                await websocket.send_json(
                    {"type": "partial", "text": result.text, "language": result.language}
                )
                continue
            text = message.get("text")
            if text is None:
                continue
            payload = text.strip()
            if payload in {"finish", "final", "stop"} or '"finish"' in payload:
                result = await run_in_threadpool(session.finish)
                await websocket.send_json(
                    {"type": "final", "text": result.text, "language": result.language}
                )
                break
    except WebSocketDisconnect:
        pass
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001
                logger.exception("Error closing ASR streaming session")


# ---------------------------------------------------------------------------
# Vocal2Midi (v2m)
# ---------------------------------------------------------------------------
@app.get("/api/v2m/models", response_model=V2MModelListResponse, tags=[V2M_TAG])
def list_v2m_models(
    task: str | None = None, downloaded: bool | None = None
) -> V2MModelListResponse:
    manager = _v2m_manager()
    items: list[V2MModelInfo] = []
    for entry in manager.models():
        if task and task not in entry.tasks:
            continue
        is_downloaded = manager.is_downloaded(entry)
        if downloaded is not None and is_downloaded != downloaded:
            continue
        items.append(_v2m_model_info(manager, entry))
    return V2MModelListResponse(total=len(items), models=items)


@app.get("/api/v2m/models/{model_id}", response_model=V2MModelInfo, tags=[V2M_TAG])
def get_v2m_model(model_id: str) -> V2MModelInfo:
    manager = _v2m_manager()
    return _v2m_model_info(manager, manager.get_entry(model_id))


@app.post("/api/v2m/models/{model_id}/download", response_model=V2MModelInfo, tags=[V2M_TAG])
def download_v2m_model(model_id: str, source: str | None = None) -> V2MModelInfo:
    manager = _v2m_manager()
    manager.resolve_path(model_id, source=source)
    return _v2m_model_info(manager, manager.get_entry(model_id))


_V2M_TRUE = {"1", "true", "yes", "on", "y", "t"}


def _form_str(form, key: str, default: str | None = None) -> str | None:  # noqa: ANN001
    value = form.get(key)
    if value is None:
        return default
    text = str(value).strip()
    return text if text != "" else default


def _form_float(form, key: str, default: float) -> float:  # noqa: ANN001
    value = form.get(key)
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _form_int(form, key: str, default: int) -> int:  # noqa: ANN001
    value = form.get(key)
    try:
        return int(float(value)) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _form_bool(form, key: str, default: bool = False) -> bool:  # noqa: ANN001
    value = form.get(key)
    if value in (None, ""):
        return default
    return str(value).strip().lower() in _V2M_TRUE


def _build_v2m_request(form) -> object:  # noqa: ANN001
    from .v2m.pipeline import SUPPORTED_OUTPUT_FORMATS, V2MRequest

    formats_raw = _form_str(form, "output_formats") or _form_str(form, "output_format") or "mid"
    formats = [item.strip().lower() for item in formats_raw.split(",") if item.strip()]
    invalid = [fmt for fmt in formats if fmt not in SUPPORTED_OUTPUT_FORMATS]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid output format(s): {invalid}. Expected {sorted(SUPPORTED_OUTPUT_FORMATS)}",
        )
    return V2MRequest(
        language=_form_str(form, "language", "auto") or "auto",
        lyric_output_mode=_form_str(form, "lyric_output_mode", "auto") or "auto",
        original_lyrics=_form_str(form, "original_lyrics", "") or "",
        output_formats=formats,
        slicing_method=_form_str(form, "slicing_method", "auto") or "auto",
        slice_min_sec=_form_float(form, "slice_min_sec", 5.0),
        slice_max_sec=_form_float(form, "slice_max_sec", 10.0),
        tempo=_form_float(form, "tempo", 120.0),
        quantization_step=_form_int(form, "quantization_step", 0),
        quantization_mode=_form_str(form, "quantization_mode", "smart") or "smart",
        quant_simplicity=_form_float(form, "quant_simplicity", 0.0),
        pitch_format=_form_str(form, "pitch_format", "name") or "name",
        round_pitch=_form_bool(form, "round_pitch", True),
        seg_threshold=_form_float(form, "seg_threshold", 0.2),
        seg_radius=_form_float(form, "seg_radius", 0.02),
        est_threshold=_form_float(form, "est_threshold", 0.2),
        batch_size=_form_int(form, "batch_size", 1),
        asr_batch_size=_form_int(form, "asr_batch_size", 2),
        output_lyrics=_form_bool(form, "output_lyrics", True),
        output_pitch_curve=_form_bool(form, "output_pitch_curve", False),
        chinese_asr_engine=_form_str(form, "chinese_asr_engine", "qwen") or "qwen",
        japanese_asr_engine=_form_str(form, "japanese_asr_engine", "romaji") or "romaji",
        alignment_engine=_form_str(form, "alignment_engine", "tifa") or "tifa",
        japanese_g2p_engine=_form_str(form, "japanese_g2p_engine", "kashi-g2p-onnx") or "kashi-g2p-onnx",
        asr_model=_form_str(form, "asr_model"),
        pinyin_asr_model=_form_str(form, "pinyin_asr_model", "pinyin-asr") or "pinyin-asr",
        romaji_asr_model=_form_str(form, "romaji_asr_model", "romaji-asr") or "romaji-asr",
        game_model=_form_str(form, "game_model", "game") or "game",
        tifa_model=_form_str(form, "tifa_model", "tifa") or "tifa",
        hfa_model=_form_str(form, "hfa_model", "hubertfa") or "hubertfa",
        rmvpe_model=_form_str(form, "rmvpe_model", "rmvpe") or "rmvpe",
        kashi_g2p_model=_form_str(form, "kashi_g2p_model", "kashi-g2p-onnx") or "kashi-g2p-onnx",
        device=_form_str(form, "device"),
        source=_form_str(form, "source"),
    )


@app.post("/api/v2m/inference", tags=[V2M_TAG])
async def v2m_inference(request: Request):
    """Vocal2Midi pipeline: vocal audio -> lyric-aligned MIDI (zip)."""

    settings = _settings()
    started = time.time()
    form = await request.form()
    upload = form.get("audio") or form.get("file") or form.get("input")
    if upload is None or isinstance(upload, str):
        raise HTTPException(status_code=400, detail="multipart field 'audio' (file) is required.")

    v2m_request = _build_v2m_request(form)
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds MSST_MAX_UPLOAD_MB ({settings.max_upload_mb} MB)",
        )

    output_key = Path(upload.filename or "vocal.wav").stem or "vocal"
    temp_dir = Path(tempfile.mkdtemp(prefix="v2m-", dir=str(settings.temp_dir)))
    output_dir = temp_dir / "out"
    try:
        upload_path = save_upload(
            data, Path(upload.filename or "vocal.wav").suffix or ".wav", temp_dir
        )
        from .v2m.pipeline import run_v2m_pipeline

        def _run():
            return run_v2m_pipeline(
                settings=settings,
                asr_manager=_asr_manager(),
                v2m=_v2m_manager(),
                audio_path=upload_path,
                output_filename=f"{output_key}.wav",
                output_dir=output_dir,
                request=v2m_request,
            )

        try:
            outputs = await run_in_threadpool(_run)
        except (V2MError, ASRError):
            raise
        except Exception as error:  # noqa: BLE001
            logger.exception("v2m inference failed")
            raise HTTPException(status_code=500, detail=f"v2m inference failed: {error}") from error

        metadata = V2MInferenceMetadata(
            language=v2m_request.language,
            lyric_output_mode=v2m_request.lyric_output_mode,
            output_formats=v2m_request.output_formats,
            alignment_engine=v2m_request.alignment_engine,
            chinese_asr_engine=v2m_request.chinese_asr_engine,
            japanese_asr_engine=v2m_request.japanese_asr_engine,
            files=sorted(outputs),
            elapsed_seconds=round(time.time() - started, 3),
        )
        files = dict(outputs)
        files["manifest.json"] = json.dumps(metadata.model_dump(), indent=2).encode("utf-8")
        archive = build_zip(files)
        headers = {"Content-Disposition": f'attachment; filename="{output_key}.zip"'}
        return Response(content=archive, media_type="application/zip", headers=headers)
    finally:
        if not settings.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


async def _decode_capability_audio(request: Request, sample_rate: int):  # noqa: ANN202
    form = await request.form()
    upload = form.get("audio") or form.get("file") or form.get("input")
    if upload is None or isinstance(upload, str):
        raise HTTPException(status_code=400, detail="multipart field 'audio' (file) is required.")
    settings = _settings()
    data = await upload.read()
    temp_dir = Path(tempfile.mkdtemp(prefix="v2m-cap-", dir=str(settings.temp_dir)))
    upload_path = save_upload(
        data, Path(upload.filename or "input.wav").suffix or ".wav", temp_dir
    )
    return form, upload_path, temp_dir


@app.post("/api/v2m/slice", tags=[V2M_TAG])
async def v2m_slice(request: Request):
    """Capability: slice audio into chunks and return a zip of wavs."""

    settings = _settings()
    form = await request.form()
    upload = form.get("audio") or form.get("file") or form.get("input")
    if upload is None or isinstance(upload, str):
        raise HTTPException(status_code=400, detail="multipart field 'audio' (file) is required.")
    sample_rate = _form_int(form, "sample_rate", 44100)
    data = await upload.read()
    temp_dir = Path(tempfile.mkdtemp(prefix="v2m-slice-", dir=str(settings.temp_dir)))
    try:
        upload_path = save_upload(data, Path(upload.filename or "a.wav").suffix or ".wav", temp_dir)
        wav = decode_audio_mono(upload_path, sample_rate)

        from .v2m._vendor.inference.API.slicer_api import (
            normalize_slicing_method,
            slice_audio_with_custom_bounds,
        )

        raw_chunks = slice_audio_with_custom_bounds(
            wav,
            sample_rate,
            normalize_slicing_method(_form_str(form, "slicing_method", "auto") or "auto"),
            min_len_sec=_form_float(form, "slice_min_sec", 5.0),
            max_len_sec=_form_float(form, "slice_max_sec", 10.0),
        )
        manifest = []
        files: dict[str, bytes] = {}
        for idx, chunk in enumerate(raw_chunks):
            waveform = np.asarray(chunk["waveform"], dtype=np.float32)
            if waveform.ndim == 1:
                waveform = waveform[:, np.newaxis]
            files[f"chunk_{idx:03d}.wav"] = encode_audio(waveform, sample_rate, fmt="wav")
            manifest.append({"offset": float(chunk["offset"]), "num_samples": int(waveform.shape[0])})
        files["manifest.json"] = json.dumps({"chunks": manifest}, indent=2).encode("utf-8")
        archive = build_zip(files)
        return Response(
            content=archive,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="chunks.zip"'},
        )
    finally:
        if not settings.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/api/v2m/pitch", tags=[V2M_TAG])
async def v2m_pitch(request: Request):
    """Capability: RMVPE F0 curve for an audio clip."""

    settings = _settings()
    form, upload_path, temp_dir = await _decode_capability_audio(request, 16000)
    try:
        wav = decode_audio_mono(upload_path, 16000)
        from .v2m.capabilities import extract_pitch

        return JSONResponse(
            extract_pitch(
                _v2m_manager(),
                wav,
                16000,
                device=_form_str(form, "device"),
                source=_form_str(form, "source"),
            )
        )
    finally:
        if not settings.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/api/v2m/notes", tags=[V2M_TAG])
async def v2m_notes(request: Request):
    """Capability: GAME note extraction (pitch-only) for an audio clip."""

    settings = _settings()
    form, upload_path, temp_dir = await _decode_capability_audio(request, 44100)
    try:
        wav = decode_audio_mono(upload_path, 44100)
        from .v2m.capabilities import extract_notes

        notes = extract_notes(
            _v2m_manager(),
            wav,
            44100,
            device=_form_str(form, "device"),
            source=_form_str(form, "source"),
        )
        return JSONResponse({"notes": notes, "count": len(notes)})
    finally:
        if not settings.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/api/v2m/align", tags=[V2M_TAG])
async def v2m_align(request: Request):
    """Capability: forced alignment of a known transcript (TiFA)."""

    settings = _settings()
    form, upload_path, temp_dir = await _decode_capability_audio(request, 44100)
    try:
        text = _form_str(form, "text") or _form_str(form, "lyrics")
        if not text:
            raise HTTPException(status_code=400, detail="Field 'text' is required for alignment.")
        wav = decode_audio_mono(upload_path, 44100)
        from .v2m.capabilities import align_text

        return JSONResponse(
            align_text(
                _v2m_manager(),
                wav,
                44100,
                text,
                language=_form_str(form, "language", "zh") or "zh",
                device=_form_str(form, "device"),
                source=_form_str(form, "source"),
            )
        )
    finally:
        if not settings.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/api/v2m/g2p", tags=[V2M_TAG])
def v2m_g2p(
    text: str = Form(..., description="Text to convert to a reading."),
    language: str = Form("ja"),
    engine: str = Form("kashi-g2p-onnx"),
    source: str | None = Form(None),
):
    """Capability: grapheme-to-phoneme reading (kashi-g2p / pinyin)."""

    from .v2m.capabilities import g2p_reading

    return JSONResponse(
        g2p_reading(
            _v2m_manager(),
            text,
            language=language,
            engine=engine,
            source=source,
        )
    )


# ---------------------------------------------------------------------------
# DiffSinger (SVS)
# ---------------------------------------------------------------------------
def _svs_voice_info(voice) -> SVSVoiceInfo:  # noqa: ANN001
    from .svs.voicebank import probe_voicebank

    return SVSVoiceInfo(**probe_voicebank(voice.root))


@app.get("/api/diffsinger/models", response_model=SVSVoiceListResponse, tags=[SVS_TAG])
def list_svs_models() -> SVSVoiceListResponse:
    items: list[SVSVoiceInfo] = []
    for voice in _svs_manager().library.all():
        try:
            items.append(_svs_voice_info(voice))
        except Exception:  # noqa: BLE001 - skip unreadable voicebanks
            logger.exception("Skipping unreadable voicebank %s", voice.id)
    return SVSVoiceListResponse(total=len(items), voices=items)


@app.get("/api/diffsinger/models/{voice_id}", response_model=SVSVoiceInfo, tags=[SVS_TAG])
def get_svs_model(voice_id: str) -> SVSVoiceInfo:
    voice = _svs_manager().library.get(voice_id)
    return _svs_voice_info(voice)


_SVS_MEDIA_TYPES = {"wav": "audio/wav", "flac": "audio/flac", "mp3": "audio/mpeg"}


def _header_safe(value: str) -> str:
    """HTTP headers are latin-1; percent-encode anything outside it."""

    try:
        value.encode("latin-1")
        return value
    except UnicodeEncodeError:
        from urllib.parse import quote

        return quote(value)


def _build_svs_request(form) -> SVSRequest:  # noqa: ANN001
    return SVSRequest(
        language=_form_str(form, "language", "auto") or "auto",
        pitch_steps=_form_int(form, "pitch_steps", 20),
        variance_steps=_form_int(form, "variance_steps", 20),
        depth=_form_float(form, "depth", 1.0),
        steps=_form_int(form, "steps", 20),
        generate_pitch=_form_bool(form, "generate_pitch", True),
        render=_form_bool(form, "render", True),
    )


async def _read_svs_form(request: Request):  # noqa: ANN202
    form = await request.form()
    upload = (
        form.get("ustx")
        or form.get("ustx_file")
        or form.get("file")
        or form.get("input")
    )
    if upload is None or isinstance(upload, str):
        raise HTTPException(
            status_code=400, detail="multipart field 'ustx' (file) is required."
        )
    model_value = _form_str(form, "model") or _settings().svs_default_model
    if not model_value:
        raise HTTPException(
            status_code=400,
            detail="Field 'model' (voicebank id) is required.",
        )
    voice = _svs_manager().library.get(str(model_value))
    return form, upload, voice


@app.post("/api/diffsinger/inference", tags=[SVS_TAG])
async def diffsinger_inference(request: Request):
    """Render an OpenUtau ``ustx`` score to audio with a DiffSinger voicebank.

    Runs auto-pitch (``pitd``) then the variance/acoustic/vocoder chain.  Returns
    a single audio file by default; set ``return_ustx=true`` to get a zip with
    the audio and the regenerated ``pitd`` score.
    """

    settings = _settings()
    started = time.time()
    form, upload, voice = await _read_svs_form(request)

    output_format = (
        _form_str(form, "output_format", "wav") or "wav"
    ).strip().lower().lstrip(".")
    if output_format not in _SVS_MEDIA_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid output_format: {output_format}. Expected wav | flac | mp3",
        )
    bit_depth = _form_str(form, "bit_depth")
    device = _form_str(form, "device") or settings.device
    svs_request = _build_svs_request(form)
    return_ustx = _form_bool(form, "return_ustx", False)

    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty USTX upload")
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds MSST_MAX_UPLOAD_MB ({settings.max_upload_mb} MB)",
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="svs-", dir=str(settings.temp_dir)))
    try:
        ustx_path = save_upload(data, ".ustx", temp_dir)
        spec = SVSSpec(root=voice.root, device=str(device))
        try:
            result = await run_in_threadpool(
                _svs_manager().render, spec, str(ustx_path), svs_request
            )
        except (SVSVoiceNotFoundError, SVSError):
            raise
        except Exception as error:  # noqa: BLE001
            logger.exception("DiffSinger inference failed")
            raise HTTPException(
                status_code=500, detail=f"DiffSinger inference failed: {error}"
            ) from error

        metadata = {
            "model": voice.id,
            "language": result.language,
            "phonemizer": result.phonemizer,
            "phrases": result.phrases,
            "notes": result.notes,
            "sample_rate": result.sample_rate,
            "duration_seconds": round(result.duration_seconds, 3),
            "pitch_steps": svs_request.pitch_steps,
            "variance_steps": svs_request.variance_steps,
            "depth": svs_request.depth,
            "steps": svs_request.steps,
            "elapsed_seconds": round(time.time() - started, 3),
        }

        if return_ustx or not svs_request.render:
            files: dict[str, bytes] = {
                f"{ustx_path.stem}.pitd.ustx": result.ustx_bytes,
                "manifest.json": json.dumps(metadata, indent=2).encode("utf-8"),
            }
            if svs_request.render:
                content = encode_audio(
                    result.samples,
                    result.sample_rate,
                    fmt=output_format,
                    subtype=bit_depth,
                )
                files[f"{ustx_path.stem}.{output_format}"] = content
            archive = build_zip(files)
            headers = {
                "Content-Disposition": f'attachment; filename="{ustx_path.stem}.zip"',
                "X-SVS-Model": _header_safe(voice.id),
                "X-SVS-Language": result.language,
            }
            return Response(
                content=archive, media_type="application/zip", headers=headers
            )

        content = encode_audio(
            result.samples, result.sample_rate, fmt=output_format, subtype=bit_depth
        )
        headers = {
            "Content-Disposition": f'inline; filename="output.{output_format}"',
            "X-SVS-Model": _header_safe(voice.id),
            "X-SVS-Language": result.language,
            "X-SVS-Sample-Rate": str(result.sample_rate),
            "X-SVS-Elapsed": f"{time.time() - started:.3f}",
        }
        return Response(
            content=content, media_type=_SVS_MEDIA_TYPES[output_format], headers=headers
        )
    finally:
        if not settings.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/api/diffsinger/pitch", tags=[SVS_TAG])
async def diffsinger_pitch(request: Request):
    """Auto-pitch only: ``ustx -> ustx`` with a generated ``pitd`` curve."""

    settings = _settings()
    started = time.time()
    form, upload, voice = await _read_svs_form(request)
    svs_request = _build_svs_request(form)
    svs_request.render = False
    svs_request.generate_pitch = True

    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty USTX upload")
    temp_dir = Path(tempfile.mkdtemp(prefix="svs-pitch-", dir=str(settings.temp_dir)))
    try:
        ustx_path = save_upload(data, ".ustx", temp_dir)
        spec = SVSSpec(root=voice.root, device=str(_form_str(form, "device") or settings.device))
        try:
            result = await run_in_threadpool(
                _svs_manager().render, spec, str(ustx_path), svs_request
            )
        except (SVSVoiceNotFoundError, SVSError):
            raise
        except Exception as error:  # noqa: BLE001
            logger.exception("DiffSinger pitch generation failed")
            raise HTTPException(
                status_code=500, detail=f"DiffSinger pitch generation failed: {error}"
            ) from error
        headers = {
            "Content-Disposition": f'inline; filename="{ustx_path.stem}.pitd.ustx"',
            "X-SVS-Model": _header_safe(voice.id),
            "X-SVS-Language": result.language,
            "X-SVS-Elapsed": f"{time.time() - started:.3f}",
        }
        return Response(
            content=result.ustx_bytes,
            media_type="application/x-yaml",
            headers=headers,
        )
    finally:
        if not settings.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


def run() -> None:
    """Console entry point (``msst-api``)."""

    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "msst_api.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    run()
