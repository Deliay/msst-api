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

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

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
from .rvc_engine import (
    RVCError,
    RVCManager,
    RVCParams,
    RVCVoiceNotFoundError,
    RVCVoiceSpec,
)
from .schemas import (
    HealthResponse,
    InferenceMetadata,
    ModelInfo,
    ModelListResponse,
    RVCVoiceInfo,
    RVCVoiceListResponse,
)
from .separator import InferenceError, ModelManager, ModelSpec, resolve_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("msst_api")

TAG = "MSST"
RVC_TAG = "RVC"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.rvc_model_dir.mkdir(parents=True, exist_ok=True)
    settings.rvc_asset_dir.mkdir(parents=True, exist_ok=True)
    app.state.settings = settings
    app.state.manager = ModelManager(settings)
    app.state.rvc_manager = RVCManager(settings)
    logger.info(
        "MSST API ready. model_dir=%s rvc_model_dir=%s device=%s",
        settings.model_dir,
        settings.rvc_model_dir,
        settings.device,
    )
    try:
        yield
    finally:
        app.state.manager.unload()
        app.state.rvc_manager.unload()


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
