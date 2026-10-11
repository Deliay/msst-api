"""Runtime configuration for the MSST API, sourced from environment variables.

Every setting has a sensible default so the container can run with a single
mounted model directory.  The two settings that matter most for deployment are
:data:`Settings.model_dir` (where downloaded checkpoints live) and
:data:`Settings.download_backend` (which mirror is preferred when a model is
missing).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
BUNDLED_CONFIG_DIR = _PACKAGE_ROOT / "data" / "configs"
BUNDLED_REGISTRY = _PACKAGE_ROOT / "data" / "registry.json"

_TRUE = {"1", "true", "yes", "on", "y", "t"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return Path(raw).expanduser()


def _env_path_opt(name: str) -> Path | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return Path(raw).expanduser()


@dataclass(slots=True)
class Settings:
    """All tunables for the service, resolved from the environment."""

    # --- Storage -----------------------------------------------------------
    #: Directory that holds downloaded checkpoints (``MSST_MODEL_DIR``).
    model_dir: Path = field(
        default_factory=lambda: _env_path("MSST_MODEL_DIR", Path("./models"))
    )
    #: Directory with per-model YAML configs (``MSST_CONFIG_DIR``).  Defaults to
    #: the configs bundled inside the package.
    config_dir: Path = field(
        default_factory=lambda: _env_path("MSST_CONFIG_DIR", BUNDLED_CONFIG_DIR)
    )

    # --- RVC (voice conversion) -------------------------------------------
    #: Directory holding RVC voice models (``*.pth`` + optional ``*.index``).
    #: Defaults to ``<model_dir>/rvc_models`` (``MSST_RVC_MODEL_DIR``).
    rvc_model_dir: Path | None = field(
        default_factory=lambda: _env_path_opt("MSST_RVC_MODEL_DIR")
    )
    #: Directory caching auxiliary RVC assets (embedders, predictors).
    #: Defaults to ``<model_dir>/rvc_assets`` (``MSST_RVC_ASSET_DIR``).
    rvc_asset_dir: Path | None = field(
        default_factory=lambda: _env_path_opt("MSST_RVC_ASSET_DIR")
    )
    #: Default feature extractor for RVC conversion (``MSST_RVC_EMBEDDER``).
    rvc_embedder: str = field(
        default_factory=lambda: os.environ.get("MSST_RVC_EMBEDDER", "contentvec").strip().lower()
    )

    # --- ASR (speech / lyric recognition) ---------------------------------
    #: Directory holding ASR model bundles.  Defaults to ``<model_dir>/asr``
    #: (``MSST_ASR_MODEL_DIR``).
    asr_model_dir: Path | None = field(
        default_factory=lambda: _env_path_opt("MSST_ASR_MODEL_DIR")
    )
    #: Optional JSON catalog extending/replacing the bundled ASR catalog.
    asr_catalog: Path | None = field(
        default_factory=lambda: _env_path_opt("MSST_ASR_CATALOG")
    )
    #: ASR model used when a request omits ``model`` (``MSST_ASR_DEFAULT_MODEL``).
    asr_default_model: str = field(
        default_factory=lambda: os.environ.get("MSST_ASR_DEFAULT_MODEL", "qwen3-asr-1.7b").strip()
    )
    #: Preferred backend for streaming Qwen3-ASR: ``auto`` (vLLM if installed,
    #: else chunked transformers), ``vllm`` or ``transformers``
    #: (``MSST_ASR_STREAMING_BACKEND``).
    asr_streaming_backend: str = field(
        default_factory=lambda: os.environ.get("MSST_ASR_STREAMING_BACKEND", "auto").strip().lower()
    )
    #: dtype used to load Qwen3-ASR (``MSST_QWEN_ASR_DTYPE``).
    qwen_asr_dtype: str = field(
        default_factory=lambda: os.environ.get("MSST_QWEN_ASR_DTYPE", "bfloat16").strip().lower()
    )

    # --- Vocal2Midi --------------------------------------------------------
    #: Directory holding Vocal2Midi model bundles (GAME, TiFA, RMVPE, ...).
    #: Defaults to ``<model_dir>/v2m`` (``MSST_V2M_MODEL_DIR``).
    v2m_model_dir: Path | None = field(
        default_factory=lambda: _env_path_opt("MSST_V2M_MODEL_DIR")
    )
    #: Optional JSON catalog extending/replacing the bundled v2m catalog.
    v2m_catalog: Path | None = field(
        default_factory=lambda: _env_path_opt("MSST_V2M_CATALOG")
    )
    #: Backend for the v2m pitch curve: ``rvc`` (reuse the RVC/Applio RMVPE
    #: torch predictor, no extra download) or ``onnx`` (the RMVPE ONNX bundle).
    #: (``MSST_V2M_RMVPE_BACKEND``)
    v2m_rmvpe_backend: str = field(
        default_factory=lambda: os.environ.get("MSST_V2M_RMVPE_BACKEND", "rvc").strip().lower()
    )
    #: ONNX opset (18-20) used when exporting the TiFA aligner from its PyTorch
    #: checkpoint. (``MSST_V2M_TIFA_OPSET``)
    v2m_tifa_opset: int = field(
        default_factory=lambda: _env_int("MSST_V2M_TIFA_OPSET", 18)
    )
    #: Keep ``model.pt`` after exporting the TiFA ONNX graphs. (``MSST_V2M_TIFA_KEEP_SOURCE``)
    v2m_tifa_keep_source: bool = field(
        default_factory=lambda: _env_bool("MSST_V2M_TIFA_KEEP_SOURCE", False)
    )
    #: Hard cap (seconds) on each audio chunk fed to GAME.  GAME's estimator VRAM
    #: grows steeply with chunk length (roughly 2.8 GB at 30 s, 6 GB at 50 s,
    #: 8+ GB at 60 s on a 24 GB card), so long chunks are split before note
    #: extraction.  Set 0 to disable. (``MSST_V2M_MAX_CHUNK_SEC``)
    v2m_max_chunk_sec: float = field(
        default_factory=lambda: _env_float("MSST_V2M_MAX_CHUNK_SEC", 30.0)
    )

    # --- Server ------------------------------------------------------------
    host: str = field(default_factory=lambda: os.environ.get("MSST_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("MSST_PORT", 8000))

    # --- Inference ---------------------------------------------------------
    #: Default torch device, e.g. ``cuda:0`` or ``cpu``.  GPU-only by default.
    device: str = field(
        default_factory=lambda: os.environ.get("MSST_DEVICE", "cuda:0")
    )
    #: When false (default) inference refuses to run without CUDA.
    allow_cpu: bool = field(
        default_factory=lambda: _env_bool("MSST_ALLOW_CPU", False)
    )
    #: Maximum number of model instances kept resident (LRU evicted).
    max_loaded_models: int = field(
        default_factory=lambda: _env_int("MSST_MAX_LOADED_MODELS", 1)
    )
    #: Maximum number of concurrent inference requests.
    max_concurrency: int = field(
        default_factory=lambda: _env_int("MSST_MAX_CONCURRENCY", 1)
    )

    # --- Uploads -----------------------------------------------------------
    max_upload_mb: float = field(
        default_factory=lambda: _env_float("MSST_MAX_UPLOAD_MB", 512.0)
    )
    #: Delete per-request temporary directories once the response is produced.
    keep_temp: bool = field(
        default_factory=lambda: _env_bool("MSST_KEEP_TEMP", False)
    )
    temp_dir: Path = field(
        default_factory=lambda: _env_path("MSST_TEMP_DIR", Path("/tmp/msst-api"))
    )

    # --- Downloads ---------------------------------------------------------
    #: Preferred backend when a checkpoint is missing.  One of ``modelscope``
    #: (default), ``huggingface`` or ``auto`` (ModelScope first, then HF).
    download_backend: str = field(
        default_factory=lambda: os.environ.get(
            "MSST_DOWNLOAD_BACKEND", "modelscope"
        ).strip().lower()
    )
    #: Optional ModelScope repository that mirrors the Hugging Face layout
    #: (e.g. ``my-org/MSST-WebUI``).  When set, every catalog model is fetched
    #: from ModelScope using the same relative path as the HF repository.
    modelscope_mirror: str | None = field(
        default_factory=lambda: os.environ.get("MSST_MODELSCOPE_MIRROR") or None
    )
    #: When true the downloader fails instead of falling back to another mirror.
    strict_backend: bool = field(
        default_factory=lambda: _env_bool("MSST_STRICT_BACKEND", False)
    )
    #: Optional Hugging Face endpoint/mirror, e.g. ``https://hf-mirror.com``.
    hf_endpoint: str | None = field(
        default_factory=lambda: os.environ.get("MSST_HF_ENDPOINT")
        or os.environ.get("HF_ENDPOINT")
        or None
    )
    hf_token: str | None = field(
        default_factory=lambda: os.environ.get("MSST_HF_TOKEN")
        or os.environ.get("HF_TOKEN")
        or None
    )
    modelscope_token: str | None = field(
        default_factory=lambda: os.environ.get("MSST_MODELSCOPE_TOKEN")
        or os.environ.get("MODELSCOPE_API_TOKEN")
        or None
    )
    #: Per-read timeout (seconds) for HTTP downloads.  0 disables the timeout.
    download_timeout: float = field(
        default_factory=lambda: _env_float("MSST_DOWNLOAD_TIMEOUT", 60.0)
    )
    #: Verify the registry sha256 after download (disable for faster first run).
    verify_download: bool = field(
        default_factory=lambda: _env_bool("MSST_VERIFY_DOWNLOAD", True)
    )
    #: GitHub download proxy prefix. GitHub URLs are rewritten as
    #: ``<proxy><github_url>`` (e.g. ``https://ghfast.top/https://github.com/...``).
    #: Set to an empty string to disable. (``MSST_GITHUB_PROXY``)
    github_proxy: str = field(
        default_factory=lambda: os.environ.get(
            "MSST_GITHUB_PROXY", "https://ghfast.top/"
        ).strip()
    )

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir).expanduser().resolve()
        self.config_dir = Path(self.config_dir).expanduser().resolve()
        self.temp_dir = Path(self.temp_dir).expanduser().resolve()
        if self.rvc_model_dir is None:
            self.rvc_model_dir = self.model_dir / "rvc_models"
        if self.rvc_asset_dir is None:
            self.rvc_asset_dir = self.model_dir / "rvc_assets"
        self.rvc_model_dir = Path(self.rvc_model_dir).expanduser().resolve()
        self.rvc_asset_dir = Path(self.rvc_asset_dir).expanduser().resolve()
        if self.asr_model_dir is None:
            self.asr_model_dir = self.model_dir / "asr"
        if self.v2m_model_dir is None:
            self.v2m_model_dir = self.model_dir / "v2m"
        self.asr_model_dir = Path(self.asr_model_dir).expanduser().resolve()
        self.v2m_model_dir = Path(self.v2m_model_dir).expanduser().resolve()
        if self.asr_streaming_backend not in {"auto", "vllm", "transformers"}:
            self.asr_streaming_backend = "auto"
        if self.v2m_rmvpe_backend not in {"rvc", "onnx"}:
            self.v2m_rmvpe_backend = "rvc"
        if not 18 <= self.v2m_tifa_opset <= 20:
            self.v2m_tifa_opset = 18
        if self.v2m_max_chunk_sec < 0:
            self.v2m_max_chunk_sec = 0.0
        if self.download_backend not in {"modelscope", "huggingface", "auto"}:
            self.download_backend = "auto"

    def backend_order(self) -> list[str]:
        """Return the ordered list of download backends to try."""
        if self.download_backend == "huggingface":
            return ["huggingface", "modelscope"]
        # ``modelscope`` and ``auto`` both prefer ModelScope first.
        return ["modelscope", "huggingface"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""

    return Settings()
