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

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir).expanduser().resolve()
        self.config_dir = Path(self.config_dir).expanduser().resolve()
        self.temp_dir = Path(self.temp_dir).expanduser().resolve()
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
