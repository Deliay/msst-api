"""Locate and download the auxiliary assets RVC inference needs.

An RVC voice ``.pth`` only contains the synthesizer weights.  Feature
extraction and F0 estimation additionally need:

* an **embedder** (HuBERT / ContentVec) under ``<assets>/embedders/<name>/``;
* the **RMVPE** F0 predictor under ``<assets>/predictors/rmvpe.pt``.

Both are fetched on demand from the Applio Hugging Face repository.  FCPE is
bundled inside the ``torchfcpe`` wheel, so it needs no download.

The asset root is configured once at startup through :func:`set_assets_dir`;
it defaults to ``./rvc_assets`` so importing this module never touches the
network.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import requests

from .hf_endpoint import get_hf_endpoint

logger = logging.getLogger("msst_api.rvc.assets")

#: Hugging Face repository that hosts the Applio inference resources.
_APPLIO_REPO = "IAHispano/Applio"
_APPLIO_RESOURCES = "Resources"

_CHUNK = 1024 * 1024

#: Embedder name -> directory inside the HF ``Resources/embedders`` folder.
_EMBEDDER_DIRS = {
    "contentvec": "contentvec",
    "spin": "spin",
    "spin-v2": "spin-v2",
    "chinese-hubert-base": "chinese_hubert_base",
    "japanese-hubert-base": "japanese_hubert_base",
    "korean-hubert-base": "korean_hubert_base",
}

_ASSETS_DIR: Path | None = None


def set_assets_dir(path: str | Path) -> None:
    """Set the directory that holds downloaded embedders and predictors."""

    global _ASSETS_DIR
    _ASSETS_DIR = Path(path).expanduser()


def get_assets_dir() -> Path:
    """Return the configured asset root (``./rvc_assets`` by default)."""

    return _ASSETS_DIR if _ASSETS_DIR is not None else Path("./rvc_assets")


def predictor_path(name: str) -> Path:
    return get_assets_dir() / "predictors" / name


def embedder_dir(name: str) -> Path:
    return get_assets_dir() / "embedders" / _EMBEDDER_DIRS.get(name, name)


def _hf_url(relative: str) -> str:
    return f"{get_hf_endpoint()}/{_APPLIO_REPO}/resolve/main/{relative}"


def _download(relative: str, dest: Path) -> Path:
    """Download ``relative`` from the Applio repo to ``dest`` atomically."""

    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = _hf_url(relative)
    tmp = dest.with_name(f".{dest.name}.part")
    token = os.environ.get("MSST_HF_TOKEN") or os.environ.get("HF_TOKEN")
    timeout_env = os.environ.get("MSST_DOWNLOAD_TIMEOUT", "60")
    try:
        read_timeout = float(timeout_env) if timeout_env else None
    except ValueError:
        read_timeout = 60.0
    headers = {"User-Agent": "msst-api/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    timeout = (30.0, read_timeout) if read_timeout else None

    logger.info("Downloading RVC asset %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=_CHUNK):
                if chunk:
                    handle.write(chunk)
    os.replace(tmp, dest)
    return dest


def ensure_predictor(name: str = "rmvpe.pt") -> Path:
    """Ensure an F0 predictor weight is present and return its path."""

    return _download(f"{_APPLIO_RESOURCES}/predictors/{name}", predictor_path(name))


def ensure_embedder(name: str = "contentvec") -> Path:
    """Ensure the embedder directory is populated and return its path."""

    folder = _EMBEDDER_DIRS.get(name, name)
    target = embedder_dir(name)
    target.mkdir(parents=True, exist_ok=True)
    for filename in ("pytorch_model.bin", "config.json"):
        _download(
            f"{_APPLIO_RESOURCES}/embedders/{folder}/{filename}", target / filename
        )
    return target
