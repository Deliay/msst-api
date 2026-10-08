"""Checkpoint download helpers with Hugging Face and ModelScope backends.

The catalog stores, for every model, one or more *sources*.  A source is a
dictionary describing how to fetch the weights:

``{"backend": "huggingface", "repo_id": "...", "file": "...", "url": "..."}``

At fetch time the downloader walks the configured backend order (ModelScope
first by default), deriving a ModelScope source from the mirrored HF layout when
``MSST_MODELSCOPE_MIRROR`` is configured, and falls back to the next backend
when a source is unavailable or fails.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Iterable

import requests

from .config import Settings, get_settings

logger = logging.getLogger("msst_api.download")

_CHUNK = 1024 * 1024


class DownloadError(RuntimeError):
    """Raised when a model or config cannot be downloaded from any backend."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_target(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest.with_name(f".{dest.name}.part")


def _clean_stale_staging(directory: Path, max_age: float = 3600.0) -> None:
    """Remove ``.hf-*``/``.ms-*`` staging dirs left by a crashed download."""

    import time as _time

    now = _time.time()
    try:
        children = list(directory.iterdir())
    except FileNotFoundError:
        return
    for child in children:
        if not child.is_dir():
            continue
        if not (child.name.startswith(".hf-") or child.name.startswith(".ms-")):
            continue
        try:
            if now - child.stat().st_mtime > max_age:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue


def _finalize(tmp_path: Path, dest: Path) -> Path:
    os.replace(tmp_path, dest)
    return dest


def _apply_hf_endpoint(url: str, settings: Settings) -> str:
    """Rewrite a huggingface.co URL to the configured mirror endpoint."""

    if settings.hf_endpoint and "huggingface.co" in url:
        base = settings.hf_endpoint.rstrip("/")
        return url.replace("https://huggingface.co", base)
    return url


def _download_http(url: str, dest: Path, settings: Settings) -> Path:
    url = _apply_hf_endpoint(url, settings)
    headers = {"User-Agent": "msst-api/0.1"}
    if settings.hf_token and "huggingface" in url:
        headers["Authorization"] = f"Bearer {settings.hf_token}"
    timeout = (30.0, settings.download_timeout) if settings.download_timeout else None
    tmp = _atomic_target(dest)
    logger.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=_CHUNK):
                if chunk:
                    handle.write(chunk)
    return _finalize(tmp, dest)


def _download_huggingface(
    repo_id: str, filename: str, dest: Path, settings: Settings
) -> Path:
    import tempfile

    # Bound the hub client so a stalled connection fails instead of hanging.
    os.environ.setdefault(
        "HF_HUB_DOWNLOAD_TIMEOUT", str(int(settings.download_timeout or 60))
    )
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    if settings.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    _clean_stale_staging(dest.parent)
    staging = Path(tempfile.mkdtemp(prefix=".hf-", dir=str(dest.parent)))
    logger.info("Downloading hf://%s/%s -> %s", repo_id, filename, dest)
    try:
        local_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(staging),
                endpoint=settings.hf_endpoint,
                token=settings.hf_token,
            )
        ).resolve()
        tmp = _atomic_target(dest)
        shutil.move(str(local_path), tmp)
        return _finalize(tmp, dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _download_modelscope(
    repo_id: str, filename: str, dest: Path, settings: Settings
) -> Path:
    import tempfile

    from modelscope.hub.file_download import model_file_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    _clean_stale_staging(dest.parent)
    staging = Path(tempfile.mkdtemp(prefix=".ms-", dir=str(dest.parent)))
    logger.info("Downloading modelscope://%s/%s -> %s", repo_id, filename, dest)
    try:
        local_path = Path(
            model_file_download(
                model_id=repo_id,
                file_path=filename,
                local_dir=str(staging),
                token=settings.modelscope_token,
            )
        ).resolve()
        tmp = _atomic_target(dest)
        shutil.move(str(local_path), tmp)
        return _finalize(tmp, dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _derive_modelscope_sources(
    sources: Iterable[dict], settings: Settings
) -> list[dict]:
    derived = [dict(src) for src in sources if src.get("backend") == "modelscope"]
    mirror = settings.modelscope_mirror
    if mirror:
        for src in sources:
            if src.get("backend") == "huggingface" and src.get("file"):
                derived.append(
                    {
                        "backend": "modelscope",
                        "repo_id": mirror,
                        "file": src["file"],
                    }
                )
    return derived


def _candidates_for(backend: str, sources: Iterable[dict], settings: Settings) -> list[dict]:
    if backend == "modelscope":
        return _derive_modelscope_sources(sources, settings)
    return [dict(src) for src in sources if src.get("backend") == backend]


def _fetch_candidate(candidate: dict, dest: Path, settings: Settings) -> Path:
    backend = candidate["backend"]
    repo_id = candidate.get("repo_id")
    filename = candidate.get("file")
    url = candidate.get("url")

    attempts: list[tuple[str, Callable[[], Path]]] = []
    if backend == "huggingface" and repo_id and filename:
        attempts.append(
            ("huggingface", lambda: _download_huggingface(repo_id, filename, dest, settings))
        )
    if backend == "modelscope" and repo_id and filename:
        attempts.append(
            ("modelscope", lambda: _download_modelscope(repo_id, filename, dest, settings))
        )
    if url:
        attempts.append(("url", lambda: _download_http(url, dest, settings)))
    if not attempts:
        raise DownloadError(
            f"Incomplete {backend} source, expected repo_id+file or url: {candidate!r}"
        )

    last_error: Exception | None = None
    for label, attempt in attempts:
        try:
            return attempt()
        except Exception as error:  # noqa: BLE001 - fall back to the next transport
            last_error = error
            logger.warning("%s transport failed for %s: %s", label, dest.name, error)
    raise DownloadError(f"All {backend} transports failed for {dest.name}: {last_error}")


def _verify(path: Path, expected_size: int | None, expected_sha: str | None, settings: Settings) -> None:
    if expected_size is not None:
        actual = path.stat().st_size
        if actual != int(expected_size):
            path.unlink(missing_ok=True)
            raise DownloadError(
                f"Size mismatch for {path.name}: expected {expected_size} bytes, got {actual}"
            )
    if settings.verify_download and expected_sha:
        actual_sha = sha256_file(path)
        if actual_sha.lower() != expected_sha.lower():
            path.unlink(missing_ok=True)
            raise DownloadError(
                f"sha256 mismatch for {path.name}: expected {expected_sha}, got {actual_sha}"
            )


def download_model(
    sources: Iterable[dict],
    dest: Path,
    *,
    expected_size: int | None = None,
    expected_sha: str | None = None,
    settings: Settings | None = None,
) -> Path:
    """Fetch a checkpoint to ``dest`` using the configured backend order."""

    settings = settings or get_settings()
    dest = Path(dest)
    if dest.is_file():
        try:
            _verify(dest, expected_size, expected_sha, settings)
            logger.info("Model already present: %s", dest)
            return dest
        except DownloadError:
            logger.warning("Existing model failed verification, re-downloading: %s", dest)

    sources = list(sources)
    if not sources:
        raise DownloadError("No download sources available for this model")

    errors: list[str] = []
    for backend in settings.backend_order():
        candidates = _candidates_for(backend, sources, settings)
        if not candidates:
            continue
        backend_error: Exception | None = None
        for candidate in candidates:
            try:
                path = _fetch_candidate(candidate, dest, settings)
                _verify(path, expected_size, expected_sha, settings)
                logger.info("Model ready: %s", dest)
                return dest
            except Exception as error:  # noqa: BLE001 - report and try next source
                backend_error = error
                logger.warning(
                    "Download via %s failed (%s): %s", backend, candidate, error
                )
        errors.append(f"{backend}: {backend_error}")
        if settings.strict_backend:
            raise DownloadError(
                f"Download failed for {dest.name} using {backend}: {backend_error}"
            )

    raise DownloadError(
        f"Could not download {dest.name} from any backend. Tried: {'; '.join(errors)}"
    )


def download_url(url: str, dest: Path, settings: Settings | None = None) -> Path:
    """Download an arbitrary URL to ``dest`` (used for ad-hoc configs/weights)."""

    settings = settings or get_settings()
    dest = Path(dest)
    tmp = _atomic_target(dest)
    headers = {"User-Agent": "msst-api/0.1"}
    timeout = (30.0, settings.download_timeout) if settings.download_timeout else None
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=_CHUNK):
                if chunk:
                    handle.write(chunk)
    return _finalize(tmp, dest)


def materialize(
    *,
    source_url: str | None,
    local_path: Path,
    expected_size: int | None = None,
    expected_sha: str | None = None,
    settings: Settings | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Ensure a file exists locally, downloading it from ``source_url`` if needed."""

    settings = settings or get_settings()
    local_path = Path(local_path)
    if local_path.is_file():
        return local_path
    if not source_url:
        raise DownloadError(f"File {local_path} is missing and no download URL was given")
    if progress:
        progress(f"downloading {source_url}")
    path = download_url(source_url, local_path, settings)
    _verify(path, expected_size, expected_sha, settings)
    return path
