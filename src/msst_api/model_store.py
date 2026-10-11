"""A small catalog + download store shared by the ASR and Vocal2Midi subsystems.

The MSST registry (:mod:`msst_api.registry`) maps one catalog entry to a single
checkpoint file.  The ASR and Vocal2Midi models, in contrast, are *bundles* of
several files (an ONNX graph plus vocab/metadata, or a whole Qwen snapshot), so
they need a slightly different store:

* an entry describes a model directory under a capability root
  (``<model_dir>/asr/<dir_name>`` and ``<model_dir>/v2m/<dir_name>``);
* ``required_files`` are checked to decide whether the bundle is present;
* ``snapshots`` list the Hub repositories a bundle can be fetched from, tried
  in the configured backend order, using :func:`msst_api.download.download_snapshot`.

Operators can point the store at an alternate JSON catalog with
``MSST_ASR_CATALOG`` / ``MSST_V2M_CATALOG``, or drop models into the capability
directory manually.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings, get_settings
from .download import DownloadError, download_snapshot

logger = logging.getLogger("msst_api.model_store")


class ModelAssetNotFound(KeyError):
    """Raised when a requested ASR / v2m model id is not in the catalog."""


@dataclass(frozen=True)
class ModelEntry:
    """A multi-file model bundle in the ASR / v2m catalog."""

    id: str
    capability: str
    kind: str
    name: str
    description: str = ""
    tasks: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    dir_name: str = ""
    required_files: tuple[str, ...] = ()
    required_globs: tuple[str, ...] = ()
    snapshots: tuple[dict, ...] = ()
    single_files: tuple[dict, ...] = ()
    archives: tuple[dict, ...] = ()
    extra: dict = field(default_factory=dict)

    @property
    def subdir(self) -> str:
        return self.dir_name or self.id

    def to_public_dict(self, *, downloaded: bool) -> dict:
        return {
            "id": self.id,
            "capability": self.capability,
            "kind": self.kind,
            "name": self.name,
            "description": self.description,
            "tasks": list(self.tasks),
            "languages": list(self.languages),
            "downloaded": downloaded,
            **({"extra": self.extra} if self.extra else {}),
        }


def _entry_from_dict(raw: dict, capability: str) -> ModelEntry:
    return ModelEntry(
        id=raw["id"],
        capability=capability,
        kind=raw.get("kind", ""),
        name=raw.get("name", raw["id"]),
        description=raw.get("description", ""),
        tasks=tuple(raw.get("tasks") or ()),
        languages=tuple(raw.get("languages") or ()),
        dir_name=raw.get("dir_name", ""),
        required_files=tuple(raw.get("required_files") or ()),
        required_globs=tuple(raw.get("required_globs") or ()),
        snapshots=tuple(raw.get("snapshots") or ()),
        single_files=tuple(raw.get("single_files") or ()),
        archives=tuple(raw.get("archives") or ()),
        extra=dict(raw.get("extra") or {}),
    )


def _safe_members(names, staging: Path):
    for name in names:
        target = (staging / name).resolve()
        if not str(target).startswith(str(staging.resolve())):
            raise DownloadError(f"Refusing to extract archive entry outside target: {name}")


def _extract_archive(archive: Path, staging: Path) -> None:
    """Extract a zip/tar archive into ``staging`` (guarding against traversal)."""

    staging.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            _safe_members(handle.namelist(), staging)
            handle.extractall(staging)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            _safe_members(handle.getnames(), staging)
            handle.extractall(staging)
        return
    raise DownloadError(f"Unsupported archive format: {archive.name}")


class ModelStore:
    """Resolve model bundles and download them on demand."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._entries: dict[str, ModelEntry] = {}

    # -- registration ------------------------------------------------------
    def register(self, entries: list[ModelEntry] | list[dict], capability: str) -> None:
        for raw in entries:
            entry = raw if isinstance(raw, ModelEntry) else _entry_from_dict(raw, capability)
            self._entries[entry.id] = entry

    def load_json(self, path: Path, capability: str, *, override: bool = True) -> None:
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        raw_entries = payload.get("models", payload) if isinstance(payload, dict) else payload
        if not override:
            raw_entries = [e for e in raw_entries if e.get("id") not in self._entries]
        self.register([{**e, "capability": capability} for e in raw_entries], capability)

    # -- lookup ------------------------------------------------------------
    def all(self, capability: str | None = None) -> list[ModelEntry]:
        items = [e for e in self._entries.values() if capability is None or e.capability == capability]
        return sorted(items, key=lambda e: e.id)

    def get(self, model_id: str) -> ModelEntry:
        try:
            return self._entries[model_id]
        except KeyError as error:
            raise ModelAssetNotFound(model_id) from error

    def root(self, capability: str) -> Path:
        if capability == "asr":
            return self.settings.asr_model_dir
        if capability == "v2m":
            return self.settings.v2m_model_dir
        return self.settings.model_dir / capability

    def path_for(self, entry: ModelEntry) -> Path:
        return self.root(entry.capability) / entry.subdir

    # -- presence / download ----------------------------------------------
    @staticmethod
    def _bundle_present(path: Path, entry: ModelEntry) -> bool:
        if not path.is_dir():
            return False
        for rel in entry.required_files:
            if not (path / rel).is_file():
                return False
        for pattern in entry.required_globs:
            if not any(path.glob(pattern)):
                return False
        if entry.required_files or entry.required_globs:
            return True
        return any(path.iterdir())

    def is_downloaded(self, entry: ModelEntry) -> bool:
        return self._bundle_present(self.path_for(entry), entry)

    def ensure(self, entry: ModelEntry, source: str | None = None) -> Path:
        """Return the local path for ``entry``, downloading it when missing."""

        dest = self.path_for(entry)
        if self.is_downloaded(entry):
            return dest
        self._download(entry, dest, source=source)
        if not self.is_downloaded(entry):
            raise DownloadError(
                f"Model {entry.id!r} was downloaded but the expected files are missing "
                f"under {dest}. Check the archive layout or set a custom catalog."
            )
        return dest

    def _download(self, entry: ModelEntry, dest: Path, *, source: str | None) -> None:
        settings = self.settings
        backends = settings.backend_order()
        if source:
            source = source.strip().lower()
            if source not in {"modelscope", "huggingface", "auto"}:
                raise DownloadError(f"Invalid source: {source}")
            backends = ["huggingface", "modelscope"] if source == "huggingface" else ["modelscope", "huggingface"]

        if not entry.archives and not entry.snapshots and not entry.single_files:
            raise DownloadError(
                f"No download source is configured for model {entry.id!r}. "
                f"Place its files in {dest} or set MSST_{entry.capability.upper()}_CATALOG."
            )

        errors: list[str] = []

        # Release-asset archives (e.g. GitHub release zips) are backend-agnostic.
        for archive in entry.archives:
            try:
                self._download_archive(entry, dest, archive, settings)
                logger.info("Model %s ready at %s", entry.id, dest)
                return
            except Exception as error:  # noqa: BLE001 - try next source
                errors.append(f"archive:{archive.get('url')}: {error}")
                logger.warning("Archive download failed for %s: %s", entry.id, error)

        for backend in backends:
            candidates = [s for s in entry.snapshots if s.get("backend") == backend]
            for candidate in candidates:
                try:
                    download_snapshot(
                        candidate["repo_id"],
                        dest,
                        backend=backend,
                        repo_type=candidate.get("repo_type", "model"),
                        allow_patterns=candidate.get("allow_patterns"),
                        settings=settings,
                    )
                    logger.info("Model %s ready at %s", entry.id, dest)
                    return
                except Exception as error:  # noqa: BLE001 - try next source
                    errors.append(f"{backend}:{candidate.get('repo_id')}: {error}")
                    logger.warning("Snapshot download failed for %s: %s", entry.id, error)
            # single-file fallbacks mirror the MSST downloader behaviour
            files = [s for s in entry.single_files if s.get("backend") == backend]
            if files:
                try:
                    self._download_single(entry, dest, files, settings)
                    return
                except Exception as error:  # noqa: BLE001
                    errors.append(f"{backend}:files: {error}")
        raise DownloadError(
            f"Could not download model {entry.id!r}. Tried: {'; '.join(errors) or '(no sources)'}"
        )

    def _download_single(self, entry: ModelEntry, dest: Path, files: list[dict], settings: Settings) -> None:
        from .download import download_model

        for spec in files:
            target = dest / spec.get("file", Path(spec.get("url", "")).name)
            download_model(
                [spec],
                target,
                expected_size=spec.get("size"),
                expected_sha=spec.get("sha256"),
                settings=settings,
            )
        logger.info("Model %s ready at %s", entry.id, dest)

    # -- archive extraction ------------------------------------------------
    def _download_archive(
        self, entry: ModelEntry, dest: Path, archive: dict, settings: Settings
    ) -> None:
        """Download a zip/tar release asset and unpack the model bundle."""

        from .download import download_file

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_archive = dest.parent / f".{entry.subdir}{Path(archive.get('url', '')).suffix or '.zip'}"
        try:
            download_file(
                archive["url"],
                tmp_archive,
                expected_size=archive.get("size"),
                expected_sha=archive.get("sha256"),
                settings=settings,
            )
            staging = Path(tempfile.mkdtemp(prefix=".unpack-", dir=str(dest.parent)))
            try:
                _extract_archive(tmp_archive, staging)
                root = self._locate_bundle_root(staging, entry)
                if dest.exists():
                    shutil.rmtree(dest, ignore_errors=True)
                if root == staging:
                    os.replace(staging, dest)
                else:
                    os.replace(root, dest)
                    shutil.rmtree(staging, ignore_errors=True)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        finally:
            tmp_archive.unlink(missing_ok=True)

    @staticmethod
    def _locate_bundle_root(staging: Path, entry: ModelEntry) -> Path:
        """Find the shallowest directory that holds the expected bundle."""

        candidates = [staging]
        candidates.extend(sorted((p for p in staging.rglob("*") if p.is_dir()), key=lambda p: len(p.parts)))
        if entry.required_files or entry.required_globs:
            for candidate in candidates:
                if ModelStore._bundle_present(candidate, entry):
                    return candidate
        # No explicit requirement: descend through a single wrapping folder.
        while True:
            children = list(staging.iterdir())
            dirs = [child for child in children if child.is_dir()]
            files = [child for child in children if child.is_file()]
            if len(dirs) == 1 and not files:
                staging = dirs[0]
                continue
            return staging

