"""Vocal2Midi model manager.

Owns the v2m model catalog + store and registers itself (and the shared model
cache) with :mod:`msst_api.v2m.runtime` so every vendored ``load_*`` helper
routes through the process-wide LRU / concurrency budget.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import Settings, get_settings
from ..model_store import ModelAssetNotFound, ModelEntry, ModelStore
from ..resources import SharedModelCache
from . import runtime
from .catalog import V2M_CAPABILITY, build_store

logger = logging.getLogger("msst_api.v2m.manager")


class V2MError(RuntimeError):
    """Raised when a v2m model is missing or the pipeline fails."""


class V2MModelNotFoundError(V2MError):
    """Raised when a requested v2m model id is not in the catalog."""


class V2MManager:
    def __init__(
        self,
        settings: Settings | None = None,
        store: ModelStore | None = None,
        cache: SharedModelCache | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or build_store(self.settings)
        self._cache = cache or SharedModelCache(
            self.settings.max_loaded_models, self.settings.max_concurrency
        )
        runtime.configure(settings=self.settings, cache=self._cache, v2m_manager=self)

    # -- catalog -----------------------------------------------------------
    def models(self) -> list[ModelEntry]:
        return self.store.all(V2M_CAPABILITY)

    def get_entry(self, model_id: str) -> ModelEntry:
        try:
            return self.store.get(model_id)
        except ModelAssetNotFound as error:
            available = ", ".join(e.id for e in self.models()) or "(none)"
            raise V2MModelNotFoundError(
                f"Unknown v2m model {model_id!r}. Available: {available}"
            ) from error

    def is_downloaded(self, entry: ModelEntry) -> bool:
        return self.store.is_downloaded(entry)

    def resolve_path(self, model_id: str, source: str | None = None) -> Path:
        entry = self.get_entry(model_id)
        try:
            return self.store.ensure(entry, source=source)
        except Exception as error:  # noqa: BLE001
            raise V2MError(
                f"Could not prepare v2m model {model_id!r}: {error}"
            ) from error

    def path_or_none(self, model_id: str) -> Path | None:
        entry = self.get_entry(model_id)
        path = self.store.path_for(entry)
        return path if self.store.is_downloaded(entry) else None

    # -- lifecycle ---------------------------------------------------------
    def unload(self) -> None:
        self._cache.clear("v2m")

    @property
    def loaded(self) -> list[str]:
        return self._cache.loaded("v2m")

    @property
    def cache(self) -> SharedModelCache:
        return self._cache
