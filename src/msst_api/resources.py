"""Shared inference resources for the MSST and RVC subsystems.

Both the MSST separator and the RVC converter are long-lived, GPU-resident
model caches.  To honour ``MSST_MAX_LOADED_MODELS`` and ``MSST_MAX_CONCURRENCY``
as *global* budgets rather than per-subsystem ones, both managers register their
loaded models in a single :class:`SharedModelCache`:

* the LRU eviction is global, so at most ``MSST_MAX_LOADED_MODELS`` models stay
  resident across MSST **and** RVC combined;
* a single semaphore bounds the total number of concurrent inferences across
  both subsystems to ``MSST_MAX_CONCURRENCY``.

Entries are namespaced (``"msst"`` / ``"rvc"``) so each manager can still report
and unload only its own models.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Callable, Iterator

logger = logging.getLogger("msst_api.resources")


class _Entry:
    __slots__ = ("obj", "label", "namespace", "closer")

    def __init__(
        self, obj: object, label: str, namespace: str, closer: Callable[[object], None]
    ) -> None:
        self.obj = obj
        self.label = label
        self.namespace = namespace
        self.closer = closer


class SharedModelCache:
    """Global LRU of loaded models plus a shared inference semaphore."""

    def __init__(self, max_loaded_models: int = 1, max_concurrency: int = 1) -> None:
        self._lock = threading.RLock()
        self._cache: "OrderedDict[tuple, _Entry]" = OrderedDict()
        self._max_loaded = max(1, int(max_loaded_models))
        self._semaphore = threading.Semaphore(max(1, int(max_concurrency)))

    # -- concurrency -------------------------------------------------------
    @contextmanager
    def inference_slot(self) -> Iterator[None]:
        """Bound concurrent inference across all subsystems."""

        self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()

    # -- cache -------------------------------------------------------------
    def get(self, key: tuple) -> object | None:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            self._cache.move_to_end(key)
            return entry.obj

    def put(
        self,
        key: tuple,
        obj: object,
        *,
        namespace: str,
        label: str,
        closer: Callable[[object], None],
    ) -> object:
        with self._lock:
            self._cache[key] = _Entry(obj, label, namespace, closer)
            self._cache.move_to_end(key)
            self._evict_locked()
        return obj

    def _evict_locked(self) -> None:
        while len(self._cache) > self._max_loaded:
            _, entry = self._cache.popitem(last=False)
            logger.info(
                "Evicting %s model %s to respect MSST_MAX_LOADED_MODELS",
                entry.namespace,
                entry.label,
            )
            self._close(entry)

    @staticmethod
    def _close(entry: _Entry) -> None:
        try:
            entry.closer(entry.obj)
        except Exception:  # noqa: BLE001 - best effort cleanup
            logger.exception("Error closing evicted model %s", entry.label)

    def loaded(self, namespace: str | None = None) -> list[str]:
        with self._lock:
            return [
                entry.label
                for entry in self._cache.values()
                if namespace is None or entry.namespace == namespace
            ]

    def clear(self, namespace: str | None = None) -> None:
        with self._lock:
            if namespace is None:
                entries = list(self._cache.values())
                self._cache.clear()
            else:
                keys = [
                    key
                    for key, entry in self._cache.items()
                    if entry.namespace == namespace
                ]
                entries = [self._cache.pop(key) for key in keys]
        for entry in entries:
            self._close(entry)
