"""ASR model manager: lazy loading under the shared LRU / concurrency budget.

Every engine is registered in the process-wide
:class:`~msst_api.resources.SharedModelCache`, so:

* at most ``MSST_MAX_LOADED_MODELS`` ASR / MSST / RVC / v2m models stay resident
  combined (LRU eviction);
* a single semaphore bounds concurrent inference across all subsystems
  (``MSST_MAX_CONCURRENCY``);
* a model is only loaded when a request actually needs it.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Settings, get_settings
from ..model_store import ModelAssetNotFound, ModelEntry, ModelStore
from ..resources import SharedModelCache
from .catalog import ASR_CAPABILITY, build_store
from .engines import (
    ASREngine,
    ASRResult,
    StreamingSession,
    build_engine,
)

logger = logging.getLogger("msst_api.asr.manager")


class ASRError(RuntimeError):
    """Raised when an ASR model cannot be loaded or a transcription fails."""


class ASRModelNotFoundError(ASRError):
    """Raised when a requested ASR model id is not in the catalog."""


@dataclass(frozen=True)
class ASRModelSpec:
    """Cache identity of a loaded ASR engine."""

    model_id: str
    kind: str
    path: Path
    device: str
    streaming: bool = False

    @property
    def key(self) -> tuple:
        return (
            "asr",
            self.kind,
            str(self.path),
            self.device,
            "stream" if self.streaming else "batch",
        )


class _ManagedStreamingSession:
    """Bounds each streaming decode step by the shared inference semaphore."""

    def __init__(self, session: StreamingSession, cache: SharedModelCache) -> None:
        self._session = session
        self._cache = cache

    def feed(self, pcm16k: np.ndarray) -> ASRResult:
        with self._cache.inference_slot():
            return self._session.feed(pcm16k)

    def finish(self) -> ASRResult:
        with self._cache.inference_slot():
            return self._session.finish()

    def close(self) -> None:
        self._session.close()


class ASRManager:
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
        self._lock = threading.RLock()

    # -- catalog -----------------------------------------------------------
    def models(self) -> list[ModelEntry]:
        return self.store.all(ASR_CAPABILITY)

    def get_entry(self, model_id: str) -> ModelEntry:
        try:
            return self.store.get(model_id)
        except ModelAssetNotFound as error:
            available = ", ".join(e.id for e in self.models()) or "(none)"
            raise ASRModelNotFoundError(
                f"Unknown ASR model {model_id!r}. Available: {available}"
            ) from error

    def is_downloaded(self, entry: ModelEntry) -> bool:
        return self.store.is_downloaded(entry)

    def ensure_path(self, model_id: str | None = None, source: str | None = None) -> tuple[ModelEntry, Path]:
        model_id = (model_id or self.settings.asr_default_model).strip()
        entry = self.get_entry(model_id)
        try:
            path = self.store.ensure(entry, source=source)
        except Exception as error:  # noqa: BLE001 - surface a clean service error
            raise ASRError(f"Could not prepare ASR model {model_id!r}: {error}") from error
        return entry, path

    # -- engine cache ------------------------------------------------------
    def _get_engine(self, spec: ASRModelSpec) -> ASREngine:
        cached = self._cache.get(spec.key)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._cache.get(spec.key)
            if cached is not None:
                return cached
            try:
                engine = build_engine(spec.kind, spec.path, spec.device, self.settings)
            except Exception as error:  # noqa: BLE001
                raise ASRError(f"Could not load ASR engine {spec.kind}: {error}") from error
            return self._cache.put(
                spec.key,
                engine,
                namespace="asr",
                label=f"{spec.kind}:{spec.path.name}",
                closer=lambda obj: obj.close(),
            )

    def _engine_for(self, entry: ModelEntry, path: Path, device: str) -> ASREngine:
        return self._get_engine(
            ASRModelSpec(model_id=entry.id, kind=entry.kind, path=Path(path), device=device)
        )

    # -- inference ---------------------------------------------------------
    def transcribe(
        self,
        model_id: str | None,
        audio: np.ndarray,
        *,
        language: str | None = None,
        context: str | None = None,
        device: str | None = None,
        source: str | None = None,
    ) -> ASRResult:
        model_id = (model_id or self.settings.asr_default_model).strip()
        entry, path = self.ensure_path(model_id, source=source)
        resolved_device = device or self.settings.device
        engine = self._engine_for(entry, path, resolved_device)
        with self._cache.inference_slot():
            return engine.transcribe(audio, language=language, context=context)

    def transcribe_path(
        self,
        kind: str,
        path: Path,
        audio: np.ndarray,
        *,
        language: str | None = None,
        context: str | None = None,
        device: str | None = None,
    ) -> ASRResult:
        """Transcribe using an already-resolved model directory (v2m pipeline)."""

        resolved_device = device or self.settings.device
        engine = self._get_engine(
            ASRModelSpec(model_id=f"path:{Path(path).name}", kind=kind, path=Path(path), device=resolved_device)
        )
        with self._cache.inference_slot():
            return engine.transcribe(audio, language=language, context=context)

    def open_stream(
        self,
        model_id: str | None,
        *,
        language: str | None = None,
        context: str | None = None,
        device: str | None = None,
        source: str | None = None,
    ) -> StreamingSession:
        model_id = (model_id or self.settings.asr_default_model).strip()
        entry, path = self.ensure_path(model_id, source=source)
        if entry.kind not in {"qwen3", "qwen"}:
            raise ASRError(
                f"Model {entry.id!r} ({entry.kind}) does not support streaming."
            )
        resolved_device = device or self.settings.device
        backend = self.settings.asr_streaming_backend
        use_vllm = backend in {"auto", "vllm"} and self._vllm_available()
        if backend == "vllm" and not use_vllm:
            raise ASRError(
                "MSST_ASR_STREAMING_BACKEND=vllm but vLLM is not installed. "
                "Install it with: pip install 'qwen-asr[vllm]'"
            )
        if use_vllm:
            model = self._get_streaming_model(entry, path, resolved_device)
            return _ManagedStreamingSession(
                model.new_session(language=language, context=context), self._cache
            )
        engine = self._engine_for(entry, path, resolved_device)
        return _ManagedStreamingSession(
            engine.new_chunked_session(language=language, context=context), self._cache
        )

    def _get_streaming_model(self, entry: ModelEntry, path: Path, device: str):
        from .engines import Qwen3ASRVLLMStreamingModel

        spec = ASRModelSpec(
            model_id=entry.id, kind=entry.kind, path=Path(path), device=device, streaming=True
        )
        cached = self._cache.get(spec.key)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._cache.get(spec.key)
            if cached is not None:
                return cached
            try:
                model = Qwen3ASRVLLMStreamingModel(Path(path), device, self.settings)
            except Exception as error:  # noqa: BLE001
                raise ASRError(f"Could not load streaming ASR model {entry.id!r}: {error}") from error
            return self._cache.put(
                spec.key,
                model,
                namespace="asr",
                label=f"stream:{entry.id}",
                closer=lambda obj: obj.close(),
            )

    @staticmethod
    def _vllm_available() -> bool:
        try:
            import vllm  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    # -- lifecycle ---------------------------------------------------------
    def unload(self) -> None:
        self._cache.clear("asr")

    @property
    def loaded(self) -> list[str]:
        return self._cache.loaded("asr")
