"""Process-wide runtime wiring for the v2m capability.

The vendored Vocal2Midi runtimes own their own ``load_*`` helpers.  To make
every v2m model honour the project's global ``MSST_MAX_LOADED_MODELS`` /
``MSST_MAX_CONCURRENCY`` budgets, those helpers are redirected here, where a
single :class:`~msst_api.resources.SharedModelCache` (the same instance used by
MSST / RVC / ASR) backs a small ``get_or_load`` helper.

When the runtime has not been configured (e.g. a vendored module imported
standalone in a unit test) a private fallback cache is created lazily so the
loaders keep working.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable

from ..config import Settings, get_settings
from ..resources import SharedModelCache

logger = logging.getLogger("msst_api.v2m.runtime")

_lock = threading.RLock()
_fallback_cache: SharedModelCache | None = None
_state: dict[str, Any] = {"settings": None, "cache": None, "asr": None, "v2m": None}


def configure(
    *,
    settings: Settings | None = None,
    cache: SharedModelCache | None = None,
    asr_manager: Any = None,
    v2m_manager: Any = None,
) -> None:
    with _lock:
        if settings is not None:
            _state["settings"] = settings
        if cache is not None:
            _state["cache"] = cache
        if asr_manager is not None:
            _state["asr"] = asr_manager
        if v2m_manager is not None:
            _state["v2m"] = v2m_manager


def settings() -> Settings:
    return _state["settings"] or get_settings()


def cache() -> SharedModelCache:
    global _fallback_cache
    with _lock:
        if _state["cache"] is None:
            if _fallback_cache is None:
                s = settings()
                _fallback_cache = SharedModelCache(s.max_loaded_models, s.max_concurrency)
            return _fallback_cache
        return _state["cache"]


def asr_manager() -> Any:
    manager = _state["asr"]
    if manager is None:
        raise RuntimeError("The ASR manager has not been configured for the v2m runtime.")
    return manager


def v2m_manager() -> Any:
    manager = _state["v2m"]
    if manager is None:
        raise RuntimeError("The v2m manager has not been configured.")
    return manager


@contextmanager
def inference_slot():
    with cache().inference_slot():
        yield


def get_or_load(
    key: tuple,
    *,
    namespace: str,
    label: str,
    builder: Callable[[], Any],
    closer: Callable[[Any], None],
) -> Any:
    """Return a cached model, building (and LRU-registering) it on demand."""

    shared = cache()
    cached = shared.get(key)
    if cached is not None:
        return cached
    with _lock:
        cached = shared.get(key)
        if cached is not None:
            return cached
        # Release an older model *before* building this one so two models never
        # overlap on the GPU (honours MSST_MAX_LOADED_MODELS).
        shared.reserve()
        model = builder()
        return shared.put(key, model, namespace=namespace, label=label, closer=closer)


def unload(namespace: str | None = None) -> None:
    cache().clear(namespace)


def build_rmvpe_transcriber(model_path: str, device: str | None):
    """Return a loaded RMVPE transcriber, reusing the RVC predictor by default."""

    settings_obj = settings()
    backend = getattr(settings_obj, "v2m_rmvpe_backend", "rvc") or "rvc"
    if backend == "onnx":
        from ._vendor.inference.API.rmvpe_api import RmvpeTranscriber

        if not model_path:
            raise RuntimeError(
                "MSST_V2M_RMVPE_BACKEND=onnx requires an RMVPE .onnx model "
                "(place it in MSST_V2M_MODEL_DIR/RMVPE/)."
            )
        return get_or_load(
            ("rmvpe", "onnx", str(model_path), str(device or "")),
            namespace="v2m",
            label=f"rmvpe-onnx:{model_path}",
            builder=lambda: RmvpeTranscriber(model_path, device=device),
            closer=lambda model: model.shutdown(),
        )

    from .rmvpe import RvcRmvpeTranscriber

    return get_or_load(
        ("rmvpe", "rvc", str(device or "")),
        namespace="v2m",
        label="rmvpe:rvc",
        builder=lambda: RvcRmvpeTranscriber(device, settings_obj),
        closer=lambda model: model.shutdown(),
    )
