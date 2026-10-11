"""Device / ONNX Runtime provider selection for the vendored Vocal2Midi engines.

This is a modified copy of Vocal2Midi's ``inference/device_utils.py``.  The
upstream project targets Windows + DirectML; this service targets CUDA + CPU,
so the DirectML/DXGI machinery has been replaced by a small CUDA/CPU provider
resolver that keeps the same public function names used throughout the
vendored code (``normalize_runtime_device``, ``resolve_onnx_providers``,
``use_dml``).

Legacy device spellings are still accepted and normalized so the pipeline code
that passes ``"dml"``/``"gpu"``/``"cuda"`` keeps working.
"""

from __future__ import annotations

import logging

import onnxruntime as ort

logger = logging.getLogger(__name__)


RUNTIME_DEVICE_CHOICES = ("cuda", "cpu")
VISIBLE_RUNTIME_DEVICE_CHOICES = ("cuda", "cpu")

ProviderSpec = str | tuple[str, dict[str, str]]

#: Legacy spellings accepted from upstream configuration.
_DEVICE_ALIASES = {
    "": "cpu",
    "auto": "cuda",
    "gpu": "cuda",
    "cuda": "cuda",
    "cuda:0": "cuda",
    "dml": "cuda",
    "directml": "cuda",
    "metal": "cpu",  # Metal is not available in this service; CPU is the fallback.
    "cpu": "cpu",
}


def _cuda_provider_available() -> bool:
    try:
        return "CUDAExecutionProvider" in set(ort.get_available_providers())
    except Exception:  # noqa: BLE001 - onnxruntime always exposes this, but be safe
        return False


def default_runtime_device() -> str:
    """Return the accelerated backend when a CUDA ONNX Runtime is installed."""

    return "cuda" if _cuda_provider_available() else "cpu"


def normalize_runtime_device(device: str | None, default: str | None = None) -> str:
    """Map a requested device to ``"cuda"`` or ``"cpu"``."""

    fallback = default if default is not None else default_runtime_device()
    value = str(fallback if device is None else device).strip().lower()
    if not value:
        value = fallback
    if value.startswith("cuda"):
        return "cuda"
    return _DEVICE_ALIASES.get(value, value)


def resolve_onnx_providers(
    device: str | None, *, label: str = "ONNX"
) -> tuple[str, list[ProviderSpec]]:
    """Return ``(provider_name, provider_specs)`` for an ONNX session."""

    normalized = normalize_runtime_device(device)
    if normalized == "cpu":
        return "cpu", ["CPUExecutionProvider"]

    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in available:
        logger.info("[%s] Using CUDAExecutionProvider.", label)
        # kSameAsRequested avoids the BFC arena rounding each allocation up to
        # the next power of two (kNextPowerOfTwo), which otherwise inflates
        # peak VRAM for the large attention buffers in GAME/TiFA and can OOM.
        return "cuda", [
            ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
            "CPUExecutionProvider",
        ]

    logger.info(
        "[%s] CUDAExecutionProvider unavailable; falling back to CPUExecutionProvider.",
        label,
    )
    return "cpu", ["CPUExecutionProvider"]


def use_dml(device: str | None) -> bool:
    """Compatibility shim: upstream used this to mean "use the GPU provider"."""

    if normalize_runtime_device(device) == "cpu":
        return False
    return _cuda_provider_available()
