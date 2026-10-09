"""MSST separator wrapper and an LRU model manager.

This module adapts the ``msst`` library runtime for a long-lived HTTP service:

* :class:`MSSeparator` loads one checkpoint onto a device and separates
  waveforms in memory, exposing inference parameters (batch size, overlap,
  chunk size, normalisation) that the high-level ``msst.Separator`` hides.
* :class:`ModelManager` caches loaded separators keyed by their full
  configuration so repeated requests do not reload multi-gigabyte weights.
"""

from __future__ import annotations

import gc
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from msst.utils.audio_utils import denormalize_audio, normalize_audio
from msst.utils.model_utils import apply_tta, bigshifts_wrapper, prefer_target_instrument
from msst.utils.settings import build_model_from_config, get_model_from_config

from .config import Settings, get_settings

logger = logging.getLogger("msst_api.separator")


class InferenceError(RuntimeError):
    """Raised when a model cannot be loaded or an inference run fails."""


def _config_get(config, section: str, key: str, default=None):
    try:
        node = config[section]
        return node.get(key, default)
    except Exception:  # noqa: BLE001 - config objects vary between model types
        return default


def _config_set(config, section: str, key: str, value) -> None:
    config[section][key] = value


def _state_matches(model: torch.nn.Module, checkpoint: dict) -> bool:
    """True when ``model`` and ``checkpoint`` have identical keys and shapes."""

    if not isinstance(checkpoint, dict):
        return False
    model_state = model.state_dict()
    if set(model_state) != set(checkpoint):
        return False
    for key, tensor in checkpoint.items():
        if not hasattr(tensor, "shape"):
            return False
        if tuple(model_state[key].shape) != tuple(tensor.shape):
            return False
    return True


#: Config fields whose historical meaning changed across releases.  When a
#: checkpoint does not match the configured architecture we rebuild the model
#: trying these alternatives and keep the one that matches exactly.  This is
#: what lets a single catalog serve both legacy and current checkpoints.
_ARCH_VARIANTS: dict[str, list[tuple[str, tuple]]] = {
    "mel_band_roformer": [
        ("mask_estimator_depth", (1, 2, 3, 4)),
        ("mlp_expansion_factor", (2, 4, 8)),
    ],
    "bs_roformer": [
        ("mlp_expansion_factor", (2, 4, 8)),
    ],
    "mel_band_conformer": [
        ("mask_estimator_depth", (1, 2, 3, 4)),
    ],
}


@dataclass(frozen=True)
class ModelSpec:
    """Everything that identifies a loaded model + inference configuration."""

    model_type: str
    config_path: Path
    checkpoint_path: Path
    device: str = "cuda:0"
    use_tta: bool = False
    bigshifts: int = 1
    extract_instrumental: bool = False
    batch_size: Optional[int] = None
    num_overlap: Optional[int] = None
    chunk_size: Optional[int] = None
    normalize: Optional[bool] = None

    @property
    def key(self) -> tuple:
        return (
            str(self.config_path),
            str(self.checkpoint_path),
            self.model_type,
            self.device,
            self.use_tta,
            self.bigshifts,
            self.extract_instrumental,
            self.batch_size,
            self.num_overlap,
            self.chunk_size,
            self.normalize,
        )


def resolve_device(requested: str, settings: Settings) -> str:
    """Map a requested device to a runnable one, honouring the GPU-only policy."""

    wants_cuda = requested.startswith("cuda")
    if wants_cuda and not torch.cuda.is_available():
        if not settings.allow_cpu:
            raise InferenceError(
                "CUDA is not available but GPU-only inference is enforced. "
                "Set MSST_ALLOW_CPU=true to allow CPU fallback."
            )
        logger.warning("CUDA unavailable; falling back to CPU (MSST_ALLOW_CPU=true)")
        return "cpu"
    if not wants_cuda and requested == "cpu" and not settings.allow_cpu:
        raise InferenceError(
            "CPU inference is disabled (MSST_ALLOW_CPU=false); request a CUDA device."
        )
    return requested


class _MonoInputAdapter(torch.nn.Module):
    """Feeds one channel of a (possibly duplicated) waveform to a mono model.

    ``msst.utils.model_utils.demix`` unconditionally expands mono input to two
    channels (``mix.repeat(2, 1)``) and slices the result back to one channel
    afterwards.  That is correct for stereo architectures receiving mono, but
    mono architectures (``model.stereo`` is ``False``) assert that the input
    has exactly one channel, so the expansion trips their forward pass (e.g.
    ``dereverb_room_anvuew_sdr_13.7432.ckpt``).

    This adapter keeps the real model untouched and simply drops the duplicated
    channel before the forward call.  It is a no-op once the upstream bug is
    fixed, because a single-channel input is left as-is.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        # Preserved so ``MSSeparator._coerce_channels`` can still read it.
        self.stereo = bool(getattr(model, "stereo", True))

    def forward(self, x: torch.Tensor, *args, **kwargs):  # noqa: ANN002, ANN003
        return self.model(x[:, :1], *args, **kwargs)


class MSSeparator:
    """Loads one MSST checkpoint and separates in-memory audio."""

    def __init__(self, spec: ModelSpec, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.spec = spec
        self.device = resolve_device(spec.device, self.settings)
        self.model = None
        self.config = None
        self._load()

    # -- loading -----------------------------------------------------------
    def _load(self) -> None:
        config_path = Path(self.spec.config_path)
        checkpoint_path = Path(self.spec.checkpoint_path)
        if not config_path.is_file():
            raise InferenceError(f"Config not found: {config_path}")
        if not checkpoint_path.is_file():
            raise InferenceError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(
            "Loading model_type=%s config=%s checkpoint=%s device=%s",
            self.spec.model_type,
            config_path.name,
            checkpoint_path.name,
            self.device,
        )
        model, config = get_model_from_config(self.spec.model_type, str(config_path))
        # Configs may declare their own model type (e.g. rotated checkpoints).
        resolved_type = _config_get(config, "training", "model_type", self.spec.model_type)
        self.model_type = str(resolved_type or self.spec.model_type)

        self._apply_overrides(config)

        state_dict = self._read_state_dict(checkpoint_path)
        model, config = self._match_checkpoint(model, config, state_dict)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as error:
            raise InferenceError(
                f"Checkpoint {checkpoint_path.name} is incompatible with "
                f"model_type={self.spec.model_type!r}: {error}"
            ) from error

        model = model.to(self.device)
        model.eval()
        if getattr(model, "stereo", True) is False:
            # Work around msst's unconditional mono->stereo expansion in demix.
            model = _MonoInputAdapter(model)
        self.model = model
        self.config = config

    def _read_state_dict(self, checkpoint_path: Path):
        suffix = checkpoint_path.suffix.lower()
        if self.spec.model_type in {"htdemucs", "apollo"}:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        elif suffix == ".safetensors":
            from safetensors.torch import load_file

            return load_file(str(checkpoint_path))
        else:
            try:
                state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            except Exception:  # noqa: BLE001 - legacy checkpoints need full pickle
                state = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
        if isinstance(state, dict):
            for wrapper in ("state", "state_dict", "model_state_dict"):
                if wrapper in state and isinstance(state[wrapper], dict):
                    state = state[wrapper]
                    break
        return state

    def _match_checkpoint(self, model, config, checkpoint):
        """Rebuild ``model`` when the config architecture does not match weights.

        Several pretrained checkpoints predate parameter renames (most notably
        ``mask_estimator_depth`` for mel-band RoFormers).  Rather than shipping
        a hand-tuned config per checkpoint we try the known alternatives and
        keep the architecture whose state dict matches the checkpoint exactly.
        """

        if _state_matches(model, checkpoint):
            return model, config

        variants = _ARCH_VARIANTS.get(self.model_type, [])
        if not variants:
            return model, config

        for field, values in variants:
            original = _config_get(config, "model", field)
            for value in values:
                if value == original:
                    continue
                try:
                    _config_set(config, "model", field, value)
                    candidate = build_model_from_config(self.model_type, config)
                except Exception as error:  # noqa: BLE001
                    logger.debug("Architecture candidate %s=%s rejected: %s", field, value, error)
                    continue
                if _state_matches(candidate, checkpoint):
                    logger.info(
                        "Matched checkpoint architecture: %s=%s (config had %s)",
                        field,
                        value,
                        original,
                    )
                    return candidate, config
            try:
                _config_set(config, "model", field, original)
            except Exception:  # noqa: BLE001
                pass

        logger.warning(
            "No architecture variant matched checkpoint %s; loading as-is",
            self.spec.checkpoint_path,
        )
        return model, config

    def _apply_overrides(self, config) -> None:
        """Merge request-level inference parameters into the model config."""

        if self.spec.batch_size is not None:
            self._set(config, "inference", "batch_size", int(self.spec.batch_size))
        if self.spec.num_overlap is not None:
            self._set(config, "inference", "num_overlap", int(self.spec.num_overlap))
        if self.spec.normalize is not None:
            self._set(config, "inference", "normalize", bool(self.spec.normalize))
        if self.spec.chunk_size is not None:
            chunk = int(self.spec.chunk_size)
            # ``demix`` prefers ``inference.chunk_size`` and falls back to
            # ``audio.chunk_size``; keep both in sync.
            self._set(config, "audio", "chunk_size", chunk)
            self._set(config, "inference", "chunk_size", chunk)

    @staticmethod
    def _set(config, section: str, key: str, value) -> None:
        try:
            _config_set(config, section, key, value)
        except Exception as error:  # noqa: BLE001 - OmegaConf struct configs vary
            logger.warning("Could not override %s.%s: %s", section, key, error)

    @property
    def sample_rate(self) -> int:
        return int(_config_get(self.config, "audio", "sample_rate", 44100))

    @property
    def instruments(self) -> list[str]:
        target = _config_get(self.config, "training", "target_instrument")
        if target:
            return [str(target)]
        return [str(item) for item in _config_get(self.config, "training", "instruments", [])]

    # -- inference ---------------------------------------------------------
    def separate(self, mix: np.ndarray) -> dict[str, np.ndarray]:
        """Separate a ``(channels, samples)`` or ``(samples,)`` waveform.

        Returns a mapping of stem name to ``(samples, channels)`` arrays.
        """

        mix = np.asarray(mix, dtype=np.float32)
        if mix.ndim == 1:
            mix = mix[np.newaxis, :]

        mix = self._coerce_channels(mix)
        mix_orig = mix.copy()

        normalize = bool(_config_get(self.config, "inference", "normalize", False))
        norm_params = None
        if normalize:
            mix, norm_params = normalize_audio(mix)

        with torch.no_grad():
            waveforms = bigshifts_wrapper(
                self.config,
                self.model,
                mix,
                self.device,
                model_type=self.model_type,
                pbar=False,
                bigshifts=max(1, int(self.spec.bigshifts)),
            )
            if isinstance(waveforms, np.ndarray):
                # Demucs returns a bare array when a single stem is produced.
                first = prefer_target_instrument(self.config)[0]
                waveforms = {first: waveforms}
            if self.spec.use_tta:
                waveforms = apply_tta(
                    self.config,
                    self.model,
                    mix,
                    waveforms,
                    self.device,
                    self.model_type,
                    bigshifts=max(1, int(self.spec.bigshifts)),
                    pbar=False,
                )

        results: dict[str, np.ndarray] = {}
        for instr in prefer_target_instrument(self.config):
            estimate = waveforms[instr]
            if norm_params is not None:
                estimate = denormalize_audio(estimate, norm_params)
            results[instr] = np.ascontiguousarray(estimate.T)

        target = _config_get(self.config, "training", "target_instrument")
        if target:
            others = [
                instr
                for instr in _config_get(self.config, "training", "instruments", [])
                if instr != target
            ]
            if others:
                estimate = mix_orig - waveforms[target]
                if norm_params is not None:
                    estimate = denormalize_audio(estimate, norm_params)
                results.setdefault(others[0], np.ascontiguousarray(estimate.T))

        if self.spec.extract_instrumental and "instrumental" not in results:
            base = "vocals" if "vocals" in results else self.instruments[0]
            estimate = mix_orig - waveforms[base]
            if norm_params is not None:
                estimate = denormalize_audio(estimate, norm_params)
            results["instrumental"] = np.ascontiguousarray(estimate.T)

        return results

    def _coerce_channels(self, mix: np.ndarray) -> np.ndarray:
        # Match the channel layout the built model expects.  Configs may omit
        # ``model.stereo``, and several model classes default it to ``False``;
        # assuming ``True`` here would feed stereo audio to a mono model and
        # trip the model's forward assertion.  Reading the value back from the
        # instantiated model keeps the two in sync regardless of the config.
        stereo = bool(getattr(self.model, "stereo", True))
        if stereo and mix.shape[0] == 1:
            return np.repeat(mix, 2, axis=0)
        if stereo and mix.shape[0] > 2:
            mono = mix.mean(axis=0, keepdims=True)
            return np.repeat(mono, 2, axis=0)
        if not stereo and mix.shape[0] > 1:
            return mix.mean(axis=0, keepdims=True)
        return mix

    def close(self) -> None:
        self.model = None
        self.config = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ModelManager:
    """Thread-safe LRU cache of loaded :class:`MSSeparator` instances."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._cache: "OrderedDict[tuple, MSSeparator]" = OrderedDict()
        self._lock = threading.RLock()
        self._semaphore = threading.Semaphore(max(1, self.settings.max_concurrency))

    def get(self, spec: ModelSpec) -> MSSeparator:
        with self._lock:
            cached = self._cache.get(spec.key)
            if cached is not None:
                self._cache.move_to_end(spec.key)
                return cached
            separator = MSSeparator(spec, self.settings)
            self._cache[spec.key] = separator
            self._evict_locked()
            return separator

    def separate(self, spec: ModelSpec, mix: np.ndarray) -> dict[str, np.ndarray]:
        separator = self.get(spec)
        with self._semaphore:
            return separator.separate(mix)

    def _evict_locked(self) -> None:
        limit = max(1, self.settings.max_loaded_models)
        while len(self._cache) > limit:
            _, evicted = self._cache.popitem(last=False)
            logger.info("Evicting cached model to respect MSST_MAX_LOADED_MODELS")
            evicted.close()

    def unload(self) -> None:
        with self._lock:
            for separator in self._cache.values():
                separator.close()
            self._cache.clear()

    @property
    def loaded(self) -> list[str]:
        with self._lock:
            return [sep.spec.checkpoint_path.name for sep in self._cache.values()]
