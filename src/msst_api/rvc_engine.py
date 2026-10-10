"""High-level RVC voice-conversion service layer.

This wraps the vendored :class:`msst_api.rvc.infer.convert.VoiceConverter` in the
same long-lived-service shape as :mod:`msst_api.separator`:

* :class:`RVCVoiceLibrary` discovers voice models (flat ``*.pth`` files with an
  optional sibling ``*.index``) under ``MSST_RVC_MODEL_DIR``;
* :class:`RVCManager` keeps loaded converters resident with LRU eviction so
  repeated requests do not reload the weights.

The heavy RVC engine is imported lazily so model listing (and the HTTP app)
works even before the optional engine dependencies are installed.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .config import Settings, get_settings
from .resources import SharedModelCache

logger = logging.getLogger("msst_api.rvc")


class RVCError(RuntimeError):
    """Raised when a voice model cannot be found/loaded or a conversion fails."""


class RVCVoiceNotFoundError(RVCError):
    """Raised when a requested voice id does not match any installed model."""


# ---------------------------------------------------------------------------
# Voice discovery
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RVCVoice:
    """A single RVC voice model on disk."""

    id: str
    name: str
    path: Path
    index_path: Optional[Path]
    size: Optional[int]

    @property
    def has_index(self) -> bool:
        return self.index_path is not None

    def to_public_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "has_index": self.has_index,
            "size": self.size,
            "downloaded": self.path.is_file(),
        }


class RVCVoiceLibrary:
    """Scans ``MSST_RVC_MODEL_DIR`` for flat ``*.pth`` voice models."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)

    def _scan(self) -> list[RVCVoice]:
        if not self.model_dir.is_dir():
            return []
        voices: list[RVCVoice] = []
        for path in sorted(self.model_dir.glob("*.pth")):
            if not path.is_file():
                continue
            index_path = path.with_suffix(".index")
            voices.append(
                RVCVoice(
                    id=path.stem,
                    name=path.name,
                    path=path,
                    index_path=index_path if index_path.is_file() else None,
                    size=path.stat().st_size if path.is_file() else None,
                )
            )
        return voices

    def all(self) -> list[RVCVoice]:
        return self._scan()

    def get(self, voice: str) -> RVCVoice:
        """Resolve a user-supplied voice id/name to an installed model."""

        value = (voice or "").strip().strip('"').strip("'")
        if not value:
            raise RVCVoiceNotFoundError("Voice id is required")

        voices = self._scan()
        if not voices:
            raise RVCVoiceNotFoundError(
                f"No RVC models installed in {self.model_dir}"
            )

        # 1) exact file name or stem.
        for candidate in voices:
            if candidate.name == value or candidate.id == value:
                return candidate

        # 2) unique match by stem, then by substring.
        stem = Path(value).stem.lower()
        by_stem = [v for v in voices if v.id.lower() == stem]
        if len(by_stem) == 1:
            return by_stem[0]
        by_substring = [v for v in voices if stem and stem in v.id.lower()]
        if len(by_substring) == 1:
            return by_substring[0]
        if len(by_substring) > 1:
            names = ", ".join(v.name for v in by_substring[:10])
            raise RVCVoiceNotFoundError(f"Ambiguous voice {value!r}. Matches: {names}")

        available = ", ".join(v.name for v in voices[:20]) or "(none)"
        raise RVCVoiceNotFoundError(f"Voice {value!r} not found. Available: {available}")

    def resolve_index(self, value: str) -> Optional[Path]:
        """Resolve an index name to a ``*.index`` file inside the model dir."""

        value = (value or "").strip().strip('"').strip("'")
        if not value:
            return None
        if not self.model_dir.is_dir():
            return None

        indexes = sorted(p for p in self.model_dir.glob("*.index") if p.is_file())
        # Exact name, then stem, then unique substring.
        for path in indexes:
            if path.name == value or path.stem == value:
                return path
        stem = Path(value).stem.lower()
        by_stem = [p for p in indexes if p.stem.lower() == stem]
        if len(by_stem) == 1:
            return by_stem[0]
        by_substring = [p for p in indexes if stem and stem in p.stem.lower()]
        if len(by_substring) == 1:
            return by_substring[0]
        if len(by_substring) > 1:
            names = ", ".join(p.name for p in by_substring[:10])
            raise RVCVoiceNotFoundError(f"Ambiguous index {value!r}. Matches: {names}")
        available = ", ".join(p.name for p in indexes[:20]) or "(none)"
        raise RVCVoiceNotFoundError(f"Index {value!r} not found. Available: {available}")


# ---------------------------------------------------------------------------
# Conversion specification + parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RVCVoiceSpec:
    """Identity of a loaded voice + device + embedder (the cache key)."""

    model_path: Path
    index_path: Optional[Path]
    device: str
    embedder: str = "contentvec"
    embedder_custom: Optional[str] = None

    @property
    def key(self) -> tuple:
        return (
            str(self.model_path),
            str(self.index_path or ""),
            self.device,
            self.embedder,
            self.embedder_custom or "",
        )


@dataclass
class RVCParams:
    """Per-request conversion parameters."""

    pitch: int = 0
    f0_method: str = "rmvpe"
    index_rate: float = 0.75
    volume_envelope: float = 1.0
    protect: float = 0.5
    split_audio: bool = False
    f0_autotune: bool = False
    f0_autotune_strength: float = 1.0
    clean_audio: bool = False
    clean_strength: float = 0.5
    resample_sr: int = 0
    sid: int = 0
    proposed_pitch: bool = False
    proposed_pitch_threshold: float = 155.0
    post_process: bool = False
    effects: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
class RVCManager:
    """Thread-safe LRU cache of loaded :class:`VoiceConverter` instances.

    Uses the same :class:`~msst_api.resources.SharedModelCache` as the MSST
    manager so ``MSST_MAX_LOADED_MODELS`` and ``MSST_MAX_CONCURRENCY`` are
    global budgets across both subsystems.
    """

    def __init__(
        self, settings: Settings | None = None, cache: SharedModelCache | None = None
    ) -> None:
        self.settings = settings or get_settings()
        from .rvc.lib.assets import set_assets_dir

        set_assets_dir(self.settings.rvc_asset_dir)
        self.library = RVCVoiceLibrary(self.settings.rvc_model_dir)
        self._cache = cache or SharedModelCache(
            self.settings.max_loaded_models, self.settings.max_concurrency
        )
        self._lock = threading.RLock()

    # -- loading -----------------------------------------------------------
    def _build(self, spec: RVCVoiceSpec):
        from .rvc.infer.convert import RVCConversionError, VoiceConverter

        logger.info(
            "Loading RVC voice=%s index=%s device=%s embedder=%s",
            spec.model_path.name,
            spec.index_path.name if spec.index_path else "-",
            spec.device,
            spec.embedder,
        )
        try:
            converter = VoiceConverter(spec.device)
            converter.get_vc(str(spec.model_path), 0)
            if converter.vc is None:
                raise RVCConversionError(f"Failed to load {spec.model_path.name}")
            # Warm the embedder so the first conversion does not pay for it.
            converter.load_hubert(spec.embedder, spec.embedder_custom)
            converter.last_embedder_model = spec.embedder
        except RVCConversionError:
            raise
        except Exception as error:  # noqa: BLE001 - surface a clean service error
            raise RVCError(f"Could not load RVC voice {spec.model_path.name}: {error}") from error
        return converter

    def get(self, spec: RVCVoiceSpec):
        cached = self._cache.get(spec.key)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._cache.get(spec.key)
            if cached is not None:
                return cached
            converter = self._build(spec)
            return self._cache.put(
                spec.key,
                converter,
                namespace="rvc",
                label=spec.model_path.name,
                closer=lambda obj: obj.cleanup_model(),
            )

    # -- inference ---------------------------------------------------------
    def convert(
        self, spec: RVCVoiceSpec, audio: np.ndarray, params: RVCParams
    ) -> tuple[np.ndarray, int]:
        converter = self.get(spec)
        with self._cache.inference_slot():
            return converter.convert(
                audio,
                model_path=str(spec.model_path),
                index_path=str(spec.index_path or ""),
                pitch=params.pitch,
                f0_method=params.f0_method,
                index_rate=params.index_rate,
                volume_envelope=params.volume_envelope,
                protect=params.protect,
                split_audio=params.split_audio,
                f0_autotune=params.f0_autotune,
                f0_autotune_strength=params.f0_autotune_strength,
                embedder_model=spec.embedder,
                embedder_model_custom=spec.embedder_custom,
                clean_audio=params.clean_audio,
                clean_strength=params.clean_strength,
                resample_sr=params.resample_sr,
                sid=params.sid,
                proposed_pitch=params.proposed_pitch,
                proposed_pitch_threshold=params.proposed_pitch_threshold,
                post_process=params.post_process,
                **params.effects,
            )

    # -- lifecycle ---------------------------------------------------------
    def unload(self) -> None:
        self._cache.clear("rvc")

    @property
    def loaded(self) -> list[str]:
        return self._cache.loaded("rvc")
